"""PPO in torch over the warp Env: rollout, GAE, clipped-surrogate updates.

Throughput notes: the policy sample and post-step bookkeeping are torch.compile'd, so no
eager torch ops sit between warp launches in the rollout. The minibatch update is a
torch.compile'd loss with autograd and a fused Adam; the GEMMs run in bf16 (fp16 on
pre-bf16 GPUs) under autocast (fp32 master weights, ~2x tensor-core rate). The only
blocking reads are the per-epoch KL check and the per-iteration log scalars; episode
stats are recorded into GPU ring buffers and shipped to pinned host memory once per
iteration.

The env self-resets on done, so the observation after a done step is the fresh spawn of a
new episode — the value bootstrap is zeroed there in GAE."""

from collections import deque

import numpy as np
import torch

from warporacer.agent import LOGSTD_MAX, LOGSTD_MIN
from warporacer.sim import ACT_DIM, OBS_DIM

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
VF_CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
LR_MIN, LR_MAX, LR_FACTOR = 1e-6, 3e-3, 1.5
NORM_CLIP = 10.0
HALF_LOG_2PI = float(0.5 * np.log(2.0 * np.pi))


class RunningMeanStd:
    """All state lives in device tensors (count included) so updates fuse under compile."""

    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.inv_std = torch.ones(shape, device=device)
        self.count = torch.full((), 1e-4, device=device)

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = float(x.shape[0])
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta * (bc / tot))
        self.var.copy_((self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)) / tot)
        self.count.copy_(tot)
        self.inv_std.copy_(torch.rsqrt(self.var + 1e-8))

    def normalize(self, x):
        return ((x - self.mean) * self.inv_std).clamp(-NORM_CLIP, NORM_CLIP)


class ReturnNormalizer:
    """Scales rewards by the running std of the discounted return."""

    def __init__(self, num_envs, device):
        self.returns = torch.zeros(num_envs, device=device)
        self.rms = RunningMeanStd((), device)

    def normalize(self, reward, done):
        self.returns.copy_(self.returns * GAMMA * (1.0 - done) + reward)
        self.rms.update(self.returns)
        return reward * self.rms.inv_std


@torch.jit.script
def _gae(rew_b, done_b, val_ext, gamma: float, lam: float):
    live = 1.0 - done_b
    deltas = rew_b + gamma * val_ext[1:] * live - val_ext[:-1]
    adv_b = torch.empty_like(rew_b)
    last = torch.zeros_like(val_ext[0])
    for t in range(rew_b.shape[0] - 1, -1, -1):
        last = deltas[t] + gamma * lam * live[t] * last
        adv_b[t] = last
    return adv_b


class PPO:
    def __init__(self, env, agent, rollouts: int = 24, epochs: int = 5, minibatches: int = 4,
                 lr: float = 3e-4, compile: bool = True):
        assert (rollouts * env.num_envs) % minibatches == 0
        self.env, self.agent = env, agent
        self.epochs, self.minibatches = epochs, minibatches
        self.global_step = 0
        self.iteration = 0
        T, N = rollouts, env.num_envs
        self.T, self.N = T, N
        self.batch_size = T * N
        dev = torch.device(env.torch_device)
        self.cuda = dev.type == "cuda"
        # Autocast the loss GEMMs on any GPU: bf16 where the hardware runs it natively
        # (Ampere+), else fp16 (T4 5x over its emulated-bf16 fallback, mps 1.8x over
        # fp32). fp16 only gives up exponent range vs bf16, which this normalized/
        # clipped loss never needs. Masters stay fp32 either way; no grad scaler.
        self.amp_dev = dev.type
        self.amp = dev.type != "cpu"
        self.amp_dtype = (
            torch.bfloat16
            if self.cuda and torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float16
        )
        self.lr = lr
        self.opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5,
                                    fused=dev.type in ("cuda", "mps"))

        # Rollout buffers; obs_ext row T holds the bootstrap observation so the value
        # pass runs over all T+1 rows without a cat.
        self.obs_ext = torch.zeros((T + 1, N, OBS_DIM), device=dev)
        self.obs_b = self.obs_ext[:T]
        self.act_b = torch.zeros((T, N, ACT_DIM), device=dev)
        self.logp_b = torch.zeros((T, N), device=dev)
        self.rew_b = torch.zeros((T, N), device=dev)
        self.done_b = torch.zeros((T, N), device=dev)
        self.obs = torch.zeros((N, OBS_DIM), device=dev)  # current normalized observation
        self.t_idx = torch.arange(T, device=dev)

        self.obs_rms = RunningMeanStd((OBS_DIM,), dev)
        self.ret_rms = ReturnNormalizer(N, dev)

        # Episode stats: GPU ring buffers, copied to pinned host memory once per iteration
        # (event-synced before reading) so the rollout never blocks on .any()/.item().
        self.ep_ret = torch.zeros(N, device=dev)
        self.ep_len = torch.zeros(N, device=dev)
        self.fin_hist = torch.zeros((T, N), dtype=torch.bool, device=dev)
        self.ret_hist = torch.zeros((T, N), device=dev)
        self.len_hist = torch.zeros((T, N), device=dev)
        pin = self.cuda
        self.fin_cpu = torch.zeros((T, N), dtype=torch.bool, pin_memory=pin)
        self.ret_cpu = torch.zeros((T, N), pin_memory=pin)
        self.len_cpu = torch.zeros((T, N), pin_memory=pin)
        self.copy_done = torch.cuda.Event() if self.cuda else None
        self.fin_rets, self.fin_lens = deque(maxlen=100), deque(maxlen=100)

        self._policy = self._policy_step
        self._process = self._process_step
        self._loss = self._loss_fn
        if compile and self.cuda:
            from torch._inductor import config as _inductor_config

            # Blackwell triton miscompiles the fused bookkeeping kernel with persistent
            # reductions ("PassManager::run failed"); disabling them is near-free here.
            _inductor_config.triton.persistent_reductions = False
            self._policy = torch.compile(self._policy_step)
            self._process = torch.compile(self._process_step)
            self._loss = torch.compile(self._loss_fn)

        self.reset_env_stats()

    def reset_env_stats(self):
        """(Re)sync with the env after construction or a map rotation."""
        self.obs_rms.update(self.env.obs)
        self.obs.copy_(self.obs_rms.normalize(self.env.obs))
        self.ep_ret.zero_()
        self.ep_len.zero_()
        self.ret_rms.returns.zero_()

    def _policy_step(self, t):
        """Sample the policy for step t and record obs/act/logp into the ring buffers."""
        obs = self.obs
        self.obs_b.index_copy_(0, t, obs.unsqueeze(0))
        mean = self.agent.actor(obs)
        ls = self.agent.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX)
        noise = torch.randn_like(mean)
        act = mean + noise * ls.exp()
        logp = (-0.5 * noise.square() - ls - HALF_LOG_2PI).sum(-1)
        self.act_b.index_copy_(0, t, act.unsqueeze(0))
        self.logp_b.index_copy_(0, t, logp.unsqueeze(0))
        return act

    def _process_step(self, raw, rew, done_i, t):
        """Post-step bookkeeping: reward normalization, episode stats, obs normalization."""
        done = done_i.float()
        live = 1.0 - done
        self.rew_b.index_copy_(0, t, self.ret_rms.normalize(rew, done).unsqueeze(0))
        self.done_b.index_copy_(0, t, done.unsqueeze(0))
        self.ep_ret += rew
        self.ep_len += 1.0
        self.fin_hist.index_copy_(0, t, (done_i != 0).unsqueeze(0))
        self.ret_hist.index_copy_(0, t, self.ep_ret.unsqueeze(0))
        self.len_hist.index_copy_(0, t, self.ep_len.unsqueeze(0))
        self.ep_ret *= live
        self.ep_len *= live
        self.obs_rms.update(raw)
        self.obs.copy_(self.obs_rms.normalize(raw))

    def _loss_fn(self, o, a, logp_old, adv, ret, v_old):
        """Clipped-surrogate PPO loss; GEMMs in bf16/fp16 under autocast (fp32 masters)."""
        with torch.autocast(self.amp_dev, dtype=self.amp_dtype, enabled=self.amp):
            mean = self.agent.actor(o)
            ls = self.agent.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX)
            z = (a - mean.float()) / ls.exp()
            new_logp = (-0.5 * z.square() - ls - HALF_LOG_2PI).sum(-1)
            logratio = new_logp - logp_old
            ratio = logratio.exp()
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -torch.min(ratio * advn, ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advn).mean()

            v = self.agent.value(o).float()
            vc = v_old + (v - v_old).clamp(-VF_CLIP, VF_CLIP)
            v_loss = 0.5 * torch.max((v - ret).square(), (vc - ret).square()).mean()
            ent = (0.5 + HALF_LOG_2PI + ls).sum()
            loss = pg + VF_COEF * v_loss - ENT_COEF * ent

            kl = ((ratio - 1.0) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > CLIP).float().mean()
        return loss, torch.stack([pg, v_loss, ent, kl, clipfrac]).detach()

    @torch.no_grad()
    def _rollout(self):
        viewer = getattr(self.env, "viewer", None)
        for t in range(self.T):
            ti = self.t_idx[t : t + 1]
            act = self._policy(ti)
            raw, rew, done_i = self.env.step(act)
            self._process(raw, rew, done_i, ti)
            if viewer is not None and t % 10 == 0:
                viewer.render()

        # Ship episode stats host-side in the background (async only on CUDA, where the
        # event below fences the read); elsewhere the copies block, which is cheap.
        self.fin_cpu.copy_(self.fin_hist, non_blocking=self.cuda)
        self.ret_cpu.copy_(self.ret_hist, non_blocking=self.cuda)
        self.len_cpu.copy_(self.len_hist, non_blocking=self.cuda)
        if self.copy_done is not None:
            self.copy_done.record()

        # One batched value pass over all T+1 observations.
        self.obs_ext[self.T] = self.obs
        val_ext = self.agent.value(self.obs_ext.reshape((self.T + 1) * self.N, OBS_DIM))
        val_ext = val_ext.reshape(self.T + 1, self.N)
        adv_b = _gae(self.rew_b, self.done_b, val_ext, GAMMA, GAE_LAMBDA)
        return adv_b, val_ext[: self.T]

    def iterate(self) -> dict:
        adv_b, val_b = self._rollout()
        B = self.batch_size
        b_obs = self.obs_b.reshape(B, OBS_DIM)
        b_act = self.act_b.reshape(B, ACT_DIM)
        b_logp = self.logp_b.reshape(B)
        b_adv = adv_b.reshape(B)
        b_val = val_b.reshape(B)
        b_ret = b_adv + b_val
        mb = B // self.minibatches
        dev = b_obs.device

        stats_acc = torch.zeros(5, device=dev)
        n_upd = 0
        kl_stop = False
        for _ in range(self.epochs):
            perm = torch.randperm(B, device=dev)
            epoch_kl = torch.zeros((), device=dev)
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                loss, stats = self._loss(b_obs[idx], b_act[idx], b_logp[idx],
                                         b_adv[idx], b_ret[idx], b_val[idx])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                self.opt.step()
                with torch.no_grad():
                    self.agent.log_std.clamp_(LOGSTD_MIN, LOGSTD_MAX)
                stats_acc += stats
                epoch_kl += stats[3]
                n_upd += 1
            if epoch_kl.item() / self.minibatches > 1.5 * TARGET_KL:  # one sync per epoch
                kl_stop = True
                break

        pg, v_loss, ent, kl, clipfrac = (stats_acc / max(n_upd, 1)).tolist()
        if kl > 2.0 * TARGET_KL:
            self.lr = max(LR_MIN, self.lr / LR_FACTOR)
        elif kl < 0.5 * TARGET_KL:
            self.lr = min(LR_MAX, self.lr * LR_FACTOR)
        self.opt.param_groups[0]["lr"] = self.lr

        if self.copy_done is not None:
            self.copy_done.synchronize()
        fin = self.fin_cpu.numpy()
        if fin.any():
            self.fin_rets.extend(self.ret_cpu.numpy()[fin].tolist())
            self.fin_lens.extend(self.len_cpu.numpy()[fin].tolist())

        self.global_step += B
        self.iteration += 1
        log = {"policy_loss": pg, "value_loss": v_loss, "entropy": ent,
               "approx_kl": kl, "clipfrac": clipfrac, "kl_stop": int(kl_stop),
               "log_std": self.agent.log_std.mean().item(), "lr": self.lr,
               "iteration": self.iteration}
        if self.fin_rets:
            log["ep_return"] = float(np.mean(self.fin_rets))
            log["ep_length"] = float(np.mean(self.fin_lens))
        return log
