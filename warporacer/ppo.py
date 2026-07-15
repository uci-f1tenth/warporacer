"""PPO in torch over the warp Env: rollout, GAE, clipped-surrogate updates.

Throughput notes: the policy step and the loss step are torch.compile'd (CUDA graphs via
"reduce-overhead"), Adam runs fused, and the hot loops are sync-free — episode stats are
recorded into GPU ring buffers and shipped to pinned host memory once per iteration; the
only blocking reads are the per-epoch KL check and the per-iteration log scalars.

The env self-resets on done, so the observation after a done step is the fresh spawn of a
new episode — the value bootstrap is zeroed there in GAE."""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.inv_std = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        self.var = (self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)) / tot
        self.count = tot
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x):
        return ((x - self.mean) * self.inv_std).clamp(-NORM_CLIP, NORM_CLIP)


class ReturnNormalizer:
    """Scales rewards by the running std of the discounted return."""

    def __init__(self, num_envs, device):
        self.returns = torch.zeros(num_envs, device=device)
        self.rms = RunningMeanStd((), device)

    def normalize(self, reward, done):
        self.returns = self.returns * GAMMA * (1.0 - done) + reward
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
        self.opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5, fused=self.cuda)

        self.obs_b = torch.zeros((T, N, OBS_DIM), device=dev)
        self.act_b = torch.zeros((T, N, ACT_DIM), device=dev)
        self.logp_b = torch.zeros((T, N), device=dev)
        self.rew_b = torch.zeros((T, N), device=dev)
        self.done_b = torch.zeros((T, N), device=dev)

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
        self._loss = self._loss_step
        if compile and self.cuda:
            self._policy = torch.compile(self._policy_step, mode="reduce-overhead")
            self._loss = torch.compile(self._loss_step, mode="reduce-overhead")

        self.reset_env_stats()

    def reset_env_stats(self):
        """(Re)sync with the env after construction or a map rotation."""
        self.obs_rms.update(self.env.obs)
        self.obs = self.obs_rms.normalize(self.env.obs)
        self.ep_ret.zero_()
        self.ep_len.zero_()
        self.ret_rms.returns.zero_()

    def _policy_step(self, obs):
        mean = self.agent.actor(obs)
        ls = self.agent.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX)
        noise = torch.randn_like(mean)
        act = mean + noise * ls.exp()
        logp = (-0.5 * noise.square() - ls - HALF_LOG_2PI).sum(-1)
        return act, logp

    def _loss_step(self, obs, act, logp_old, adv, ret, val_old):
        mean = self.agent.actor(obs)
        ls = self.agent.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX)
        z = (act - mean) / ls.exp()
        new_logp = (-0.5 * z.square() - ls - HALF_LOG_2PI).sum(-1)
        logratio = new_logp - logp_old
        ratio = logratio.exp()

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()

        new_val = self.agent.value(obs)
        v_clip = val_old + (new_val - val_old).clamp(-VF_CLIP, VF_CLIP)
        v_loss = 0.5 * torch.max((new_val - ret).square(), (v_clip - ret).square()).mean()
        ent = (0.5 + HALF_LOG_2PI + ls).sum()

        loss = pg + VF_COEF * v_loss - ENT_COEF * ent
        with torch.no_grad():
            kl = ((ratio - 1.0) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > CLIP).float().mean()
            stats = torch.stack([pg.detach(), v_loss.detach(), ent.detach(), kl, clipfrac])
        return loss, stats

    def _mark_step(self):
        if self.cuda:
            torch.compiler.cudagraph_mark_step_begin()

    @torch.no_grad()
    def _rollout(self):
        for t in range(self.T):
            self.obs_b[t] = self.obs
            self._mark_step()
            act, logp = self._policy(self.obs)
            self.act_b[t] = act
            self.logp_b[t] = logp
            raw, rew, done_i = self.env.step(act)
            done = done_i.float()
            self.rew_b[t] = self.ret_rms.normalize(rew, done)
            self.done_b[t] = done
            self.ep_ret += rew
            self.ep_len += 1.0
            # Branch-free episode bookkeeping (no host syncs in the hot loop).
            self.fin_hist[t] = done_i.bool()
            self.ret_hist[t] = self.ep_ret
            self.len_hist[t] = self.ep_len
            live = 1.0 - done
            self.ep_ret *= live
            self.ep_len *= live
            self.obs_rms.update(raw)
            self.obs = self.obs_rms.normalize(raw)
            if getattr(self.env, "viewer", None) is not None and t % 10 == 0:
                self.env.viewer.render()

        # Ship episode stats host-side in the background; read after the update.
        self.fin_cpu.copy_(self.fin_hist, non_blocking=True)
        self.ret_cpu.copy_(self.ret_hist, non_blocking=True)
        self.len_cpu.copy_(self.len_hist, non_blocking=True)
        if self.copy_done is not None:
            self.copy_done.record()

        # One batched value pass over all T+1 observations.
        obs_all = torch.cat([self.obs_b.reshape(self.batch_size, OBS_DIM), self.obs])
        val_ext = self.agent.value(obs_all).reshape(self.T + 1, self.N)
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
                self._mark_step()
                loss, stats = self._loss(b_obs[idx], b_act[idx], b_logp[idx],
                                         b_adv[idx], b_ret[idx], b_val[idx])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
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
        lr = self.opt.param_groups[0]["lr"]
        if kl > 2.0 * TARGET_KL:
            lr = max(LR_MIN, lr / LR_FACTOR)
        elif kl < 0.5 * TARGET_KL:
            lr = min(LR_MAX, lr * LR_FACTOR)
        self.opt.param_groups[0]["lr"] = lr

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
               "log_std": self.agent.log_std.mean().item(), "lr": lr, "iteration": self.iteration}
        if self.fin_rets:
            log["ep_return"] = float(np.mean(self.fin_rets))
            log["ep_length"] = float(np.mean(self.fin_lens))
        return log
