"""PPO in torch over the warp Env: rollout, GAE, clipped-surrogate updates.

Throughput notes: the rollout runs exactly two torch.compile'd (CUDA-graph) calls per env
step — the policy sample and the post-step bookkeeping — so no eager torch ops sit between
warp launches. The minibatch update is one compiled step with hand-derived gradients for
the actor/critic MLPs and an inlined Adam, so there is no autograd, optimizer, or
grad-clip launch overhead; gradcheck() verifies the analytic gradients against autograd
(~5e-5 relative, TF32 noise) — rerun it if the Agent architecture or the loss changes.
The only blocking reads are the per-epoch KL check and the per-iteration log scalars;
episode stats are recorded into GPU ring buffers and shipped to pinned host memory once
per iteration.

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
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-5
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
        if self.cuda:
            from torch import _dynamo
            from torch._inductor import config as _inductor_config

            # Blackwell triton miscompiles the fused bookkeeping kernel with persistent
            # reductions ("PassManager::run failed"); disabling them is near-free here.
            _inductor_config.triton.persistent_reductions = False

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

        # Static minibatch slots (gathered into once per update) and manual Adam state.
        mb = self.batch_size // minibatches
        self.mb_obs = torch.zeros((mb, OBS_DIM), device=dev)
        self.mb_act = torch.zeros((mb, ACT_DIM), device=dev)
        self.mb_logp = torch.zeros(mb, device=dev)
        self.mb_adv = torch.zeros(mb, device=dev)
        self.mb_ret = torch.zeros(mb, device=dev)
        self.mb_val = torch.zeros(mb, device=dev)
        self.params = [agent.actor[0].weight, agent.actor[0].bias,
                       agent.actor[2].weight, agent.actor[2].bias,
                       agent.actor[4].weight, agent.actor[4].bias,
                       agent.critic[0].weight, agent.critic[0].bias,
                       agent.critic[2].weight, agent.critic[2].bias,
                       agent.critic[4].weight, agent.critic[4].bias,
                       agent.log_std]
        self.exp_avg = [torch.zeros_like(p) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]
        self.adam_step = torch.zeros((), device=dev)
        self.lr = lr
        self.lr_t = torch.full((), lr, device=dev)

        if self.cuda:  # keep CUDA-graph inputs in place instead of copying per replay
            for t in (self.obs_ext, self.act_b, self.logp_b, self.rew_b, self.done_b,
                      self.obs, self.ep_ret, self.ep_len, self.fin_hist, self.ret_hist,
                      self.len_hist, self.mb_obs, self.mb_act, self.mb_logp, self.mb_adv,
                      self.mb_ret, self.mb_val, self.adam_step, self.lr_t,
                      self.ret_rms.returns, env.obs, env.rew, env.done,
                      *self.params, *self.exp_avg, *self.exp_avg_sq,
                      *(v for r in (self.obs_rms, self.ret_rms.rms)
                        for v in (r.mean, r.var, r.inv_std, r.count))):
                _dynamo.mark_static_address(t)

        self._policy = self._policy_step
        self._process = self._process_step
        self._update = self._update_step
        if compile and self.cuda:
            self._policy = torch.compile(self._policy_step, mode="reduce-overhead")
            self._process = torch.compile(self._process_step, mode="reduce-overhead")
            self._update = torch.compile(self._update_step, mode="reduce-overhead")

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

    def _grads_and_stats(self, o, a, logp_old, adv, ret, v_old):
        """Analytic forward+backward for the clipped-surrogate loss through both MLPs.
        Returns per-parameter gradients (ordered like self.params) and the loss stats."""
        aW1, ab1, aW2, ab2, aW3, ab3, cW1, cb1, cW2, cb2, cW3, cb3, lsp = self.params
        M = float(o.shape[0])

        # Actor forward and policy-gradient backward.
        h1 = torch.tanh(o @ aW1.t() + ab1)
        h2 = torch.tanh(h1 @ aW2.t() + ab2)
        mean = h2 @ aW3.t() + ab3
        ls = lsp.clamp(LOGSTD_MIN, LOGSTD_MAX)
        e = ls.exp()
        z = (a - mean) / e
        new_logp = (-0.5 * z.square() - ls - HALF_LOG_2PI).sum(-1)
        logratio = new_logp - logp_old
        ratio = logratio.exp()
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        inside = (ratio - (1.0 - CLIP)) * ((1.0 + CLIP) - ratio) >= 0.0
        s1 = ratio * advn
        s2 = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advn
        d_ratio = torch.where(s1 <= s2, advn, torch.where(inside, advn, torch.zeros_like(advn))) * (-1.0 / M)
        d_nl = ratio * d_ratio
        d_mean = d_nl.unsqueeze(-1) * z / e
        ls_pass = (lsp >= LOGSTD_MIN) & (lsp <= LOGSTD_MAX)
        g_ls = torch.where(ls_pass, (d_nl.unsqueeze(-1) * (z.square() - 1.0)).sum(0) - ENT_COEF,
                           torch.zeros_like(lsp))
        g_aW3 = d_mean.t() @ h2
        g_ab3 = d_mean.sum(0)
        d_u2 = (d_mean @ aW3) * (1.0 - h2.square())
        g_aW2 = d_u2.t() @ h1
        g_ab2 = d_u2.sum(0)
        d_u1 = (d_u2 @ aW2) * (1.0 - h1.square())
        g_aW1 = d_u1.t() @ o
        g_ab1 = d_u1.sum(0)

        # Critic forward and clipped-value backward.
        k1 = torch.tanh(o @ cW1.t() + cb1)
        k2 = torch.tanh(k1 @ cW2.t() + cb2)
        v = (k2 @ cW3.t() + cb3).squeeze(-1)
        vc = v_old + (v - v_old).clamp(-VF_CLIP, VF_CLIP)
        e1 = v - ret
        e2 = vc - ret
        b1s = e1.square()
        b2s = e2.square()
        vin = (v - v_old).abs() <= VF_CLIP
        d_v = torch.where(b1s >= b2s, 2.0 * e1,
                          torch.where(vin, 2.0 * e2, torch.zeros_like(e2))) * (VF_COEF * 0.5 / M)
        d_k2 = d_v.unsqueeze(-1) * cW3
        d_w2 = d_k2 * (1.0 - k2.square())
        g_cW3 = d_v.unsqueeze(0) @ k2
        g_cb3 = d_v.sum().reshape(1)
        g_cW2 = d_w2.t() @ k1
        g_cb2 = d_w2.sum(0)
        d_w1 = (d_w2 @ cW2) * (1.0 - k1.square())
        g_cW1 = d_w1.t() @ o
        g_cb1 = d_w1.sum(0)

        grads = [g_aW1, g_ab1, g_aW2, g_ab2, g_aW3, g_ab3,
                 g_cW1, g_cb1, g_cW2, g_cb2, g_cW3, g_cb3, g_ls]

        pg = -torch.min(s1, s2).mean()
        v_loss = 0.5 * torch.max(b1s, b2s).mean()
        ent = (0.5 + HALF_LOG_2PI + ls).sum()
        kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > CLIP).float().mean()
        return grads, torch.stack([pg, v_loss, ent, kl, clipfrac])

    @torch.no_grad()
    def _update_step(self, o, a, logp_old, adv, ret, v_old):
        """One minibatch update: analytic gradients, global grad-norm clip, and Adam
        (matches torch's fused Adam, eps applied after bias correction) in one fused step."""
        grads, stats = self._grads_and_stats(o, a, logp_old, adv, ret, v_old)
        total = torch.sqrt(sum(g.square().sum() for g in grads))
        coef = (MAX_GRAD_NORM / (total + 1e-6)).clamp(max=1.0)
        self.adam_step += 1.0
        bc1 = 1.0 - torch.pow(ADAM_B1, self.adam_step)
        bc2 = 1.0 - torch.pow(ADAM_B2, self.adam_step)
        sd = self.lr_t / bc1
        for p, g, m, v in zip(self.params, grads, self.exp_avg, self.exp_avg_sq):
            gc = g * coef
            m.mul_(ADAM_B1).add_(gc, alpha=1.0 - ADAM_B1)
            v.mul_(ADAM_B2).addcmul_(gc, gc, value=1.0 - ADAM_B2)
            p.add_(-sd * m / ((v / bc2).sqrt() + ADAM_EPS))
        self.agent.log_std.clamp_(LOGSTD_MIN, LOGSTD_MAX)
        return stats

    def _mark_step(self):
        if self.cuda:
            torch.compiler.cudagraph_mark_step_begin()

    @torch.no_grad()
    def _rollout(self):
        for t in range(self.T):
            ti = self.t_idx[t : t + 1]
            self._mark_step()
            act = self._policy(ti)
            raw, rew, done_i = self.env.step(act)
            self._process(raw, rew, done_i, ti)
            if getattr(self.env, "viewer", None) is not None and t % 10 == 0:
                self.env.viewer.render()

        # Ship episode stats host-side in the background; read after the update.
        self.fin_cpu.copy_(self.fin_hist, non_blocking=True)
        self.ret_cpu.copy_(self.ret_hist, non_blocking=True)
        self.len_cpu.copy_(self.len_hist, non_blocking=True)
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
                torch.index_select(b_obs, 0, idx, out=self.mb_obs)
                torch.index_select(b_act, 0, idx, out=self.mb_act)
                torch.index_select(b_logp, 0, idx, out=self.mb_logp)
                torch.index_select(b_adv, 0, idx, out=self.mb_adv)
                torch.index_select(b_ret, 0, idx, out=self.mb_ret)
                torch.index_select(b_val, 0, idx, out=self.mb_val)
                self._mark_step()
                stats = self._update(self.mb_obs, self.mb_act, self.mb_logp,
                                     self.mb_adv, self.mb_ret, self.mb_val)
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
        self.lr_t.fill_(self.lr)

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


def gradcheck(agent, device="cuda", mb=1024):
    """Compare _grads_and_stats against autograd on random data; returns max rel error
    (~5e-5 under TF32). Rerun after changing the Agent architecture or the loss."""
    torch.manual_seed(0)
    o = torch.randn(mb, OBS_DIM, device=device)
    a = torch.randn(mb, ACT_DIM, device=device)
    lp = torch.randn(mb, device=device)
    adv = torch.randn(mb, device=device)
    ret = torch.randn(mb, device=device)
    vo = torch.randn(mb, device=device)

    mean = agent.actor(o)
    ls = agent.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX)
    z = (a - mean) / ls.exp()
    new_logp = (-0.5 * z.square() - ls - HALF_LOG_2PI).sum(-1)
    ratio = (new_logp - lp).exp()
    advn = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -torch.min(ratio * advn, ratio.clamp(1 - CLIP, 1 + CLIP) * advn).mean()
    v = agent.value(o)
    vc = vo + (v - vo).clamp(-VF_CLIP, VF_CLIP)
    v_loss = 0.5 * torch.max((v - ret).square(), (vc - ret).square()).mean()
    ent = (0.5 + HALF_LOG_2PI + ls).sum()
    (pg + VF_COEF * v_loss - ENT_COEF * ent).backward()

    class _Env:  # minimal stand-in so PPO can build without a sim
        num_envs, torch_device = mb, device
        obs = torch.zeros((mb, OBS_DIM), device=device)
        rew = torch.zeros(mb, device=device)
        done = torch.zeros(mb, dtype=torch.int32, device=device)

    ppo = PPO(_Env(), agent, rollouts=1, minibatches=1, compile=False)
    with torch.no_grad():
        grads, _ = ppo._grads_and_stats(o, a, lp, adv, ret, vo)
    return max(
        ((g - p.grad).abs().max() / (p.grad.abs().max() + 1e-12)).item()
        for g, p in zip(grads, ppo.params)
    )
