"""PPO in torch over the warp Env: rollout, GAE, clipped-surrogate updates.

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


class PPO:
    def __init__(self, env, agent, rollouts: int = 24, epochs: int = 5, minibatches: int = 4,
                 lr: float = 3e-4):
        assert (rollouts * env.num_envs) % minibatches == 0
        self.env, self.agent = env, agent
        self.epochs, self.minibatches = epochs, minibatches
        self.opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        self.global_step = 0
        self.iteration = 0
        T, N = rollouts, env.num_envs
        self.T, self.N = T, N
        self.batch_size = T * N
        dev = torch.device(env.torch_device)

        self.obs_b = torch.zeros((T, N, OBS_DIM), device=dev)
        self.act_b = torch.zeros((T, N, ACT_DIM), device=dev)
        self.logp_b = torch.zeros((T, N), device=dev)
        self.rew_b = torch.zeros((T, N), device=dev)
        self.done_b = torch.zeros((T, N), device=dev)

        self.obs_rms = RunningMeanStd((OBS_DIM,), dev)
        self.ret_rms = ReturnNormalizer(N, dev)
        self.obs_rms.update(env.obs)
        self.obs = self.obs_rms.normalize(env.obs)

        self.ep_ret = torch.zeros(N, device=dev)
        self.ep_len = torch.zeros(N, device=dev)
        self.fin_rets, self.fin_lens = deque(maxlen=100), deque(maxlen=100)

    @torch.no_grad()
    def _rollout(self):
        for t in range(self.T):
            self.obs_b[t] = self.obs
            dist = self.agent.dist(self.obs)
            act = dist.sample()
            self.act_b[t] = act
            self.logp_b[t] = dist.log_prob(act).sum(-1)
            raw, rew, done_i = self.env.step(act)
            done = done_i.float()
            self.rew_b[t] = self.ret_rms.normalize(rew, done)
            self.done_b[t] = done
            self.ep_ret += rew
            self.ep_len += 1.0
            fin = done_i.bool()
            if fin.any():
                self.fin_rets.extend(self.ep_ret[fin].tolist())
                self.fin_lens.extend(self.ep_len[fin].tolist())
                self.ep_ret[fin] = 0.0
                self.ep_len[fin] = 0.0
            self.obs_rms.update(raw)
            self.obs = self.obs_rms.normalize(raw)

        # One batched value pass over all T+1 observations.
        obs_all = torch.cat([self.obs_b.reshape(self.batch_size, OBS_DIM), self.obs])
        val_ext = self.agent.value(obs_all).reshape(self.T + 1, self.N)

        adv_b = torch.zeros_like(self.rew_b)
        last = torch.zeros(self.N, device=self.rew_b.device)
        for t in reversed(range(self.T)):
            live = 1.0 - self.done_b[t]
            delta = self.rew_b[t] + GAMMA * val_ext[t + 1] * live - val_ext[t]
            last = delta + GAMMA * GAE_LAMBDA * live * last
            adv_b[t] = last
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

        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd = 0
        kl_stop = False
        for _ in range(self.epochs):
            perm = torch.randperm(B, device=b_obs.device)
            epoch_kl = 0.0
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                dist = self.agent.dist(b_obs[idx])
                new_logp = dist.log_prob(b_act[idx]).sum(-1)
                logratio = new_logp - b_logp[idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    stats["clipfrac"] += ((ratio - 1.0).abs() > CLIP).float().mean().item()
                epoch_kl += approx_kl

                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()

                new_val = self.agent.value(b_obs[idx])
                v_clip = b_val[idx] + (new_val - b_val[idx]).clamp(-VF_CLIP, VF_CLIP)
                v_loss = 0.5 * torch.max(
                    (new_val - b_ret[idx]).square(), (v_clip - b_ret[idx]).square()
                ).mean()
                ent = dist.entropy().sum(-1).mean()

                self.opt.zero_grad(set_to_none=True)
                (pg + VF_COEF * v_loss - ENT_COEF * ent).backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                self.opt.step()
                with torch.no_grad():
                    self.agent.log_std.clamp_(LOGSTD_MIN, LOGSTD_MAX)
                stats["pg"] += pg.item()
                stats["v"] += v_loss.item()
                stats["ent"] += ent.item()
                stats["kl"] += approx_kl
                n_upd += 1
            if epoch_kl / self.minibatches > 1.5 * TARGET_KL:
                kl_stop = True
                break

        for k in stats:
            stats[k] /= max(n_upd, 1)
        lr = self.opt.param_groups[0]["lr"]
        if stats["kl"] > 2.0 * TARGET_KL:
            lr = max(LR_MIN, lr / LR_FACTOR)
        elif stats["kl"] < 0.5 * TARGET_KL:
            lr = min(LR_MAX, lr * LR_FACTOR)
        self.opt.param_groups[0]["lr"] = lr

        self.global_step += B
        self.iteration += 1
        log = {"policy_loss": stats["pg"], "value_loss": stats["v"], "entropy": stats["ent"],
               "approx_kl": stats["kl"], "clipfrac": stats["clipfrac"], "kl_stop": int(kl_stop),
               "log_std": self.agent.log_std.mean().item(), "lr": lr, "iteration": self.iteration}
        if self.fin_rets:
            log["ep_return"] = float(np.mean(self.fin_rets))
            log["ep_length"] = float(np.mean(self.fin_lens))
        return log
