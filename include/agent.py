import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import wandb
from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines
from torch.distributions import Normal

from include.constants import *
from include.environment import Environment
from include.map import Map

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], device: torch.device) -> None:
        self.mean: torch.Tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var: torch.Tensor = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std: torch.Tensor = torch.ones(shape, dtype=torch.float32, device=device)
        self.count: float = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape).float()
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc: int = x.shape[0]
        delta: torch.Tensor = bm - self.mean
        tot: float = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        self.var = (
            self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)
        ) / tot
        self.count = tot
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


class ReturnNormalizer:
    def __init__(self, num_envs: int, gamma: float, device: torch.device) -> None:
        self.gamma: float = gamma
        self.returns: torch.Tensor = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms: RunningMeanStd = RunningMeanStd((), device)

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        self.returns = self.returns * self.gamma * (1.0 - done) + reward
        self.rms.update(self.returns)

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        return reward * self.rms.inv_std


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    LOGSTD_MIN: float = -1.6
    LOGSTD_MAX: float = -0.3

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.actor: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std: nn.Parameter = nn.Parameter(torch.full((1, act_dim), -0.5))
        self._compiled: bool = False

    def _dist(self, obs: torch.Tensor, mean: Optional[torch.Tensor] = None) -> Normal:
        if mean is None:
            mean = self.actor(obs)
        ls: torch.Tensor = self.log_std.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return Normal(mean, ls.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.actor(obs)
        ls: torch.Tensor = self.log_std.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        std: torch.Tensor = ls.exp()
        
        noise: torch.Tensor = torch.randn_like(mean)
        action: torch.Tensor = mean + noise * std
        
        var: torch.Tensor = std.pow(2)
        log_prob: torch.Tensor = -((action - mean) ** 2) / (2 * var) - ls - math.log(math.sqrt(2 * math.pi))
        log_prob = log_prob.sum(-1)
        
        entropy: torch.Tensor = (0.5 + 0.5 * math.log(2 * math.pi) + ls).sum(-1).expand_as(log_prob).clone()
        
        return action, log_prob, entropy, self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.actor(obs)
        ls: torch.Tensor = self.log_std.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        var: torch.Tensor = ls.exp().pow(2)
        
        log_prob: torch.Tensor = -((action - mean) ** 2) / (2 * var) - ls - math.log(math.sqrt(2 * math.pi))
        entropy: torch.Tensor = (0.5 + 0.5 * math.log(2 * math.pi) + ls).sum(-1).expand_as(log_prob[:, 0]).clone()
        
        return log_prob.sum(-1), entropy, self.critic(obs).squeeze(-1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class KLAdaptiveLR:
    def __init__(self, opt: torch.optim.Optimizer, target_kl: float = 0.02, factor: float = 1.5, lr_min: float = 1e-6, lr_max: float = 3e-3) -> None:
        self.opt: torch.optim.Optimizer = opt
        self.target: float = target_kl
        self.factor: float = factor
        self.lr_min: float = lr_min
        self.lr_max: float = lr_max

    def step(self, kl: float) -> None:
        for pg in self.opt.param_groups:
            lr: float = pg["lr"]
            if kl > 2.0 * self.target:
                pg["lr"] = max(self.lr_min, lr / self.factor)
            elif kl < 0.5 * self.target:
                pg["lr"] = min(self.lr_max, lr * self.factor)

    @property
    def lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])


def record_rollout(env: "Environment", agent: Agent, num_steps: int, out_path: Path, obs_rms: Optional[RunningMeanStd] = None) -> None:
    snap: Dict[str, torch.Tensor] = env.save_state()
    was_training: bool = agent.training
    agent.eval()
    
    try:
        m: "Map" = env.map
        corners: np.ndarray = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        base_frame: np.ndarray = cvtColor(m.raw, COLOR_GRAY2RGB)

        def w2p_vec(pts: np.ndarray) -> np.ndarray:
            px = (pts[:, 0] - m.ox) / m.res
            py = m.h - 1 - (pts[:, 1] - m.oy) / m.res
            return np.column_stack((px, py)).astype(np.int32)

        raw, _ = env.reset()
        obs: torch.Tensor = obs_rms.normalize(raw) if obs_rms else raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ---------------------------------------------------------
        # PHASE 1: Pure GPU Simulation
        # ---------------------------------------------------------
        device = env.cars_buf.device
        
        # FIX 1: Allocate enough space for the FULL tensor row to avoid Advanced Indexing
        num_features = env.cars_buf.shape[1] 
        traj_states = torch.empty((num_steps, num_features), dtype=torch.float32, device=device)
        resets_gpu = torch.empty(num_steps, dtype=torch.bool, device=device)

        with torch.no_grad():
            for i in range(num_steps):
                a: torch.Tensor = agent.deterministic(obs)
                raw, _, term, trunc, _ = env.step(a)
                obs = obs_rms.normalize(raw) if obs_rms else raw
                
                # Fast contiguous memory copy instead of indexing kernel
                traj_states[i] = env.cars_buf[0]
                resets_gpu[i] = term[0] | trunc[0]

        # Sync to CPU once
        full_states_cpu = traj_states.cpu().numpy()
        resets_cpu = resets_gpu.cpu().numpy()

        # Extract exactly what we need on the CPU side
        x_arr = full_states_cpu[:, 0]
        y_arr = full_states_cpu[:, 1]
        psi_arr = full_states_cpu[:, 4]

        # ---------------------------------------------------------
        # PHASE 2: Pre-Compute Rendering Math
        # ---------------------------------------------------------
        # FIX 2: Vectorize the map transformation for all 10,000 steps instantly
        centers = np.column_stack((x_arr, y_arr))
        px_centers = w2p_vec(centers) 
        
        # Pre-compute rotation sin/cos for all steps
        c_arr = np.cos(psi_arr)
        s_arr = np.sin(psi_arr)

        # ---------------------------------------------------------
        # PHASE 3: Pure CPU Rendering
        # ---------------------------------------------------------
        trail: deque = deque(maxlen=300)
        
        with imageio.get_writer(str(out_path), fps=int(1 / DT), macro_block_size=2) as w:
            for i in range(num_steps):
                if resets_cpu[i]:
                    trail.clear()
                    
                # The trail now stores raw pixel coordinates directly!
                trail.append(px_centers[i])
                
                frame: np.ndarray = base_frame.copy()
                
                if len(trail) > 1:
                    # Convert deque to array (much faster without the w2p_vec math inside)
                    polylines(frame, [np.array(trail)], False, (0, 200, 0), 2)
                    
                # Build rotation matrix for this specific frame
                c, s = c_arr[i], s_arr[i]
                R: np.ndarray = np.array([[c, -s], [s, c]])
                
                # Calculate corners
                world_pts: np.ndarray = corners @ R.T + centers[i]
                px_world = w2p_vec(world_pts)
                
                fillPoly(frame, [px_world], (255, 50, 50))
                
                w.append_data(frame)

    finally:
        env.restore_state(snap)
        agent.train(was_training)


@torch.compile(mode="reduce-overhead")
def _train_step(
    agent: Agent,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    b_obs_idx: torch.Tensor,
    b_act_idx: torch.Tensor,
    b_logp_idx: torch.Tensor,
    b_adv_idx: torch.Tensor,
    b_ret_idx: torch.Tensor,
    b_val_idx: torch.Tensor,
    clip: float,
    vf_coef: float,
    vf_clip: float,
    ent_coef: float,
    max_grad_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Hardcode "cuda" to prevent graph breaks from checking tensor attributes
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        new_logp, ent, new_val = agent.evaluate(b_obs_idx, b_act_idx)
        
        logratio: torch.Tensor = new_logp - b_logp_idx
        ratio: torch.Tensor = logratio.exp()
        
        approx_kl: torch.Tensor = ((ratio - 1.0) - logratio).mean()
        clipfrac: torch.Tensor = ((ratio - 1.0).abs() > clip).float().mean()
        
        adv_mb: torch.Tensor = (b_adv_idx - b_adv_idx.mean()) / (b_adv_idx.std() + 1e-8)
        s1: torch.Tensor = ratio * adv_mb
        s2: torch.Tensor = ratio.clamp(1 - clip, 1 + clip) * adv_mb
        pg: torch.Tensor = -torch.min(s1, s2).mean()
        
        v_err: torch.Tensor = new_val - b_ret_idx
        if vf_clip > 0:
            v_clipped: torch.Tensor = b_val_idx + (new_val - b_val_idx).clamp(-vf_clip, vf_clip)
            v_loss: torch.Tensor = 0.5 * torch.max(v_err.square(), (v_clipped - b_ret_idx).square()).mean()
        else:
            v_loss = 0.5 * v_err.square().mean()
            
        loss: torch.Tensor = pg + vf_coef * v_loss - ent_coef * ent.mean()

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    scaler.step(opt)
    scaler.update()
    
    return pg, v_loss, ent.mean(), approx_kl, clipfrac


@torch.compile(mode="reduce-overhead", fullgraph=True)
def compute_gae(
    rew_b: torch.Tensor,
    val_b: torch.Tensor,
    next_val: torch.Tensor,
    term_b: torch.Tensor,
    done_b: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    rollouts: int,
) -> torch.Tensor:
    adv_b: torch.Tensor = torch.zeros_like(rew_b)
    last: torch.Tensor = torch.zeros_like(next_val)
    
    for t in range(rollouts - 1, -1, -1):
        nonterm: torch.Tensor = 1.0 - term_b[t]
        nondone: torch.Tensor = 1.0 - done_b[t]
        
        next_v: torch.Tensor = next_val if t == rollouts - 1 else val_b[t + 1]
        
        delta: torch.Tensor = rew_b[t] + gamma * next_v * nonterm - val_b[t]
        last = delta + gamma * gae_lambda * nondone * last
        adv_b[t] = last
        
    return adv_b

#git@profile
def train(
    env: Environment,
    agent: Agent,
    iterations: int = 2000,
    rollouts: int = 24,
    epochs: int = 5,
    minibatches: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    vf_clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    lr: float = 3e-4,
    target_kl: float = 0.02,
    log_dir: Path = Path("./logs"),
    record_every_iteration: int = 100,
    record_duration_steps: int = 2000,
    switch_map_iter: int = 10,
    use_wandb_train: bool = False
) -> Tuple[float, RunningMeanStd, ReturnNormalizer, int]:
    device: torch.device = next(agent.parameters()).device
    N: int = env.num_envs
    
    # Enable Fused Adam for massive CPU overhead reduction
    opt: torch.optim.Optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5, fused=True)
    sched: KLAdaptiveLR = KLAdaptiveLR(opt, target_kl=target_kl)
    obs_rms: RunningMeanStd = RunningMeanStd((OBS_DIM,), device)
    ret_rms: ReturnNormalizer = ReturnNormalizer(N, gamma, device)

    if not agent._compiled:
        agent.evaluate = torch.compile(agent.evaluate, mode="reduce-overhead")
        agent.act_value = torch.compile(agent.act_value, mode="reduce-overhead")
        agent._compiled = True

    scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda")

    obs_b: torch.Tensor = torch.zeros((rollouts, N, OBS_DIM), device=device)
    raw_obs_b: torch.Tensor = torch.zeros((rollouts, N, OBS_DIM), device=device)
    act_b: torch.Tensor = torch.zeros((rollouts, N, ACT_DIM), device=device)
    logp_b: torch.Tensor = torch.zeros((rollouts, N), device=device)
    rew_b: torch.Tensor = torch.zeros((rollouts, N), device=device)
    done_b: torch.Tensor = torch.zeros((rollouts, N), device=device)
    term_b: torch.Tensor = torch.zeros((rollouts, N), device=device)
    val_b: torch.Tensor = torch.zeros((rollouts, N), device=device)

    raw, _ = env.reset()
    obs_rms.update(raw)
    obs: torch.Tensor = obs_rms.normalize(raw)
    ep_ret: torch.Tensor = torch.zeros(N, device=device)
    ep_len: torch.Tensor = torch.zeros(N, device=device)
    finished_rets: deque = deque(maxlen=100)
    finished_lens: deque = deque(maxlen=100)

    global_step: int = 0
    t0: float = time.time()
    last_t: float = t0
    
    # Initialize dynamic epochs
    current_epochs: int = epochs

    for it in range(iterations):
        agent.eval()
        with torch.no_grad():
            for t in range(rollouts):
                obs_b[t] = obs
                act, logp, _, val = agent.act_value(obs)
                act_b[t] = act
                logp_b[t] = logp
                val_b[t] = val
                raw, raw_rew, term, trunc, _ = env.step(act)
                raw_obs_b[t] = raw

                done: torch.Tensor = (term | trunc).float()
                ret_rms.update(raw_rew, done)
                rew_b[t] = ret_rms.normalize(raw_rew)
                done_b[t] = done
                term_b[t] = term.float()
                ep_ret.add_(raw_rew)
                ep_len.add_(1.0)
                
                fin: torch.Tensor = done.bool()
                if fin.any():
                    finished_rets.extend(ep_ret[fin].detach().cpu().numpy())
                    finished_lens.extend(ep_len[fin].detach().cpu().numpy())
                    ep_ret[fin] = 0.0
                    ep_len[fin] = 0.0
                obs = obs_rms.normalize(raw)

                # Render every 4th frame
                if env.vs and t % 4 == 0:
                    env.vs.render()
                
            next_val: torch.Tensor = agent.value(obs)

        obs_rms.update(raw_obs_b)

        # GAE Calculation - We MUST use .clone() here to escape CUDAGraph static memory
        adv_b: torch.Tensor = compute_gae(rew_b, val_b, next_val, term_b, done_b, gamma, gae_lambda, rollouts).clone()
        ret_b: torch.Tensor = adv_b + val_b
        
        B: int = rollouts * N
        global_step += B

        b_obs: torch.Tensor = obs_b.reshape(B, OBS_DIM)
        b_act: torch.Tensor = act_b.reshape(B, ACT_DIM)
        b_logp: torch.Tensor = logp_b.reshape(B)
        b_adv: torch.Tensor = adv_b.reshape(B)
        b_ret: torch.Tensor = ret_b.reshape(B)
        b_val: torch.Tensor = val_b.reshape(B)
        mb: int = B // minibatches

        agent.train()
        
        stats: Dict[str, torch.Tensor] = {
            "pg": torch.tensor(0.0, device=device),
            "v": torch.tensor(0.0, device=device),
            "ent": torch.tensor(0.0, device=device),
            "kl": torch.tensor(0.0, device=device),
            "clipfrac": torch.tensor(0.0, device=device)
        }
        
        n_upd: int = 0
        
        torch.compiler.cudagraph_mark_step_begin()
        for epoch in range(current_epochs):  
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                
                pg, v_loss, ent_m, approx_kl, clipfrac = _train_step(
                    agent, opt, scaler, b_obs[idx], b_act[idx], b_logp[idx], 
                    b_adv[idx], b_ret[idx], b_val[idx], clip, vf_coef, vf_clip, 
                    ent_coef, max_grad_norm
                )

                # We MUST use .clone() here to escape CUDAGraph static memory
                stats["pg"] += pg.clone()
                stats["v"] += v_loss.clone()
                stats["ent"] += ent_m.clone()
                stats["kl"] += approx_kl.clone()
                stats["clipfrac"] += clipfrac.clone()
                n_upd += 1
                
        divisor = max(n_upd, 1)
        for k in stats:
            stats[k] = stats[k] / divisor  # Vectorized division, stays on the GPU
            
        final_kl = float(stats["kl"].item())
        sched.step(final_kl)
        
        # Dynamic Epoch Logic
        if final_kl > 1.5 * target_kl:
            # We drifted too far. Do fewer epochs next time.
            current_epochs = max(1, current_epochs - 1)
        elif final_kl < target_kl / 1.5:
            # Cap the max epochs at your initial default (epochs variable) instead of 10
            current_epochs = min(epochs, current_epochs + 1)

        now: float = time.time()
        sps: int = int(rollouts * N / max(now - last_t, 1e-9))
        last_t = now
        
        log: Dict[str, Any] = {
            "policy_loss": stats["pg"].item(),
            "value_loss": stats["v"].item(),
            "entropy": stats["ent"].item(),
            "approx_kl": final_kl,
            "clipfrac": stats["clipfrac"].item(),
            "current_epochs": current_epochs,  # Replaced kl_stop with current_epochs
            "log_std": agent.log_std.mean().item(),
            "iter_lr": sched.lr,
            "sps": sps,
            "iteration": it,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))

        # Update wandb
        if use_wandb_train:
            try:
                wandb.log(log, step=global_step)
            except Exception as e:
                print(f"[WandB] Log failed: {e}")

        if it % 10 == 0:
            er: float = log.get("ep_return", float("nan"))
            # Removed kl_stop conditional from this print statement
            print(
                f"[it {it:4d}] step={global_step:>9d} sps={sps:>6d} "
                f"ret={er:8.2f} kl={final_kl:.4f} lr={sched.lr:.2e} epochs={current_epochs}"
            )
            
        if record_every_iteration > 0 and (it + 1) % record_every_iteration == 0:
            out: Path = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            print(f"record_rollout: out={out}")
            record_rollout(env, agent, record_duration_steps, out, obs_rms)

            if use_wandb_train:
                try:
                    wandb.log({"rollout": wandb.Video(str(out), format="mp4")}, step=global_step)
                except Exception as e:
                    print(f"[WandB] Rollout video failed: {e}")
        
        if switch_map_iter > 0 and (it + 1) % switch_map_iter == 0:
            env.cycle_next_map() # TODO: randomize
            
    return time.time() - t0, obs_rms, ret_rms, global_step