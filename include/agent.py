from collections import deque
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines
import imageio.v2 as imageio
import numpy as np
import psutil
import torch
import torch.nn as nn
import wandb

from include.constants import *
from include.environment import Environment

# Fast matrix multiplication layout alignment configurations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], device: torch.device) -> None:
        self.mean: torch.Tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var: torch.Tensor = torch.zeros(shape, dtype=torch.float32, device=device)
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
        self.returns = self.returns * self.gamma + reward
        self.rms.update(self.returns)
        self.returns = self.returns * (1.0 - done) # Reset fresh for the next step

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        return reward * self.rms.inv_std


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    LOGSTD_MIN: float = -2.0
    LOGSTD_MAX: float = -0.5
    HALF_LOG_TWO_PI: float = 0.9189385332046727

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.actor: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.actor(obs)
        ls = self.LOGSTD_MIN + (self.LOGSTD_MAX - self.LOGSTD_MIN) * torch.sigmoid(self.log_std)
        std: torch.Tensor = ls.exp()
        
        # Sample from standard normal distribution
        noise = torch.randn_like(mean)
        raw_action = mean + noise * std
        
        # Squash cleanly to [-1, 1] for physics engine safety
        action_squashed = torch.tanh(raw_action)
        
        # Analytical Jacobian Correction for Tanh Squashing
        log_prob = -((raw_action - mean) ** 2) / (2 * std.pow(2)) - ls - self.HALF_LOG_TWO_PI
        log_prob = log_prob.sum(-1) - torch.log(1.0 - action_squashed.pow(2) + 1e-6).sum(-1)
        
        # Use standard normal entropy as a proxy for exploration tracking
        entropy = (0.5 + self.HALF_LOG_TWO_PI + ls).sum(-1)
        
        return raw_action, action_squashed, log_prob, entropy, self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean: torch.Tensor = self.actor(obs)
        ls = self.LOGSTD_MIN + (self.LOGSTD_MAX - self.LOGSTD_MIN) * torch.sigmoid(self.log_std)
        std: torch.Tensor = ls.exp()
        
        action_squashed = torch.tanh(raw_action)
        
        log_prob = -((raw_action - mean) ** 2) / (2 * std.pow(2)) - ls - self.HALF_LOG_TWO_PI
        log_prob = log_prob.sum(-1) - torch.log(1.0 - action_squashed.pow(2) + 1e-6).sum(-1)
        
        entropy = (0.5 + self.HALF_LOG_TWO_PI + ls).sum(-1)
        
        return log_prob, entropy, self.critic(obs).squeeze(-1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(obs), -1.0, 1.0)


class KLAdaptiveLR:
    def __init__(self, opt: torch.optim.Optimizer, target_kl: float = 0.015, factor: float = 1.5, lr_min: float = 1e-5, lr_max: float = 1e-3) -> None:
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


def process_observations(raw_tensor: torch.Tensor, rms_module: RunningMeanStd) -> torch.Tensor:
    kinematics = raw_tensor[..., :3]
    sensory_normalized = rms_module.normalize(raw_tensor[..., 3:])
    return torch.cat([kinematics, sensory_normalized], dim=-1)


def record_rollout(env: "Environment", agent: Agent, num_steps: int, out_path: Path, obs_rms: Optional[RunningMeanStd] = None) -> None:
    snap: Dict[str, torch.Tensor] = env.save_state()
    was_training: bool = agent.training
    agent.eval()
    
    try:
        m = env.map
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
        obs: torch.Tensor = process_observations(raw, obs_rms) if obs_rms else raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        device = env.cars_buf.device
        num_features = env.cars_buf.shape[1] 
        traj_states = torch.empty((num_steps, num_features), dtype=torch.float32, device=device)
        resets_gpu = torch.empty(num_steps, dtype=torch.bool, device=device)

        with torch.no_grad():
            for i in range(num_steps):
                a: torch.Tensor = agent.deterministic(obs)
                raw, _, term, trunc, _ = env.step(a)
                obs = process_observations(raw, obs_rms) if obs_rms else raw
                traj_states[i] = env.cars_buf[0]
                resets_gpu[i] = term[0] | trunc[0]

        full_states_cpu = traj_states.cpu().numpy()
        resets_cpu = resets_gpu.cpu().numpy()

        x_arr = full_states_cpu[:, 0]
        y_arr = full_states_cpu[:, 1]
        psi_arr = full_states_cpu[:, 4]

        centers = np.column_stack((x_arr, y_arr))
        px_centers = w2p_vec(centers) 
        c_arr = np.cos(psi_arr)
        s_arr = np.sin(psi_arr)

        trail: deque = deque(maxlen=300)
        
        with imageio.get_writer(str(out_path), fps=int(1 / DT), macro_block_size=2) as w:
            for i in range(num_steps):
                if resets_cpu[i]:
                    trail.clear()
                    
                trail.append(px_centers[i])
                frame: np.ndarray = base_frame.copy()
                
                if len(trail) > 1:
                    polylines(frame, [np.array(trail)], False, (0, 200, 0), 2)
                    
                c, s = c_arr[i], s_arr[i]
                R: np.ndarray = np.array([[c, -s], [s, c]])
                
                world_pts: np.ndarray = corners @ R.T + centers[i]
                px_world = w2p_vec(world_pts)
                
                fillPoly(frame, [px_world], (255, 50, 50))
                w.append_data(frame)

    finally:
        env.restore_state(snap)
        agent.train(was_training)


@torch.compile(mode="reduce-overhead")
def _train_step_compiled(
    agent: Agent,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        new_logp, ent, new_val = agent.evaluate(b_obs_idx, b_act_idx)
        
        # Cast to float32 to prevent exponentiation overflow/underflow in half-precision
        logratio: torch.Tensor = new_logp.float() - b_logp_idx.float()
        ratio: torch.Tensor = logratio.exp()
        
        approx_kl: torch.Tensor = ((ratio - 1.0) - logratio).mean()
        clipfrac: torch.Tensor = ((ratio - 1.0).abs() > clip).float().mean()
        
        adv_mb: torch.Tensor = (b_adv_idx - b_adv_idx.mean()) / (b_adv_idx.std() + 1e-8)
        s1: torch.Tensor = ratio * adv_mb
        s2: torch.Tensor = ratio.clamp(1 - clip, 1 + clip) * adv_mb
        pg: torch.Tensor = -torch.min(s1, s2).mean()
        
        v_err: torch.Tensor = new_val.float() - b_ret_idx.float()
        if vf_clip > 0:
            v_clipped: torch.Tensor = b_val_idx.float() + (new_val.float() - b_val_idx.float()).clamp(-vf_clip, vf_clip)
            v_loss: torch.Tensor = 0.5 * torch.max(v_err.square(), (v_clipped - b_ret_idx).square()).mean()
        else:
            v_loss = 0.5 * v_err.square().mean()
            
        loss: torch.Tensor = pg + vf_coef * v_loss - ent_coef * ent.mean()
        
    return loss, pg, v_loss, ent.mean(), approx_kl, clipfrac


@torch.compile
def compute_gae(
    rew_b: torch.Tensor,
    val_b: torch.Tensor,
    next_val: torch.Tensor,
    term_b: torch.Tensor,
    done_b: torch.Tensor,  # done_b = (term | trunc)
    gamma: float,
    gae_lambda: float,
    rollouts: int,
) -> torch.Tensor:
    adv_b: torch.Tensor = torch.zeros_like(rew_b)
    last: torch.Tensor = torch.zeros_like(next_val)
    
    for t in range(rollouts - 1, -1, -1):
        # CRITICAL: Treat both crashes AND timeouts as trajectory cutoffs 
        # to prevent bootstrapping from contaminated in-place reset observations.
        nondone: torch.Tensor = 1.0 - done_b[t]
        
        next_v: torch.Tensor = next_val if t == rollouts - 1 else val_b[t + 1]
        
        # Using nondone here drops the bootstrap cleanly at any reset event
        delta: torch.Tensor = rew_b[t] + gamma * next_v * nondone - val_b[t]
        last = delta + gamma * gae_lambda * nondone * last
        adv_b[t] = last
        
    return adv_b


def train(
    env: Environment,
    agent: Agent,
    iterations: int = 5000,
    rollouts: int = 256,
    epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    vf_clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    lr: float = 3.0e-4,
    target_kl: float = 0.010,
    log_dir: Path = Path("./logs"),
    record_every_iteration: int = 200,
    record_duration_steps: int = 2000,
    switch_map_iter: int = 20,
    use_wandb_train: bool = False
) -> Tuple[float, RunningMeanStd, ReturnNormalizer, int]:
    device: torch.device = next(agent.parameters()).device
    process_profile = psutil.Process()
    N: int = env.num_envs
    
    # -------------------------------------------------------------------------
    # PARAMETER & HYPERPARAMETER TELEMETRY
    # -------------------------------------------------------------------------
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total_batch_size = rollouts * N
    
    print("=" * 80)
    print(f"[{'TRAINING PIPELINE INITIALIZATION':^74}]")
    print("=" * 80)
    print(f" -> Compute Device      : {device}")
    print(f" -> Parallel Envs (N)   : {N:,}")
    print(f" -> Rollout Steps (T)   : {rollouts}")
    print(f" -> Total Batch Size (B): {total_batch_size:,} steps/iteration")
    print(f" -> Optimization Epochs : {epochs}")
    print(f" -> Base Learning Rate  : {lr:<10} | Target KL Threshold: {target_kl}")
    print(f" -> GAE Gamma           : {gamma:<10} | GAE Lambda        : {gae_lambda}")
    print(f" -> Policy Clip Bounds  : {clip:<10} | Value Clip Bounds  : {vf_clip}")
    print(f" -> Loss Weights        : Value: {vf_coef:<6} | Entropy: {ent_coef}")
    print("-" * 80)
    print(f" -> Total Parameters    : {total_params:,}")
    print(f" -> Trainable Params    : {trainable_params:,}")
    print("=" * 80)
    # -------------------------------------------------------------------------

    opt: torch.optim.Optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5, fused=True)
    
    # Drop base optimizer LR down slightly for smoother high-batch step mechanics
    for pg in opt.param_groups:
        pg["lr"] = lr

    # Cosine annealing that safely floors at 4e-5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=iterations,
        eta_min=4e-5
    )
    
    sensory_dim = OBS_DIM - 3
    obs_rms: RunningMeanStd = RunningMeanStd((sensory_dim,), device)
    ret_rms: ReturnNormalizer = ReturnNormalizer(N, gamma, device)

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
    obs_rms.update(raw[..., 3:]) 
    obs: torch.Tensor = process_observations(raw, obs_rms)
    ep_ret: torch.Tensor = torch.zeros(N, device=device)
    ep_len: torch.Tensor = torch.zeros(N, device=device)
    finished_rets: deque = deque(maxlen=100)
    finished_lens: deque = deque(maxlen=100)

    global_step: int = 0
    t0: float = time.time()
    last_t: float = t0
    current_epochs: int = epochs

    start_wall_clock = time.time()
    for it in range(iterations):
        agent.eval()
        with torch.no_grad():
            for t in range(rollouts):
                # Inside your data generation loop:
                obs_b[t] = obs
                act_raw, act_clamped, logp, _, val = agent.act_value(obs)
                act_b[t] = act_raw # Store the raw action for mathematical consistency during training
                logp_b[t] = logp
                val_b[t] = val

                raw, raw_rew, term, trunc, _ = env.step(act_clamped) # Step the environment with the clamped action
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
                    res_rets = ep_ret[fin].to("cpu", non_blocking=True).numpy()
                    res_lens = ep_len[fin].to("cpu", non_blocking=True).numpy()
                    finished_rets.extend(res_rets)
                    finished_lens.extend(res_lens)
                    ep_ret[fin] = 0.0
                    ep_len[fin] = 0.0
                obs = process_observations(raw, obs_rms)

                if env.vs and t % 4 == 0:
                    env.vs.render()
                
            next_val: torch.Tensor = agent.value(obs)

        obs_rms.update(raw_obs_b[..., 3:])
        # Re-normalize the trailing state using the fresh statistics before the next iteration
        obs = process_observations(raw, obs_rms)

        adv_b: torch.Tensor = compute_gae(rew_b, val_b, next_val, term_b, done_b, gamma, gae_lambda, rollouts)
        ret_b: torch.Tensor = adv_b + val_b
        
        B: int = rollouts * N
        global_step += B

        TARGET_MINIBATCH_SIZE = 16384 * 4
        calculated_minibatches = max(1, B // TARGET_MINIBATCH_SIZE)
        mb: int = B // calculated_minibatches

        b_obs: torch.Tensor = obs_b.reshape(B, OBS_DIM)
        b_act: torch.Tensor = act_b.reshape(B, ACT_DIM)
        b_logp: torch.Tensor = logp_b.reshape(B)
        b_adv: torch.Tensor = adv_b.reshape(B)
        b_ret: torch.Tensor = ret_b.reshape(B)
        b_val: torch.Tensor = val_b.reshape(B)

        agent.train()
        
        tot_pg, tot_v, tot_ent, tot_kl, tot_clip = 0.0, 0.0, 0.0, 0.0, 0.0
        n_upd: int = 0
        current_ent_coef = max(0.001, ent_coef * (1.0 - (it / iterations)))
        permutation_indices = torch.arange(B, device=device)

        for epoch in range(current_epochs):  
            perm = permutation_indices[torch.randperm(B, device=device)]
            epoch_kl = 0.0
            minibatches_run = 0
            
            # Drop the remainder batch if it isn't an exact match
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                if idx.shape[0] != mb: 
                    continue # Skip the odd-sized remainder to protect torch.compile

                opt.zero_grad(set_to_none=True)
                
                loss, pg, v_loss, ent_m, approx_kl, clipfrac = _train_step_compiled(
                    agent, b_obs[idx], b_act[idx], b_logp[idx], 
                    b_adv[idx], b_ret[idx], b_val[idx], clip, vf_coef, vf_clip, 
                    current_ent_coef
                )
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()

                tot_pg += pg.item()
                tot_v += v_loss.item()
                tot_ent += ent_m.item()
                tot_kl += approx_kl.item()
                tot_clip += clipfrac.item()
                n_upd += 1
                
                epoch_kl += approx_kl.item()
                minibatches_run += 1
            
            # Post-epoch check: Stop optimizing if the average drift this epoch is too high
            if minibatches_run > 0 and (epoch_kl / minibatches_run) > target_kl * 2.0:
                break
                
        denom = max(n_upd, 1)
        avg_pg, avg_v, avg_ent, final_kl, avg_clip = tot_pg/denom, tot_v/denom, tot_ent/denom, tot_kl/denom, tot_clip/denom
        
        # Scheduler execution step
        sched.step()
        
        if final_kl > 1.5 * target_kl:
            current_epochs = max(1, current_epochs - 1)
        elif final_kl < target_kl / 1.5:
            current_epochs = min(epochs, current_epochs + 1)

        now: float = time.time()
        sps: int = int(rollouts * N / max(now - last_t, 1e-9))
        last_t = now
        
        # Extracted active learning rate securely from the base optimizer parameter group
        current_lr = float(opt.param_groups[0]["lr"])

        log: Dict[str, Any] = {
            "policy_loss": avg_pg,
            "value_loss": avg_v,
            "entropy": avg_ent,
            "approx_kl": final_kl,
            "clipfrac": avg_clip,
            "current_epochs": current_epochs,  
            "log_std": agent.log_std.mean().item(),
            "iter_lr": current_lr,
            "sps": sps,
            "iteration": it,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))

        if use_wandb_train:
            try:
                wandb.log(log, step=global_step)
            except Exception as e:
                print(f"[WandB] Log failed: {e}")

        if it % 10 == 0:
            er = log.get("ep_return", float("nan"))
            el = log.get("ep_length", float("nan"))
            cpu = process_profile.cpu_percent()
            ram = process_profile.memory_info().rss / 1048576

            real_elapsed = time.time() - start_wall_clock
            rh, rem = divmod(real_elapsed, 3600)
            rm, rs = divmod(rem, 60)
            real_str = f"{int(rh):02d}:{int(rm):02d}:{int(rs):02d}"

            sim_elapsed_seconds = global_step * DT  
            sh, srem = divmod(sim_elapsed_seconds, 3600)
            sm, ss = divmod(srem, 60)
            sim_str = f"{int(sh):02d}:{int(sm):02d}:{int(ss):02d}"

            print(
                f"[{it:4d}] Real:{real_str} Sim:{sim_str} | {sps:>5d} SPS | "
                f"R:{er:6.1f} L:{el:5.1f} Ent:{avg_ent:.3f} | "
                f"V:{avg_v:.3f} P:{avg_pg:.3f} Clp:{avg_clip:.2f} KL:{final_kl:.3f} | "
                f"LR:{current_lr:.1e} Ep:{current_epochs} | RAM:{ram:4.0f}M CPU:{cpu:4.1f}%"
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
            env.cycle_next_map(randomize=True)
            
            #Re-synchronize state representation with the new layout allocation
            raw = env.obs_buf
            obs_rms.update(raw[..., 3:])
            obs = process_observations(raw, obs_rms)
            
    return time.time() - t0, obs_rms, ret_rms, global_step