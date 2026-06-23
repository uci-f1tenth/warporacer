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

from include.constants import ACT_DIM, DT, LENGTH, OBS_DIM, WIDTH
from include.environment import Environment

# Fast matrix multiplication layout alignment configurations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class RunningMeanStd:
    """Tracks running mean, variance, and inverse standard deviation metrics."""

    def __init__(self, shape: Tuple[int, ...], device: torch.device) -> None:
        self.mean: torch.Tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var: torch.Tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        self.inv_std: torch.Tensor = torch.ones(shape, dtype=torch.float32, device=device)
        self.count: float = 1e-4

    def update(self, x: torch.Tensor) -> None:
        """Updates internal moments using an online parallel variance update algorithm."""
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
        """Standardizes inputs using current tracking statistics."""
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


class ReturnNormalizer:
    """Normalizes episodic rewards using a discounted running tracking strategy."""

    def __init__(self, num_envs: int, gamma: float, device: torch.device) -> None:
        self.gamma: float = gamma
        self.returns: torch.Tensor = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms: RunningMeanStd = RunningMeanStd((), device)

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """Accrues rolling rewards and shifts tracking windows on environment steps."""
        self.returns = self.returns * self.gamma + reward
        self.rms.update(self.returns)
        self.returns = self.returns * (1.0 - done)

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Scales active rewards across standard tracking limits."""
        return reward * self.rms.inv_std


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """Applies orthogonal parameter initializations to target projection layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Asymmetric Actor-Critic implementation supporting continuous environments."""

    LOGSTD_MIN: float = -2.0
    LOGSTD_MAX: float = -0.5
    HALF_LOG_TWO_PI: float = 0.9189385332046727

    def __init__(self, obs_dim: int, critic_obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        
        self.actor: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(), 
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        
        self.critic: nn.Sequential = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Evaluates state values from privileged observation feeds."""
        return self.critic(critic_obs).squeeze(-1)

    def act_value(self, obs: torch.Tensor, critic_obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples exploratory actions and gathers companion state values."""
        if critic_obs is None:
            critic_obs = obs
            
        mean: torch.Tensor = self.actor(obs)
        ls = self.LOGSTD_MIN + (self.LOGSTD_MAX - self.LOGSTD_MIN) * torch.sigmoid(self.log_std)
        std: torch.Tensor = ls.exp()
        
        noise = torch.randn_like(mean)
        raw_action = mean + noise * std
        action_squashed = torch.tanh(raw_action)
        
        log_prob = -((raw_action - mean) ** 2) / (2 * std.pow(2)) - ls - self.HALF_LOG_TWO_PI
        log_prob = log_prob.sum(-1) - torch.log(1.0 - action_squashed.pow(2) + 1e-6).sum(-1)
        entropy = (0.5 + self.HALF_LOG_TWO_PI + ls).sum(-1)
        
        return raw_action, action_squashed, log_prob, entropy, self.critic(critic_obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, critic_obs: torch.Tensor, raw_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates targeted action probabilities during policy optimization phases."""
        mean: torch.Tensor = self.actor(obs)
        ls = self.LOGSTD_MIN + (self.LOGSTD_MAX - self.LOGSTD_MIN) * torch.sigmoid(self.log_std)
        std: torch.Tensor = ls.exp()
        
        action_squashed = torch.tanh(raw_action)
        log_prob = -((raw_action - mean) ** 2) / (2 * std.pow(2)) - ls - self.HALF_LOG_TWO_PI
        log_prob = log_prob.sum(-1) - torch.log(1.0 - action_squashed.pow(2) + 1e-6).sum(-1)
        entropy = (0.5 + self.HALF_LOG_TWO_PI + ls).sum(-1)
        
        return log_prob, entropy, self.critic(critic_obs).squeeze(-1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Generates deterministic actions matching bounds expected by the environment physics."""
        return torch.tanh(self.actor(obs))


def process_observations(raw_tensor: torch.Tensor, rms_module: RunningMeanStd) -> torch.Tensor:
    """Splits rigid kinematics from structural ray scans to scale distance horizons."""
    kinematics = raw_tensor[..., :3]
    sensory_normalized = rms_module.normalize(raw_tensor[..., 3:])
    return torch.cat([kinematics, sensory_normalized], dim=-1)


@torch.compile(mode="reduce-overhead")
def _train_step_compiled(
    agent: Agent,
    b_obs_idx: torch.Tensor,
    b_critic_obs_idx: torch.Tensor,
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
    """Executes a single compiled policy loss calculation pass under half-precision settings."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        new_logp, ent, new_val = agent.evaluate(b_obs_idx, b_critic_obs_idx, b_act_idx)
        
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
    done_b: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    rollouts: int,
) -> torch.Tensor:
    """Backpropagates multi-step Generalized Advantage Estimations from rollout horizons."""
    adv_b: torch.Tensor = torch.zeros_like(rew_b)
    last: torch.Tensor = torch.zeros_like(next_val)
    
    for t in range(rollouts - 1, -1, -1):
        nondone: torch.Tensor = 1.0 - done_b[t]
        next_v: torch.Tensor = next_val if t == rollouts - 1 else val_b[t + 1]
        
        delta: torch.Tensor = rew_b[t] + gamma * next_v * nondone - val_b[t]
        last = delta + gamma * gae_lambda * nondone * last
        adv_b[t] = last
        
    return adv_b


class RolloutBuffers:
    """Manages tensor allocation memory pools to avoid re-allocating space in the loop."""

    def __init__(self, rollouts: int, num_envs: int, device: torch.device) -> None:
        self.obs_b: torch.Tensor = torch.zeros((rollouts, num_envs, OBS_DIM), device=device)
        self.raw_obs_b: torch.Tensor = torch.zeros((rollouts, num_envs, OBS_DIM), device=device)
        self.critic_obs_b: torch.Tensor = torch.zeros((rollouts, num_envs, OBS_DIM + 5), device=device)
        self.act_b: torch.Tensor = torch.zeros((rollouts, num_envs, ACT_DIM), device=device)
        self.logp_b: torch.Tensor = torch.zeros((rollouts, num_envs), device=device)
        self.rew_b: torch.Tensor = torch.zeros((rollouts, num_envs), device=device)
        self.done_b: torch.Tensor = torch.zeros((rollouts, num_envs), device=device)
        self.term_b: torch.Tensor = torch.zeros((rollouts, num_envs), device=device)
        self.val_b: torch.Tensor = torch.zeros((rollouts, num_envs), device=device)


def _log_training_summary(it: int, global_step: int, sps: int, avg_ent: float, avg_v: float, avg_pg: float, avg_clip: float, final_kl: float, current_lr: float, current_epochs: int, log: Dict[str, Any], process_profile: psutil.Process, start_wall_clock: float) -> None:
    """Formats and prints an iteration summary line."""
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


def record_rollout(env: Environment, agent: Agent, num_steps: int, out_path: Path, obs_rms: Optional[RunningMeanStd] = None) -> None:
    """Saves a rollout validation video to track behavior on a random active track without interfering with training."""
    snap: Dict[str, torch.Tensor] = env.save_state()
    was_training: bool = agent.training
    agent.eval()
    
    try:
        # 1. Reset the environment states normally
        raw, _, _ = env.reset()

        # 2. Randomly select an index from the currently active maps pool
        import random
        map_idx = random.randint(0, env.num_maps - 1)
        m = env.maps[map_idx]
        print(f"Recording validation rollout on sampled map: {m.path_name}")

        # 3. Calculate Global Map Alignment Shifts
        cl_vec3_array = env.maps_storage.centerline_buf.numpy()
        global_wp0 = cl_vec3_array[map_idx, 0]
        shift_x = float(global_wp0[0] - m.centerline[0, 0])
        shift_y = float(global_wp0[1] - m.centerline[0, 1])

        # 4. Force environment 0 to use this specific map and spawn point safely
        cl_idx = random.randint(0, len(m.centerline) - 1)
        
        env.views.cars_buf[0, 0] = m.centerline[cl_idx, 0] + shift_x # Global X
        env.views.cars_buf[0, 1] = m.centerline[cl_idx, 1] + shift_y # Global Y
        env.views.cars_buf[0, 4] = m.angles[cl_idx]                  # Yaw
        
        env.views.cars_int_buf[0, 0] = 0       # Step counter reset
        env.views.cars_int_buf[0, 1] = cl_idx  # Closest centerline index
        
        map_ids_np = env.agents.env_map_ids.numpy()
        map_ids_np[0] = map_idx
        env.agents.env_map_ids.assign(map_ids_np)

        # 5. Step once with zero actions to compute the clean observation vector for the new position
        # FIXED: Correct unpacking layout matching the environment configuration
        raw, _, _, _, _, _ = env.step(torch.zeros((env.num_envs, 2), device=env.views.obs_buf.device))
        obs: torch.Tensor = process_observations(raw, obs_rms) if obs_rms else raw

        # --- SETUP GRAPHICS RENDERING BASE ---
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
            local_x = pts[:, 0] - shift_x
            local_y = pts[:, 1] - shift_y
            px = (local_x - m.ox) / m.res
            py = m.h - 1 - (local_y - m.oy) / m.res
            return np.column_stack((px, py)).astype(np.int32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        device = env.views.cars_buf.device
        num_features = env.views.cars_buf.shape[1] 
        
        traj_states = torch.empty((num_steps, num_features), dtype=torch.float32, device=device)
        resets_gpu = torch.empty(num_steps, dtype=torch.bool, device=device)

        # --- COLLECT TRAJECTORY ---
        with torch.no_grad():
            for i in range(num_steps):
                a: torch.Tensor = agent.deterministic(obs)
                # FIXED: Corrected unpacking from 5 arguments to 6 here to clear the ValueError
                raw, _, _, term, trunc, _ = env.step(a)
                obs = process_observations(raw, obs_rms) if obs_rms else raw
                traj_states[i] = env.views.cars_buf[0]
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
        
        # --- RENDER VIDEO ---
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
        # 5. Restore original tensors perfectly back into memory blocks
        env.restore_state(snap)
        agent.train(was_training)


def train(
    env: Environment,
    agent: Agent,
    iterations: int = 5000,
    rollouts: int = 64,
    epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    vf_clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    lr: float = 5.0e-4,
    target_kl: float = 0.010,
    log_dir: Path = Path("./logs"),
    record_every_iteration: int = 200,
    record_duration_steps: int = 2000,
    switch_map_iter: int = 20,
    use_wandb_train: bool = False
) -> Tuple[float, RunningMeanStd, ReturnNormalizer, int]:
    """Orchestrates high-throughput Proximal Policy Optimization loops across parallel tracking buffers."""
    device: torch.device = next(agent.parameters()).device
    process_profile = psutil.Process()
    N: int = env.num_envs
    
    B: int = rollouts * N
    TARGET_MINIBATCH_SIZE = 16384 * 4
    calculated_minibatches = max(1, B // TARGET_MINIBATCH_SIZE)
    mb: int = B // calculated_minibatches
    permutation_indices = torch.arange(B, device=device)
    
    print("=" * 80)
    print(f" -> Compute Device      : {device} | Parallel Envs (N): {N:,}")
    print(f" -> Total Batch Size (B): {B:,} steps/iteration | Minibatches: {calculated_minibatches}")
    print("=" * 80)

    opt: torch.optim.Optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations, eta_min=4e-5)
    
    sensory_dim = OBS_DIM - 3
    obs_rms: RunningMeanStd = RunningMeanStd((sensory_dim,), device)
    ret_rms: ReturnNormalizer = ReturnNormalizer(N, gamma, device)
    scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda")
    buffers = RolloutBuffers(rollouts, N, device)

    raw, raw_critic, _ = env.reset()
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
                buffers.obs_b[t] = obs
                buffers.critic_obs_b[t] = raw_critic
                act_raw, act_clamped, logp, _, val = agent.act_value(obs, critic_obs=raw_critic)
                buffers.act_b[t] = act_raw 
                buffers.logp_b[t] = logp
                buffers.val_b[t] = val

                raw, raw_critic, raw_rew, term, trunc, _ = env.step(act_clamped) 
                buffers.raw_obs_b[t] = raw

                done: torch.Tensor = (term | trunc).float()
                ret_rms.update(raw_rew, done)
                buffers.rew_b[t] = ret_rms.normalize(raw_rew)
                buffers.done_b[t] = done
                buffers.term_b[t] = term.float()
                
                ep_ret.add_(raw_rew)
                ep_len.add_(1.0)
                
                fin: torch.Tensor = done.bool()
                if fin.any():
                    finished_rets.extend(ep_ret[fin].to("cpu", non_blocking=True).numpy())
                    finished_lens.extend(ep_len[fin].to("cpu", non_blocking=True).numpy())
                    ep_ret[fin] = 0.0
                    ep_len[fin] = 0.0
                obs = process_observations(raw, obs_rms)

                if env.vs and t % 4 == 0:
                    env.vs.render()
                
            next_val: torch.Tensor = agent.value(raw_critic)

        obs_rms.update(buffers.raw_obs_b[..., 3:])
        obs = process_observations(raw, obs_rms)

        adv_b: torch.Tensor = compute_gae(buffers.rew_b, buffers.val_b, next_val, buffers.term_b, buffers.done_b, gamma, gae_lambda, rollouts)
        ret_b: torch.Tensor = adv_b + buffers.val_b
        global_step += B

        b_obs = buffers.obs_b.reshape(B, OBS_DIM)
        b_act = buffers.act_b.reshape(B, ACT_DIM)
        b_logp = buffers.logp_b.reshape(B)
        b_adv = adv_b.reshape(B)
        b_ret = ret_b.reshape(B)
        b_val = buffers.val_b.reshape(B)
        b_critic_obs = buffers.critic_obs_b.reshape(B, buffers.critic_obs_b.shape[-1])

        agent.train()
        tot_pg, tot_v, tot_ent, tot_kl, tot_clip = 0.0, 0.0, 0.0, 0.0, 0.0
        n_upd: int = 0
        current_ent_coef = max(0.001, ent_coef * (1.0 - (it / iterations)))

        for epoch in range(current_epochs):  
            perm = permutation_indices[torch.randperm(B, device=device)]
            epoch_kl = 0.0
            minibatches_run = 0
            
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                if idx.shape[0] != mb: 
                    continue 

                opt.zero_grad(set_to_none=True)
                loss, pg, v_loss, ent_m, approx_kl, clipfrac = _train_step_compiled(
                    agent, b_obs[idx], b_critic_obs[idx], b_act[idx], b_logp[idx], 
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
            
            if minibatches_run > 0 and (epoch_kl / minibatches_run) > target_kl * 2.0:
                break
                
        denom = max(n_upd, 1)
        avg_pg, avg_v, avg_ent, final_kl, avg_clip = tot_pg/denom, tot_v/denom, tot_ent/denom, tot_kl/denom, tot_clip/denom
        sched.step()
        
        if final_kl > 1.5 * target_kl:
            current_epochs = max(1, current_epochs - 1)
        elif final_kl < target_kl / 1.5:
            current_epochs = min(epochs, current_epochs + 1)

        now: float = time.time()
        sps: int = int(rollouts * N / max(now - last_t, 1e-9))
        last_t = now
        current_lr = float(opt.param_groups[0]["lr"])

        log: Dict[str, Any] = {
            "policy_loss": avg_pg, "value_loss": avg_v, "entropy": avg_ent,
            "approx_kl": final_kl, "clipfrac": avg_clip, "current_epochs": current_epochs,  
            "log_std": agent.log_std.mean().item(), "iter_lr": current_lr, "sps": sps, "iteration": it,
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
            _log_training_summary(it, global_step, sps, avg_ent, avg_v, avg_pg, avg_clip, final_kl, current_lr, current_epochs, log, process_profile, start_wall_clock)
            
        if record_every_iteration > 0 and (it + 1) % record_every_iteration == 0:
            out: Path = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            record_rollout(env, agent, record_duration_steps, out, obs_rms)
            if use_wandb_train:
                try:
                    wandb.log({"rollout": wandb.Video(str(out), format="mp4")}, step=global_step)
                except Exception as e:
                    print(f"[WandB] Rollout video upload failed: {e}")
        
        if switch_map_iter > 0 and (it + 1) % switch_map_iter == 0:
            env.trigger_map_rotation()
            env._launch(env._zero_act)
            env._sanitize()
            raw = env.obs_buf
            obs_rms.update(raw[..., 3:])
            obs = process_observations(raw, obs_rms)
            
    return time.time() - t0, obs_rms, ret_rms, global_step