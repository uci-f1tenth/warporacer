import os
import sys
from pathlib import Path
from typing import List, Optional

# --- Cross-OS Isolated Compilation Cache Config ---
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

base_cache_dir = Path("./.torch_compile_cache").resolve()
os_suffix = "windows" if sys.platform == "win32" else "linux"
isolated_cache_path = base_cache_dir / os_suffix
isolated_cache_path.mkdir(parents=True, exist_ok=True)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(isolated_cache_path)
# --------------------------------------------------

import numpy as np
import torch
import typer
import wandb
import warp as wp

from include.agent import Agent, record_rollout, train
from include.constants import ACT_DIM, OBS_DIM
from include.environment import Environment


def _configure_hardware_performance(seed: int, target_device: wp.Device) -> str:
    """Configures multi-framework seeds, device paths, and Ampere+ tensor optimizations."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return f"cuda:{target_device.ordinal}" if target_device.is_cuda else "cpu"


def _validate_and_load_maps(maps_dir: Path, switch_map_iter: int) -> List[Path]:
    """Validates the directory structural integrity and verifies available map configurations."""
    if not maps_dir.is_dir():
        raise NotADirectoryError(
            f"[Error] maps_dir must be a valid directory containing layout assets: {maps_dir}"
        )
        
    available_maps = list(maps_dir.glob("*.yaml"))
    if not available_maps:
        raise FileNotFoundError(f"[Error] No .yaml map files found in directory: {maps_dir}")

    if switch_map_iter == 0:
        print(f"[Mode] Single Map Mode. Locking baseline layout: {available_maps[0].name}")
    else:
        print(f"[Mode] Multi-Map Mode. Found {len(available_maps)} maps.")
        
    return available_maps


def _initialize_wandb(use_wandb: bool, seed: int, num_envs: int, iterations: int, switch_map_iter: int, maps_dir: Path) -> bool:
    """Safely triggers a Weights & Biases telemetry tracking session."""
    if not use_wandb:
        return False
    try:
        wandb.init(
            project="warporacer",
            name=f"seed{seed}_n{num_envs}",
            config={
                "num_envs": num_envs,
                "iterations": iterations,
                "seed": seed,
                "maps_directory": str(maps_dir),
                "switch_map_iter": switch_map_iter,
            },
        )
        return True
    except Exception as e:
        print(f"[WandB] Initialization failed, falling back to local logs: {e}")
        return False


def _save_agent_checkpoint(agent: torch.nn.Module, obs_rms: any, log_dir: Path) -> None:
    """Persists policy network weights and running normalization statistics safely to disk."""
    clean_state_dict = getattr(agent, "_orig_mod", agent).state_dict()
    checkpoint_data = {
        "agent": clean_state_dict,
        "obs_mean": obs_rms.mean.cpu(),
        "obs_var": obs_rms.var.cpu(),
        "obs_count": obs_rms.count,
    }
    torch.save(checkpoint_data, log_dir / "agent_final.pt")
    print("[Saved!] State file weights dumped successfully.")


def main(
    maps_dir_str: str = typer.Option("maps/", help="Path to maps file or directory"),
    num_envs: int = 16384,
    seed: int = 0,
    interactive: bool = False,
    live_viewer: bool = False,
    iterations: int = 1000,
    record_every_iteration: int = 100,
    record_duration_steps: int = 2000,
    switch_map_iter: int = 20,
    max_active_maps: int = 16,
    device: Optional[str] = None,
    use_wandb: bool = False,
    log_dir_str: str = typer.Option("./logs", help="Target output logging directory"),
) -> None:
    """Main orchestrator handling parallel reinforcement learning or interactive car runs."""
    maps_dir = Path(maps_dir_str).resolve()
    log_dir = Path(log_dir_str).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if interactive:
        num_envs = 1
        live_viewer = True

    target_device = wp.get_device(device) if device else wp.get_device()
    torch_device_str = _configure_hardware_performance(seed, target_device)
    _validate_and_load_maps(maps_dir, switch_map_iter)

    with wp.ScopedDevice(target_device):
        env = Environment(maps_dir, num_envs, seed, target_device, live_viewer, max_active_maps)

        # Execution Stream A: Manual driving viewport loop
        if interactive:
            if env.vs is not None:
                env.vs.interactive_render_loop()
            else:
                print("[Error] Live viewer initialization failed. Interactive loop unavailable.")
            return

        # Execution Stream B: Highly parallelized RL training
        use_wandb = _initialize_wandb(use_wandb, seed, num_envs, iterations, switch_map_iter, maps_dir)
        
        raw_agent = Agent(obs_dim=OBS_DIM, critic_obs_dim=(OBS_DIM + 5), act_dim=ACT_DIM).to(torch_device_str)
        agent = torch.compile(raw_agent, mode="reduce-overhead")
        
        elapsed, obs_rms, _, step = train(
            env,
            agent,
            iterations=iterations,
            log_dir=log_dir,
            record_every_iteration=record_every_iteration,
            record_duration_steps=record_duration_steps,
            switch_map_iter=switch_map_iter,
            use_wandb_train=use_wandb
        )
        print(f"[Done!] Optimization path complete in {elapsed:.1f}s")

        _save_agent_checkpoint(agent, obs_rms, log_dir)

        # Final tracking validation rollout run video processing
        out_video_path = log_dir / "rollout_final.mp4"
        record_rollout(env, agent, record_duration_steps, out_video_path, obs_rms=obs_rms)

        if use_wandb:
            try:
                wandb.log({"rollout_final": wandb.Video(str(out_video_path), format="mp4")}, step=step)
                wandb.finish()
            except Exception as e:
                print(f"[WandB] Video payload upload or session finalization failed: {e}")


if __name__ == "__main__":
    typer.run(main)