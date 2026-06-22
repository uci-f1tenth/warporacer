import os
import sys
from pathlib import Path
from typing import Optional

# --- Cross-OS Isolated Compilation Cache Config ---
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

# Dynamically route the cache path so Windows and Linux allocations never collide
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
from include.constants import *
from include.environment import Environment

# ---------------------------------------------------------------------------------
# Hardware Troubleshooting Note:
# If you encounter the following error across dual-GPU environments (iGPU & eGPU setups):
#   "Warp UserWarning: Could not register GL buffer since CUDA/OpenGL interoperability is not available.
#    Falling back to copy operations between the Warp array and the OpenGL buffer."
# ---------------------------------------------------------------------------------

def main(
    maps_dir_str: str = typer.Option("maps/", help="Path to maps file or directory"),
    num_envs: int = 1024,
    seed: int = 0,
    interactive: bool = False,
    live_viewer: bool = False,
    iterations: int = 5000,
    record_every_iteration: int = 100,
    record_duration_steps: int = 2000,
    switch_map_iter: int = 100,  # Training step interval between layout rotations (0 to disable)
    device: Optional[str] = None,
    use_wandb: bool = False,
    log_dir_str: str = typer.Option("./logs", help="Target output logging directory"),
):
    """
    Main entry point for handling parallelized reinforcement learning or interactive car runs.
    Manages global random seed distribution, CUDA/TensorFloat32 compilation pathways, 
    and bootstraps physical map setups before handing control off to execution streams.
    """
    maps_dir = Path(maps_dir_str).resolve()
    log_dir = Path(log_dir_str).resolve()
    
    # Force localized parameter overrides whenever direct manual driving is requested
    if interactive:
        num_envs = 1
        live_viewer = True

    # Fall back to native NVIDIA Warp global runtime selection configurations if unspecified
    if not device:
        target_device = wp.get_device()
    else:
        target_device = wp.get_device(device)

    # Safely assert log target presence ahead of downstream validation check blocks
    log_dir.mkdir(parents=True, exist_ok=True)

    # Distribute tracking values across separate execution frameworks to ensure run reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Fast compilation and execution optimizations for modern Ampere+ GPU architectures
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Map Mode Selection & Validation Pipelines
    # -----------------------------------------------------------------------------
    if switch_map_iter == 0:
        if not maps_dir.is_file():
            raise FileNotFoundError(
                f"[Error] switch_map_iter is 0 (Single Map Mode), but maps_dir is not a valid file: {maps_dir}"
            )
        print(f"[Mode] Single Map Mode. Running exclusively on: {maps_dir.name}")
    else:
        if not maps_dir.is_dir():
            raise NotADirectoryError(
                f"[Error] switch_map_iter is {switch_map_iter} (Multi-Map Mode), but maps_dir is not a valid directory: {maps_dir}"
            )
        available_maps = list(maps_dir.glob("*.yaml"))
        if not available_maps:
            raise FileNotFoundError(f"[Error] No .yaml map files found in directory: {maps_dir}")
        print(f"[Mode] Multi-Map Mode. Loaded {len(available_maps)} maps from: {maps_dir.name}")

    # Bind the contextual physical compute resource block
    with wp.ScopedDevice(target_device):
        env = Environment(maps_dir, num_envs, seed, target_device, live_viewer)

        if interactive:
            if env.vs is not None:
                env.vs.interactive_render_loop()
            else:
                print("[Error] Live viewer initialization failed. Interactive loop unavailable.")
        else:
            # Instantiate Weights & Biases (WandB) logger sessions
            if use_wandb:
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
                except Exception as e:
                    print(f"[WandB] Init failed: {e}")
                    use_wandb = False  
                    
            # Safe cross-OS device mapping using Warp's native naming properties
            torch_device_str = f"cuda:{target_device.ordinal}" if target_device.is_cuda else "cpu"
            raw_agent = Agent(obs_dim=OBS_DIM).to(torch_device_str)
            
            # 2. OPTIMIZATION: Wrap with reduce-overhead to align optimization speeds with your RTX 2070
            agent = torch.compile(raw_agent, mode="reduce-overhead")
            
            # Explicitly pass multi-map tracking arguments down to the trainer pipeline
            elapsed, obs_rms, ret_rms, step = train(
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

            # Extract clean state_dict without compiler prefix noise (_orig_mod.)
            clean_state_dict = getattr(agent, "_orig_mod", agent).state_dict()

            # Persist policy network configurations and running observation statistics
            torch.save(
                {
                    "agent": clean_state_dict,
                    "obs_mean": obs_rms.mean.cpu(),
                    "obs_var": obs_rms.var.cpu(),
                    "obs_count": obs_rms.count,
                },
                log_dir / "agent_final.pt",
            )
            print(f"[Saved!] State file weights dumped successfully.")

            # Record a standalone baseline test tracking validation run video
            out = log_dir / "rollout_final.mp4"
            record_rollout(env, agent, record_duration_steps, out, obs_rms=obs_rms)

            # Export validation video telemetry channels back up to active logging dashboards
            if use_wandb:
                try:
                    wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
                    wandb.finish() 
                except Exception as e:
                    print(f"[WandB] Final log cleanup or video upload failed: {e}")

if __name__ == "__main__":
    typer.run(main)