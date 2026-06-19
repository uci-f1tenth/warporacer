from pathlib import Path
from typing import Optional

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
#
# Fix: Ensure all underlying Python execution stacks utilize the exact same high-performance GPU.
# On Windows, register the target python.exe binary explicitly inside "Graphics Settings" 
# and switch its resource allocation profile preference to "High Performance".
# ---------------------------------------------------------------------------------

def main(
    map_path: Path = Path("maps/berlin.yaml"),  # FIXED: Platform-agnostic forward slashes
    num_envs: int = 1024,
    seed: int = 0,
    interactive: bool = True,
    live_viewer: bool = False,
    iterations: int = 2000,
    record_every: int = 100,
    record_steps: int = 2000,
    switch_map_iter: int = 10,  # Training step interval between layout rotations (0 to disable)
    device: Optional[str] = None, # FIXED: Added Optional type hint
    use_wandb: bool = False,
    log_dir: Path = Path("./logs"),
):
    """
    Main entry point for handling parallelized reinforcement learning or interactive car runs.
    Manages global random seed distribution, CUDA/TensorFloat32 compilation pathways, 
    and bootstraps physical map setups before handing control off to execution streams.
    """
    
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
        # Single-map mode validation: map_path target must point explicitly to a file asset
        if not map_path.is_file():
            raise FileNotFoundError(
                f"[Error] switch_map_iter is 0 (Single Map Mode), "
                f"but map_path is not a valid file: {map_path}"
            )
        available_maps = [map_path]
        print(f"[Mode] Single Map Mode. Running exclusively on: {map_path.name}")
    else:
        # Multi-map mode validation: map_path target must point explicitly to a directory asset
        if not map_path.is_dir():
            raise NotADirectoryError(
                f"[Error] switch_map_iter is {switch_map_iter} (Multi-Map Mode), "
                f"but map_path is not a valid directory: {map_path}"
            )
        available_maps = list(map_path.glob("*.yaml"))
        if not available_maps:
            raise FileNotFoundError(f"[Error] No .yaml map files found in directory: {map_path}")
        print(f"[Mode] Multi-Map Mode. Loaded {len(available_maps)} maps from: {map_path.name}")

    # Bind the contextual physical compute resource block
    with wp.ScopedDevice(target_device):
        # Simply pass whatever path argument parameter choice was specified at execution runtime.
        # The Environment auto-sorts files vs directories and mounts individual collection indexes itself.
        env = Environment(map_path, num_envs, seed, live_viewer)

        if interactive:
            # Drop structural execution logic straight over to manual keyboard loop threads
            env.vs.interactive_render_loop()
        else:
            # Instantiate Weights & Biases (WandB) logger sessions for hyperparameter analytics tracking
            if use_wandb:
                try:
                    wandb.init(
                        project="warporacer",
                        name=f"seed{seed}_n{num_envs}",
                        config={
                            "num_envs": num_envs,
                            "iterations": iterations,
                            "seed": seed,
                            "maps_directory": str(map_path),
                            "switch_map_iter": switch_map_iter,
                        },
                    )
                except Exception as e:
                    print(f"[WandB] Init failed: {e}")
                    use_wandb = False  # FIXED: Soft fallback to prevent downstream logging crashes
                    
            # Instantiate the network onto the environment's target device execution context
            raw_agent = Agent(obs_dim=OBS_DIM).to(str(env.device))
            
            # Wrap standard Agent modules inside torch.compile paths to trigger Graph optimization benefits
            agent = torch.compile(raw_agent)
            
            # FIXED: Explicitly pass multi-map tracking arguments down to the trainer pipeline
            elapsed, obs_rms, ret_rms, step = train(
                env,
                agent,
                iterations=iterations,
                log_dir=log_dir,
                record_every=record_every,
                record_steps=record_steps,
                use_wandb_train=use_wandb,
                available_maps=available_maps,       # Added parameter
                switch_map_iter=switch_map_iter,     # Added parameter
            )

            print(f"[Done!] Optimization path complete in {elapsed:.1f}s")

            # FIXED: Extract clean state_dict without compiler prefix noise (_orig_mod.)
            clean_state_dict = getattr(agent, "_orig_mod", agent).state_dict()

            # Persist policy network configurations and running observation statistics down onto disk structures
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
            record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)

            # Export validation video telemetry channels back up to active logging dashboards
            if use_wandb:
                try:
                    wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
                    wandb.finish()  # FIXED: Cleanly close connection context loops
                except Exception as e:
                    print(f"[WandB] Final log cleanup or video upload failed: {e}")

if __name__ == "__main__":
    # Typer command parsing layer wraps execution parameters cleanly
    typer.run(main)