"""CLI: PPO-train a racing agent on one track image or a directory of them."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import wandb
import warp as wp
from typer import run

from warporacer.agent import Agent
from warporacer.ppo import PPO
from warporacer.sim import Env
from warporacer.track import Track
from warporacer.video import record_rollout


_track_cache = {}


def load_tracks(paths, k, rng):
    """Sample up to k maps and build Tracks concurrently (skeletonization is CPU-heavy),
    caching by path so later rotations are cheap. Unloadable maps are skipped."""
    chosen = list(rng.choice(paths, size=min(k, len(paths)), replace=False))

    def build(p):
        if p not in _track_cache:
            try:
                _track_cache[p] = Track(p)
            except Exception as e:
                print(f"[maps] skipping {p.name}: {e}")
                _track_cache[p] = None
        return _track_cache[p]

    with ThreadPoolExecutor() as pool:
        tracks = [t for t in pool.map(build, chosen) if t is not None]
    if not tracks:
        raise RuntimeError("no loadable maps in sample")
    return tracks


def main(
    maps: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("logs"),
    device: str = "",
    record_every: int = 100,
    record_steps: int = 1800,
    max_active_maps: int = 8,
    switch_map_iter: int = 0,
    compile: bool = True,
    live_viewer: bool = False,
    interactive: bool = False,
    use_wandb: bool = True,
):
    wp.init()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")  # TF32 tensor cores for the update GEMMs
    dev = wp.get_device(device or None)
    log_dir.mkdir(parents=True, exist_ok=True)

    map_paths = sorted(maps.glob("*.yaml")) if maps.is_dir() else [maps]
    if not map_paths:
        raise FileNotFoundError(f"no map yamls under {maps}")
    rng = np.random.default_rng(seed)

    if interactive:
        env = Env(load_tracks(map_paths, 1, rng), 1, seed=seed, device=dev)
        from warporacer.viewer import Viewer

        Viewer(env).interactive()
        return

    env = Env(load_tracks(map_paths, max_active_maps, rng), num_envs, seed=seed, device=dev)
    env.viewer = None
    if live_viewer:
        from warporacer.viewer import Viewer

        env.viewer = Viewer(env)
    agent = Agent().to(env.torch_device)
    ppo = PPO(env, agent, compile=compile and dev.is_cuda)

    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}",
                config={"num_envs": num_envs, "iterations": iterations, "seed": seed,
                        "maps": str(maps), "switch_map_iter": switch_map_iter},
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    t0 = last = time.time()
    for it in range(iterations):
        log = ppo.iterate()
        now = time.time()
        log["sps"] = int(ppo.batch_size / (now - last))
        last = now
        try:
            wandb.log(log, step=ppo.global_step)
        except Exception:
            pass
        if it % 10 == 0:
            print(
                f"[it {it:4d}] step={ppo.global_step:>9d} sps={log['sps']:>7d} "
                f"ret={log.get('ep_return', float('nan')):8.2f} kl={log['approx_kl']:.4f} "
                f"lr={log['lr']:.2e}{' KL-STOP' if log['kl_stop'] else ''}"
            )
        if record_every > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, ppo.obs_rms, record_steps, out)
            except Exception as e:
                print(f"[rollout {it + 1}] failed: {e}")
            else:
                try:
                    wandb.log({"rollout": wandb.Video(str(out), format="mp4")}, step=ppo.global_step)
                except Exception:
                    pass
        if switch_map_iter > 0 and (it + 1) % switch_map_iter == 0 and len(map_paths) > 1:
            env.rotate(load_tracks(map_paths, max_active_maps, rng))
            ppo.reset_env_stats()
            if env.viewer is not None:
                env.viewer.reset()
    print(f"[done] {time.time() - t0:.1f}s")

    torch.save(
        {
            "agent": agent.state_dict(),
            "obs_mean": ppo.obs_rms.mean.cpu(),
            "obs_var": ppo.obs_rms.var.cpu(),
            "obs_count": ppo.obs_rms.count,
        },
        log_dir / "agent_final.pt",
    )
    print(f"[checkpoint] {log_dir / 'agent_final.pt'}")
    record_rollout(env, agent, ppo.obs_rms, record_steps, log_dir / "rollout_final.mp4")


if __name__ == "__main__":
    run(main)
