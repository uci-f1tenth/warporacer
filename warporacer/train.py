"""CLI: PPO-train a racing agent on a track image."""

import time
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


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("logs"),
    device: str = "",
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
):
    wp.init()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")  # TF32 tensor cores for the update GEMMs
    dev = wp.get_device(device or None)
    log_dir.mkdir(parents=True, exist_ok=True)

    track = Track(map_yaml)
    env = Env(track, num_envs, seed=seed, device=dev)
    agent = Agent().to(env.torch_device)
    ppo = PPO(env, agent)

    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}",
                config={"num_envs": num_envs, "iterations": iterations, "seed": seed, "map": str(map_yaml)},
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
                f"[it {it:4d}] step={ppo.global_step:>9d} sps={log['sps']:>6d} "
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
