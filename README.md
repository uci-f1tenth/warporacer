# warporacer

GPU-parallel autonomous racing RL, written end to end in [NVIDIA Warp](https://github.com/NVIDIA/warp) +
[warp-nn](https://github.com/NVIDIA/warp-nn) — no PyTorch. Give it a ROS-style track image and it PPO-trains a
lidar-driven racing policy.

```bash
uv run python main.py maps/my_map.yaml            # train (wandb on by default)
uv run python main.py maps/my_map.yaml --num-envs 256 --device cpu --no-use-wandb  # laptop-sized
uv run python viz_centerline.py maps/my_map.yaml  # debug centerline extraction
```

## Layout

| file | what it does |
|---|---|
| `warporacer/track.py` | map yaml + image → wall-distance field, closed centerline loop, nearest-waypoint LUT |
| `warporacer/sim.py` | the sim: one Warp kernel per step (RK4 kinematic bicycle with traction cap, reward, lidar, respawn) |
| `warporacer/agent.py` | actor-critic MLP (warp-nn) + Gaussian sampling kernel |
| `warporacer/ppo.py` | PPO entirely on Warp arrays: rollout, obs/return normalization, GAE, clipped update, KL-adaptive LR |
| `warporacer/video.py` | mp4 render of a deterministic rollout |
| `warporacer/train.py` | CLI, logging, checkpoints (`.npz`) |

Everything stays on the device: the env kernel writes obs/reward/done straight into the rollout buffers, values are
computed in one batched pass, and the only host syncs per iteration are the KL check (per epoch) and the logging
scalars. State per car is `(x, y, psi, v, delta)`; per-episode domain randomization jitters friction and wheelbase
by ±15%.
