# warporacer

GPU-parallel autonomous racing RL: the sim is [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels, the
learning is PyTorch. Give it a ROS-style track image and it PPO-trains a lidar-driven racing policy at
~1.8M env-steps/s on a single GPU.

```bash
uv run python main.py maps/my_map.yaml            # train (wandb on by default)
uv run python main.py maps/my_map.yaml --num-envs 256 --device cpu --no-use-wandb  # laptop-sized
uv run python viz_centerline.py maps/my_map.yaml  # debug centerline extraction
```

## Layout

| file | what it does |
|---|---|
| `warporacer/track.py` | map yaml + image → wall-distance field, closed centerline loop, nearest-waypoint LUT |
| `warporacer/sim.py` | the sim: a physics/reward/respawn kernel (RK4 kinematic bicycle with traction cap) + a lidar kernel parallelized over (car, beam); torch tensors in and out |
| `warporacer/agent.py` | actor-critic MLP (torch) with a Gaussian policy |
| `warporacer/ppo.py` | PPO: rollout, obs/return normalization, GAE, clipped update, KL-adaptive LR |
| `warporacer/video.py` | mp4 render of a deterministic rollout |
| `warporacer/train.py` | CLI, logging, checkpoints (`.pt`) |

The env writes obs/reward/done into warp buffers that torch reads zero-copy; on CUDA the kernels launch on
torch's stream, so there are no cross-framework syncs. Values are computed in one batched pass after the
rollout. State per car is `(x, y, psi, v, delta)`; per-episode domain randomization jitters friction and
wheelbase by ±15%.
