# warporacer

GPU-parallel autonomous racing RL: the sim is [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels, the
learning is PyTorch. Give it a ROS-style track image and it PPO-trains a lidar-driven racing policy at
~7M env-steps/s at 8k envs (~10M at 16k) on a single GPU.

```bash
uv run python main.py maps/my_map.yaml            # train on one track (wandb on by default)
uv run python main.py maps/ --switch-map-iter 20  # multi-map: rotate 8 random tracks every 20 iters
uv run python main.py maps/my_map.yaml --num-envs 256 --device cpu --no-use-wandb  # laptop-sized
uv run python main.py maps/ --live-viewer         # 3D OpenGL view while training (uv sync --extra viz)
uv run python main.py maps/my_map.yaml --interactive  # drive the car yourself (I/K throttle, J/L steer)
uv run python viz_centerline.py maps/my_map.yaml  # debug centerline extraction
```

## Layout

| file | what it does |
|---|---|
| `warporacer/track.py` | map yaml + image → wall-distance field, closed centerline loop, nearest-waypoint LUT |
| `warporacer/sim.py` | the sim: a physics/reward/respawn kernel (RK4 kinematic bicycle with traction cap) + a lidar kernel parallelized over (car, beam); multi-map via padded raster stacks + per-env map ids; torch tensors in and out |
| `warporacer/agent.py` | actor-critic MLP (torch) with a Gaussian policy |
| `warporacer/ppo.py` | PPO: rollout, obs/return normalization, GAE, clipped update, KL-adaptive LR |
| `warporacer/viewer.py` | optional 3D OpenGL live viewer (walls/centerline/cars/lidar) with keyboard drive mode |
| `warporacer/video.py` | mp4 render of a deterministic rollout |
| `warporacer/train.py` | CLI, logging, checkpoints (`.pt`) |

The env writes obs/reward/done into warp buffers that torch reads zero-copy; on CUDA the kernels launch on
torch's stream, so there are no cross-framework syncs. The policy, bookkeeping, and loss are
torch.compile'd, the update GEMMs run in bf16 under autocast, Adam runs fused, and the hot loops
never block on the host (episode stats stream to pinned
memory once per iteration). Values are computed in one batched pass after the rollout. State per car is
`(x, y, psi, v, delta)`; per-episode domain randomization jitters friction and wheelbase by ±15%.
