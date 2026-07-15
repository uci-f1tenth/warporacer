"""Render a deterministic-policy rollout of env 0 to mp4."""

from collections import deque

import imageio.v2 as imageio
import numpy as np
import torch
from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines

from warporacer.sim import DT, LENGTH, WIDTH

TRAIL_LEN = 300


@torch.no_grad()
def record_rollout(env, agent, obs_rms, num_steps: int, out_path):
    snap = env.snapshot()
    track = env.track
    corners = np.array([[-LENGTH, -WIDTH], [LENGTH, -WIDTH], [LENGTH, WIDTH], [-LENGTH, WIDTH]]) / 2.0
    trail = deque(maxlen=TRAIL_LEN)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def to_px(pts):
        cols, rows = track.world_to_px(pts[:, 0], pts[:, 1])
        return np.column_stack([cols, rows]).astype(np.int32)

    env._launch()  # refresh obs from current state
    with imageio.get_writer(str(out_path), fps=round(1 / DT), macro_block_size=2) as writer:
        for _ in range(num_steps):
            action = agent.actor(obs_rms.normalize(env.obs))
            _, _, done = env.step(action)
            x, y, psi = env.cars_t[0, :3].tolist()
            if done[0]:
                trail.clear()
            trail.append((x, y))

            frame = cvtColor(track.image, COLOR_GRAY2RGB)
            if len(trail) > 1:
                polylines(frame, [to_px(np.array(trail))], False, (0, 200, 0), 2)
            rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            fillPoly(frame, [to_px(corners @ rot.T + (x, y))], (255, 50, 50))
            writer.append_data(frame)
    env.restore(snap)
