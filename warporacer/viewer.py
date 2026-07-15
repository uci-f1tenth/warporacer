"""Optional 3D live viewer on warp.render's OpenGL backend (requires pyglet).

Shows one track (env 0's map): walls as a red point cloud, centerline in green, a
subsample of cars as boxes, and env 0's lidar hits in cyan. `interactive()` drives env 0
with I/K (throttle) and J/L (steering)."""

import numpy as np
import torch
import warp as wp
import warp.render
from scipy.ndimage import binary_dilation

from warporacer.track import OCC_THRESH
from warporacer.sim import ACT_DIM, DT, LENGTH, LIDAR_FOV, LIDAR_MOUNT_X, NUM_LIDAR, WIDTH

MAX_CARS = 256
CAR_Z = 0.15


class Viewer:
    def __init__(self, env, headless: bool = False):
        self.env = env
        self.renderer = warp.render.OpenGLRenderer(
            title="warporacer",
            screen_width=1280,
            screen_height=720,
            up_axis="Z",
            near_plane=0.1,
            far_plane=1000.0,
            background_color=(0.0, 0.0, 0.0),
            draw_grid=False,
            draw_axis=False,
            draw_sky=False,
            headless=headless,
            device=env.device,
        )
        self._beam_angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
        self._frame = 0
        self.reset()

    def reset(self):
        """(Re)build static geometry for env 0's current map and the car boxes."""
        self.renderer.clear()
        self.map_idx = int(self.env.map_id_t[0])
        track = self.env.tracks[self.map_idx]
        free = track.image >= OCC_THRESH

        # Bird's-eye default camera over the track (user can still fly with WASD + mouse).
        cx, cy = (track.centerline.min(axis=0) + track.centerline.max(axis=0)) / 2.0
        height = float(np.ptp(track.centerline, axis=0).max()) * 1.2
        self.renderer.camera_far_plane = max(1000.0, 5.0 * height)
        self.renderer.update_projection_matrix()
        pos = np.array([cx, cy - 0.45 * height, 0.9 * height])
        front = np.array([cx, cy, 0.0]) - pos
        self.renderer.update_view_matrix(
            cam_pos=tuple(pos), cam_front=tuple(front / np.linalg.norm(front)), cam_up=(0.0, 1.0, 0.0)
        )

        walls = np.argwhere(binary_dilation(free) & ~free)
        pts = np.zeros((len(walls), 3), np.float32)
        pts[:, 0] = track.ox + walls[:, 1] * track.res
        pts[:, 1] = track.oy + (track.h - 1 - walls[:, 0]) * track.res
        pts[:, 2] = 0.05
        self.renderer.render_points(name="walls", points=pts, radius=0.06, colors=(0.8, 0.2, 0.2))

        cl = np.column_stack([track.centerline, np.full(len(track.centerline), 0.04, np.float32)])
        self.renderer.render_points(name="centerline", points=cl.astype(np.float32),
                                    radius=0.04, colors=(0.2, 0.7, 0.2))

        # Cars on this map (env 0 first), subsampled.
        on_map = (self.env.map_id_t == self.map_idx).nonzero().squeeze(-1).cpu().numpy()
        on_map = np.concatenate([[0], on_map[on_map != 0]])
        self.car_ids = on_map[: MAX_CARS]
        self.car_ids_t = torch.as_tensor(self.car_ids.copy(), device=self.env.obs.device)
        for k in range(len(self.car_ids)):
            f = k / max(len(self.car_ids) - 1, 1)
            self.renderer.render_box(
                name=f"car_{k}", pos=[0.0, 0.0, CAR_Z], rot=[0.0, 0.0, 0.0, 1.0],
                extents=[LENGTH / 2, WIDTH / 2, 0.1],
                color=(1.0, 1.0 - f, 0.1) if k else (0.2, 0.6, 1.0),
            )

    def render(self):
        self._frame += 1
        self.renderer.begin_frame(self._frame * DT)
        cars = self.env.cars_t[self.car_ids_t].cpu().numpy()
        half = np.sin(cars[:, 2] / 2.0)
        for k in range(len(cars)):
            self.renderer.update_shape_instance(
                name=f"car_{k}", pos=[cars[k, 0], cars[k, 1], CAR_Z],
                rot=[0.0, 0.0, float(half[k]), float(np.cos(cars[k, 2] / 2.0))],
            )
        # Lidar hits for env 0, reconstructed from the observation distances.
        x, y, psi = cars[0, :3]
        dist = self.env.obs[0, 2:].cpu().numpy()
        ang = psi + self._beam_angles
        hits = np.stack([
            x + LIDAR_MOUNT_X * np.cos(psi) + dist * np.cos(ang),
            y + LIDAR_MOUNT_X * np.sin(psi) + dist * np.sin(ang),
            np.full(NUM_LIDAR, CAR_Z),
        ], axis=1).astype(np.float32)
        self.renderer.render_points(name="lidar", points=hits, radius=0.05, colors=(0.0, 1.0, 1.0))
        self.renderer.end_frame()

    def interactive(self):
        """Drive env 0 with the keyboard: I/K throttle, J/L steering."""
        import time

        from pyglet.window import key

        handler = key.KeyStateHandler()
        self.renderer.window.push_handlers(handler)
        actions = torch.zeros((self.env.num_envs, ACT_DIM), device=self.env.obs.device)
        last = time.perf_counter()
        while self.renderer.is_running():
            now = time.perf_counter()
            if now - last < DT:
                continue
            last = now
            actions[0, 0] = float(handler[key.J] - handler[key.L])
            actions[0, 1] = float(handler[key.I] - handler[key.K])
            self.env.step(actions)
            self.render()

    def screenshot(self):
        """Render pixels to a numpy image (for headless testing)."""
        img = wp.zeros((self.renderer.screen_height, self.renderer.screen_width, 3),
                       dtype=wp.float32, device=self.env.device)
        self.renderer.get_pixels(img, split_up_tiles=False, mode="rgb")
        return (img.numpy() * 255).astype(np.uint8)
