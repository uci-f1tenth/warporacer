import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import wandb
import warp as wp
from cv2 import (
    COLOR_GRAY2RGB,
    IMREAD_GRAYSCALE,
    cvtColor,
    fillPoly,
    imread,
    polylines,
)
from scipy.ndimage import convolve, distance_transform_edt, label
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from torch.distributions import Normal
from typer import run
from yaml import safe_load

MU = 1.0489
LF = 0.15875
LR = 0.17145
LWB = LF + LR
MASS = 3.74

STEER_MIN = -0.4189
STEER_MAX = 0.4189
STEER_V_MAX = 3.2
A_MAX = 9.51
V_MIN = -5.0
V_MAX = 20.0
PSI_PRIME_MAX = 6.0
BETA_MAX = 1.2

# Car
WIDTH = 0.31
LENGTH = 0.58
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))
G = 9.81
DT = 1.0 / 60.0
SUBSTEPS = 6
DT_SUB = DT / float(SUBSTEPS)
DT_SUB_HALF = DT_SUB * 0.5
DT_SUB_SIX = DT_SUB / 6.0

DR_FRAC = 0.15

PROGRESS_SCALE = 100.0
PROGRESS_V_COEF = 10.0
TERM_PENALTY = 10.0

NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
LIDAR_RANGE = 20.0
OBS_LIDAR_OFF = 2
OBS_DIM = OBS_LIDAR_OFF + NUM_LIDAR
ACT_DIM = 2
MAX_STEPS = 10_000

OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1
DONE_TRUNCATED = 2


@wp.struct
class VDeriv:
    d_x: float
    d_y: float
    d_psi: float
    d_psip: float
    d_beta: float
    d_v: float


@wp.func
def st_deriv(
    delta: float,
    v: float,
    psi: float,
    psip: float,
    beta: float,
    steer_v: float,
    accel: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
) -> VDeriv:
    lf = LF * lf_s
    lr = LR * lr_s
    lwb = lf + lr
    mu = MU * mu_s
    a_max = mu * G

    tand = wp.tan(delta)
    d_psi_kin = v * tand / lwb
    d_psi_cap = a_max / wp.max(wp.abs(v), 0.5)
    d_psi = wp.clamp(d_psi_kin, -d_psi_cap, d_psi_cap)

    a_lat = v * d_psi
    a_long_max = wp.sqrt(wp.max(a_max * a_max - a_lat * a_lat, 0.0))

    cp = wp.cos(psi)
    sp = wp.sin(psi)
    out = VDeriv()
    out.d_x = v * cp
    out.d_y = v * sp
    out.d_psi = d_psi
    out.d_v = wp.clamp(accel, -a_long_max, a_long_max)
    out.d_psip = 0.0
    out.d_beta = 0.0
    return out


@wp.func
def rk4_step(
    delta: float,
    v: float,
    psi: float,
    psip: float,
    beta: float,
    steer_v: float,
    accel: float,
    mu_s: float,
    mass_s: float,
    lf_s: float,
    lr_s: float,
) -> VDeriv:
    dd = steer_v * DT_SUB_HALF
    dd_full = steer_v * DT_SUB

    k1 = st_deriv(delta, v, psi, psip, beta, steer_v, accel, mu_s, mass_s, lf_s, lr_s)
    k2 = st_deriv(
        delta + dd,
        v + k1.d_v * DT_SUB_HALF,
        psi + k1.d_psi * DT_SUB_HALF,
        psip + k1.d_psip * DT_SUB_HALF,
        beta + k1.d_beta * DT_SUB_HALF,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k3 = st_deriv(
        delta + dd,
        v + k2.d_v * DT_SUB_HALF,
        psi + k2.d_psi * DT_SUB_HALF,
        psip + k2.d_psip * DT_SUB_HALF,
        beta + k2.d_beta * DT_SUB_HALF,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    k4 = st_deriv(
        delta + dd_full,
        v + k3.d_v * DT_SUB,
        psi + k3.d_psi * DT_SUB,
        psip + k3.d_psip * DT_SUB,
        beta + k3.d_beta * DT_SUB,
        steer_v,
        accel,
        mu_s,
        mass_s,
        lf_s,
        lr_s,
    )
    out = VDeriv()
    out.d_x = (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT_SUB_SIX
    out.d_y = (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT_SUB_SIX
    out.d_psi = (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT_SUB_SIX
    out.d_v = (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT_SUB_SIX
    out.d_psip = (
        k1.d_psip + 2.0 * k2.d_psip + 2.0 * k3.d_psip + k4.d_psip
    ) * DT_SUB_SIX
    out.d_beta = (
        k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta
    ) * DT_SUB_SIX
    return out


@wp.kernel
def step_kernel(
    actions: wp.array(dtype=wp.vec2),
    obs: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    cars: wp.array2d(dtype=wp.float32),
    cars_int: wp.array2d(dtype=wp.int32),
    car_dr: wp.array2d(dtype=wp.float32),
    origin: wp.vec2,
    res: float,
    dt_map: wp.array2d(dtype=wp.float32),
    cl_lut: wp.array2d(dtype=wp.int32),
    centerline: wp.array(dtype=wp.vec3),
    n_cl: int,
    lidar_dirs: wp.array(dtype=wp.vec2),
    seed_base: int,
):
    i = wp.tid()
    x = cars[i, 0]
    y = cars[i, 1]
    delta = cars[i, 2]
    v = cars[i, 3]
    psi = cars[i, 4]
    psip = cars[i, 5]
    beta = cars[i, 6]
    steps = cars_int[i, 0]
    wp_i = cars_int[i, 1]
    mu_s = car_dr[i, 0]
    mass_s = car_dr[i, 1]
    lf_s = car_dr[i, 2]
    lr_s = car_dr[i, 3]

    mw = dt_map.shape[0]
    mh = dt_map.shape[1]
    mh_f = wp.float32(mh) - 1.0

    # Input
    steer_v = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0.0 and delta <= STEER_MIN) or (steer_v > 0.0 and delta >= STEER_MAX):
        steer_v = 0.0
    accel = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    if (accel < 0.0 and v <= V_MIN) or (accel > 0.0 and v >= V_MAX):
        accel = 0.0

    dd_sub = steer_v * DT_SUB
    for _ in range(SUBSTEPS):
        d = rk4_step(
            delta,
            v,
            psi,
            psip,
            beta,
            steer_v,
            accel,
            mu_s,
            mass_s,
            lf_s,
            lr_s,
        )
        x += d.d_x
        y += d.d_y
        delta += dd_sub
        v += d.d_v
        psi += d.d_psi
        psip += d.d_psip
        beta += d.d_beta

    delta = wp.clamp(delta, STEER_MIN, STEER_MAX)
    v = wp.clamp(v, V_MIN, V_MAX)
    psip = wp.clamp(psip, -PSI_PRIME_MAX, PSI_PRIME_MAX)
    beta = wp.clamp(beta, -BETA_MAX, BETA_MAX)

    # Reward + done
    px = wp.clamp(wp.int32((x - origin[0]) / res), 0, mw - 1)
    py = wp.clamp(wp.int32(mh_f - (y - origin[1]) / res), 0, mh - 1)
    edt_val = dt_map[px, py] * res
    term = edt_val < CAR_HALF_DIAG
    trunc = steps >= MAX_STEPS
    steps += 1

    new_wp = cl_lut[px, py]
    d_wp = new_wp - wp_i
    if 2 * d_wp > n_cl:
        d_wp -= n_cl
    elif 2 * d_wp < -n_cl:
        d_wp += n_cl

    cth = centerline[new_wp][2]
    v_along = v * wp.cos(beta + psi - cth)
    progress = (
        wp.float32(d_wp)
        / wp.float32(n_cl)
        * PROGRESS_SCALE
        * (1.0 + wp.max(v_along, 0.0) / PROGRESS_V_COEF)
    )

    term_pen = wp.where(term, -TERM_PENALTY, 0.0)
    reward[i] = progress + term_pen

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    # Reset
    if term or trunc:
        rng = wp.rand_init(seed_base + i * 73 + steps * 31 + new_wp * 17)
        rnd = wp.int32(wp.randf(rng) * wp.float32(n_cl)) % n_cl
        rpt = centerline[rnd]
        x = rpt[0]
        y = rpt[1]
        psi = rpt[2]
        delta = 0.0
        v = 0.0
        psip = 0.0
        beta = 0.0
        steps = 0
        new_wp = rnd
        car_dr[i, 0] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 1] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 2] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        car_dr[i, 3] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)

    # Lidar
    sh = wp.sin(psi)
    ch = wp.cos(psi)
    lx = x + LF * ch
    ly = y + LF * sh
    lpx = wp.clamp(wp.int32((lx - origin[0]) / res), 0, mw - 1)
    lpy = wp.clamp(wp.int32(mh_f - (ly - origin[1]) / res), 0, mh - 1)
    lpos = wp.vec2(wp.float32(lpx), wp.float32(lpy))
    lrange_px = LIDAR_RANGE / res
    for j in range(lidar_dirs.shape[0]):
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        dpx = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        ray = lpos
        dist = float(0.0)
        while dist < lrange_px:
            rx = wp.int32(ray[0])
            ry = wp.int32(ray[1])
            if rx < 0 or rx >= mw or ry < 0 or ry >= mh:
                break
            step_px = dt_map[rx, ry]
            ray = ray + dpx * step_px
            dist += step_px
            if step_px == 0.0:
                break
        obs[i, OBS_LIDAR_OFF + j] = wp.min(dist, lrange_px) * res

    obs[i, 0] = delta
    obs[i, 1] = v
    cars[i, 0] = x
    cars[i, 1] = y
    cars[i, 2] = delta
    cars[i, 3] = v
    cars[i, 4] = psi
    cars[i, 5] = psip
    cars[i, 6] = beta
    cars_int[i, 0] = steps
    cars_int[i, 1] = new_wp


# Map
class Map:
    def __init__(self, path: Path):
        self.meta = safe_load(path.read_text())
        img_path = path.parent / self.meta["image"]
        self.raw = imread(str(img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(img_path)
        free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(free)
        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(free)
        self._build_lut()

    @staticmethod
    def _neighbors(skel, r, c, h, w):
        return [
            (r + dr, c + dc)
            for dr, dc in ADJ
            if 0 <= r + dr < h and 0 <= c + dc < w and skel[r + dr, c + dc]
        ]

    @staticmethod
    def _prune_endpoints(skel):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        skel = skel.copy()
        while True:
            counts = convolve(
                skel.astype(np.uint8), kernel, mode="constant", cval=0
            )
            endpoints = skel & (counts <= 1)
            if not endpoints.any():
                break
            skel &= ~endpoints
        return skel

    @staticmethod
    def _best_loop_through(start, skel, h, w):
        nbrs = Map._neighbors(skel, start[0], start[1], h, w)
        if len(nbrs) < 2:
            return None
        best = None
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                src, target = nbrs[i], nbrs[j]
                parent = {src: src}
                q = deque([src])
                while q:
                    u = q.popleft()
                    if u == target:
                        break
                    for v in Map._neighbors(skel, u[0], u[1], h, w):
                        if v in parent or v == start:
                            continue
                        parent[v] = u
                        q.append(v)
                if target not in parent:
                    continue
                path = [start]
                n = target
                while n != src:
                    path.append(n)
                    n = parent[n]
                path.append(src)
                path.reverse()
                if best is None or len(path) > len(best):
                    best = path
        return best

    def _compute_centerline(self, free):
        skel = self._prune_endpoints(skeletonize(free))
        # Keep the largest connected component — stray skeleton islands
        # (often from noise outside the track) confuse the loop walk.
        labels, n_lab = label(skel, structure=np.ones((3, 3), dtype=int))
        if n_lab == 0:
            raise RuntimeError("empty skeleton after pruning")
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        skel = labels == int(np.argmax(sizes))

        h, w = skel.shape
        pts = np.argwhere(skel)
        if len(pts) == 0:
            raise RuntimeError("largest skeleton component is empty")
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        order = np.argsort(((pts - origin_px) ** 2).sum(1))

        # Try several candidate starts close to origin and BFS between
        # each pair of the start's neighbours (avoiding start). The
        # longest path found is the track centerline; this stays robust
        # if start lands on a junction or articulation point.
        best_path = None
        for k in range(min(32, len(order))):
            start = tuple(int(x) for x in pts[order[k]])
            loop = self._best_loop_through(start, skel, h, w)
            if loop is None:
                continue
            if best_path is None or len(loop) > len(best_path):
                best_path = loop
        if best_path is None:
            raise RuntimeError("could not extract a closed centerline loop")

        rc = np.array(best_path)
        world = np.column_stack(
            [
                self.ox + rc[:, 1] * self.res,
                self.oy + (self.h - 1 - rc[:, 0]) * self.res,
            ]
        )
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    def _build_lut(self):
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        tree = KDTree(cl_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        self.lut = tree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)


# Env
class RacingEnv:
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def __init__(
        self,
        map_paths,
        num_envs: int,
        seed: int = 0,
        device: str | None = None,
    ):
        wp.init()
        if isinstance(map_paths, (str, Path)):
            map_paths = [map_paths]
        map_paths = [Path(p) for p in map_paths]
        if not map_paths:
            raise ValueError("at least one map path is required")

        self.num_envs = num_envs
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = wp.get_device(requested)
        self.device = str(d)
        self.seed_base = int(seed)

        self.maps = []
        for p in map_paths:
            try:
                self.maps.append(Map(p))
            except Exception as e:
                raise RuntimeError(f"failed to load map {p}: {e}") from e
        n_maps = len(self.maps)
        if num_envs < n_maps:
            raise ValueError(
                f"num_envs ({num_envs}) must be >= number of maps ({n_maps})"
            )

        # Partition envs across maps as evenly as possible.
        sizes = [
            num_envs // n_maps + (1 if i < num_envs % n_maps else 0)
            for i in range(n_maps)
        ]
        splits = []
        cursor = 0
        for sz in sizes:
            splits.append((cursor, cursor + sz))
            cursor += sz

        rng = np.random.default_rng(seed)
        cars = np.zeros((num_envs, 7), dtype=np.float32)
        cars_int = np.zeros((num_envs, 2), dtype=np.int32)
        for i, m in enumerate(self.maps):
            a, b = splits[i]
            idxs = rng.integers(0, len(m.centerline), size=b - a)
            cars[a:b, 0] = m.centerline[idxs, 0]
            cars[a:b, 1] = m.centerline[idxs, 1]
            cars[a:b, 4] = m.angles[idxs]
            cars_int[a:b, 1] = idxs
        dr_init = (
            1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((num_envs, 4), dtype=np.float32)
        )

        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.car_dr = wp.array(dr_init, dtype=float, device=d)
        self.obs = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(num_envs, dtype=float, device=d)
        self.done = wp.zeros(num_envs, dtype=int, device=d)

        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self.cars_buf = wp.to_torch(self.cars)
        self.cars_int_buf = wp.to_torch(self.cars_int)
        self.car_dr_buf = wp.to_torch(self.car_dr)
        self._step_counter = self.cars_int_buf[:, 0]

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )
        self._zero_act = wp.zeros(num_envs, dtype=wp.vec2, device=d)
        self._zero_act_torch = wp.to_torch(self._zero_act)

        # Per-map warp buffers + cached sliced views into the per-env tensors.
        self.map_buffers = []
        for i, m in enumerate(self.maps):
            a, b = splits[i]
            self.map_buffers.append(
                {
                    "map": m,
                    "env_start": a,
                    "env_end": b,
                    "n_envs": b - a,
                    "origin": wp.vec2(m.ox, m.oy),
                    "res": float(m.res),
                    "n_cl": int(len(m.centerline)),
                    "dt": wp.array(m.dt.T.astype(np.float32), dtype=float, device=d),
                    "lut": wp.array(m.lut.T.astype(np.int32), dtype=int, device=d),
                    "cl": wp.array(
                        np.column_stack([m.centerline, m.angles]).astype(np.float32),
                        dtype=wp.vec3,
                        device=d,
                    ),
                    "cars_v": wp.from_torch(self.cars_buf[a:b]),
                    "cars_int_v": wp.from_torch(self.cars_int_buf[a:b]),
                    "car_dr_v": wp.from_torch(self.car_dr_buf[a:b]),
                    "obs_v": wp.from_torch(self.obs_buf[a:b]),
                    "rew_v": wp.from_torch(self.rew_buf[a:b]),
                    "done_v": wp.from_torch(self.done_buf[a:b]),
                    "zero_act_v": wp.from_torch(
                        self._zero_act_torch[a:b], dtype=wp.vec2
                    ),
                }
            )

        self._call = 0
        # Warm-up reset
        self._launch_zero()
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()

    def _launch_one(self, mb, act_w, seed):
        wp.launch(
            step_kernel,
            dim=mb["n_envs"],
            inputs=[
                act_w,
                mb["obs_v"],
                mb["rew_v"],
                mb["done_v"],
                mb["cars_v"],
                mb["cars_int_v"],
                mb["car_dr_v"],
                mb["origin"],
                mb["res"],
                mb["dt"],
                mb["lut"],
                mb["cl"],
                mb["n_cl"],
                self.lidar_buf,
                int(seed),
            ],
            device=self.cars.device,
        )

    def _launch(self, act_torch):
        seed_step = (
            self.seed_base * 2654435761 + self._call * 83492791
        ) & 0x7FFFFFFF
        for i, mb in enumerate(self.map_buffers):
            a, b = mb["env_start"], mb["env_end"]
            act_w = wp.from_torch(act_torch[a:b], dtype=wp.vec2)
            self._launch_one(mb, act_w, seed_step ^ ((i + 1) * 982451653))
        wp.synchronize_device(self.cars.device)
        self._call += 1

    def _launch_zero(self):
        seed_step = (
            self.seed_base * 2654435761 + self._call * 83492791
        ) & 0x7FFFFFFF
        for i, mb in enumerate(self.map_buffers):
            self._launch_one(mb, mb["zero_act_v"], seed_step ^ ((i + 1) * 982451653))
        wp.synchronize_device(self.cars.device)
        self._call += 1

    def _sanitize(self):
        bad = ~(
            torch.isfinite(self.obs_buf).all(1) & torch.isfinite(self.cars_buf).all(1)
        )
        if not bad.any():
            return
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self._step_counter[bad] = MAX_STEPS
        self.done_buf[bad] = DONE_TRUNCATED

    def reset(self):
        self._step_counter.fill_(MAX_STEPS)
        self._launch_zero()
        self._sanitize()
        self._step_counter.zero_()
        self.rew_buf.zero_()
        self.done_buf.zero_()
        return self.obs_buf, {}

    def step(self, action):
        act = action.detach().to(self.obs_buf.device, non_blocking=True).contiguous()
        self._launch(act)
        self._sanitize()
        return (
            self.obs_buf,
            self.rew_buf,
            self.done_buf == DONE_TERMINATED,
            self.done_buf == DONE_TRUNCATED,
            {},
        )

    def save_state(self):
        return {
            k: getattr(self, k).clone()
            for k in (
                "cars_buf",
                "cars_int_buf",
                "car_dr_buf",
                "obs_buf",
                "rew_buf",
                "done_buf",
            )
        }

    def restore_state(self, s):
        self.cars_buf.copy_(s["cars_buf"])
        self.cars_int_buf.copy_(s["cars_int_buf"])
        self.car_dr_buf.copy_(s["car_dr_buf"])
        self.obs_buf.copy_(s["obs_buf"])
        self.rew_buf.copy_(s["rew_buf"])
        self.done_buf.copy_(s["done_buf"])


# PPO components
class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape).float()
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        self.var = (
            self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)
        ) / tot
        self.count = tot
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x, clip: float = 10.0):
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


class ReturnNormalizer:
    def __init__(self, num_envs, gamma, device):
        self.gamma = gamma
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.rms = RunningMeanStd((), device)

    def update(self, reward, done):
        self.returns = self.returns * self.gamma * (1.0 - done) + reward
        self.rms.update(self.returns)

    def normalize(self, reward):
        return reward * self.rms.inv_std


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.full((1, act_dim), -0.5))

    def _dist(self, obs):
        mean = self.actor(obs)
        ls = self.log_std.expand_as(mean).clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return Normal(mean, ls.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act_value(self, obs, action=None):
        d = self._dist(obs)
        if action is None:
            action = d.sample()
        return action, d.log_prob(action).sum(-1), d.entropy().sum(-1), self.value(obs)

    def deterministic(self, obs):
        return self.actor(obs)


class KLAdaptiveLR:
    def __init__(self, opt, target_kl=0.02, factor=1.5, lr_min=1e-6, lr_max=3e-3):
        self.opt = opt
        self.target = target_kl
        self.factor = factor
        self.lr_min = lr_min
        self.lr_max = lr_max

    def step(self, kl):
        for pg in self.opt.param_groups:
            lr = pg["lr"]
            if kl > 2.0 * self.target:
                pg["lr"] = max(self.lr_min, lr / self.factor)
            elif kl < 0.5 * self.target:
                pg["lr"] = min(self.lr_max, lr * self.factor)

    @property
    def lr(self):
        return self.opt.param_groups[0]["lr"]


# Rollout video
def record_rollout(env, agent, num_steps, out_path, obs_rms=None):
    snap = env.save_state()
    was_training = agent.training
    agent.eval()
    try:
        m = env.maps[0]
        corners = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        def w2p(x, y):
            return int((x - m.ox) / m.res), int(m.h - 1 - (y - m.oy) / m.res)

        trail = deque(maxlen=300)
        raw, _ = env.reset()
        obs = obs_rms.normalize(raw) if obs_rms else raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(out_path), fps=int(1 / DT), macro_block_size=2
        ) as w:
            with torch.no_grad():
                for _ in range(num_steps):
                    a = agent.deterministic(obs)
                    raw, _, term, trunc, _ = env.step(a)
                    obs = obs_rms.normalize(raw) if obs_rms else raw
                    row = env.cars_buf[0].tolist()
                    x, y, psi = row[0], row[1], row[4]
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        trail.clear()
                    trail.append((x, y))
                    frame = cvtColor(m.raw, COLOR_GRAY2RGB)
                    if len(trail) > 1:
                        polylines(
                            frame,
                            [np.array([w2p(*p) for p in trail], dtype=np.int32)],
                            False,
                            (0, 200, 0),
                            2,
                        )
                    R = np.array(
                        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                    )
                    world = corners @ R.T + (x, y)
                    fillPoly(
                        frame,
                        [np.array([w2p(*p) for p in world], dtype=np.int32)],
                        (255, 50, 50),
                    )
                    w.append_data(frame)
    finally:
        env.restore_state(snap)
        agent.train(was_training)


# PPO training
def train(
    env,
    agent,
    iterations=2000,
    rollouts=24,
    epochs=5,
    minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    vf_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    max_grad_norm=0.5,
    lr=3e-4,
    target_kl=0.02,
    log_dir=Path("./logs"),
    record_every=100,
    record_steps=1800,
):
    device = next(agent.parameters()).device
    N = env.num_envs
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    sched = KLAdaptiveLR(opt, target_kl=target_kl)
    obs_rms = RunningMeanStd((OBS_DIM,), device)
    ret_rms = ReturnNormalizer(N, gamma, device)

    obs_b = torch.zeros((rollouts, N, OBS_DIM), device=device)
    act_b = torch.zeros((rollouts, N, ACT_DIM), device=device)
    logp_b = torch.zeros((rollouts, N), device=device)
    rew_b = torch.zeros((rollouts, N), device=device)
    done_b = torch.zeros((rollouts, N), device=device)
    val_b = torch.zeros((rollouts, N), device=device)

    raw, _ = env.reset()
    obs_rms.update(raw)
    obs = obs_rms.normalize(raw)
    ep_ret = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    finished_rets, finished_lens = deque(maxlen=100), deque(maxlen=100)

    global_step = 0
    t0 = time.time()
    last_t = t0

    for it in range(iterations):
        agent.eval()
        with torch.no_grad():
            for t in range(rollouts):
                obs_b[t] = obs
                act, logp, _, val = agent.act_value(obs)
                act_b[t] = act
                logp_b[t] = logp
                val_b[t] = val
                raw, raw_rew, term, trunc, _ = env.step(act)
                done = (term | trunc).float()
                ret_rms.update(raw_rew, done)
                rew_b[t] = ret_rms.normalize(raw_rew)
                done_b[t] = done
                ep_ret.add_(raw_rew)
                ep_len.add_(1.0)
                fin = done.bool()
                if fin.any():
                    finished_rets.extend(ep_ret[fin].cpu().tolist())
                    finished_lens.extend(ep_len[fin].cpu().tolist())
                    ep_ret[fin] = 0.0
                    ep_len[fin] = 0.0
                obs_rms.update(raw)
                obs = obs_rms.normalize(raw)
            next_val = agent.value(obs)

        # GAE — the kernel resets terminated/truncated envs in-place, so the
        # obs (and val_ext[t+1]) at a done step is the FRESH spawn of a new
        # episode, not the dying state. Zero the bootstrap on any done so we
        # don't credit the new episode's value to the old one's advantage.
        val_ext = torch.cat([val_b, next_val.unsqueeze(0)], 0)
        adv_b = torch.zeros_like(rew_b)
        last = torch.zeros_like(next_val)
        for t in reversed(range(rollouts)):
            nondone = 1.0 - done_b[t]
            delta = rew_b[t] + gamma * val_ext[t + 1] * nondone - val_b[t]
            last = delta + gamma * gae_lambda * nondone * last
            adv_b[t] = last
        ret_b = adv_b + val_b
        global_step += rollouts * N

        # Flatten
        B = rollouts * N
        b_obs = obs_b.reshape(B, OBS_DIM)
        b_act = act_b.reshape(B, ACT_DIM)
        b_logp = logp_b.reshape(B)
        b_adv = adv_b.reshape(B)
        b_ret = ret_b.reshape(B)
        b_val = val_b.reshape(B)
        mb = B // minibatches

        agent.train()
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd = 0
        kl_stop = False
        for epoch in range(epochs):
            perm = torch.randperm(B, device=device)
            epoch_kl = 0.0
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                _, new_logp, ent, new_val = agent.act_value(b_obs[idx], b_act[idx])
                logratio = new_logp - b_logp[idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()
                epoch_kl += approx_kl
                adv_mb = b_adv[idx]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                s1 = ratio * adv_mb
                s2 = ratio.clamp(1 - clip, 1 + clip) * adv_mb
                pg = -torch.min(s1, s2).mean()

                v_err = new_val - b_ret[idx]
                if vf_clip > 0:
                    v_clipped = b_val[idx] + (new_val - b_val[idx]).clamp(
                        -vf_clip, vf_clip
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            v_err.square(), (v_clipped - b_ret[idx]).square()
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * v_err.square().mean()
                ent_m = ent.mean()
                loss = pg + vf_coef * v_loss - ent_coef * ent_m
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    agent.log_std.clamp_(Agent.LOGSTD_MIN, Agent.LOGSTD_MAX)
                stats["pg"] += pg.item()
                stats["v"] += v_loss.item()
                stats["ent"] += ent_m.item()
                stats["kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                n_upd += 1
            if epoch_kl / max(minibatches, 1) > 1.5 * target_kl:
                kl_stop = True
                break
        for k in stats:
            stats[k] /= max(n_upd, 1)
        sched.step(stats["kl"])

        now = time.time()
        sps = int(rollouts * N / max(now - last_t, 1e-9))
        last_t = now
        log = {
            "policy_loss": stats["pg"],
            "value_loss": stats["v"],
            "entropy": stats["ent"],
            "approx_kl": stats["kl"],
            "clipfrac": stats["clipfrac"],
            "kl_stop": int(kl_stop),
            "log_std": agent.log_std.mean().item(),
            "lr": sched.lr,
            "sps": sps,
            "iteration": it,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))
        try:
            wandb.log(log, step=global_step)
        except Exception:
            pass
        if it % 10 == 0:
            er = log.get("ep_return", float("nan"))
            print(
                f"[it {it:4d}] step={global_step:>9d} sps={sps:>6d} "
                f"ret={er:8.2f} kl={stats['kl']:.4f} lr={sched.lr:.2e}"
                f"{' KL-STOP' if kl_stop else ''}"
            )
        if record_every > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
                wandb.log(
                    {"rollout": wandb.Video(str(out), format="mp4")}, step=global_step
                )
            except Exception as e:
                print(f"[rollout {it + 1}] failed: {e}")
    return time.time() - t0, obs_rms, ret_rms, global_step


def main(
    map_yamls: list[Path],
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = RacingEnv(map_yamls, num_envs=num_envs, seed=seed, device=device or None)
    agent = Agent(obs_dim=OBS_DIM).to(env.device)

    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}_m{len(map_yamls)}",
                config={
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "seed": seed,
                    "maps": [str(p) for p in map_yamls],
                },
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    elapsed, obs_rms, ret_rms, step = train(
        env,
        agent,
        iterations=iterations,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(f"[done] {elapsed:.1f}s")

    torch.save(
        {
            "agent": agent.state_dict(),
            "obs_mean": obs_rms.mean.cpu(),
            "obs_var": obs_rms.var.cpu(),
            "obs_count": obs_rms.count,
            "config": {
                "obs_dim": OBS_DIM,
                "act_dim": ACT_DIM,
                "hidden": 256,
                "num_lidar": NUM_LIDAR,
                "lidar_fov": float(LIDAR_FOV),
                "lidar_range": LIDAR_RANGE,
                "steer_v_max": STEER_V_MAX,
                "a_max": A_MAX,
                "steer_min": STEER_MIN,
                "steer_max": STEER_MAX,
                "v_min": V_MIN,
                "v_max": V_MAX,
                "dt": DT,
            },
        },
        log_dir / "agent_final.pt",
    )

    out = log_dir / "rollout_final.mp4"
    record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
    try:
        wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
    except Exception:
        pass


if __name__ == "__main__":
    run(main)
