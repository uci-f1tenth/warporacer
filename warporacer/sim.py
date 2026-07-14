"""Vectorized racing sim: one Warp kernel steps physics, reward, lidar, and respawns."""

import numpy as np
import warp as wp

# Car (F1TENTH)
MU = 1.0489  # tire friction
WHEELBASE = 0.3302  # m
WIDTH, LENGTH = 0.42, 0.58  # m
HALF_DIAG = float(np.hypot(WIDTH, LENGTH) / 2.0)
LIDAR_MOUNT_X = 0.2733  # m, base_link -> laser offset
G = 9.81

# Actuation limits
STEER_MAX = 0.4189  # rad
STEER_V_MAX = 3.2  # rad/s
A_MAX = 9.51  # m/s^2
V_MIN, V_MAX = -5.0, 5.0  # m/s

DT = 1.0 / 60.0
SUBSTEPS = 6
DT_SUB = DT / SUBSTEPS
MAX_STEPS = 10_000
DR_FRAC = 0.15  # per-episode +-15% domain randomization of mu and wheelbase

# Lidar
NUM_LIDAR = 108
LIDAR_FOV = float(np.radians(270.0))
LIDAR_RANGE = 20.0  # m

# Observation = [steer, speed, lidar...]
OBS_DIM = 2 + NUM_LIDAR
ACT_DIM = 2  # [steer rate, accel], both in [-1, 1]

# Reward
PROGRESS_SCALE = 100.0
PROGRESS_V_COEF = 10.0
TERM_PENALTY = 25.0
WALL_MARGIN = 0.20  # m, wall-proximity penalty band
WALL_PROX_COEF = 0.2
CENTER_COEF = 1.0  # symmetric centerline-offset penalty (opposes wall hugging)


@wp.func
def deriv(s: wp.vec4, delta: float, accel: float, mu: float, lwb: float) -> wp.vec4:
    """Kinematic bicycle with a friction-circle cap. State s = (x, y, psi, v)."""
    a_max = mu * G
    cap = a_max / wp.max(wp.abs(s[3]), 0.5)
    d_psi = wp.clamp(s[3] * wp.tan(delta) / lwb, -cap, cap)
    a_lon = wp.sqrt(wp.max(a_max * a_max - (s[3] * d_psi) * (s[3] * d_psi), 0.0))
    return wp.vec4(s[3] * wp.cos(s[2]), s[3] * wp.sin(s[2]), d_psi, wp.clamp(accel, -a_lon, a_lon))


@wp.func
def rk4(s: wp.vec4, delta: float, steer_v: float, accel: float, mu: float, lwb: float) -> wp.vec4:
    d_mid = delta + steer_v * DT_SUB * 0.5
    k1 = deriv(s, delta, accel, mu, lwb)
    k2 = deriv(s + k1 * (DT_SUB * 0.5), d_mid, accel, mu, lwb)
    k3 = deriv(s + k2 * (DT_SUB * 0.5), d_mid, accel, mu, lwb)
    k4 = deriv(s + k3 * DT_SUB, delta + steer_v * DT_SUB, accel, mu, lwb)
    return s + (k1 + 2.0 * (k2 + k3) + k4) * (DT_SUB / 6.0)


@wp.func
def raycast(edt: wp.array2d(dtype=float), p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """Sphere-march from pixel p along unit direction d (col, row); returns distance in px."""
    dist = float(0.0)
    while dist < max_px:
        col = wp.int32(p[0])
        row = wp.int32(p[1])
        if col < 0 or col >= edt.shape[1] or row < 0 or row >= edt.shape[0]:
            break
        step = edt[row, col]
        if step == 0.0:
            break
        p += d * step
        dist += step
    return wp.min(dist, max_px)


@wp.kernel
def bump_kernel(tick: wp.array(dtype=wp.int32)):
    tick[0] += 1


@wp.kernel
def step_kernel(
    actions: wp.array2d(dtype=float),
    cars: wp.array2d(dtype=float),  # x, y, psi, v, delta
    cars_i: wp.array2d(dtype=wp.int32),  # steps, waypoint
    dr: wp.array2d(dtype=float),  # mu scale, wheelbase scale
    edt: wp.array2d(dtype=float),  # (h, w) px distance to nearest wall
    lut: wp.array2d(dtype=wp.int32),  # (h, w) nearest centerline waypoint
    centerline: wp.array(dtype=wp.vec3),  # x, y, theta
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.vec2,
    res: float,
    seed: int,
    tick: wp.array(dtype=wp.int32),  # device RNG clock: keeps randomness fresh across CUDA graph replays
    obs: wp.array2d(dtype=float),
    rew: wp.array(dtype=float),
    done: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    s = wp.vec4(cars[i, 0], cars[i, 1], cars[i, 2], cars[i, 3])
    delta = cars[i, 4]
    mu = MU * dr[i, 0]
    lwb = WHEELBASE * dr[i, 1]

    steer_v = wp.clamp(actions[i, 0], -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0.0 and delta <= -STEER_MAX) or (steer_v > 0.0 and delta >= STEER_MAX):
        steer_v = 0.0
    accel = wp.clamp(actions[i, 1], -1.0, 1.0) * A_MAX
    if (accel < 0.0 and s[3] <= V_MIN) or (accel > 0.0 and s[3] >= V_MAX):
        accel = 0.0

    for _ in range(SUBSTEPS):
        s = rk4(s, delta, steer_v, accel, mu, lwb)
        delta += steer_v * DT_SUB
    delta = wp.clamp(delta, -STEER_MAX, STEER_MAX)
    s[3] = wp.clamp(s[3], V_MIN, V_MAX)

    h = edt.shape[0]
    w = edt.shape[1]
    col = wp.clamp(wp.int32((s[0] - origin[0]) / res), 0, w - 1)
    row = wp.clamp(wp.int32(wp.float32(h - 1) - (s[1] - origin[1]) / res), 0, h - 1)
    clearance = edt[row, col] * res - HALF_DIAG

    # Signed centerline progress, wrapped to the shorter way around the loop.
    n_cl = centerline.shape[0]
    wpt = lut[row, col]
    d_wp = wpt - cars_i[i, 1]
    if 2 * d_wp > n_cl:
        d_wp -= n_cl
    elif 2 * d_wp < -n_cl:
        d_wp += n_cl

    cpt = centerline[wpt]
    v_along = s[3] * wp.cos(s[2] - cpt[2])
    progress = float(d_wp) / float(n_cl) * PROGRESS_SCALE * (1.0 + wp.max(v_along, 0.0) / PROGRESS_V_COEF)
    prox = wp.max((WALL_MARGIN - clearance) / WALL_MARGIN, 0.0)
    prox_pen = WALL_PROX_COEF * prox * prox * (1.0 + wp.max(s[3], 0.0) / PROGRESS_V_COEF)
    offset = wp.abs(-(s[0] - cpt[0]) * wp.sin(cpt[2]) + (s[1] - cpt[1]) * wp.cos(cpt[2]))

    crashed = clearance < 0.0
    steps = cars_i[i, 0] + 1
    rew[i] = wp.where(crashed, -TERM_PENALTY, progress - prox_pen - CENTER_COEF * offset * offset)
    done[i] = wp.int32(0)

    if crashed or steps >= MAX_STEPS:  # respawn at a random waypoint with fresh randomization
        done[i] = 1
        rng = wp.rand_init(seed, tick[0] * actions.shape[0] + i)
        wpt = wp.int32(wp.randf(rng) * float(n_cl)) % n_cl
        rpt = centerline[wpt]
        s = wp.vec4(rpt[0], rpt[1], rpt[2], 0.0)
        delta = 0.0
        steps = 0
        dr[i, 0] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        dr[i, 1] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)

    # Lidar from the mount point, beams rotated into the heading frame (row axis flips y).
    ch = wp.cos(s[2])
    sh = wp.sin(s[2])
    p = wp.vec2(
        (s[0] + LIDAR_MOUNT_X * ch - origin[0]) / res,
        wp.float32(h - 1) - (s[1] + LIDAR_MOUNT_X * sh - origin[1]) / res,
    )
    for j in range(NUM_LIDAR):
        d = wp.vec2(
            ch * lidar_dirs[j][0] - sh * lidar_dirs[j][1],
            -(sh * lidar_dirs[j][0] + ch * lidar_dirs[j][1]),
        )
        obs[i, 2 + j] = raycast(edt, p, d, LIDAR_RANGE / res) * res
    obs[i, 0] = delta
    obs[i, 1] = s[3]

    cars[i, 0] = s[0]
    cars[i, 1] = s[1]
    cars[i, 2] = s[2]
    cars[i, 3] = s[3]
    cars[i, 4] = delta
    cars_i[i, 0] = steps
    cars_i[i, 1] = wpt


class Env:
    """num_envs cars on one track; step() writes obs/rew/done into caller-provided buffers."""

    def __init__(self, track, num_envs: int, seed: int = 0, device=None):
        self.track = track
        self.num_envs = num_envs
        self.device = wp.get_device(device)
        self.seed = seed
        d = self.device
        self.tick = wp.zeros(1, dtype=wp.int32, device=d)

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(track.centerline), num_envs)
        cars = np.zeros((num_envs, 5), np.float32)
        cars[:, :2] = track.centerline[idx]
        cars[:, 2] = track.angles[idx]
        cars_i = np.zeros((num_envs, 2), np.int32)
        cars_i[:, 1] = idx
        self.cars = wp.array(cars, device=d)
        self.cars_i = wp.array(cars_i, device=d)
        self.dr = wp.array(
            (1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((num_envs, 2))).astype(np.float32), device=d
        )

        self.edt = wp.array(track.edt.astype(np.float32), device=d)
        self.lut = wp.array(track.lut, device=d)
        self.centerline = wp.array(
            np.column_stack([track.centerline, track.angles]).astype(np.float32), dtype=wp.vec3, device=d
        )
        beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
        self.lidar_dirs = wp.array(
            np.column_stack([np.cos(beams), np.sin(beams)]).astype(np.float32), dtype=wp.vec2, device=d
        )
        self.origin = wp.vec2(track.ox, track.oy)

        # Scratch outputs, used for the initial observation and by video recording.
        self.obs = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(num_envs, dtype=float, device=d)
        self.done = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.zero_actions = wp.zeros((num_envs, ACT_DIM), dtype=float, device=d)
        self.step(self.zero_actions, self.obs, self.rew, self.done)  # v=0, so cars stay put

    def step(self, actions, obs, rew, done):
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                actions, self.cars, self.cars_i, self.dr, self.edt, self.lut,
                self.centerline, self.lidar_dirs, self.origin, self.track.res,
                self.seed, self.tick,
            ],
            outputs=[obs, rew, done],
            device=self.device,
        )
        wp.launch(bump_kernel, dim=1, inputs=[self.tick], device=self.device)

    def snapshot(self):
        return [wp.clone(a) for a in (self.cars, self.cars_i, self.dr)]

    def restore(self, snap):
        for dst, src in zip((self.cars, self.cars_i, self.dr), snap):
            wp.copy(dst, src)
