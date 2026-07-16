"""Vectorized racing sim on Warp: one kernel steps physics/reward/respawn, one raycasts
lidar. Supports many tracks at once — map rasters are stacked into padded (n_maps, H, W)
buffers and each env carries a map id. The Env speaks torch: obs/reward/done live in
zero-copy torch views, and on CUDA the kernels launch on torch's stream."""

import numpy as np
import torch
import warp as wp

# fast_math: SFU trig for the RK4/lidar transcendentals (+22% sim throughput; domain
# randomization dwarfs the precision loss). No warp autodiff is used anywhere.
wp.set_module_options({"fast_math": True, "enable_backward": False})

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
def raycast(edt: wp.array3d(dtype=float), mid: int, p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """Sphere-march from pixel p along unit direction d (col, row); returns distance in px.
    Padding rows/cols are zero (= wall), so rays stop at each map's true bounds."""
    dist = float(0.0)
    while dist < max_px:
        col = wp.int32(p[0])
        row = wp.int32(p[1])
        if col < 0 or col >= edt.shape[2] or row < 0 or row >= edt.shape[1]:
            break
        step = edt[mid, row, col]
        if step == 0.0:
            break
        p += d * step
        dist += step
    return wp.min(dist, max_px)


@wp.func
def raycast_tex(tex: wp.Texture3D, midf: float, p: wp.vec2, d: wp.vec2, max_px: float) -> float:
    """raycast() against the edt texture: border sampling returns 0 (wall) outside the
    volume, so no explicit bounds check is needed."""
    dist = float(0.0)
    while dist < max_px:
        step = wp.texture_sample(tex, wp.vec3(p[0], p[1], midf), dtype=float)
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
    map_id: wp.array(dtype=wp.int32),
    edt: wp.array3d(dtype=float),  # (n_maps, H, W) px distance to nearest wall
    lut: wp.array3d(dtype=wp.int32),  # (n_maps, H, W) nearest centerline waypoint
    centerline: wp.array2d(dtype=wp.vec3),  # (n_maps, CL) x, y, theta (padded)
    n_cl: wp.array(dtype=wp.int32),  # centerline length per map
    origin: wp.array(dtype=wp.vec2),  # world origin per map
    res: wp.array(dtype=float),  # meters/px per map
    map_h: wp.array(dtype=wp.int32),  # unpadded rows per map
    map_w: wp.array(dtype=wp.int32),  # unpadded cols per map
    seed: int,
    tick: wp.array(dtype=wp.int32),  # device RNG clock: fresh respawn randomness every step
    obs: wp.array2d(dtype=float),
    rew: wp.array(dtype=float),
    done: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    mid = map_id[i]
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

    r = res[mid]
    org = origin[mid]
    col = wp.clamp(wp.int32((s[0] - org[0]) / r), 0, map_w[mid] - 1)
    row = wp.clamp(wp.int32(wp.float32(map_h[mid] - 1) - (s[1] - org[1]) / r), 0, map_h[mid] - 1)
    clearance = edt[mid, row, col] * r - HALF_DIAG

    # Signed centerline progress, wrapped to the shorter way around the loop.
    ncl = n_cl[mid]
    wpt = lut[mid, row, col]
    d_wp = wpt - cars_i[i, 1]
    if 2 * d_wp > ncl:
        d_wp -= ncl
    elif 2 * d_wp < -ncl:
        d_wp += ncl

    cpt = centerline[mid, wpt]
    v_along = s[3] * wp.cos(s[2] - cpt[2])
    progress = float(d_wp) / float(ncl) * PROGRESS_SCALE * (1.0 + wp.max(v_along, 0.0) / PROGRESS_V_COEF)
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
        wpt = wp.int32(wp.randf(rng) * float(ncl)) % ncl
        rpt = centerline[mid, wpt]
        s = wp.vec4(rpt[0], rpt[1], rpt[2], 0.0)
        delta = 0.0
        steps = 0
        dr[i, 0] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)
        dr[i, 1] = 1.0 - DR_FRAC + 2.0 * DR_FRAC * wp.randf(rng)

    obs[i, 0] = delta
    obs[i, 1] = s[3]
    cars[i, 0] = s[0]
    cars[i, 1] = s[1]
    cars[i, 2] = s[2]
    cars[i, 3] = s[3]
    cars[i, 4] = delta
    cars_i[i, 0] = steps
    cars_i[i, 1] = wpt


@wp.kernel
def lidar_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=wp.int32),
    edt: wp.array3d(dtype=float),
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=float),
):
    """One thread per (car, beam): rays from the mount point, rotated into the heading
    frame (row axis flips y). Runs after step_kernel so respawned poses are used."""
    i, j = wp.tid()
    mid = map_id[i]
    r = res[mid]
    org = origin[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = wp.vec2(
        (cars[i, 0] + LIDAR_MOUNT_X * ch - org[0]) / r,
        wp.float32(map_h[mid] - 1) - (cars[i, 1] + LIDAR_MOUNT_X * sh - org[1]) / r,
    )
    d = wp.vec2(
        ch * lidar_dirs[j][0] - sh * lidar_dirs[j][1],
        -(sh * lidar_dirs[j][0] + ch * lidar_dirs[j][1]),
    )
    obs[i, 2 + j] = raycast(edt, mid, p, d, LIDAR_RANGE / r) * r


@wp.kernel
def lidar_tex_kernel(
    cars: wp.array2d(dtype=float),
    map_id: wp.array(dtype=wp.int32),
    edt_tex: wp.Texture3D,
    lidar_dirs: wp.array(dtype=wp.vec2),
    origin: wp.array(dtype=wp.vec2),
    res: wp.array(dtype=float),
    map_h: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=float),
):
    """lidar_kernel sampling the edt as a 3D texture (CUDA only): the texture cache fits
    the scattered sphere-march reads much better than global memory (+14% sim throughput).
    Nearest-sampling at (col, row, map) matches edt[mid, int(row), int(col)] exactly."""
    i, j = wp.tid()
    mid = map_id[i]
    r = res[mid]
    org = origin[mid]
    ch = wp.cos(cars[i, 2])
    sh = wp.sin(cars[i, 2])
    p = wp.vec2(
        (cars[i, 0] + LIDAR_MOUNT_X * ch - org[0]) / r,
        wp.float32(map_h[mid] - 1) - (cars[i, 1] + LIDAR_MOUNT_X * sh - org[1]) / r,
    )
    d = wp.vec2(
        ch * lidar_dirs[j][0] - sh * lidar_dirs[j][1],
        -(sh * lidar_dirs[j][0] + ch * lidar_dirs[j][1]),
    )
    obs[i, 2 + j] = raycast_tex(edt_tex, wp.float32(mid) + 0.5, p, d, LIDAR_RANGE / r) * r


class Env:
    """num_envs cars spread across one or more tracks.
    step(actions) -> (obs, reward, done) torch tensors; rotate() swaps the track pool."""

    def __init__(self, tracks, num_envs: int, seed: int = 0, device=None):
        self.num_envs = num_envs
        self.device = wp.get_device(device)
        self.torch_device = wp.device_to_torch(self.device)
        self.seed = seed
        self._loads = 0
        d = self.device
        self.tick = wp.zeros(1, dtype=wp.int32, device=d)

        # Persistent per-env buffers (zero-copy torch views must stay valid across rotate()).
        self.cars = wp.zeros((num_envs, 5), dtype=float, device=d)
        self.cars_i = wp.zeros((num_envs, 2), dtype=wp.int32, device=d)
        self.dr = wp.zeros((num_envs, 2), dtype=float, device=d)
        self.map_id = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.act = wp.zeros((num_envs, ACT_DIM), dtype=float, device=d)
        self.obs_w = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew_w = wp.zeros(num_envs, dtype=float, device=d)
        self.done_w = wp.zeros(num_envs, dtype=wp.int32, device=d)
        self.act_t = wp.to_torch(self.act)
        self.obs = wp.to_torch(self.obs_w)
        self.rew = wp.to_torch(self.rew_w)
        self.done = wp.to_torch(self.done_w)
        self.cars_t = wp.to_torch(self.cars)
        self.map_id_t = wp.to_torch(self.map_id)

        beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
        self.lidar_dirs = wp.array(
            np.column_stack([np.cos(beams), np.sin(beams)]).astype(np.float32), dtype=wp.vec2, device=d
        )

        # Launch on torch's stream so torch reads warp results without syncing.
        self.stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.torch_device))
            if self.device.is_cuda
            else None
        )
        self.rotate(tracks)

    def rotate(self, tracks):
        """Load a track pool: stack rasters into padded buffers, respawn every env."""
        if not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        self.tracks = list(tracks)
        d = self.device
        m = len(self.tracks)
        hmax = max(t.h for t in self.tracks)
        wmax = max(t.w for t in self.tracks)
        clmax = max(len(t.centerline) for t in self.tracks)

        edt = np.zeros((m, hmax, wmax), np.float32)
        lut = np.zeros((m, hmax, wmax), np.int32)
        cl = np.zeros((m, clmax, 3), np.float32)
        for k, t in enumerate(self.tracks):
            edt[k, : t.h, : t.w] = t.edt
            lut[k, : t.h, : t.w] = t.lut
            cl[k, : len(t.centerline), :2] = t.centerline
            cl[k, : len(t.centerline), 2] = t.angles
        self.edt = wp.array(edt, device=d)
        if self.device.is_cuda:
            self.edt_tex = wp.Texture3D(
                self.edt,
                filter_mode=wp.TextureFilterMode.CLOSEST,
                address_mode=wp.TextureAddressMode.BORDER,
                normalized_coords=False,
                device=d,
            )
        self.lut = wp.array(lut, device=d)
        self.centerline = wp.array(cl, dtype=wp.vec3, device=d)
        self.n_cl = wp.array([len(t.centerline) for t in self.tracks], dtype=wp.int32, device=d)
        self.origin = wp.array([wp.vec2(t.ox, t.oy) for t in self.tracks], dtype=wp.vec2, device=d)
        self.res = wp.array([t.res for t in self.tracks], dtype=float, device=d)
        self.map_h = wp.array([t.h for t in self.tracks], dtype=wp.int32, device=d)
        self.map_w = wp.array([t.w for t in self.tracks], dtype=wp.int32, device=d)

        # Assign envs to maps and spawn on random waypoints (host-side; rotation is rare).
        self._loads += 1
        rng = np.random.default_rng(self.seed + self._loads)
        mids = rng.integers(0, m, self.num_envs).astype(np.int32)
        cars = np.zeros((self.num_envs, 5), np.float32)
        cars_i = np.zeros((self.num_envs, 2), np.int32)
        for k, t in enumerate(self.tracks):
            mask = mids == k
            idx = rng.integers(0, len(t.centerline), int(mask.sum()))
            cars[mask, :2] = t.centerline[idx]
            cars[mask, 2] = t.angles[idx]
            cars_i[mask, 1] = idx
        self.map_id.assign(mids)
        self.cars.assign(cars)
        self.cars_i.assign(cars_i)
        self.dr.assign((1.0 - DR_FRAC + 2.0 * DR_FRAC * rng.random((self.num_envs, 2))).astype(np.float32))
        self.act_t.zero_()
        self._launch()  # initial observation (v=0, so cars stay put)

    def _launch(self):
        cuda = self.device.is_cuda
        kw = {"stream": self.stream} if self.stream else {"device": self.device}
        if cuda:
            kw["block_dim"] = 128
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                self.act, self.cars, self.cars_i, self.dr, self.map_id, self.edt, self.lut,
                self.centerline, self.n_cl, self.origin, self.res, self.map_h, self.map_w,
                self.seed, self.tick,
            ],
            outputs=[self.obs_w, self.rew_w, self.done_w],
            **kw,
        )
        wp.launch(
            lidar_tex_kernel if cuda else lidar_kernel,
            dim=(self.num_envs, NUM_LIDAR),
            inputs=[self.cars, self.map_id, self.edt_tex if cuda else self.edt,
                    self.lidar_dirs, self.origin, self.res, self.map_h, self.obs_w],
            **kw,
        )
        wp.launch(bump_kernel, dim=1, inputs=[self.tick], **kw)

    def step(self, actions: torch.Tensor):
        self.act_t.copy_(actions.detach())
        self._launch()
        return self.obs, self.rew, self.done

    def snapshot(self):
        torch.cuda.synchronize() if self.device.is_cuda else None
        return [wp.clone(a) for a in (self.cars, self.cars_i, self.dr)]

    def restore(self, snap):
        torch.cuda.synchronize() if self.device.is_cuda else None
        for dst, src in zip((self.cars, self.cars_i, self.dr), snap):
            wp.copy(dst, src)
