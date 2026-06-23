import warp as wp

from include.constants import *


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
    # Scale static physical dimensions by the randomized environment coefficients
    lf = LF * lf_s
    lr = LR * lr_s
    lwb = lf + lr
    mu = MU * mu_s
    a_max = mu * G

    # Calculate kinematic angular velocity bounds
    tand = wp.tan(delta)
    d_psi_kin = v * tand / lwb
    d_psi_cap = a_max / wp.max(wp.abs(v), 0.5)
    d_psi = wp.clamp(d_psi_kin, -d_psi_cap, d_psi_cap)

    # Project tire traction constraints into friction ellipse boundaries
    a_lat = v * d_psi
    a_long_max = wp.sqrt(wp.max(a_max * a_max - a_lat * a_lat, 0.0))

    cp = wp.cos(psi)
    sp = wp.sin(psi)
    
    # Populate the output structural representation
    out = VDeriv()
    out.d_x = v * cp
    out.d_y = v * sp
    out.d_psi = d_psi
    out.d_v = wp.clamp(accel, -A_MAX, a_long_max) # Always allow maximum braking force (-A_MAX)
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
    
    # Perform weighted average blending for the definitive integration update
    out = VDeriv()
    out.d_x = (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT_SUB_SIX
    out.d_y = (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT_SUB_SIX
    out.d_psi = (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT_SUB_SIX
    out.d_v = (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT_SUB_SIX
    out.d_psip = (k1.d_psip + 2.0 * k2.d_psip + 2.0 * k3.d_psip + k4.d_psip) * DT_SUB_SIX
    out.d_beta = (k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta) * DT_SUB_SIX
    return out

@wp.kernel
def step_kernel(
    actions: wp.array[wp.vec2],
    obs: wp.array2d[wp.float32],
    critic_obs: wp.array2d[wp.float32],
    reward: wp.array[wp.float32],
    done: wp.array[wp.int32],
    cars: wp.array2d[wp.float32],
    cars_int: wp.array2d[wp.int32],
    car_dr: wp.array2d[wp.float32],
    centerline: wp.array2d[wp.vec3],       # Shape: (num_maps, max_n_cl)
    dt_map: wp.array3d[wp.float32],        # Shape: (num_maps, max_w, max_h)
    cl_lut: wp.array3d[wp.int32],          # Shape: (num_maps, max_w, max_h)
    maps_origin: wp.array[wp.vec2],        # Shape: (num_maps)
    maps_res: wp.array[wp.float32],        # Shape: (num_maps)
    maps_n_cl: wp.array[wp.int32],         # Shape: (num_maps)
    maps_look_step: wp.array[wp.int32],    # Shape: (num_maps)
    env_map_ids: wp.array[wp.int32],       # Shape: (num_envs)
    lidar_dirs: wp.array[wp.vec2],
    seed_base: int,
):
    i = wp.tid()
    
    # Extract dynamic track profile context for this thread
    map_idx = env_map_ids[i]
    origin = maps_origin[map_idx]
    res = maps_res[map_idx]
    n_cl = maps_n_cl[map_idx]
    
    # Extract current physical state
    x = cars[i, 0]
    y = cars[i, 1]
    delta = cars[i, 2]
    v = cars[i, 3]
    psi = cars[i, 4]
    psip = cars[i, 5]
    beta = cars[i, 6]
    
    steps = cars_int[i, 0]
    wp_i = cars_int[i, 1]
    stall_steps = cars_int[i, 2]
    
    mu_s = car_dr[i, 0]
    mass_s = car_dr[i, 1]
    lf_s = car_dr[i, 2]
    lr_s = car_dr[i, 3]

    # Matrix shapes
    mw = dt_map.shape[1]
    mh = dt_map.shape[2]
    df32 = wp.float32(DR_FRAC)

    # DYNAMIC UNPADDED HEIGHT TRACKING
    # Scan back from the max padded height boundary to find this map slice's actual valid image row size
    actual_mh = mh
    for h_chk in range(mh):
        reverse_idx = mh - 1 - h_chk
        if cl_lut[map_idx, 0, reverse_idx] != 0 or cl_lut[map_idx, mw - 1, reverse_idx] != 0 or cl_lut[map_idx, mw // 2, reverse_idx] != 0:
            actual_mh = reverse_idx + 1
            break
    
    # Base height inversion target calculated cleanly off the specific map's actual dimensions
    mh_f = wp.float32(actual_mh) - 1.0

    # Calculate unified dynamic vision lookahead stride based on velocity
    dynamic_look_step = wp.int32(wp.float32(BASE_STRIDE) + (wp.float32(VELOCITY_SCALE) * wp.max(0.0, v)))

    # Actuation processing and physical constraint clamping
    steer_v = wp.clamp(actions[i][0], -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0.0 and delta <= STEER_MIN) or (steer_v > 0.0 and delta >= STEER_MAX):
        steer_v = 0.0
        
    accel = wp.clamp(actions[i][1], -1.0, 1.0) * A_MAX
    if (accel < 0.0 and v <= V_MIN) or (accel > 0.0 and v >= V_MAX):
        accel = 0.0

    # Clean multi-pass RK4 integration over the sub-stepping interval
    dd_sub = steer_v * DT_SUB
    for _ in range(SUBSTEPS):
        d = rk4_step(delta, v, psi, psip, beta, steer_v, accel, mu_s, mass_s, lf_s, lr_s)
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

    # Coordinate mapping to grid space indices
    px = wp.clamp(wp.int32((x - origin[0]) / res), 0, mw - 1)
    py = wp.clamp(wp.int32(mh_f - (y - origin[1]) / res), 0, mh - 1)
    
    # Lookups now use 3D map index and 2D centerline row
    new_wp = cl_lut[map_idx, px, py]
    d_wp = new_wp - wp_i
    if 2 * d_wp > n_cl:
        d_wp -= n_cl
    elif 2 * d_wp < -n_cl:
        d_wp += n_cl

    # Pull local track geometry constants
    cpt_local = centerline[map_idx, new_wp]
    cx_local = cpt_local[0]
    cy_local = cpt_local[1]
    cth_local = cpt_local[2]
    s_cth_local = wp.sin(cth_local)
    c_cth_local = wp.cos(cth_local)
    
    # Handle progress stall verification
    if wp.abs(v) < STALL_VELOCITY:
        stall_steps += 1
    else:
        stall_steps = 0
    is_stalled = stall_steps > wp.int32(STALL_SECONDS_TO_STEPS)

    # Compute termination limits
    true_lateral_dist = wp.abs(-(x - cx_local) * s_cth_local + (y - cy_local) * c_cth_local)
    edt_val = dt_map[map_idx, px, py] * res
    is_stable = wp.isfinite(x) and wp.isfinite(y) and wp.isfinite(v) and wp.isfinite(psi)
    is_off_track = true_lateral_dist > MAX_CENTERLINE_DEV  
    
    term = (edt_val < CAR_HALF_DIAG) or (not is_stable) or is_off_track or is_stalled
    trunc = steps >= MAX_STEPS
    steps += 1

    if term:
        done[i] = DONE_TERMINATED
    elif trunc:
        done[i] = DONE_TRUNCATED
    else:
        done[i] = 0

    # =========================================================================
    # REWARD MATRIX GENERATION
    # =========================================================================
    base_progress = (wp.float32(d_wp) / wp.float32(n_cl)) * PROGRESS_SCALE
    if d_wp < 0:
        base_progress = base_progress * BACKWARDS_PROGRESS_PENALTY_MUL

    target_wp = (new_wp + dynamic_look_step) % n_cl
    cth_target = centerline[map_idx, target_wp][2]
    v_along = v * wp.max(0.0, wp.cos(beta + psi - cth_target))
    vel_progress = v_along * PROGRESS_V_COEF

    term_pen = TERM_PENALTY if term else 0.0
    lat_err_reward = -(x - cx_local) * s_cth_local + (y - cy_local) * c_cth_local
    lat_pen = (lat_err_reward * lat_err_reward) * LATERAL_PENALTY

    reward[i] = base_progress + vel_progress + term_pen + IDLE_PENALTY + lat_pen

    # =========================================================================
    # COMPACT AUTO-RESET ENGINE
    # =========================================================================
    if term or trunc:
        rng = wp.rand_init(seed_base + i * 73 + steps * 31 + new_wp * 17)
        rnd = wp.int32(wp.randf(rng) * wp.float32(n_cl))
        if rnd >= n_cl:
            rnd = n_cl - 1
            
        # Query thread-specific map slice
        rpt = centerline[map_idx, rnd]
        x = rpt[0]
        y = rpt[1]
        psi = rpt[2]
        
        delta = 0.0
        v = 0.0
        psip = 0.0
        beta = 0.0
        steps = 0
        stall_steps = 0 
        new_wp = rnd
        
        # Apply domain randomization parameter fluxes
        car_dr[i, 0] = wp.float32(1.0) - df32 + wp.float32(2.0) * df32 * wp.randf(rng)
        car_dr[i, 1] = wp.float32(1.0) - df32 + wp.float32(2.0) * df32 * wp.randf(rng)
        car_dr[i, 2] = wp.float32(1.0) - df32 + wp.float32(2.0) * df32 * wp.randf(rng)
        car_dr[i, 3] = wp.float32(1.0) - df32 + wp.float32(2.0) * df32 * wp.randf(rng)

        # Query thread-specific map slice
        cpt_local = centerline[map_idx, new_wp]
        cx_local = cpt_local[0]
        cy_local = cpt_local[1]
        cth_local = cpt_local[2]
        s_cth_local = wp.sin(cth_local)
        c_cth_local = wp.cos(cth_local)
        
        dynamic_look_step = wp.int32(wp.float32(BASE_STRIDE))

    # =========================================================================
    # POST-RESET OBSERVATION CALCULATION
    # =========================================================================
    sh = wp.sin(psi)
    ch = wp.cos(psi)
    
    lx = x + LF * ch
    ly = y + LF * sh
    
    lpx = wp.clamp(wp.int32((lx - origin[0]) / res), 0, mw - 1)
    lpy = wp.clamp(wp.int32(mh_f - (ly - origin[1]) / res), 0, mh - 1)
    lpos = wp.vec2(wp.float32(lpx), wp.float32(lpy))
    lrange_px = LIDAR_RANGE / res
    
    # Bounded LiDAR raymarching loop
    max_ray_steps = 250
    for j in range(lidar_dirs.shape[0]):
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        dpx = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        ray = lpos
        dist = wp.float32(0.0)
        
        ray_count = wp.int32(0)
        while dist < lrange_px and ray_count < max_ray_steps:
            ray_count += 1
            rx = wp.int32(ray[0])
            ry = wp.int32(ray[1])
            if rx < 0 or rx >= mw or ry < 0 or ry >= mh:
                break
            # Query distance value via 3D map reference
            step_px = dt_map[map_idx, rx, ry]
            ray = ray + dpx * step_px
            dist += step_px
            if step_px == 0.0:
                break
        obs[i, 3 + j] = wp.min(dist, lrange_px) * res

    # Populate Frenet error matrices
    obs[i, OBS_FRENET_OFF] = wp.atan2(s_cth_local * ch - c_cth_local * sh, c_cth_local * ch + s_cth_local * sh)
    obs[i, OBS_FRENET_OFF + 1] = -(x - cx_local) * s_cth_local + (y - cy_local) * c_cth_local

    # Multi-point metric lookahead path projection
    idx = new_wp
    for k in range(NUM_LOOKAHEAD):
        idx = (idx + dynamic_look_step) % n_cl
        w = centerline[map_idx, idx]
        dx = w[0] - x
        dy = w[1] - y
        obs[i, OBS_LOOK_OFF + k * 2] = dx * ch + dy * sh
        obs[i, OBS_LOOK_OFF + k * 2 + 1] = -dx * sh + dy * ch

    # Normalize system values into observation spaces
    obs[i, 0] = delta / STEER_MAX
    obs[i, 1] = v / V_MAX
    obs[i, 2] = psip / PSI_PRIME_MAX
    
    # Commit synchronized parameters back to shared buffers
    cars[i, 0] = x
    cars[i, 1] = y
    cars[i, 2] = delta
    cars[i, 3] = v
    cars[i, 4] = psi
    cars[i, 5] = psip
    cars[i, 6] = beta

    # Populate the Critic's Privileged State
    for k in range(OBS_DIM):
        critic_obs[i, k] = obs[i, k]
        
    critic_obs[i, OBS_DIM] = car_dr[i, 0]     # Ground Truth Friction
    critic_obs[i, OBS_DIM + 1] = car_dr[i, 1] # Ground Truth Mass
    critic_obs[i, OBS_DIM + 2] = x            # Global X
    critic_obs[i, OBS_DIM + 3] = y            # Global Y
    critic_obs[i, OBS_DIM + 4] = beta         # Exact Slip Angle
    
    cars_int[i, 0] = steps
    cars_int[i, 1] = new_wp
    cars_int[i, 2] = stall_steps