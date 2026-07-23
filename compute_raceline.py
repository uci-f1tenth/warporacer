import numpy as np
import casadi as ca
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

try:
    from main import Map
except ImportError as e:
    print("Error: Could not import 'Map' from 'main.py'. Ensure main.py is in the same directory.")
    raise SystemExit(1)

g = 9.81


def _resample_uniform_arclength(centerline: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed polyline to n_points evenly spaced by arc length."""
    closed = np.vstack([centerline, centerline[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))[:-1]
    total_len = s[-1] + seg_len[-1]

    s_uniform = np.linspace(0.0, total_len, n_points, endpoint=False)
    x_new = np.interp(s_uniform, s, centerline[:, 0], period=total_len)
    y_new = np.interp(s_uniform, s, centerline[:, 1], period=total_len)
    return np.column_stack((x_new, y_new))


def _periodic_first_derivative(arr: np.ndarray) -> np.ndarray:
    """Centered first difference that wraps around a closed loop."""
    return (np.roll(arr, -1) - np.roll(arr, 1)) / 2.0


def _periodic_second_derivative(arr: np.ndarray) -> np.ndarray:
    """Centered second difference that wraps around a closed loop."""
    return np.roll(arr, -1) - 2.0 * arr + np.roll(arr, 1)


def _apply_kinematic_limits(v_target: np.ndarray, ds: float, a_accel: float = 2.0, a_decel: float = 4.0) -> np.ndarray:
    """Applies forward/backward sweeps to ensure velocity transitions are physically possible."""
    N = len(v_target)
    for _ in range(2):
        for i in range(N - 2, -1, -1):
            v_target[i] = min(v_target[i], np.sqrt(v_target[i+1]**2 + 2 * a_decel * ds))
        v_target[-1] = min(v_target[-1], np.sqrt(v_target[0]**2 + 2 * a_decel * ds))

    for _ in range(2):
        for i in range(N - 1):
            v_target[i+1] = min(v_target[i+1], np.sqrt(v_target[i]**2 + 2 * a_accel * ds))
        v_target[0] = min(v_target[0], np.sqrt(v_target[-1]**2 + 2 * a_accel * ds))

    return v_target


def _numpy_curvature_energy(x: np.ndarray, y: np.ndarray) -> float:
    """Plain-numpy version of the true curvature expression, for sanity checks
    against forced/perturbed alpha values (no CasADi graph involved)."""
    # Wrap exactly 1 point to get exactly N segments, matching the CasADi logic
    x_ext = np.concatenate((x, [x[0]]))
    y_ext = np.concatenate((y, [y[0]]))
    dx = np.diff(x_ext)
    dy = np.diff(y_ext)
    
    ds = np.sqrt(dx**2 + dy**2 + 1e-8)
    tx = dx / ds
    ty = dy / ds
    
    tx_next = np.roll(tx, -1)
    ty_next = np.roll(ty, -1)
    ds_next = np.roll(ds, -1)
    
    sin_dtheta = tx * ty_next - ty * tx_next
    kappa = sin_dtheta / (0.5 * (ds + ds_next))
    return float(np.sum(kappa**2))


# Optimizes centerline of a track to find optimal raceline by solving for minimum curvature
def compute_raceline(
    map_yaml: str,
    width: float = 0.8,
    mu: float = 1.0489,
    v_cap: float = 10.0,
    output_dir: str = "racelines",
    debug: bool = True,
):
    path = Path(map_yaml)
    track_map = Map(path, force_geometric=True)
    centerline = track_map.centerline

    # some loaders return the closing point duplicated
    if np.allclose(centerline[0], centerline[-1]):
        centerline = centerline[:-1]

    N = len(centerline)
    centerline = _resample_uniform_arclength(centerline, N)

    # --- PRE-SMOOTHING STEP ---
    # Smooth the uniform centerline BEFORE computing normals.
    # Removes pixel-aliasing artifacts that cause normal vectors to cross.
    window = min(21, len(centerline) - (len(centerline) % 2 == 0))
    centerline[:, 0] = savgol_filter(centerline[:, 0], window_length=window, polyorder=3)
    centerline[:, 1] = savgol_filter(centerline[:, 1], window_length=window, polyorder=3)

    # 1. DYNAMIC TRACK WIDTH CALCULATION
    h, w = track_map.raw.shape
    res = track_map.res
    ox, oy = track_map.ox, track_map.oy

    col = np.clip(np.int32((centerline[:, 0] - ox) / res), 0, w - 1)
    row = np.clip(np.int32(h - 1 - (centerline[:, 1] - oy) / res), 0, h - 1)

    dist_to_wall = track_map.dt[row, col] * res

    # Pad by 0.2m (approx half car width + safety margin) to avoid clipping walls
    max_shifts = np.clip(dist_to_wall - 0.2, 0.01, None)

    # normals perpendicular to centerline (periodic, matches curvature computation)
    dx = _periodic_first_derivative(centerline[:, 0])
    dy = _periodic_first_derivative(centerline[:, 1])
    ddx_c = _periodic_second_derivative(centerline[:, 0])
    ddy_c = _periodic_second_derivative(centerline[:, 1])
    normals = np.column_stack((-dy, dx))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # --- GEOMETRIC SINGULARITY BOUNDS (Swallowtail Prevention) ---
    # If alpha > local radius of curvature (R_c), the offset curve loops backwards 
    # causing a swallowtail singularity. We cap the shift to prevent this.
    kappa_c = (dx * ddy_c - dy * ddx_c) / np.maximum((dx**2 + dy**2)**1.5, 1e-6)
    R_c = 1.0 / (np.abs(kappa_c) + 1e-6)
    
    max_alpha = max_shifts.copy()
    min_alpha = -max_shifts.copy()
    
    # Cap shifts at 80% of the local radius of curvature
    safe_R = R_c * 0.8
    left_mask = kappa_c > 0  # Turning left, center of curvature is on the inside
    max_alpha[left_mask] = np.minimum(max_alpha[left_mask], safe_R[left_mask])
    
    right_mask = kappa_c < 0 # Turning right
    min_alpha[right_mask] = np.maximum(min_alpha[right_mask], -safe_R[right_mask])

    if debug:
        print("\n--- [DEBUG] track width / bounds ---")
        print(f"dist_to_wall: min={dist_to_wall.min():.4f} max={dist_to_wall.max():.4f} mean={dist_to_wall.mean():.4f}")
        print(f"alpha_bounds: min={min_alpha.min():.4f} max={max_alpha.max():.4f}")
        
        angles = np.arctan2(normals[:, 1], normals[:, 0])
        dangles = np.abs(np.diff(np.unwrap(angles)))
        print(f"max angle jump between adjacent normals: {dangles.max():.4f} rad")

    # casadi optimization setup
    opti = ca.Opti()
    alpha = opti.variable(N)

    opti.set_initial(alpha, np.zeros(N))

    x = centerline[:, 0] + alpha * normals[:, 0]
    y = centerline[:, 1] + alpha * normals[:, 1]

    # Exactly N segments (no double-seam weighting)
    x_ext = ca.vertcat(x, x[0])
    y_ext = ca.vertcat(y, y[0])
    dx_path = ca.diff(x_ext)
    dy_path = ca.diff(y_ext)

    ds_nominal = np.linalg.norm(centerline[1] - centerline[0])

    # --- FORWARD PROGRESS CONSTRAINT (Anti-180 Kink Exploit) ---
    c_x_ext = np.concatenate([centerline[:, 0], [centerline[0, 0]]])
    c_y_ext = np.concatenate([centerline[:, 1], [centerline[0, 1]]])
    c_dx = np.diff(c_x_ext)
    c_dy = np.diff(c_y_ext)
    
    forward_progress = dx_path * c_dx + dy_path * c_dy
    opti.subject_to(forward_progress >= 0.1 * (ds_nominal**2))
    
    # --- TRUE CURVATURE (Scale-Independent) ---
    ds_path = ca.sqrt(dx_path**2 + dy_path**2 + 1e-8)
    
    tx = dx_path / ds_path
    ty = dy_path / ds_path
    
    tx_next = ca.vertcat(tx[1:], tx[0])
    ty_next = ca.vertcat(ty[1:], ty[0])
    ds_next = ca.vertcat(ds_path[1:], ds_path[0])
    
    sin_dtheta = tx * ty_next - ty * tx_next
    true_kappa = sin_dtheta / (0.5 * (ds_path + ds_next))
    
    curvature_energy = ca.sumsqr(true_kappa)

    # 2. STEERING SMOOTHNESS REGULARIZATION
    d_alpha = ca.diff(ca.vertcat(alpha, alpha[0]))
    dd_alpha = ca.diff(ca.vertcat(d_alpha, d_alpha[0]))

    w_steer = 0.1   # Stops high frequency changes
    w_jerk = 0.1    # Acts as a shock absorber
    w_center = 0.001 # Fixes singular Hessians on perfectly straight sections

    steering_cost = (w_steer * ca.sumsqr(d_alpha) + 
                     w_jerk * ca.sumsqr(dd_alpha) + 
                     w_center * ca.sumsqr(alpha))

    opti.minimize(curvature_energy + steering_cost)

    # track boundary offset
    opti.subject_to(opti.bounded(min_alpha, alpha, max_alpha))

    # solve
    opti.solver('ipopt', {'expand': True}, {'max_iter': 1000, 'print_level': 0, 'acceptable_tol': 1e-4})
    
    is_success = True
    try:
        sol = opti.solve()
        opt_alpha = sol.value(alpha)
    except RuntimeError as e:
        print(f"\nIPOPT stopped early: {e}")
        print("Harvesting best-effort solution (may be suboptimal)...")
        opt_alpha = opti.debug.value(alpha)
        is_success = False

    if debug:
        print("\n--- [DEBUG] solver result ---")
        print(f"alpha range: {opt_alpha.min():.4f} to {opt_alpha.max():.4f}")
        print(f"bound range: {min_alpha.min():.4f} to {max_alpha.max():.4f}")
        
        if is_success:
            print(f"iterations: {sol.stats()['iter_count']}  status: {sol.stats()['return_status']}")
            ce_zero = sol.value(ca.substitute(curvature_energy, alpha, np.zeros(N)))
            ce_solved = sol.value(curvature_energy)
            sc_solved = sol.value(steering_cost)
        else:
            print("Status: Harvested early (Partial Solution)")
            ce_zero = opti.debug.value(ca.substitute(curvature_energy, alpha, np.zeros(N)))
            ce_solved = opti.debug.value(curvature_energy)
            sc_solved = opti.debug.value(steering_cost)

        print(f"curvature_energy at alpha=0:      {ce_zero:.4f}")
        print(f"curvature_energy at solved alpha: {ce_solved:.4f}")
        print(f"steering_cost (weighted) at solved: {sc_solved:.4f}")

        # Smooth Gaussian Perturbation Test
        worst_idx = int(np.argmax(np.abs(kappa_c)))
        
        alpha_forced = opt_alpha.copy()
        window = max(5, N // 20)
        indices = np.arange(N)
        dist = np.minimum(np.abs(indices - worst_idx), N - np.abs(indices - worst_idx))
        bump = np.exp(-0.5 * (dist / (window / 3.0))**2)
        
        target_alpha = max_alpha if kappa_c[worst_idx] > 0 else min_alpha
        alpha_forced += bump * (target_alpha - opt_alpha) * 0.8
        alpha_forced = np.clip(alpha_forced, min_alpha, max_alpha)
        
        x_forced = centerline[:, 0] + alpha_forced * normals[:, 0]
        y_forced = centerline[:, 1] + alpha_forced * normals[:, 1]
        ce_forced = _numpy_curvature_energy(x_forced, y_forced)
        
        print(f"\n--- [DEBUG] Gaussian Perturbation Test ---")
        print(f"Tightest geometric corner at index {worst_idx} (kappa={kappa_c[worst_idx]:.4f})")
        print(f"curvature_energy at solved alpha:          {ce_solved:.4f}")
        print(f"curvature_energy with Gaussian bump pushed: {ce_forced:.4f}")
        if ce_forced < ce_solved:
            print(">> The smooth bump LOWERED the objective -> Solver is trapped in a local minimum!")
        else:
            print(">> The smooth bump RAISED the objective -> The solved path is genuinely optimal here.")
        print("--- [END DEBUG] ---\n")

    opt_x = centerline[:, 0] + opt_alpha * normals[:, 0]
    opt_y = centerline[:, 1] + opt_alpha * normals[:, 1]

    # max velocity calculation based on curvature
    opt_dx = _periodic_first_derivative(opt_x)
    opt_dy = _periodic_first_derivative(opt_y)
    opt_ddx = _periodic_second_derivative(opt_x)
    opt_ddy = _periodic_second_derivative(opt_y)

    kappa = np.abs(opt_dx * opt_ddy - opt_dy * opt_ddx) / np.maximum((opt_dx**2 + opt_dy**2)**(1.5), 1e-6)
    radius = 1.0 / np.maximum(kappa, 1e-6)

    v_max = np.sqrt(mu * g * radius)
    v_max = np.clip(v_max, 0, v_cap)
    
    # 3. KINEMATIC SWEEP
    v_max = _apply_kinematic_limits(v_max, ds_nominal, a_accel=2.0, a_decel=4.0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Flag filename if solver didn't fully converge
    suffix = "" if is_success else "_PARTIAL"
    output_file = out_dir / f"{path.stem}_raceline{suffix}.csv"

    df = pd.DataFrame({
        'x': opt_x,
        'y': opt_y,
        'v_target': v_max,
    })
    df.to_csv(output_file, index=False)
    print(f"Optimal raceline saved to '{output_file}'.")


from typer import run
if __name__ == "__main__":
    run(compute_raceline)