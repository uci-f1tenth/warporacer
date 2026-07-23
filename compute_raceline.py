import numpy as np
import casadi as ca
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

try:
    from main import Map
except ImportError:
    print("Warning: Could not import 'Map' from 'main.py'. Ensure main.py is in the same directory.")

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
    """Plain-numpy version of the true curvature expression, for sanity checks."""
    x_ext = np.concatenate(([x[-1]], x, [x[0]]))
    y_ext = np.concatenate(([y[-1]], y, [y[0]]))
    dx = np.diff(x_ext)
    dy = np.diff(y_ext)
    ds = np.sqrt(dx**2 + dy**2) + 1e-5
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
    mu: float = 1.0489,
    v_cap: float = 5.0,
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

    # Pre-smooth jagged grid noise to prevent normal vectors from crossing
    window = min(21, len(centerline) - (len(centerline) % 2 == 0))
    centerline[:, 0] = savgol_filter(centerline[:, 0], window_length=window, polyorder=3)
    centerline[:, 1] = savgol_filter(centerline[:, 1], window_length=window, polyorder=3)

    # 1. DYNAMIC TRACK WIDTH CALCULATION
    # Extract exact distance to the wall for every point using the map's distance transform
    h, w = track_map.raw.shape
    res = track_map.res
    ox, oy = track_map.ox, track_map.oy
    
    col = np.clip(np.int32((centerline[:, 0] - ox) / res), 0, w - 1)
    row = np.clip(np.int32(h - 1 - (centerline[:, 1] - oy) / res), 0, h - 1)
    
    dist_to_wall = track_map.dt[row, col] * res
    
    # Pad by 0.2m (approx half car width + safety margin) to avoid clipping walls
    max_shifts = np.clip(dist_to_wall - 0.2, 0.01, None)

    if debug:
        print("\n--- [DEBUG] track width / bounds ---")
        print(f"dist_to_wall: min={dist_to_wall.min():.4f} max={dist_to_wall.max():.4f} mean={dist_to_wall.mean():.4f}")
        print(f"max_shifts:   min={max_shifts.min():.4f} max={max_shifts.max():.4f}")

    # normals perpendicular to centerline
    dx = _periodic_first_derivative(centerline[:, 0])
    dy = _periodic_first_derivative(centerline[:, 1])
    ddx_c = _periodic_second_derivative(centerline[:, 0])
    ddy_c = _periodic_second_derivative(centerline[:, 1])
    normals = np.column_stack((-dy, dx))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # casadi optimization setup
    opti = ca.Opti()
    alpha = opti.variable(N)
    
    opti.set_initial(alpha, np.zeros(N))

    x = centerline[:, 0] + alpha * normals[:, 0]
    y = centerline[:, 1] + alpha * normals[:, 1]

    # differentiation to compute curvature of centerline, wrapped around the seam
    x_ext = ca.vertcat(x[-1], x, x[0])
    y_ext = ca.vertcat(y[-1], y, y[0])
    dx_path = ca.diff(x_ext)
    dy_path = ca.diff(y_ext)
    
    # --- TRUE CURVATURE (Scale-Independent) ---
    # Rewards wide arcs instead of punishing path stretching
    # FIX: Epsilon must be INSIDE the square root to prevent NaN gradients (derivative of sqrt(0) is NaN)
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

    w_jerk = 0.1    # Acts as a shock absorber to prevent straight-line wobble
    w_center = 0.001 # Fixes singular Hessians on perfectly straight sections

    steering_cost = (w_jerk * ca.sumsqr(dd_alpha) + w_center * ca.sumsqr(alpha))

    opti.minimize(curvature_energy + steering_cost)

    # track boundary offset
    opti.subject_to(opti.bounded(-max_shifts, alpha, max_shifts))

    # close the loop
    opti.subject_to(alpha[0] == alpha[-1])

    # solve
    p_opts = {"expand": True}
    s_opts = {
        "max_iter": 2000,
        "print_level": 0,
        "tol": 1e-3,
        "acceptable_tol": 1e-2,
        "acceptable_iter": 10
    }
    opti.solver('ipopt', p_opts, s_opts)
    
    try:
        sol = opti.solve()
        opt_alpha = sol.value(alpha)
        is_success = True
    except Exception as e:
        print(f"\nIPOPT stopped early: {e}")
        print("Harvesting best-effort solution (usually perfectly valid)...")
        opt_alpha = opti.debug.value(alpha)
        is_success = False

    if debug:
        print("\n--- [DEBUG] solver result ---")
        print(f"alpha range: {opt_alpha.min():.4f} to {opt_alpha.max():.4f}")
        print(f"bound range: -{max_shifts.min():.4f} to {max_shifts.max():.4f}")
        
        if is_success:
            print(f"iterations: {sol.stats()['iter_count']}  status: {sol.stats()['return_status']}")
            ce_zero = sol.value(ca.substitute(curvature_energy, alpha, np.zeros(N)))
            ce_solved = sol.value(curvature_energy)
            sc_solved = sol.value(steering_cost)
        else:
            print("Status: Maximum_Iterations_Exceeded (Harvested early)")
            ce_zero = opti.debug.value(ca.substitute(curvature_energy, alpha, np.zeros(N)))
            ce_solved = opti.debug.value(curvature_energy)
            sc_solved = opti.debug.value(steering_cost)

        print(f"curvature_energy at alpha=0:      {ce_zero:.4f}")
        print(f"curvature_energy at solved alpha: {ce_solved:.4f}")
        print(f"steering_cost (weighted) at solved alpha: {sc_solved:.4f}")

        # curvature of the ORIGINAL centerline (numpy, periodic) -> find the tightest corner
        kappa_center = np.abs(dx * ddy_c - dy * ddx_c) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
        worst_idx = int(np.argmax(kappa_center))
        print("\n--- [DEBUG] tightest corner vs. solved alpha there ---")
        print(f"worst curvature at index {worst_idx}: kappa={kappa_center[worst_idx]:.4f}")
        print(f"alpha there: {opt_alpha[worst_idx]:.4f}  (bound: +-{max_shifts[worst_idx]:.4f})")

        # manual perturbation test: forcibly push alpha toward the bound at the worst corner
        alpha_forced = opt_alpha.copy()
        alpha_forced[worst_idx] = max_shifts[worst_idx] * 0.8
        x_forced = centerline[:, 0] + alpha_forced * normals[:, 0]
        y_forced = centerline[:, 1] + alpha_forced * normals[:, 1]
        ce_forced = _numpy_curvature_energy(x_forced, y_forced)
        print(f"\ncurvature_energy at solved alpha:               {ce_solved:.4f}")
        print(f"curvature_energy with worst-corner alpha forced to 80% of bound: {ce_forced:.4f}")
        if ce_forced < ce_solved:
            print(">> forcing a bigger apex shift LOWERS the objective -> solver left improvement on the table. Investigate the CasADi formulation.")
        else:
            print(">> forcing a bigger apex shift RAISES the objective -> the small solved alpha appears genuinely optimal for this cost function.")
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
    ds_nominal = np.linalg.norm(centerline[1] - centerline[0])
    v_max = _apply_kinematic_limits(v_max, ds_nominal, a_accel=2.0, a_decel=4.0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{path.stem}_raceline.csv"

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