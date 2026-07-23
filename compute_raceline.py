import numpy as np
import casadi as ca
import pandas as pd
from pathlib import Path
from main import Map

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


# Optimizes centerline of a track to find optimal raceline by solving for minimum curvature
def compute_raceline(
    map_yaml: str,
    mu: float = 1.0489,
    v_cap: float = 5.0,
    output_dir: str = "racelines",
):
    path = Path(map_yaml)
    track_map = Map(path, force_geometric=True)
    centerline = track_map.centerline

    # some loaders return the closing point duplicated
    if np.allclose(centerline[0], centerline[-1]):
        centerline = centerline[:-1]

    N = len(centerline)
    centerline = _resample_uniform_arclength(centerline, N)

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

    # normals perpendicular to centerline
    dx = _periodic_first_derivative(centerline[:, 0])
    dy = _periodic_first_derivative(centerline[:, 1])
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
    ddx_path = ca.diff(dx_path)
    ddy_path = ca.diff(dy_path)

    ds = np.linalg.norm(centerline[1] - centerline[0])
    
    # Pure curvature optimization (widest possible arcs for max speed)
    bending_energy = (ca.sumsqr(ddx_path) + ca.sumsqr(ddy_path)) / (ds**4)
    
    # 2. STEERING SMOOTHNESS REGULARIZATION
    # Prevents wobbling on straights without pulling the line tight into apexes
    d_alpha = ca.diff(ca.vertcat(alpha, alpha[0]))
    steering_cost = ca.sumsqr(d_alpha)

    w_steer = 0.1   # Stops wobbling on straights
    opti.minimize(bending_energy + w_steer * steering_cost)

    # track boundary offset
    opti.subject_to(opti.bounded(-max_shifts, alpha, max_shifts))

    # solve
    opti.solver('ipopt', {'expand': True}, {'max_iter': 1000, 'print_level': 0, 'acceptable_tol': 1e-4})
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"IPOPT failed to converge: {e}")
        print(f"Last alpha values before failure: {opti.debug.value(alpha)}")
        raise

    opt_alpha = sol.value(alpha)
    print(f"Optimization complete! Alpha shifted by max {opt_alpha.max():.3f}m, min {opt_alpha.min():.3f}m")

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
    v_max = _apply_kinematic_limits(v_max, ds, a_accel=2.0, a_decel=4.0)

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