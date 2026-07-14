import numpy as np
import casadi as ca
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typer import run
from main import Map
g = 9.81

def resample_centerline(centerline, step_distance=0.4):
    """Downsamples path to a uniform spatial distance to keep N < 1500."""
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0)
    
    total_dist = cum_dist[-1]
    num_points = int(total_dist / step_distance)
    
    uniform_dist = np.linspace(0, total_dist, num_points)
    interp_x = interp1d(cum_dist, centerline[:, 0], kind='cubic')(uniform_dist)
    interp_y = interp1d(cum_dist, centerline[:, 1], kind='cubic')(uniform_dist)
    
    return np.column_stack((interp_x, interp_y))


def compute_velocity_profile(opt_x, opt_y, mu, g, max_speed, a_accel, a_decel):
    """Generates a physical velocity profile respecting braking and acceleration limits."""
    dx = np.diff(opt_x, prepend=opt_x[-1])
    dy = np.diff(opt_y, prepend=opt_y[-1])
    ds = np.sqrt(dx**2 + dy**2)
    
    # Base curvature limit (Lateral friction)
    opt_dx = np.gradient(opt_x)
    opt_dy = np.gradient(opt_y)
    opt_ddx = np.gradient(opt_dx)
    opt_ddy = np.gradient(opt_dy)
    
    kappa = np.abs(opt_dx * opt_ddy - opt_dy * opt_ddx) / np.maximum((opt_dx**2 + opt_dy**2)**1.5, 1e-6)
    v_target = np.minimum(np.sqrt(mu * g / np.maximum(kappa, 1e-6)), max_speed)
    
    N = len(v_target)
    
    # apply braking and acceleration limits using a backward and forward pass
    # backward pass (braking)
    for _ in range(2): 
        for i in range(N - 2, -1, -1):
            max_entry_spd = np.sqrt(v_target[i+1]**2 + 2 * a_decel * ds[i+1])
            v_target[i] = min(v_target[i], max_entry_spd)
        max_entry_spd = np.sqrt(v_target[0]**2 + 2 * a_decel * ds[0])
        v_target[-1] = min(v_target[-1], max_entry_spd)

    # forward Pass (accel)
    for _ in range(2):
        for i in range(N - 1):
            max_exit_spd = np.sqrt(v_target[i]**2 + 2 * a_accel * ds[i+1])
            v_target[i+1] = min(v_target[i+1], max_exit_spd)
        max_exit_spd = np.sqrt(v_target[-1]**2 + 2 * a_accel * ds[0])
        v_target[0] = min(v_target[0], max_exit_spd)
        
    return v_target


def compute_raceline(map_yaml: str, width: float = 1.0, mu: float = 1.0489, max_speed: float = 5.0, a_accel: float = 2.0, a_decel: float = 4.0):
    path = Path(map_yaml)
    
    track_map = Map(path, force_geometric=True)
    raw_centerline = track_map.centerline

    # downscale
    # centerline = resample_centerline(raw_centerline, step_distance=0.4)
    centerline = raw_centerline
    
    # smoothing filter
    window = min(21, len(centerline) - (len(centerline) % 2 == 0))
    centerline[:, 0] = savgol_filter(centerline[:, 0], window_length=window, polyorder=3)
    centerline[:, 1] = savgol_filter(centerline[:, 1], window_length=window, polyorder=3)

    N = len(centerline)

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    normals = np.column_stack((-dy, dx))
    
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1e-6
    normals /= norms[:, np.newaxis]

    opti = ca.Opti()
    alpha = opti.variable(N)
    
    opti.set_initial(alpha, np.zeros(N))

    x = centerline[:, 0] + alpha * normals[:, 0]
    y = centerline[:, 1] + alpha * normals[:, 1]

    # curvature
    x_wrap = ca.vertcat(x, x[0])
    y_wrap = ca.vertcat(y, y[0])
    dx_c = ca.diff(x_wrap)
    dy_c = ca.diff(y_wrap)
    ds = ca.sqrt(dx_c**2 + dy_c**2) + 1e-4
    
    dx_next = ca.vertcat(dx_c[1:], dx_c[0])
    dy_next = ca.vertcat(dy_c[1:], dy_c[0])
    ds_next = ca.vertcat(ds[1:], ds[0])
    
    tx, ty = dx_c / ds, dy_c / ds
    tx_next, ty_next = dx_next / ds_next, dy_next / ds_next
    sin_dtheta = tx * ty_next - ty * tx_next
    kappa = sin_dtheta / (0.5 * (ds + ds_next))

    d_alpha = ca.diff(ca.vertcat(alpha, alpha[0]))
    dd_alpha = ca.diff(ca.vertcat(d_alpha, d_alpha[0]))
    
    w_smooth = 0.2   # smoothness weight
    w_jerk = 0.2     # oscillation penalize weight
    w_center = 0.001 # pull straights toward centerline weight

    opti.minimize(ca.sumsqr(kappa) + w_smooth * ca.sumsqr(d_alpha) + w_jerk * ca.sumsqr(dd_alpha) + w_center * ca.sumsqr(alpha))

    # distance from wall
    safety_margin = 0.05
    max_shift = width - safety_margin
    opti.subject_to(opti.bounded(-max_shift, alpha, max_shift))
    
    # Close the loop
    opti.subject_to(alpha[0] == alpha[-1])

    # optimize line
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 0, "acceptable_tol": 1e-3, "tol": 1e-4}
    opti.solver('ipopt', p_opts, s_opts)
    
    try:
        sol = opti.solve()
        opt_alpha = sol.value(alpha)
    except Exception as e:
        print(f"Optimization failed: {e}")
        opt_alpha = np.zeros(N)

    opt_x = centerline[:, 0] + opt_alpha * normals[:, 0]
    opt_y = centerline[:, 1] + opt_alpha * normals[:, 1]

    # max velocity profile
    v_max = compute_velocity_profile(opt_x, opt_y, mu, g, max_speed, a_accel, a_decel)

    output_dir = Path("racelines")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{path.stem}_raceline.csv"
    
    df = pd.DataFrame({
        'x': opt_x,
        'y': opt_y,
        'v_target': v_max,
    })
    
    df.to_csv(output_file, index=False)
    print(f"Optimal raceline saved to '{output_file}'.")


if __name__ == "__main__":
    run(compute_raceline)