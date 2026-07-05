import numpy as np
import casadi as ca
import pandas as pd
from pathlib import Path
from main import Map


g = 9.81

# Optimizes centerline of a track to find optimal raceline by solving for minimum curve
def compute_raceline(map_yaml, width = 0.4, mu = 1.0489): # physics??!?!?! unfamiliar concept
    

    path = Path(map_yaml)
    track_map = Map(path, force_geometric = True)
    centerline = track_map.centerline

    N = len(centerline)

    # normals prependicualr to centerline
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    normals = np.column_stack((-dy, dx))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # casadi optimization setup
    opti = ca.Opti()
    alpha = opti.variable(N)
    x = centerline[:, 0] + alpha * normals[:, 0]
    y = centerline[:, 1] + alpha * normals[:, 1]

    # differentiation to compute curvature of centerline
    dx_path = ca.diff(x)
    dy_path = ca.diff(y)
    ddx_path = ca.diff(dx_path)
    ddy_path = ca.diff(dy_path)

    curve_sq = ca.sumsqr(ddx_path) + ca.sumsqr(ddy_path)
    opti.minimize(curve_sq)

    # track boundary offest
    safety_margin = 0.05
    max_shift = (width / 2) - safety_margin
    opti.subject_to(opti.bounded(-max_shift, alpha, max_shift))

    # close the loop
    opti.subject_to(alpha[0] == alpha[-1])

    # solve
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})
    sol = opti.solve()

    opt_alpha = sol.value(alpha)
    opt_x = centerline[:, 0] + opt_alpha * normals[:, 0]
    opt_y = centerline[:, 1] + opt_alpha * normals[:, 1]

    # max velocity calculation based on curvature
    opt_dx = np.gradient(opt_x)
    opt_dy = np.gradient(opt_y)
    opt_ddx = np.gradient(opt_dx)
    opt_ddy = np.gradient(opt_dy)

    kappa = np.abs(opt_dx * opt_ddy - opt_dy * opt_ddx) / np.maximum((opt_dx**2 + opt_dy**2)**(1.5), 1e-6)
    radius = 1.0 / np.maximum(kappa, 1e-6)

    v_max = np.sqrt(mu * g * radius)
    v_max = np.clip(v_max, 0, 5.0)

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

from typer import run
if __name__ == "__main__":
    run(compute_raceline)