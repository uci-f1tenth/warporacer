import numpy as np
import casadi as ca
import pandas as pd


g = 9.81
def compute_raceline(centerline, width, mu): # physics??!?!?! unfamiliar concept
    N = len(centerline)
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    normals = np.column_stack((-dy, dx))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    opti = ca.Opti()
    alpha = opti.variable(N)
    x = centerline[:, 0] + alpha * normals[:, 0]
    y = centerline[:, 1] + alpha * normals[:, 1]

    dx_path = ca.diff(x)
    dy_path = ca.diff(y)
    ddx_path = ca.diff(dx_path)
    ddy_path = ca.diff(dy_path)

    curve_sq = ca.sumsqr(ddx_path) + ca.sumsqr(ddy_path)
    opti.minimize(curve_sq)

    safety_margin = 0.05
    max_shift = (width / 2) - safety_margin
    opti.subject_to(opti.bounded(-max_shift, alpha, max_shift))

    opti.subject_to(alpha[0] == alpha[-1])

    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})
    sol = opti.solve()

    opt_alpha = sol.value(alpha)
    opt_x = centerline[:, 0] + opt_alpha * normals[:, 0]
    opt_y = centerline[:, 1] + opt_alpha * normals[:, 1]

    opt_dx = np.gradient(opt_x)
    opt_dy = np.gradient(opt_y)
    opt_ddx = np.gradient(opt_dx)
    opt_ddy = np.gradient(opt_dy)

    kappa = np.abs(opt_dx * opt_ddy - opt_dy * opt_ddx) / np.maximum((opt_dx**2 + opt_dy**2)**(3/2), 1e-6)
    radius = 1.0 / np.maximum(kappa, 1e-6)

    v_max = np.sqrt(mu * g * radius)
    v_max = np.clip(v_max, 0, 5.0)

    df = pd.DataFrame({
        'x': opt_x,
        'y': opt_y,
        'v_target': v_max,
    })
    df.to_csv('optimal_raceline.csv', index=False)
    print("Optimal raceline saved to 'optimal_raceline.csv'.")
