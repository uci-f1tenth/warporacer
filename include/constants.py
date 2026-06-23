import numpy as np

# =========================================================================
# 1. Physical Footprint & Mass (1/10 Fiesta Scale with Sim2Real Margin)
# =========================================================================
WIDTH = 0.33                         # [m] Widened footprint for Fiesta rally shell
LENGTH = 0.58                        # [m] Bumper-to-bumper chassis length
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0)) * 1.10 # Expand chassis by 10%

LF = 0.135                           # [m] Distance from COG to front axle
LR = 0.155                           # [m] Distance from COG to rear axle
LWB = LF + LR                        # [m] Wheelbase
MASS = 3.85                          # [kg] Weight of upgraded 4S race chassis
G = 9.81                             # [m/s^2] Acceleration due to gravity

# =========================================================================
# 2. Kinematic, Actuation & Surface Limits (4S LiPo + Carpet Setup)
# =========================================================================
STEER_MIN = -0.4189                  # [rad] ~24 degrees max steering throw left
STEER_MAX = 0.4189                   # [rad] ~24 degrees max steering throw right
STEER_V_MAX = 4.5                    # [rad/s] Fast high-end steering servo limit
A_MAX = 16.5                         # [m/s^2] 4S instantaneous peak torque acceleration
V_MIN = -4.0                         # [m/s] Capped reverse velocity
V_MAX = 16.0                         # [m/s] ~35 mph ceiling for an indoor track setup
PSI_PRIME_MAX = 7.5                  # [rad/s] Max yaw velocity rate
BETA_MAX = 0.5                       # [rad] Tight body slip angle cap

# OPTIMIZED: Slightly increased tracking friction to reward hard carving in tight turns
MU = 1.25                            # [-] True high-grip mechanical traction on carpet
DR_FRAC = 0.15                       # [-] Tightened variance envelope to stabilize policy baseline

# =========================================================================
# 3. Temporal Steps & Sub-integration Timing
# =========================================================================
DT = 1.0 / 60.0                      # [s] Core step physics engine update interval (60Hz)
SUBSTEPS = 6                         # [-] RK4 integration passes per step
DT_SUB = DT / float(SUBSTEPS)        # [s] Individual integration step
DT_SUB_HALF = DT_SUB * 0.5           # [s] Cache constant for solver
DT_SUB_SIX = DT_SUB / 6.0            # [s] Cache constant for solver

# =========================================================================
# 4. Reward Shaping & Normalization Weights (Generalization Tuned)
# =========================================================================
PROGRESS_SCALE = 0.8                 # INCREASING: Prioritize raw downward track progression
PROGRESS_V_COEF = 0.2                # INCREASING: Reward velocity aligned with the path horizon
BACKWARDS_PROGRESS_PENALTY_MUL = 50.0 # INCREASING: Explicitly kill wrong-way wiggling immediately
TERM_PENALTY = -50.0                 # INCREASING: Give crashing a sharper, distinct penalty drop

IDLE_PENALTY = -0.5                  # INCREASING: Make loitering or oscillation hurt more
# RESTORING: Bring this back to a modest value. This forces the agent to keep 
# moving cleanly along the track vector rather than parking on the apex.
LATERAL_PENALTY = -0.01

MAX_CENTERLINE_DEV = 2.5             # [m] Track boundary containment zone
STALL_VELOCITY = 0.5
STALL_SECONDS_TO_STEPS = int(1.0 / DT)

# =========================================================================
# 5. Sensors & Observation Tensor Offsets
# =========================================================================
NUM_LIDAR = 108                      # [-] Number of radial laser scan channels
LIDAR_FOV = np.radians(270.0)        # [rad] Standard planar view sweep
LIDAR_RANGE = 20.0                   # [m] Real-world indoor proximity scan boundary ceiling
NUM_LOOKAHEAD = 20                   # [-] Number of metric horizon checkpoints tracking forward

OBS_FRENET_OFF = 3 + NUM_LIDAR       # Offset location index for Frenet features
OBS_LOOK_OFF = OBS_FRENET_OFF + 2    # Offset location index for target tracking vectors
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD # Total input size matching NN configuration
ACT_DIM = 2                          # [Steering Velocity, Longitudinal Acceleration]
MAX_STEPS = 5_000                    # [-] Reduced from 10k to prevent endless looping on fails

# =========================================================================
# 6. Map Occupancy Array Handling Constants
# =========================================================================
OCC_THRESH = 230                     # [-] Greyscale barrier threshold value
SMOOTH_WINDOW = 51                   # [-] Convoluted path smoothing filter window
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1                  # Environment terminal crash flag state
DONE_TRUNCATED = 2                   # Environment timeout truncation flag state

# =========================================================================
# 7. Dynamic Horizon Tuning Constants
# =========================================================================
# INCREASED: Gives a larger target window so the relative angle to the target
# waypoint doesn't flip wildly when entering an apex at speed.
BASE_STRIDE = 15.0           # [points] Minimum waypoint index skip at 0 m/s
VELOCITY_SCALE = 1.5         # [-] Scaling multiplier tracking forward velocity