import numpy as np

# =========================================================================
# 1. Physical Footprint & Mass (1/10 Fiesta Scale with Sim2Real Margin)
# =========================================================================
WIDTH = 0.33                         # [m] Widened footprint for Fiesta rally shell
LENGTH = 0.58                        # [m] Bumper-to-bumper chassis length
# Added a 7% structural expansion padding to naturally enforce wider corner safety margins
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0)) * 1.07 

LF = 0.135                           # [m] Distance from COG to front axle (slight front bias)
LR = 0.155                           # [m] Distance from COG to rear axle
LWB = LF + LR                        # [m] Wheelbase (~290mm, standard 1/10 short course/rally)
MASS = 3.85                          # [kg] Weight of upgraded 4S race chassis with battery
G = 9.81                             # [m/s^2] Acceleration due to gravity

# =========================================================================
# 2. Kinematic, Actuation & Surface Limits (4S LiPo + Carpet Setup)
# =========================================================================
STEER_MIN = -0.4189                  # [rad] ~24 degrees max steering throw left
STEER_MAX = 0.4189                   # [rad] ~24 degrees max steering throw right
STEER_V_MAX = 4.5                    # [rad/s] Fast high-end steering servo limit
A_MAX = 16.5                         # [m/s^2] 4S instantaneous peak torque acceleration (~1.68G)
V_MIN = -4.0                         # [m/s] Capped reverse velocity
V_MAX = 16.0                         # [m/s] ~35 mph ceiling for an indoor track setup
PSI_PRIME_MAX = 5.5                  # [rad/s] Max yaw velocity rate
BETA_MAX = 0.35                      # [rad] Tight body slip angle cap to emulate traction rolls

MU = 1.15                            # [-] Mechanical traction coefficient of carpet tires on carpet
DR_FRAC = 0.18                       # [-] Domain Randomization variance envelope (18% parameter flux)

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
PROGRESS_SCALE = 1.0                 # [-] Step progress scaling weight
PROGRESS_V_COEF = 0.4                # [-] Lookahead velocity alignment scaling factor
BACKWARDS_PROGRESS_PENALTY_MUL = 15.0 # [-] Sane deterrent modifier for wrong-way driving
TERM_PENALTY = -100.0                # [-] Punitive crash limit lowered to keep value exploration alive
IDLE_PENALTY = -0.2                  # [-] Sane structural penalty for loitering
LATERAL_PENALTY = -0.15              # [-] Squaring penalty factor emphasizing centerline adherence

MAX_CENTERLINE_DEV = 1.5             # [m] Track boundary containment zone
STALL_VELOCITY = 1.0                  # [m/s] Matches your intended racing threshold
STALL_SECONDS_TO_STEPS = 3.0 / DT     # Allow 3 full seconds of recovery time

# =========================================================================
# 5. Sensors & Observation Tensor Offsets
# =========================================================================
NUM_LIDAR = 108                      # [-] Number of radial laser scan channels
LIDAR_FOV = np.radians(270.0)        # [rad] Standard planar view sweep
LIDAR_RANGE = 20.0                    # [m] Real-world indoor proximity scan boundary ceiling
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

# Things to reward
# - Following centerline
# - Going forward fast
# - Reaching checkpoints
# - Completing laps fast
# - Smooth steering (no jitter)
#
# Things to penalize
# - Going backwards
# - Diverging from centerline a lot
# - Going in reverse of centerline direction
# - Staying in the same area
# - Going too slow
# - Hitting obstacles
# - Eratic steering

# =========================================================================
# REWARD & PENALTY CONFIGURATION NOTES (Racing Line Discovery)
# =========================================================================

# -------------------------------------------------------------------------
# PRIORITY 1: PROGRESS & SPEED (70% - 80% of total weight)
# Objective: Force the agent to find the fastest way around the track.
# -------------------------------------------------------------------------

# [+] Differential Progress (s_t - s_t-1)
# Rewards the agent strictly for advancing along the track index since the last frame.
# This naturally teaches the car to cut corners and apex correctly to shorten its path.

# [+] Velocity Alignment
# Rewards vehicle velocity vector projected along the local track heading.
# Ensures speed is only heavily rewarded if it is actively moving the car forward.


# -------------------------------------------------------------------------
# PRIORITY 2: HARD CONSTRAINTS (Immediate Episode Termination)
# Objective: Eliminate hesitation and set the physical boundaries.
# -------------------------------------------------------------------------

# [-] Terminal Wall Collision
# Massive negative penalty (e.g., -500) and instant reset when hitting a wall.
# Teaches the agent where the absolute geometric limits of traction are.

# [-] Low-Velocity Stall Death
# Instant termination if vehicle speed drops below 1.0 m/s for more than 3 seconds.
# Eliminates "reward cowardice" where the agent parks at a corner to avoid crashing.


# -------------------------------------------------------------------------
# PRIORITY 3: REFINEMENT & SMOOTHNESS (Driving Style & Stability)
# Objective: Stop erratic behavior and stabilize weight transfer.
# -------------------------------------------------------------------------

# [-] Steering Action Delta
# Penalizes the absolute difference between the current and previous frame's steering inputs.
# Essential for wiping out high-frequency physics/servo jitter.

# [-] Soft Wall Proximity
# A minor exponential penalty that activates only when the car is within centimeters of a wall.
# Provides a soft visual buffer zone to help stabilize policy convergence near edges.


# -------------------------------------------------------------------------
# PRIORITY 4: THE SAFETY NET (The "Lost Lane" Guide)
# Objective: Invisible at high speeds, active only when recovery is needed.
# -------------------------------------------------------------------------

# [-] Speed-Gated Lateral Penalty
# Squared cross-track error multiplied by a dynamic speed gate: max(0.0, 2.0 - velocity).
# Drops to zero at racing speeds (allowing wide lines), but drags the car back to the center if it spins out or stops.