import numpy as np

# TODO : Units!!!

MU = 1.5489 #1.0489
LF = 0.15875
LR = 0.17145
LWB = LF + LR
MASS = 6.74 #3.74

STEER_MIN = -0.4189
STEER_MAX = 0.4189
STEER_V_MAX = 6.4 #3.2
A_MAX = 25.0 #9.51
V_MIN = -5.0
V_MAX = 25.0 #20.0
PSI_PRIME_MAX = 6.0
BETA_MAX = 1.2

# Car
WIDTH = 0.5 #0.31
LENGTH = 1.0 #0.58
CAR_HALF_DIAG = float(np.hypot(WIDTH / 2.0, LENGTH / 2.0))
G = 9.81
DT = 1.0 / 60.0
SUBSTEPS = 6
DT_SUB = DT / float(SUBSTEPS)
DT_SUB_HALF = DT_SUB * 0.5
DT_SUB_SIX = DT_SUB / 6.0

DR_FRAC = 0.15

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

PROGRESS_SCALE = 0.5
PROGRESS_V_COEF = 0.25
BACKWARDS_PROGRESS_PENALTY_MUL = 10.0
TERM_PENALTY = -500.0
IDLE_PENALTY = -0.5
LATERAL_PENALTY = -0.1
MAX_CENTERLINE_DEV = 2.0
STALL_VELOCITY = 2.0
STALL_SECONDS_TO_STEPS = 2.0 / DT

NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
LIDAR_RANGE = 20.0
NUM_LOOKAHEAD = 10
OBS_FRENET_OFF = 3 + NUM_LIDAR
OBS_LOOK_OFF = OBS_FRENET_OFF + 2
OBS_DIM = OBS_LOOK_OFF + 2 * NUM_LOOKAHEAD
ACT_DIM = 2
MAX_STEPS = 10_000

OCC_THRESH = 230
SMOOTH_WINDOW = 51
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DONE_TERMINATED = 1
DONE_TRUNCATED = 2

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