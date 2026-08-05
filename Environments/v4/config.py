"""
Static configuration and derived constants for the v4 ATC environment.

Everything here is fixed at import time: the CONFIG dict (tunable settings), the
geometric/normalisation constants derived from it, the discrete action layout, and
the per-instruction workload costs.
"""

import random

# -- Tunable settings ----------------------------------------------------------

CONFIG = {
    # Aircraft & sector
    'ac_type':               'A320',
    'ac_speed':              450.0,
    'ac_mach':               0.78,           # nominal cruise Mach
    'ac_mach_min':           0.74,           # ATC speed-control envelope at FL350
    'ac_mach_max':           0.82,
    'mach_step':             0.04,           # Mach change per speed instruction (~24 kt TAS):
                                             # one step reaches the envelope edge from nominal
    'altitude':              350,
    'center_ll':             (0.0, 0.0),     # flat-earth equatorial: cos(0) = 1
    'n_aircraft':            lambda: random.randint(15, 30),            # sampled per episode
    'rho':                   lambda: random.uniform(1/15000, 1/5000),   # sampled per episode; area = n/rho
    'sep_nm':                5.0,
    'dest_dist_factor':      20.0,           # destination far beyond the sector, so the bearing to it is
                                             # near-constant and a held heading stays on route
    'arrival_tol_nm':        5.0,            # exit within this distance of the reference point = on-target
    'buffer_nm':             10.0,           # spawn buffer: min distance to traffic = sep_nm + buffer_nm
    'spawn_conflict_free':   True,           # reject spawns whose route hits CPA < sep within t_warn
    # Sector polygon -- varied but reasonably round (random convex shapes, circularity >= 0.7)
    'n_vertices':            lambda: random.randint(6, 12),
    'min_circularity':       0.7,
    'max_placement_tries':   50,
    # Aircraft placement jitter
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.5, 0.5),       # fully random crossing/exit directions
    # Simulation
    'sim_dt':                1.0,            # BlueSky integration timestep (DT) = 1 s
    'action_freq':           5,             # RL step = 5 s simulated (action_freq x sim_dt)
    'lookahead_s':           900.0,
    't_warn':                360.0,         # conflict-resolution horizon: 6 min (D_WARN = 45 NM)
    'crossings_per_episode': 4.0,
    'spawn_delay_s':         (0, 0),
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.67,          # ~2 min before CPA at t_warn = 360 s
    'drift_switch_margin':   0.01,
    'return_clear_nm':       20.0,          # clearance distance for the drift-fallback score (4 x sep)
    # Reward weights
    'w_los':                 10.00,         # heavy: separation violation
    'w_drift':               1.00,          # cosine drift penalty. ACT_COST scales with this, so w_drift
                                            # also sets the action-cost magnitude (doubling it doubles
                                            # both the drift penalty and every turn/speed cost).
    'w_work':                1.00,          # master scale for ACT_COST; tune magnitudes via w_drift
    'seed':                  None,
}

# -- Derived constants ---------------------------------------------------------

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 5 + N_NEIGHBOURS * 5   # 5 ownship + 4 intruders x 5 = 25

WARN_DIST_NM  = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0   # warning-horizon distance (45 NM)
CRUISE_SPD_NMS = CONFIG['ac_speed'] / 3600.0                     # nominal cruise speed (NM/s); obs scale

# -- Action layout (Discrete 10) -----------------------------------------------
#   0-2, 4-6  heading turns (stack on commanded heading)   3  hold   7  fly-direct
#   8  speed up   9  speed down
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}        # +1/-1 x mach_step on the commanded Mach
N_ACTIONS     = 10

# -- Workload cost per instruction ---------------------------------------------
# A heading instruction is one radio call, so its cost is SUB-ADDITIVE in the turn it
# commands: a single decisive turn must not cost more than splitting it across several
# instructions, or the policy is rewarded for salami-slicing. Costs are hard-coded:
# a 30-deg turn anchors at 0.5, a 60-deg turn at 0.75 (< 2x), a 45-deg turn at their
# midpoint 0.625. Hold is free; fly-direct is cheap (0.25 x the 30-deg turn); a speed
# change costs half a 30-deg turn. r_work = -w_work * ACT_COST.

_TURN_30 = 0.5                          # cost anchor: one 30-deg turn

ACT_COST = [
    0.75,                 # 0  turn -60
    0.625,                # 1  turn -45  ((0.75 + 0.5) / 2)
    0.5,                  # 2  turn -30
    0.0,                  # 3  hold (free)
    0.5,                  # 4  turn +30
    0.625,                # 5  turn +45
    0.75,                 # 6  turn +60
    0.25 * _TURN_30,      # 7  fly-direct (return to route: cheap)
    0.5  * _TURN_30,      # 8  speed up   (half a 30-deg turn)
    0.5  * _TURN_30,      # 9  speed down
]

# -- Observation labels (used by the visualiser's obs panel) -------------------
OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'a_cmd', 'v_cmd', 'retn_conf']
OBS_INTRUDER_LABELS = ['rho', 'theta', 'psi', 'vint', 'tau']
