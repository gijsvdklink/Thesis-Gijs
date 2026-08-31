# -- Tunable settings ----------------------------------------------------------

from bluesky.tools.aero import ft, kts, mach2tas

CONFIG = {
    # Aircraft & sector
    'ac_type':               'A320',
    'ac_speed':              450.0,
    'ac_mach':               0.78,           # nominal cruise Mach
    # ATC speed-control envelope at FL350. The ceiling is the A320's mmo in BlueSky's
    # performance model: commanding above M 0.80 is silently clamped, so 0.80 IS the max.
    'ac_mach_min':           0.76,           # 438.1 kt TAS
    'ac_mach_max':           0.80,           # 461.1 kt TAS -- mmo, the hard ceiling
    'mach_step':             0.02,           # Mach change per speed instruction (~11.5 kt TAS)
    'altitude':              350,
    'center_ll':             (0.0, 0.0),     # flat-earth equatorial: cos(0) = 1
    'n_aircraft':            lambda rng: rng.randint(15, 30),            # sampled per episode
    'rho':                   lambda rng: rng.uniform(1/20000, 1/10000),  # sampled per episode; area = n/rho
    'sep_nm':                5.0,
    # Arrival is a METRIC ONLY: exits within this many degrees of the initial heading count as arrived.
    'arrival_hdg_tol_deg':   5.0,
    'buffer_nm':             10.0,           # spawn buffer: min distance to traffic = sep_nm + buffer_nm
    # Sector polygon -- varied but reasonably round (random convex shapes, circularity >= 0.7)
    'n_vertices':            lambda rng: rng.randint(6, 12),
    'min_circularity':       0.7,
    'max_placement_tries':   50,
    'min_chord_nm':          15.0,           # reject spawn->ref routes too short to really be flown
    # Aircraft placement jitter
    'spawn_jitter':          lambda rng: rng.uniform(0.1, 0.9),
    'ref_jitter':            lambda rng: rng.uniform(-0.5, 0.5),      # fully random crossing/exit directions
    # Simulation
    'sim_dt':                1.0,            # BlueSky integration timestep (DT) = 1 s
    'action_freq':           5,             # RL step = 5 s simulated (action_freq x sim_dt)
    # THE conflict horizon, in seconds (360 = 6 min).
    't_warn':                360.0,
    'crossings_per_episode': 2.0,
    # Action-response delay: the timing law lives in delays.py, and the type is set per instance.
    'delay_mode':            'none',        # default delay type; see delays.DELAY_MODES
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.67,          # ~2 min before CPA at t_warn = 360 s
    # Reward weights
    'w_los':                 10.00,         # heavy: separation violation
    'w_drift':               0.50,          # cosine drift penalty on [0, 2]; also scales ACT_COST
    'w_work':                1.00,          # master scale for ACT_COST; tune magnitudes via w_drift
    # Fallback master seed, so a bare AirspaceEnv() is reproducible out of the box.
    'seed':                  0,
}

TRAINING_SCENARIOS = 1_000_000_000     # training draws a seed below this, at random

# The held-out validation set: the same 100 scenarios for every trained model.
VALIDATION_SEEDS = tuple(range(2_000_000_000, 2_000_000_100))

# -- Derived constants ---------------------------------------------------------

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

STEP_DURATION_S = CONFIG['action_freq'] * CONFIG['sim_dt']   # simulated seconds per RL step

N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 7 + N_NEIGHBOURS * 5   # 7 ownship + 4 intruders x 5 = 27

CRUISE_SPD_NMS = CONFIG['ac_speed'] / 3600.0        # nominal cruise speed (NM/s); spawn checks
NMS_TO_KT      = 3600.0                             # NM/s -> kt (observation reports kt)
# TAS is linear in Mach at a fixed altitude, so one constant is exact: the ISA speed of
# sound at cruise, straight from BlueSky's atmosphere model rather than an assumed ratio.
CRUISE_ALT_M   = CONFIG['altitude'] * 100 * ft
KT_PER_MACH    = mach2tas(1.0, CRUISE_ALT_M) / kts

# Sentinels in raw units: an empty intruder slot, and a pair that never intrudes or lies beyond the horizon.
EMPTY_RANGE_NM = 1000.0
NO_CONFLICT_S  = CONFIG['t_warn']

# -- Action layout (Discrete 10): 0-2/4-6 turns accumulating on the last EXECUTED heading, 3 hold, 7 return, 8-9 speed --
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}        # +1/-1 x mach_step on the commanded Mach
HOLD_ACTION   = 3                     # true no-op: no instruction is transmitted at all
RETURN_TO_ROUTE_ACTION = 7            # the zero-offset action: fly the initial heading
N_ACTIONS     = 10

# -- Workload cost, SUB-ADDITIVE in the turn commanded so splitting one turn is never cheaper. r_work = -w_work * ACT_COST --

_TURN_30 = 0.5                          # cost anchor: one 30-deg turn

ACT_COST = [
    0.75,                 # 0  turn -60
    0.625,                # 1  turn -45  ((0.75 + 0.5) / 2)
    0.5,                  # 2  turn -30
    0.0,                  # 3  hold (free)
    0.5,                  # 4  turn +30
    0.625,                # 5  turn +45
    0.75,                 # 6  turn +60
    0.25 * _TURN_30,      # 7  return to route (cheap: undoing a deviation)
    0.5  * _TURN_30,      # 8  speed up   (half a 30-deg turn)
    0.5  * _TURN_30,      # 9  speed down
]

# -- Observation labels (visualiser obs panel): angles in rad, speeds in kt, dist in NM, times in s --
OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'a_cmd', 'v_cmd', 'retn_conf', 'pending', 'wait_s']
OBS_INTRUDER_LABELS = ['dist', 'theta', 'psi', 'vint', 'tlos']
