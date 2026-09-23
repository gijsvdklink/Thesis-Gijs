# -- Tunable settings ----------------------------------------------------------

import math as _math
from random import Random as _Random

# Unit constants and the ISA atmosphere, previously taken from bluesky.tools.aero. Kept here
# so the environment has no BlueSky dependency at all; _speed_of_sound reproduces BlueSky's
# mach2tas at FL350 to the last digit (576.419 kt per Mach).
ft   = 0.3048
kts  = 0.514444


def _speed_of_sound(alt_m):
    # ISA troposphere: 288.15 K at sea level, lapsing 6.5 K/km up to the tropopause at 11 km.
    temp_k = 288.15 - 0.0065 * min(alt_m, 11000.0)
    return _math.sqrt(1.4 * 287.05287 * temp_k)

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
    # A METRIC ONLY: an aircraft leaving within this many degrees of its entry heading left ON ROUTE.
    'on_route_hdg_tol_deg':  5.0,
    'buffer_nm':             10.0,           # spawn buffer: min distance to traffic = sep_nm + buffer_nm
    # Sector polygon -- varied but reasonably round (random convex shapes, circularity >= 0.7)
    'n_vertices':            lambda rng: rng.randint(6, 12),
    'min_circularity':       0.7,
    'max_placement_tries':   50,
    # Aircraft enter on a random heading, so half of the draws point back out of the sector.
    # This is what rejects those, and grazing entries through a corner with them.
    'min_chord_nm':          15.0,           # shortest crossing an entry may be given
    # Simulation
    'sim_dt':                1.0,            # BlueSky integration timestep (DT) = 1 s
    'action_freq':           5,             # RL step = 5 s simulated (action_freq x sim_dt)
    # THE conflict horizon, in seconds (360 = 6 min).
    't_warn':                360.0,
    # An episode ends once this many aircraft per slot have left the sector.
    'exits_per_episode':     4.0,
    # Safety cap in sector traversals, in case traffic stops leaving at all.
    'max_crossings':         8.0,
    # ATCO behaviour: the response-delay law lives in atco.py, and the type is set per instance.
    'delay_mode':            'none',        # default delay type; see atco.DELAY_MODES
    'delay_mean_s':          30.0,          # mean time for the controller to act, in seconds
    'delay_sigma':           0.4,           # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s
    'delay_max_s':           70.0,          # cap on a drawn response time; 1.0% of lognormal draws reach it           # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s
    'revision_kappa':        0.7,           # kappa: a revision is taken up in this fraction of a full response time
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.67,          # ~2 min before CPA at t_warn = 360 s
    # Reward weights
    'w_los':                 10.00,         # heavy: separation violation
    'w_drift':               0.10,          # per simulated second, cosine drift penalty on [0, 2]
    'w_work':                1.00,          # master scale for ACT_COST; tune magnitudes via w_drift
    # Fallback master seed, so a bare AirspaceEnv() is reproducible out of the box.
    'seed':                  0,
}

# -- The eight seeds of the whole experiment -----------------------------------
#
# Seeds 1-7 are the seven training runs. Every delay type is trained at all seven, so models in
# the same column start from identical network weights and fly identical scenarios, and the
# delay is the only thing that differs between them. Seed 8 names the held-out test set.
# There is nothing else: no multipliers, no offsets, no reserved bands.

TRAINING_SEEDS  = (1, 2, 3, 4, 5, 6, 7)
VALIDATION_SEED = 8

VALIDATION_EPISODES = 300
TRAINING_SCENARIOS  = 1_000_000_000    # training draws a scenario seed below this, at random

# The 100 held-out scenarios, drawn once from VALIDATION_SEED. Training skips any scenario in this set
# (see AirspaceEnv._new_episode_rngs), so no model can ever have met one, however long it runs.
VALIDATION_SEEDS = tuple(_Random(VALIDATION_SEED).sample(range(TRAINING_SCENARIOS),
                                                         VALIDATION_EPISODES))
HELD_OUT = frozenset(VALIDATION_SEEDS)

# -- Derived constants ---------------------------------------------------------

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

STEP_DURATION_S = CONFIG['action_freq'] * CONFIG['sim_dt']   # simulated seconds per RL step

N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 10 + N_NEIGHBOURS * 5  # 10 ownship + 4 intruders x 5 = 30

CRUISE_SPD_NMS = CONFIG['ac_speed'] / 3600.0        # nominal cruise speed (NM/s); spawn checks
NMS_TO_KT      = 3600.0                             # NM/s -> kt (observation reports kt)
# TAS is linear in Mach at a fixed altitude, so one constant is exact: the ISA speed of
# sound at cruise, straight from BlueSky's atmosphere model rather than an assumed ratio.
CRUISE_ALT_M   = CONFIG['altitude'] * 100 * ft
KT_PER_MACH    = _speed_of_sound(CRUISE_ALT_M) / kts

# Sentinels in raw units: an empty intruder slot, and a pair that never intrudes or lies beyond the horizon.
EMPTY_RANGE_NM = 1000.0
NO_CONFLICT_S  = CONFIG['t_warn']

# -- Action layout (Discrete 10): 0-2/4-6 turns accumulating on the last EXECUTED heading, 3 hold, 7 return, 8-9 speed --
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}        # +1/-1 x mach_step on the commanded Mach
HOLD_ACTION   = 3                     # true no-op: no instruction is transmitted at all
RETURN_TO_INITIAL_HDG_ACTION = 7      # the zero-offset action: fly the initial heading
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
    0.25 * _TURN_30,      # 7  return to initial heading (cheap: undoing a deviation)
    0.5  * _TURN_30,      # 8  speed up   (half a 30-deg turn)
    0.5  * _TURN_30,      # 9  speed down
]

# -- Observation labels (visualiser obs panel): angles in rad, speeds in kt, dist in NM, times in s --
OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'h_cmd', 'v_cmd', 'retn_conf', 'advised_hdg', 'advised_spd',
                       't_first', 't_last', 'n_rev']
OBS_INTRUDER_LABELS = ['dist', 'theta', 'psi', 'vint', 'tlos']
