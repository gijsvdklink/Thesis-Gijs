
import math as _math
from random import Random as _Random

ft   = 0.3048
kts  = 0.514444


def _speed_of_sound(alt_m):
    temp_k = 288.15 - 0.0065 * min(alt_m, 11000.0)
    return _math.sqrt(1.4 * 287.05287 * temp_k)


CONFIG = {
    'ac_type':               'A320',
    'ac_speed':              450.0,
    'ac_mach':               0.78,
    'ac_mach_min':           0.76,
    'ac_mach_max':           0.80,
    'mach_step':             0.02,
    'altitude':              350,
    'center_ll':             (0.0, 0.0),
    'n_aircraft':            lambda rng: rng.randint(15, 30),
    'rho':                   lambda rng: rng.uniform(1/20000, 1/10000),
    'sep_nm':                5.0,
    'on_route_hdg_tol_deg':  5.0,
    'buffer_nm':             10.0,
    'n_vertices':            lambda rng: rng.randint(6, 12),
    'min_circularity':       0.7,
    'max_placement_tries':   50,
    'min_chord_nm':          15.0,
    'sim_dt':                1.0,
    'action_freq':           5,
    't_warn':                360.0,
    'exits_per_episode':     4.0,
    'max_crossings':         8.0,
    'delay_mode':            'none',
    'delay_mean_s':          30.0,
    'delay_sigma':           0.4,
    'revision_kappa':        0.7,
    'n_neighbours':          4,
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.67,
    'w_los':                 10.00,
    'w_drift':               0.10,
    'w_work':                1.00,
    'seed':                  0,
}


TRAINING_SEEDS  = (1, 2, 3, 4, 5, 6, 7)
VALIDATION_SEED = 8

VALIDATION_EPISODES = 500
TRAINING_SCENARIOS  = 1_000_000_000

VALIDATION_SEEDS = tuple(_Random(VALIDATION_SEED).sample(range(TRAINING_SCENARIOS),
                                                         VALIDATION_EPISODES))
HELD_OUT = frozenset(VALIDATION_SEEDS)


NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

STEP_DURATION_S = CONFIG['action_freq'] * CONFIG['sim_dt']

N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 9 + N_NEIGHBOURS * 5

CRUISE_SPD_NMS = CONFIG['ac_speed'] / 3600.0
NMS_TO_KT      = 3600.0
CRUISE_ALT_M   = CONFIG['altitude'] * 100 * ft
KT_PER_MACH    = _speed_of_sound(CRUISE_ALT_M) / kts

EMPTY_RANGE_NM = 1000.0
NO_CONFLICT_S  = CONFIG['t_warn']

TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}
HOLD_ACTION   = 3
RETURN_TO_INITIAL_HDG_ACTION = 7
N_ACTIONS     = 10


_TURN_30 = 0.5

ACT_COST = [
    0.75,
    0.625,
    0.5,
    0.0,
    0.5,
    0.625,
    0.75,
    0.25 * _TURN_30,
    0.5  * _TURN_30,
    0.5  * _TURN_30,
]

OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'h_cmd', 'v_cmd', 'retn_conf', 'pending', 'wait_s',
                       'pend_hdg', 'pend_spd']
OBS_INTRUDER_LABELS = ['dist', 'theta', 'psi', 'vint', 'tlos']
