"""
experimental -- v4 ATCO env + pluggable pilot/ATCO action-response delay.

Extends the v4 environment with an ACAS-Xu-style response delay (Kochenderfer et al.,
"Robust Airborne Collision Avoidance through Dynamic Programming", §6): an issued
instruction does NOT reach the simulator instantly. Instead it is held "on display"
and only executed after a sampled INITIATION DELAY, modelling the time before the
pilot/ATCO begins the manoeuvre. Until then the aircraft keeps flying its previous
command, so the cost of a slow response is felt by the agent (conflict/LoS are
computed on the real, lagged trajectory -- the delay is never masked).

The DELAY MODEL is pluggable (see action_response_delay.py): pass delay= a model
instance, or a mode string 'deterministic' / 'probabilistic' (+ delay_kwargs), to
AirspaceEnv. The env owns the deferred-command MECHANISM and the two extra
observations; the model owns how the delay is sampled and how response progress is
perceived:
  AirspaceEnv(delay='deterministic')   fixed delay -> resp_pend = known countdown (MDP)
  AirspaceEnv(delay='probabilistic')   lognormal  -> resp_pend = 1 - belief (POMDP)

The commanded heading/speed is split into two states:
  issued    -- what the agent has commanded (set immediately on the action)
  effective -- what BlueSky is actually executing (set when the delay elapses)
The s_RA analogue (which advisory is pending + response progress) is exposed to the
agent as two extra ownship observations (iss_dhdg, resp_pend; see below).

v4 -- ATCO conflict-resolution environment (multi-aircraft, ACAS Xu observation).

Reward is purely negative (no positive components):
  -w_los      x 1[LoS]                                      heavy:  separation violation during step
  -w_conflict x (1-dcpa/sep) x (1-tcpa/t_warn)             medium: linear DCPA*TCPA conflict score
  -w_drift    x ((1-cos(psi_dest-psi_cmd))/2)                medium: commanded heading deviation from route
                                                              (classic cosine drift penalty)
  -w_work     x act_cost                                    instruction workload (see ACT_COST):
    a turn costs ~ 30 s of the drift it creates; hold free; fly-direct cheap; speed = half a 30-deg turn

Observation space (28 floats) -- ACAS Xu states, ego-centric from the focus
aircraft (ownship), extended to the 4 nearest/most-urgent intruders. Angles are
in radians for unit consistency; the remaining states are scaled by their
physical ranges. VecNormalize (norm_obs=True) standardises everything for the
network, so the raw scales below need only be internally consistent. NOTE: the
"ownship real-state" features (dpsi_dest, v_own, turn_progress) are computed on the
EFFECTIVE (actually-flown) command, not the issued one -- so a pending instruction
does not yet show up there; it shows up only in the two response-delay features:
  ownship (shared, 8):
    [0]  sin(dpsi_dest)      heading error (cmd vs bearing-to-destination), sin/cos
    [1]  cos(dpsi_dest)      encoded to avoid the +-pi wrap discontinuity
    [2]  v_own               ownship speed / nominal cruise (V_NOM)         ~[0, 1]
    [3]  turn_progress       wrap(cmd_hdg - actual_hdg) in radians: outstanding
                             turn (0 = settled onto the commanded heading)
    [4]  conflict_now        conflict score on the CURRENT heading           [0, 1]
                             (0 = clear, ->1 = in / heading into conflict)
    [5]  conflict_if_return  conflict score if flying DIRECT back to route     [0, 1]
                             (0 = safe to return, ->1 = returning enters conflict).
                             No time horizon: any future predicted conflict counts,
                             so a conflict beyond t_warn still flags the return unsafe.
    [6]  iss_dhdg           PENDING advisory: wrap(issued - effective) heading, rad.
                             0 once the instruction has been executed (issued==effective).
    [7]  resp_pend          response NOT-yet-executed signal, [0, 1]. 0 = settled/no
                             pending advisory; ->1 = surely still pending. Deterministic:
                             exact remaining-fraction countdown. Probabilistic: 1 minus
                             the Bayesian belief that the pilot has begun responding.
  per intruder (4 slots x 5), urgency-desc then nearest-asc:
    rho    distance to intruder / D_WARN (warning horizon)              [0, 1]
    theta  angle to intruder relative to ownship heading, radians       [-pi, pi]
    psi    intruder heading relative to ownship heading, radians        [-pi, pi]
    v_int  intruder speed / nominal cruise (V_NOM)                      ~[0, 1]
    tau    time until loss of separation / t_warn. Horizontal sim (single
           altitude), so tau is the horizontal analogue of the ACAS Xu
           vertical-sep timer: time to CPA (0 if in LoS, 1 if diverging) [0, 1]
  empty slot sentinel: (rho = 1, theta = 0, psi = 0, v_int = 0, tau = 1)

Action space (Discrete 10) -- heading turns + hold + fly-direct + speed:
  0  turn -60 deg     psi_cmd -= 60   stacks on current commanded heading
  1  turn -45 deg     psi_cmd -= 45   stacks on current commanded heading
  2  turn -30 deg     psi_cmd -= 30   stacks on current commanded heading
  3  HOLD             true no-op: no instruction is issued
  4  turn +30 deg     psi_cmd += 30   stacks on current commanded heading
  5  turn +45 deg     psi_cmd += 45   stacks on current commanded heading
  6  turn +60 deg     psi_cmd += 60   stacks on current commanded heading
  7  FLY DIRECT       psi_cmd = fixed route heading (return to route); persistent
  8  SPEED UP         commanded Mach += mach_step, clamped to [mach_min, mach_max]
  9  SPEED DOWN       commanded Mach -= mach_step, clamped to [mach_min, mach_max]

Heading and speed are both persistent selected values (BlueSky holds them): a turn stacks
on the commanded heading, a speed action steps the commanded Mach (SPD command). Speed is
the ATCO's secondary tool for in-trail / crossing spacing; heading is primary.

Fly-direct (back to route) re-commands the fixed route heading and holds it each step.
It stays active through HOLD and speed changes, and is cancelled by any manual turn.
Intended pattern: turn(s) to avoid -> hold -> fly direct. Hold is a true no-op (nothing is
sent to the simulator); the aircraft continues executing its last instruction.

Focus aircraft = aircraft with highest single-pair urgency (total urgency burden as tiebreak).
When the sector is conflict-free, focus falls back to the drifted aircraft with the best
drift x clearance score, where clearance ramps from 0 to 1 with the nearest-neighbour
distance (return_clear_nm): drifters in open airspace are prioritised for return.
Density parameter rho = aircraft/km^2; sector area = n / rho.
Flat-earth projection: center_ll = (0, 0) so cos(lat) = 1 everywhere.
"""

import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.tools import geo
from bluesky.stack.stackbase import Stack as _BsStack
from polygenerator import random_convex_polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

from Environments.action_response.action_response_delay import make_delay_model, seed_delay

# -- Configuration -------------------------------------------------------------

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
    'center_ll':             (0.0, 0.0),      # flat-earth equatorial: cos(0)=1
    'n_aircraft':            lambda: random.randint(2, 6),  # 2-6 aircraft per episode
    'rho':                   lambda: random.uniform(1/20000, 1/10000),  # aircraft/km^2; area = n/rho.
                                                                          # medium-low density: 10-20k km^2/ac
    'sep_nm':                5.0,
    'dest_dist_factor':      20.0,           # destination far beyond the sector: bearing-to-dest is
                                               # near-constant, so a held heading stays on route (turning
                                               # back is enough; no continuous re-aiming needed)
    'arrival_tol_nm':        5.0,             # exit within this distance of t_ref counts as on-target
    'buffer_nm':             10.0,            # spawn buffer: min distance to traffic = sep_nm + buffer_nm
    # Polygon -- varied but reasonably round sectors (random convex shapes, circ >= 0.7)
    'n_vertices':            lambda: random.randint(6, 12),  # varies per episode; enough vertices to reach 0.7
    'min_circularity':       0.7,               # floor: rounder sectors, still some shape variation
    'max_placement_tries':   50,
    # Aircraft placement jitter
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.2, 0.2),  # wider spread of reference points -> more
                                                                  # varied crossing angles / entry-exit geometry
    # Simulation
    'sim_dt':                0.5,
    'action_freq':           10,              # RL step = 5 s simulated
    'lookahead_s':           900.0,
    't_warn':                360.0,           # conflict-resolution horizon: 6 min (D_WARN = 45 NM)
    'crossings_per_episode': 4.0,
    'spawn_delay_s':         (0, 0),
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.67,   # ≈ 2 min before CPA at t_warn=360 s
    'drift_switch_margin':   0.01,
    'return_clear_nm':       20.0,            # full clearance distance for the drift fallback score (4 x sep)
    # Reward weights
    'w_los':                 10.00,           # heavy: separation violation
    'w_conflict':            2.00,            # medium: imminence x miss-distance of worst conflict
    'w_drift':               1.00,            # cosine drift penalty: -w_drift * (1 - cos(dpsi)) / 2.
                                               # kept below w_conflict so avoiding conflict beats
                                               # staying on route; ACT_COST scales with this, so
                                               # turns/speed get cheaper in step with drift
    'w_work':                1.00,            # master scale for ACT_COST (already in reward units);
                                               # a turn costs ~ 30 s of the drift it creates
    # Action-response delay parameters live in action_response_delay.DELAY_DEFAULTS;
    # the env builds a delay model from the `delay` constructor argument.
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR   = CONFIG['n_neighbours']
OBS_DIM = 6 + N_NBR * 5       # 6 ownship (sin dest, v_own, turn_progress,
                              # conflict_if_return, iss_dhdg, resp_pend) + 4 intruders x 5
                              # (rho, theta, psi, v_int, tau) = 28

D_WARN = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0  # warning horizon distance (45 NM)
V_NOM  = CONFIG['ac_speed'] / 3600.0                      # nominal cruise speed (NM/s); speed-normalising reference

# Action layout (Discrete 10):
#   0-2, 4-6  heading turns (stack on commanded heading)    3  hold    7  fly-direct
#   8  speed up    9  speed down
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}        # +1/-1 x mach_step on the commanded Mach

# Workload cost per instruction, calibrated so issuing a turn costs about the same as
# letting the aircraft drift for ~30 s at the angle the turn creates:
#     cost(turn d) = steps_in_30s * w_drift * (1 - cos d) / 2
# Hold is free; fly-direct (return to route) is cheap; a speed change costs half a 30-deg
# turn (twice as cheap as the cheapest heading change). r_work = -w_work * ACT_COST[a].
_STEPS_30S = round(30.0 / (CONFIG['action_freq'] * CONFIG['sim_dt']))   # 6 RL steps

def _drift_30s_cost(delta_deg):
    """Drift penalty accrued over ~30 s if a turn of delta_deg became route drift."""
    return _STEPS_30S * CONFIG['w_drift'] * (1.0 - math.cos(math.radians(delta_deg))) / 2.0

ACT_COST = [
    _drift_30s_cost(60),          # 0  turn -60
    _drift_30s_cost(45),          # 1  turn -45
    _drift_30s_cost(30),          # 2  turn -30
    0.0,                          # 3  hold (free)
    _drift_30s_cost(30),          # 4  turn +30
    _drift_30s_cost(45),          # 5  turn +45
    _drift_30s_cost(60),          # 6  turn +60
    0.25 * _drift_30s_cost(30),   # 7  fly-direct (return to route: cheap)
    0.5 * _drift_30s_cost(30),    # 8  speed up   (half a 30-deg turn)
    0.5 * _drift_30s_cost(30),    # 9  speed down
]

_bs_initialized = False

# Labels for the visualisation obs panel (make_html.py)
OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'cmd_stack', 'retn_conf',
                       'iss_dhdg', 'b_exec']
OBS_INTRUDER_LABELS = ['rho', 'theta', 'psi', 'vint', 'tau']


__all__ = ['AirspaceEnv', 'CONFIG', 'NM_TO_KM', 'latlon_to_nm', 'wrap_to_180', 'OBS_DIM',
           'OBS_OWNSHIP_LABELS', 'OBS_INTRUDER_LABELS', 'seed_delay']

# -- Coordinate helpers --------------------------------------------------------

def latlon_to_nm(center_ll, lat, lon):
    """Convert (lat, lon) to (east_nm, north_nm) relative to center_ll."""
    ref_lat, ref_lon = center_ll
    east_nm  = (lon - ref_lon) * 60.0 * math.cos(math.radians(ref_lat))
    north_nm = (lat - ref_lat) * 60.0
    return np.array([east_nm, north_nm])

def nm_to_latlon(center_ll, east_nm, north_nm):
    """Convert (east_nm, north_nm) offsets back to (lat, lon)."""
    ref_lat, ref_lon = center_ll
    return (ref_lat + north_nm / 60.0,
            ref_lon + east_nm  / (60.0 * math.cos(math.radians(ref_lat))))

def wrap_to_180(angle_deg):
    """Wrap an angle in degrees to (-180, 180]."""
    return (angle_deg + 180) % 360 - 180

def _speed_nms(sim_idx):
    """True airspeed of aircraft at sim_idx in NM/s."""
    return float(bs.traf.tas[sim_idx]) / 1852.0

# -- Sector polygon ------------------------------------------------------------

def _make_polygon(area_km2):
    """Generate a random convex polygon with the requested area, centred at origin."""
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw    = ShapelyPolygon(random_convex_polygon(CONFIG['n_vertices']()))
        scale  = math.sqrt(target_nm2 / raw.area)
        scaled = shapely_scale(raw, xfact=scale, yfact=scale, origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= CONFIG['min_circularity']:
            break
    cx, cy = scaled.centroid.x, scaled.centroid.y
    return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])

# -- Aircraft placement --------------------------------------------------------

def _place_one(polygon, sector, n_sectors):
    """
    Place one aircraft on the polygon boundary.

    The spawn point and reference (exit) point are chosen at evenly spaced arcs with
    random jitter. The route heading is the bearing from spawn to reference computed
    in the flat NM frame, and the destination is a far point along that heading in the
    SAME frame. Working entirely in the flat frame avoids the geodesic/flat projection
    mismatch, so a held route heading reads as exactly zero drift.
    """
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx - minx)**2 + (maxy - miny)**2) * CONFIG['dest_dist_factor']

    t_spawn  = (sector + CONFIG['spawn_jitter']()) / n_sectors
    t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
    spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
    ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)

    # route heading (deg, 0 = north, clockwise) from spawn to reference, flat NM frame
    route_hdg = math.degrees(math.atan2(ref_pt.x - spawn_pt.x,
                                        ref_pt.y - spawn_pt.y)) % 360.0
    dest_e = spawn_pt.x + dest_dist * math.sin(math.radians(route_hdg))
    dest_n = spawn_pt.y + dest_dist * math.cos(math.radians(route_hdg))

    spawn_ll = nm_to_latlon(CONFIG['center_ll'], spawn_pt.x, spawn_pt.y)
    ref_ll   = nm_to_latlon(CONFIG['center_ll'], ref_pt.x,   ref_pt.y)
    dest_ll  = nm_to_latlon(CONFIG['center_ll'], dest_e,     dest_n)

    return {'sp_ll': spawn_ll, 'dest_ll': dest_ll,
            'ref_ll': ref_ll, 'heading': route_hdg}

# -- Pair urgency --------------------------------------------------------------

def _urgency_from_state(pos_i, vel_i, pos_j, vel_j):
    """
    Urgency of the conflict between two aircraft given their planar state.

    Positions are (east, north) in NM; velocities are (east, north) in NM/s.

    Returns:
      > 1   active LoS    (scales 1 at sep boundary → 10 at d=0)
      0..1  predicted LoS (scales 0 at t_warn → 1 at t_CPA=0)
      0     safe / diverging
    """
    d_east  = pos_j[0] - pos_i[0]
    d_north = pos_j[1] - pos_i[1]
    dist_sq = d_east**2 + d_north**2
    sep     = CONFIG['sep_nm']

    if dist_sq < sep**2:
        return 1.0 + 9.0 * (1.0 - math.sqrt(dist_sq) / sep)

    dv_east  = vel_j[0] - vel_i[0]
    dv_north = vel_j[1] - vel_i[1]

    rel_spd_sq = dv_east**2 + dv_north**2
    if rel_spd_sq < 1e-12:
        return 0.0

    # r · v (negative = converging, positive = diverging)
    range_rate = d_east * dv_east + d_north * dv_north
    tcpa = -range_rate / rel_spd_sq
    if tcpa < 0 or tcpa > CONFIG['lookahead_s']:
        return 0.0

    # dcpa^2 = |r|^2 - (r·v)^2 / |v|^2
    dcpa_sq = max(0.0, dist_sq - range_rate**2 / rel_spd_sq)
    if dcpa_sq >= sep**2:
        return 0.0

    return min(1.0, max(0.0, (CONFIG['t_warn'] - tcpa) / CONFIG['t_warn']))


def _bs_state(idx):
    """(pos, vel) of BlueSky aircraft idx in NM / NM/s, east-north."""
    pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
    spd = _speed_nms(idx)
    hdg = math.radians(bs.traf.hdg[idx])
    return pos, (spd * math.sin(hdg), spd * math.cos(hdg))


def _pair_urgency(i, j):
    """Urgency between BlueSky aircraft i and j (see _urgency_from_state)."""
    pos_i, vel_i = _bs_state(i)
    pos_j, vel_j = _bs_state(j)
    return _urgency_from_state(pos_i, vel_i, pos_j, vel_j)

# -- BlueSky screen stub -------------------------------------------------------

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass

# -- Environment ---------------------------------------------------------------

class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, t_warn=None, delay='probabilistic', **delay_kwargs):
        """delay: a ResponseDelay instance, or a mode string ('deterministic' /
        'probabilistic') built via make_delay_model(delay, **delay_kwargs)."""
        super().__init__()
        global _bs_initialized, D_WARN
        if t_warn is not None:
            CONFIG['t_warn'] = float(t_warn)
            D_WARN = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0
        self._delay = make_delay_model(delay, **delay_kwargs) if isinstance(delay, str) else delay
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(len(ACT_COST))

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self.n_aircraft           = 0
        self._slots               = []
        self._active_callsigns    = set()
        self._destination_ll      = {}           # far point along the route (visualisation only)
        self._ref_ll              = {}
        self._route_hdg           = {}           # fixed route bearing (deg) per callsign
        self._effective_heading   = {}           # heading BlueSky is actually executing
        self._effective_mach      = {}           # Mach BlueSky is actually executing
        self._issued_heading      = {}           # heading the agent has commanded (pre-response)
        self._issued_mach         = {}           # Mach the agent has commanded (pre-response)
        self._pending             = {}           # cs -> queued instruction awaiting response
        self._substep             = 0            # monotonic 0.5 s sub-step clock
        self._direct_mode         = {}           # fly-direct (back-to-route) active per callsign
        self._steps_since_urgency = {}
        self._focus_hold_steps    = 0
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._max_steps           = 0
        self._focus_cs            = None
        self._pending_spawns      = {}
        self._spawn_delay_range   = (1, 1)
        self._ep_stats            = {}
        self.polygon              = None
        self._polygon_shape       = None

        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
        self._los_this_step       = False

    # -- Gym interface ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)
            # seed the delay stream off the env seed too, so parallel SubprocVecEnv
            # workers (each seeded seed+rank) draw INDEPENDENT delay sequences rather
            # than the identical default-seeded stream
            seed_delay(effective_seed + 10007)

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        n_ac     = CONFIG['n_aircraft']()
        rho      = CONFIG['rho']()
        area_km2 = float(n_ac / rho)

        poly = _make_polygon(area_km2)
        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm  = math.sqrt((maxx - minx)**2 + (maxy - miny)**2)
        step_duration_s = CONFIG['action_freq'] * CONFIG['sim_dt']
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * sector_diam_nm / CONFIG['ac_speed'] * 3600 / step_duration_s
        ))

        self.n_aircraft           = n_ac
        self._slots               = [None] * n_ac
        self._active_callsigns    = set()
        self._destination_ll      = {}           # far point along the route (visualisation only)
        self._ref_ll              = {}
        self._route_hdg           = {}           # fixed route bearing (deg) per callsign
        self._effective_heading   = {}
        self._effective_mach      = {}
        self._issued_heading      = {}
        self._issued_mach         = {}
        self._pending             = {}
        self._substep             = 0
        self._direct_mode         = {}           # fly-direct (back-to-route) active per callsign
        self._steps_since_urgency = {}
        self._focus_hold_steps    = 0
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._ep_stats            = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': [],
                                     'exits': 0, 'arrivals': 0}
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
        self._los_this_step       = False
        self._exec_this_step      = {}      # cs -> True if instruction fired this RL step
        self._focus_cs            = None   # clear stale focus: callsign IDs restart each episode

        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._flat_latlon())
        bs.stack.stack('ASAS OFF')

        for slot in range(n_ac):
            for _ in range(CONFIG['max_placement_tries']):
                ac = _place_one(self._polygon_shape, slot, n_ac)
                # spawn rule: 15 NM buffer (sep_nm + buffer_nm) to all traffic
                if self._spawn_ok(ac):
                    self._spawn_aircraft(slot, ac)
                    break
            else:
                # could not place this aircraft now: retry via the respawn queue
                self._pending_spawns[slot] = 5

        self._focus_cs = self._select_focus_aircraft()
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()
        acting_cs = self._focus_cs

        if acting_cs:
            self._apply_action(acting_cs, int(action))

        # re-aim every (already-responded) fly-direct aircraft before propagating
        self._update_direct_headings()

        # propagate, releasing each queued instruction at the sub-step its
        # response delay elapses (0.5 s resolution, decoupled from the 5 s RL step)
        self._los_this_step  = False
        self._exec_this_step = {}
        for _ in range(CONFIG['action_freq']):
            self._substep += 1
            self._release_due_commands()
            bs.sim.step()
        self._check_los_now()
        self._step_count += 1

        self._process_exits()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, int(action))

        urgency_mat = self._urgency_matrix
        n_los_pairs = int((urgency_mat > 1.0).sum()) // 2 if urgency_mat.size > 0 else 0

        self._ep_stats['reward']  += reward
        self._ep_stats['steps']   += 1
        self._ep_stats['actions'].append(int(action))
        if self._los_this_step:
            self._ep_stats['los'] += 1

        truncated = self._step_count >= self._max_steps
        info = {'los_pairs': n_los_pairs, 'focus_cs': self._focus_cs,
                'n_aircraft': self.n_aircraft}
        if truncated:
            n_steps = max(self._ep_stats['steps'], 1)
            info.update({
                'mean_episode_reward': self._ep_stats['reward'] / n_steps,
                'ep_los_steps':        self._ep_stats['los'],
                'ep_length':           self._ep_stats['steps'],
                'ep_exits':            self._ep_stats['exits'],
                'ep_arrival_rate':     self._ep_stats['arrivals']
                                       / max(self._ep_stats['exits'], 1),
                'action_distribution': np.bincount(
                    self._ep_stats['actions'], minlength=len(ACT_COST)).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # -- Aircraft-state helpers ------------------------------------------------

    def _pos_nm(self, idx):
        """Position of BlueSky aircraft idx as (east, north) NM from sector centre."""
        return latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """
        Rebuild the urgency matrix and select the focus aircraft.

        Selection: highest pair_max (worst single-pair urgency), tiebreak by total_load.
        Ties go to the current focus to prevent oscillation in symmetric conflicts.
        Hysteresis keeps the current focus while it is still active; an emergency
        (pair_max >= focus_emergency_u) overrides hysteresis and forces a switch.
        """
        active = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not active:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return None

        n_active    = len(active)
        sim_indices = [bs.traf.id2idx(cs) for cs in active]

        urgency_mat = np.zeros((n_active, n_active))
        for ii in range(n_active):
            for jj in range(ii + 1, n_active):
                u = _pair_urgency(sim_indices[ii], sim_indices[jj])
                urgency_mat[ii, jj] = urgency_mat[jj, ii] = u

        self._urgency_matrix  = urgency_mat
        self._urgency_cs_list = active

        pair_max   = urgency_mat.max(axis=1)   # worst single-pair urgency per aircraft
        total_load = urgency_mat.sum(axis=1)   # total urgency burden per aircraft

        clear_steps = CONFIG['focus_clear_steps']
        for i, cs in enumerate(active):
            if pair_max[i] > 0:
                self._steps_since_urgency[cs] = 0
            else:
                self._steps_since_urgency[cs] = self._steps_since_urgency.get(cs, clear_steps) + 1

        # Current-focus state (computed once, used in both candidate selection and hysteresis)
        focus_idx      = active.index(self._focus_cs) if self._focus_cs in active else -1
        focus_pm       = pair_max[focus_idx] if focus_idx >= 0 else 0.0
        focus_resolved = self._steps_since_urgency.get(self._focus_cs, clear_steps) >= clear_steps
        emergency      = pair_max.max() >= CONFIG['focus_emergency_u']
        drift_locked   = focus_pm == 0 and self._focus_hold_steps < clear_steps

        # Candidate: highest pair_max, tiebreak by total_load; ties go to current focus
        if pair_max.max() > 0:
            tied = np.where(pair_max >= pair_max.max() - 1e-9)[0]
            if focus_idx >= 0 and np.any(tied == focus_idx):
                best_cs = self._focus_cs
            else:
                best_cs = active[tied[int(np.argmax(total_load[tied]))]]
        else:
            best_cs = self._drift_fallback(active)

        # Hysteresis: keep current focus while it is still active,
        # unless it is fully resolved or an emergency forces a switch
        if focus_idx >= 0 and best_cs != self._focus_cs:
            keep_focus = (focus_pm > 0 or not focus_resolved or drift_locked) and not emergency
            if keep_focus:
                best_cs = self._focus_cs

        if best_cs != self._focus_cs:
            self._focus_hold_steps = 0
        else:
            self._focus_hold_steps += 1
        return best_cs

    def _drift_fallback(self, active):
        """
        Return the drifted aircraft best placed to be sent back to its route.

        Each aircraft is scored as drift x clearance, where clearance ramps
        from 0 to 1 as the nearest-neighbour distance approaches
        return_clear_nm.  Drifted aircraft in open airspace are prioritised:
        they can turn back to their waypoint without creating a new conflict.
        A hysteresis margin prevents rapid focus switching.
        """
        positions = {}
        for cs in active:
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                positions[cs] = self._pos_nm(idx)

        clear_nm    = CONFIG['return_clear_nm']
        best_cs     = None
        best_score  = -1.0
        focus_score = 0.0

        for cs in sorted(active):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._route_hdg:
                continue
            route_hdg = self._route_hdg[cs]
            hdg_err = wrap_to_180(route_hdg - self._effective_heading.get(cs, bs.traf.hdg[idx]))
            drift   = (1 - math.cos(math.radians(hdg_err))) / 2

            own     = positions[cs]
            nearest = min((float(np.hypot(*(positions[o] - own)))
                           for o in positions if o != cs), default=clear_nm)
            score   = drift * min(1.0, nearest / clear_nm)

            if cs == self._focus_cs:
                focus_score = score
            if score > best_score:
                best_score, best_cs = score, cs

        margin = CONFIG['drift_switch_margin']
        if (self._focus_cs in active
                and best_cs != self._focus_cs
                and best_score <= focus_score + margin):
            return self._focus_cs
        return best_cs

    # -- Reward ----------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        # LoS: heavy binary penalty when separation is violated this step
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        # Conflict: worst predicted conflict score across all intruders
        r_conflict = -CONFIG['w_conflict'] * self._conflict_score(acting_cs)

        # Drift: per-step cost for holding the aircraft off its planned route
        r_drift = 0.0
        if acting_cs and acting_cs in self._route_hdg:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                cmd_hdg = self._effective_heading.get(acting_cs, bs.traf.hdg[idx])
                hdg_err = wrap_to_180(self._route_hdg[acting_cs] - cmd_hdg)
                # Classic cosine drift penalty: linear in (1-cos)/2.
                drift_frac = (1.0 - math.cos(math.radians(hdg_err))) / 2.0
                r_drift    = -CONFIG['w_drift'] * drift_frac

        # Workload: one-time cost per instruction. Hold (3) is free (ACT_COST 0.0);
        # fly-direct (7) carries a very low cost; turns scale with |delta|.
        r_work = 0.0
        if acting_cs:
            r_work = -CONFIG['w_work'] * ACT_COST[action_idx]

        return float(r_los + r_conflict + r_drift + r_work)

    def _conflict_score(self, cs, hdg_deg=None, infinite_horizon=False):
        """
        Returns max over all intruders of:
          (1 - tcpa/t_warn) * (1 - dcpa/sep)
        gated by dcpa < sep (only true collision courses score).
        Active LoS contributes the maximum score of 1.

        hdg_deg overrides the ownship heading used for the relative-velocity / CPA
        prediction. Pass the route heading to ask "would flying direct now create a
        conflict?"; default (None) uses the aircraft's current heading.

        infinite_horizon=True drops the time horizon entirely: any converging pair
        whose predicted CPA is inside separation scores by miss distance alone
        (1 - dcpa/sep), with no t_warn time-decay and no lookahead gate. Used for the
        "safe to return" signal so a conflict beyond the warning horizon is still seen.
        """
        if cs is None:
            return 0.0
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0

        own_hdg  = bs.traf.hdg[idx] if hdg_deg is None else hdg_deg
        pos_own  = self._pos_nm(idx)
        own_spd  = _speed_nms(idx)
        own_ve   = own_spd * math.sin(math.radians(own_hdg))
        own_vn   = own_spd * math.cos(math.radians(own_hdg))
        sep      = CONFIG['sep_nm']
        t_warn   = CONFIG['t_warn']

        worst_score = 0.0
        for other_cs in self._active_callsigns:
            int_idx = bs.traf.id2idx(other_cs)
            if other_cs == cs or int_idx < 0:
                continue

            pos_int  = self._pos_nm(int_idx)
            d_east   = pos_int[0] - pos_own[0]
            d_north  = pos_int[1] - pos_own[1]
            dist_nm  = math.sqrt(d_east**2 + d_north**2)

            if dist_nm < sep:
                worst_score = 1.0   # active LoS → maximum score; keep checking for more
                continue

            int_spd  = _speed_nms(int_idx)
            int_ve   = int_spd * math.sin(math.radians(bs.traf.hdg[int_idx]))
            int_vn   = int_spd * math.cos(math.radians(bs.traf.hdg[int_idx]))
            dv_east  = int_ve - own_ve
            dv_north = int_vn - own_vn

            rel_spd_sq = dv_east**2 + dv_north**2
            if rel_spd_sq < 1e-12:
                continue

            range_rate = d_east * dv_east + d_north * dv_north   # r·v; negative = converging
            tcpa       = -range_rate / rel_spd_sq
            if tcpa < 0:
                continue                                          # diverging: no future conflict
            if not infinite_horizon and tcpa > CONFIG['lookahead_s']:
                continue

            cpa_east  = d_east  + tcpa * dv_east
            cpa_north = d_north + tcpa * dv_north
            dcpa_sq   = cpa_east**2 + cpa_north**2
            if dcpa_sq >= sep**2:
                continue   # miss distance too large, not a conflict

            dcpa = math.sqrt(dcpa_sq)
            if infinite_horizon:
                score = max(0.0, 1.0 - dcpa / sep)                            # miss distance only
            else:
                score = max(0.0, 1.0 - tcpa / t_warn) * max(0.0, 1.0 - dcpa / sep)
            worst_score = max(worst_score, score)

        return worst_score

    def _check_los_now(self):
        """Set _los_this_step if any pair of active aircraft is within sep_nm."""
        active  = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        sep_sq  = CONFIG['sep_nm'] ** 2

        for ii in range(len(active)):
            idx_i   = bs.traf.id2idx(active[ii])
            pos_i   = self._pos_nm(idx_i)
            for jj in range(ii + 1, len(active)):
                idx_j   = bs.traf.id2idx(active[jj])
                pos_j   = self._pos_nm(idx_j)
                d_east  = pos_j[0] - pos_i[0]
                d_north = pos_j[1] - pos_i[1]
                if d_east**2 + d_north**2 < sep_sq:
                    self._los_this_step = True
                    return

    # -- Action ----------------------------------------------------------------

    def _apply_action(self, cs, action_idx):
        """Issue an instruction: update the ISSUED command immediately and queue the
        actual simulator command behind a sampled response delay (no command reaches
        BlueSky here -- see _release_due_commands). Hold issues nothing."""
        if action_idx == 3:
            return   # hold is a true no-op: no instruction is issued, nothing is queued

        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        # a superseding instruction (issued while one is still pending) draws the
        # shorter "subsequent" delay, mirroring the paper's 5 s -> 3 s
        subsequent = cs in self._pending

        # Speed instruction: step the ISSUED Mach within the ATC envelope.
        if action_idx in SPEED_ACTIONS:
            mach = self._issued_mach.get(cs, self._effective_mach.get(cs, CONFIG['ac_mach']))
            mach += SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            mach = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))
            self._issued_mach[cs] = mach
            self._queue(cs, 'spd', mach=mach, subsequent=subsequent)
            return

        # Heading instruction: stack the delta onto the ISSUED heading.
        if action_idx in TURN_DELTAS:
            current = self._issued_heading.get(cs, self._effective_heading.get(cs, bs.traf.hdg[idx]))
            self._issued_heading[cs] = (current + TURN_DELTAS[action_idx]) % 360
            self._queue(cs, 'hdg', hdg=self._issued_heading[cs], direct=False, subsequent=subsequent)
        elif action_idx == 7:
            # fly direct (back to route): issue the fixed route heading; direct tracking
            # turns on only once the pilot responds (handled in _release_due_commands).
            self._issued_heading[cs] = self._route_hdg[cs]
            self._queue(cs, 'hdg', hdg=self._route_hdg[cs], direct=True, subsequent=subsequent)

    def _queue(self, cs, typ, hdg=None, mach=None, direct=False, subsequent=False):
        """Replace any pending instruction for cs with a fresh one, due after a delay
        sampled from the delay model."""
        delay_s = self._delay.sample_delay(subsequent)
        n_sub   = int(round(delay_s / CONFIG['sim_dt']))
        self._pending[cs] = {'release': self._substep + n_sub, 'issue_sub': self._substep,
                             'delay_s': delay_s, 'type': typ,
                             'hdg': hdg, 'mach': mach, 'direct': direct}

    def _release_due_commands(self):
        """Execute any pending instruction whose response delay has elapsed: set the
        EFFECTIVE command and stack the actual BlueSky command (the pilot responds)."""
        for cs, p in list(self._pending.items()):
            if p['release'] > self._substep:
                continue
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                if p['type'] == 'spd':
                    self._effective_mach[cs] = p['mach']
                    bs.stack.stack(f"SPD {cs} {p['mach']:.3f}")
                else:
                    self._effective_heading[cs] = p['hdg']
                    self._direct_mode[cs]       = p['direct']
                    bs.stack.stack(f"HDG {cs} {p['hdg']:.1f}")
            self._exec_this_step[cs] = True
            del self._pending[cs]

    def _update_direct_headings(self):
        """Re-hold every (already-responded) fly-direct aircraft on its route heading.

        Skips aircraft with a still-pending instruction (their fly-direct hasn't been
        executed yet). The route heading is constant, so this just maintains the hold.
        """
        for cs, on in self._direct_mode.items():
            if not on or cs in self._pending:
                continue
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._route_hdg:
                continue
            self._effective_heading[cs] = self._route_hdg[cs]
            bs.stack.stack(f'HDG {cs} {self._effective_heading[cs]:.1f}')

    # -- Observation -----------------------------------------------------------

    # ACAS Xu empty-slot sentinel (normalised): rho_n=1 (far), tau_n=1 (no imminent LoS)
    _EMPTY_SLOT = [1.0, 0.0, 0.0, 0.0, 1.0]

    def _get_observation(self):
        """ACAS Xu states for the focus aircraft (ownship) against its 4
        nearest/most-urgent intruders. Angles in radians, other states scaled by
        physical range; VecNormalize standardises everything. See module docstring."""
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            # no controllable aircraft: on-route (dpsi=0), nominal speed, conflict-free,
            # no pending advisory (iss_dhdg=0, resp_pend=0)
            return np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                            + self._EMPTY_SLOT * N_NBR, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._effective_heading.get(cs, own_hdg)
        own_pos = self._pos_nm(idx)

        # heading-frame basis vectors (forward = own_hdg, lateral = right of own_hdg)
        sin_hdg = math.sin(math.radians(own_hdg))
        cos_hdg = math.cos(math.radians(own_hdg))

        own_spd = _speed_nms(idx)
        own_ve  = own_spd * math.sin(math.radians(own_hdg))
        own_vn  = own_spd * math.cos(math.radians(own_hdg))

        # heading error to route (commanded heading vs fixed route heading); matches the
        # drift the reward penalises. sin/cos avoids the +-180 wrap jump.
        route_hdg = self._route_hdg[cs]
        dpsi_act = math.radians(wrap_to_180(own_hdg - route_hdg))   # actual heading deviation, rad
        a_cmd    = math.radians(wrap_to_180(cmd_hdg - route_hdg))   # commanded deviation; persists through hold

        # conflict severity if returning to route (flying the route heading).
        # Uses an INFINITE horizon (miss-distance only) so a conflict beyond t_warn still flags unsafe.
        conflict_return = self._conflict_score(cs, route_hdg, infinite_horizon=True)

        # response-delay features: pending turn angle + binary execution flag
        issued_hdg = self._issued_heading.get(cs, cmd_hdg)
        iss_dhdg   = math.radians(wrap_to_180(issued_hdg - cmd_hdg))
        b_exec     = float(self._exec_this_step.get(cs, False))

        # ownship-global states (shared across intruder slots)
        obs = [dpsi_act,                              # actual heading deviation from route, rad
               own_spd / V_NOM,                       # v_own normalised by nominal cruise
               a_cmd,                                 # commanded heading deviation from route, rad
               conflict_return,                       # conflict if returning to route (0 = safe)
               iss_dhdg,                              # pending turn angle: issued - effective, rad
               b_exec]                                # 1 if instruction executed this step, else 0

        # fetch this aircraft's urgency row for intruder prioritisation
        urgency_row = None
        if cs in self._urgency_cs_list:
            focus_row_idx = self._urgency_cs_list.index(cs)
            if focus_row_idx < self._urgency_matrix.shape[0]:
                urgency_row = self._urgency_matrix[focus_row_idx]

        sep    = CONFIG['sep_nm']
        t_warn = CONFIG['t_warn']

        intruders = []
        for other_cs in self._active_callsigns:
            int_idx = bs.traf.id2idx(other_cs)
            if other_cs == cs or int_idx < 0:
                continue

            int_pos  = self._pos_nm(int_idx)
            d_east   = int_pos[0] - own_pos[0]
            d_north  = int_pos[1] - own_pos[1]
            dist_nm  = math.sqrt(d_east**2 + d_north**2)

            int_hdg  = bs.traf.hdg[int_idx]
            int_spd  = _speed_nms(int_idx)

            # project relative position into ego heading frame (0 = dead ahead, + = right)
            ego_lat = d_east * cos_hdg - d_north * sin_hdg
            ego_fwd = d_east * sin_hdg + d_north * cos_hdg

            # ACAS Xu states; angles in radians, others scaled by physical range:
            rho   = min(1.0, dist_nm / D_WARN)                         # [0] distance / warning horizon
            theta = math.atan2(ego_lat, ego_fwd)                      # [1] bearing to intruder, rad (0 = ahead)
            psi   = math.radians(wrap_to_180(int_hdg - own_hdg))      # [2] intruder hdg rel. ownship, rad
            v_int = int_spd / V_NOM                                    # [3] intruder speed / nominal cruise

            # tau: horizontal time-to-loss-of-separation / t_warn. 0 if already inside
            # sep, time to CPA when converging, 1 (=t_warn cap) when diverging.
            if dist_nm < sep:
                tau = 0.0
            else:
                dv_east    = int_spd * math.sin(math.radians(int_hdg)) - own_ve
                dv_north   = int_spd * math.cos(math.radians(int_hdg)) - own_vn
                rel_spd_sq = dv_east**2 + dv_north**2
                range_rate = d_east * dv_east + d_north * dv_north     # negative = converging
                tcpa       = (-range_rate / rel_spd_sq) if rel_spd_sq > 1e-12 else -1.0
                tau        = min(tcpa / t_warn, 1.0) if tcpa > 0 else 1.0

            # urgency for this pair from the pre-computed matrix
            urgency = 0.0
            if urgency_row is not None and other_cs in self._urgency_cs_list:
                other_row_idx = self._urgency_cs_list.index(other_cs)
                if other_row_idx < len(urgency_row):
                    urgency = float(urgency_row[other_row_idx])

            intruders.append((urgency, dist_nm, [rho, theta, psi, v_int, tau], other_cs))

        # fill slots: urgent pairs first (descending urgency), then nearest (ascending distance)
        urgent  = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0])
        nearest = sorted(intruders, key=lambda r: r[1])

        selected, seen = [], set()
        for rec in urgent + nearest:
            if len(selected) >= N_NBR:
                break
            if rec[3] not in seen:
                selected.append(rec)
                seen.add(rec[3])

        for slot_k in range(N_NBR):
            obs += selected[slot_k][2] if slot_k < len(selected) else self._EMPTY_SLOT

        return np.array(obs, dtype=np.float32)

    # -- Exits -----------------------------------------------------------------

    def _process_exits(self):
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                # on-target arrival: exit position within tolerance of the t_ref point
                ref_ll = self._ref_ll.get(cs)
                if ref_ll is not None:
                    dist_nm = geo.kwikdist(
                        float(bs.traf.lat[idx]), float(bs.traf.lon[idx]),
                        float(ref_ll[0]), float(ref_ll[1]))
                    self._ep_stats['exits'] += 1
                    if dist_nm <= CONFIG['arrival_tol_nm']:
                        self._ep_stats['arrivals'] += 1
                bs.traf.delete(idx)
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._destination_ll.pop(cs, None)
            self._ref_ll.pop(cs, None)
            self._route_hdg.pop(cs, None)
            self._effective_heading.pop(cs, None)
            self._effective_mach.pop(cs, None)
            self._issued_heading.pop(cs, None)
            self._issued_mach.pop(cs, None)
            self._pending.pop(cs, None)
            self._direct_mode.pop(cs, None)
            self._steps_since_urgency.pop(cs, None)
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = random.randint(*self._spawn_delay_range)

    def _find_exited(self):
        """Return callsigns that have left the sector or are no longer in BlueSky."""
        exited = []
        for cs in list(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs)
                continue
            inside = bs.tools.areafilter.checkInside(
                'SECTOR',
                np.array([bs.traf.lat[idx]]),
                np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]),
            )
            if not inside[0]:
                exited.append(cs)
        return exited

    # -- Spawning --------------------------------------------------------------

    def _process_pending_spawns(self):
        """Spawn aircraft whose countdown has reached 1, decrement all others."""
        ready    = sorted(slot for slot, t in list(self._pending_spawns.items()) if t <= 1)
        requeued = set()

        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
            else:
                self._pending_spawns[slot] = 5   # retry in 5 steps
                requeued.add(slot)

        for slot in list(self._pending_spawns):
            if slot not in requeued:
                self._pending_spawns[slot] -= 1

    def _spawn_ok(self, ac):
        """Spawn admission: geometric buffer to all active traffic."""
        pos_c = latlon_to_nm(CONFIG['center_ll'],
                             float(ac['sp_ll'][0]), float(ac['sp_ll'][1]))
        min_spawn_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        for cs in self._active_callsigns:
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue
            pos_o, _ = _bs_state(idx)
            if math.hypot(pos_c[0] - pos_o[0], pos_c[1] - pos_o[1]) < min_spawn_sep:
                return False
        return True

    def _generate_replacement(self, slot):
        """Try to place a new aircraft that clears the spawn buffer to all traffic."""
        n_ac = self.n_aircraft
        for _ in range(CONFIG['max_placement_tries']):
            ac = _place_one(self._polygon_shape, random.randint(0, n_ac - 1), n_ac)
            if self._spawn_ok(ac):
                return ac
        return None

    def _spawn_aircraft(self, slot, ac):
        cs   = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        mach = CONFIG['ac_mach']

        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['sp_ll'][0]), aclon=float(ac['sp_ll'][1]),
                    achdg=float(ac['heading']), acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')

        self._destination_ll[cs]      = ac['dest_ll']
        self._ref_ll[cs]              = ac['ref_ll']
        self._route_hdg[cs]           = float(ac['heading'])
        self._effective_heading[cs]   = float(ac['heading'])
        self._effective_mach[cs]      = CONFIG['ac_mach']
        self._issued_heading[cs]      = float(ac['heading'])
        self._issued_mach[cs]         = CONFIG['ac_mach']
        self._direct_mode[cs]         = False
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    # -- Misc ------------------------------------------------------------------

    def _flat_latlon(self):
        """Flatten polygon vertices to [lat, lon, lat, lon, ...] for BlueSky POLY."""
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
