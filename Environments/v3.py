"""
v3 -- ATCO conflict-resolution environment.

Reward is purely negative (no positive components):
  -w_los      x 1[LoS]                                      heavy:  separation violation during step
  -w_conflict x (1-dcpa/sep) x (1-tcpa/t_warn)             medium: linear DCPA*TCPA conflict score
  -w_drift    x (1-cos(psi_dest-psi_cmd))/2                 medium: commanded heading deviation from route
  -w_work     x act_cost                                    small:  instruction workload
    turn 0/6 (+-60 deg): cost 1.0    turn 1/5 (+-45 deg): cost 0.75   turn 2/4 (+-30 deg): cost 0.5
    hold (3):             cost 0.0   fly-direct (7): cost |delta_psi_cmd->dest| / 60 deg

Observation space (23 floats, ego-centric from focus aircraft):
  [0]   sin(delta_psi_dest)   heading error to destination   [-1, 1]
  [1]   cos(delta_psi_dest)                                   [-1, 1]
  [2]   turn_progress  (cmd_hdg - actual) / 180              [-1, 1]
  [3:23] 4 intruders x 5 (urgency-desc, dist-asc), local polar:
           r_norm   distance to intruder / D_WARN             [0,  1]
           theta    relative bearing / pi                     [-1, 1]
           cpa_r    CPA distance / sep                        [0,  1]
           cpa_th   CPA bearing / pi                          [-1, 1]
           tcpa_n   time to CPA / t_warn                      [0,  1]
         empty/diverging slot: (1, 0, 0, 0, 1)

Action space (Discrete 8) -- ATC-styled hybrid, consecutive:
  0  delta=-60 deg   psi_cmd -= 60   (cost 1.0)   stacks on current commanded heading
  1  delta=-45 deg   psi_cmd -= 45   (cost 0.75)  stacks on current commanded heading
  2  delta=-30 deg   psi_cmd -= 30   (cost 0.5)   stacks on current commanded heading
  3  HOLD             true no-op: no instruction is issued   (free)
  4  delta=+30 deg   psi_cmd += 30   (cost 0.5)   stacks on current commanded heading
  5  delta=+45 deg   psi_cmd += 45   (cost 0.75)  stacks on current commanded heading
  6  delta=+60 deg   psi_cmd += 60   (cost 1.0)   stacks on current commanded heading
  7  FLY DIRECT       psi_cmd = psi_dest         (cost |delta_psi_cmd->dest| / 60, 0-1.0)

Intended pattern: issue turn(s) (0-2 or 4-6, stackable) -> hold (3) -> fly direct (7).
Hold is a true no-op (nothing is sent to the simulator); the aircraft simply continues
executing its last instruction.  Turns stack on the current commanded heading.
Fly-direct cost scales with the commanded deviation being corrected: free when already on track,
up to 1.0 x w_work when correcting a full 60 deg deviation.

Focus aircraft = aircraft with highest single-pair urgency (total urgency burden as tiebreak).
When the sector is conflict-free, focus falls back to the drifted aircraft with the best
drift x clearance score, prioritising drifters in open airspace that can safely return.
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

# -- Configuration -------------------------------------------------------------

CONFIG = {
    # Aircraft & sector
    'ac_type':               'A320',
    'ac_speed':              450.0,
    'ac_mach':               0.78,
    'ac_mach_min':           0.70,
    'ac_mach_max':           0.82,
    'altitude':              350,
    'center_ll':             (0.0, 0.0),      # flat-earth equatorial: cos(0)=1
    'n_aircraft':            lambda: random.randint(10, 15),
    'rho':                   lambda: random.uniform(1/15000, 1/5000),  # aircraft/km^2; area = n/rho
    'sep_nm':                5.0,
    'buffer_nm':             10.0,
    'dest_dist_factor':      2.0,
    'arrival_tol_nm':        5.0,             # exit within this distance of t_ref counts as on-target
    # Polygon
    'n_vertices':            lambda: random.randint(5, 7),
    'min_circularity':       0.65,
    'max_placement_tries':   50,
    # Aircraft placement jitter
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.30, 0.30),
    # Simulation
    'sim_dt':                0.5,
    'action_freq':           10,              # RL step = 5 s simulated
    'lookahead_s':           900.0,
    't_warn':                600.0,
    'crossings_per_episode': 4.0,
    'spawn_delay_s':         (0, 0),
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.8,
    'drift_switch_margin':   0.05,
    'return_clear_nm':       20.0,            # full clearance distance for the drift fallback score (4 x sep)
    # Reward weights
    'w_los':                 10.00,           # heavy: separation violation
    'w_conflict':            3.00,            # medium: imminence x miss-distance of worst conflict
    'w_drift':               0.60,            # accumulates during hold, motivates return to track
    'w_work':                1.00,            # charged once per turn instruction; hold and direct are free
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR   = CONFIG['n_neighbours']
OBS_DIM = 3 + N_NBR * 5       # 3 ownship + 20 intruder = 23

D_WARN = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0  # warning horizon distance (75 NM)
V_NOM  = CONFIG['ac_speed'] / 3600.0                      # nominal cruise speed (NM/s)

# Heading offsets per turn action (actions 0-2 and 4-6); 3=hold, 7=fly-direct
TURN_DELTAS = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}

# Workload cost = |delta| / 60 deg, so equal total deviation = equal total cost.
# 2x(30 deg) = 1.0 = 1x(60 deg).  Index 7 (fly-direct) is computed dynamically.
ACT_COST = [1.0, 0.75, 0.5, 0.0, 0.5, 0.75, 1.0, None]

_bs_initialized = False

__all__ = ['AirspaceEnv', 'CONFIG', 'NM_TO_KM', 'latlon_to_nm', 'wrap_to_180', 'OBS_DIM']

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
    The spawn point and reference point (which determines heading) are chosen
    at evenly spaced arcs with random jitter.
    """
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx - minx)**2 + (maxy - miny)**2) * CONFIG['dest_dist_factor']

    t_spawn   = (sector + CONFIG['spawn_jitter']()) / n_sectors
    t_ref     = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
    spawn_pt  = polygon.exterior.interpolate(t_spawn, normalized=True)
    ref_pt    = polygon.exterior.interpolate(t_ref,   normalized=True)

    spawn_ll  = nm_to_latlon(CONFIG['center_ll'], spawn_pt.x, spawn_pt.y)
    ref_ll    = nm_to_latlon(CONFIG['center_ll'], ref_pt.x,   ref_pt.y)
    spawn_hdg, _ = geo.kwikqdrdist(*[float(v) for v in (*spawn_ll, *ref_ll)])

    dest_lat, dest_lon = geo.qdrpos(
        float(spawn_ll[0]), float(spawn_ll[1]), spawn_hdg, dest_dist)

    return {'sp_ll': spawn_ll, 'dest_ll': (dest_lat, dest_lon),
            'ref_ll': ref_ll, 'heading': spawn_hdg}

# -- Pair urgency --------------------------------------------------------------

def _pair_urgency(i, j):
    """
    Urgency of the conflict between aircraft i and j (BlueSky indices).

    Returns:
      > 1   active LoS    (scales 1 at sep boundary → 10 at d=0)
      0..1  predicted LoS (scales 0 at t_warn → 1 at t_CPA=0)
      0     safe / diverging
    """
    pos_i   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    pos_j   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    d_east  = pos_j[0] - pos_i[0]
    d_north = pos_j[1] - pos_i[1]
    dist_sq = d_east**2 + d_north**2
    sep     = CONFIG['sep_nm']

    if dist_sq < sep**2:
        return 1.0 + 9.0 * (1.0 - math.sqrt(dist_sq) / sep)

    spd_i   = _speed_nms(i)
    spd_j   = _speed_nms(j)
    ve_i    = spd_i * math.sin(math.radians(bs.traf.hdg[i]))
    vn_i    = spd_i * math.cos(math.radians(bs.traf.hdg[i]))
    ve_j    = spd_j * math.sin(math.radians(bs.traf.hdg[j]))
    vn_j    = spd_j * math.cos(math.radians(bs.traf.hdg[j]))
    dv_east  = ve_j - ve_i
    dv_north = vn_j - vn_i

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

# -- BlueSky screen stub -------------------------------------------------------

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass

# -- Environment ---------------------------------------------------------------

class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(8)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self.n_aircraft           = 0
        self._slots               = []
        self._active_callsigns    = set()
        self._destination_ll      = {}
        self._ref_ll              = {}
        self._commanded_heading   = {}
        self._steps_since_urgency = {}
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._max_steps           = 0
        self._focus_cs            = None
        self._focus_hold_steps    = 0
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
        self._destination_ll      = {}
        self._ref_ll              = {}
        self._commanded_heading   = {}
        self._steps_since_urgency = {}
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._ep_stats            = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': [],
                                     'exits': 0, 'arrivals': 0}
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
        self._los_this_step       = False
        self._focus_cs            = None   # clear stale focus: callsign IDs restart each episode
        self._focus_hold_steps    = 0

        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._flat_latlon())
        bs.stack.stack('ASAS OFF')

        min_spawn_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        for slot in range(n_ac):
            for _ in range(CONFIG['max_placement_tries']):
                ac = _place_one(self._polygon_shape, slot, n_ac)
                occupied = [
                    (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
                    for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
                ]
                clear_of_all = all(
                    geo.kwikdist(float(ac['sp_ll'][0]), float(ac['sp_ll'][1]),
                                 float(lat), float(lon)) >= min_spawn_sep
                    for lat, lon in occupied
                )
                if clear_of_all:
                    self._spawn_aircraft(slot, ac)
                    break
            else:
                # could not place this aircraft now: retry via the respawn queue
                self._pending_spawns[slot] = 5

        self._focus_cs = self._select_focus_aircraft()
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()
        acting_cs   = self._focus_cs
        pre_cmd_hdg = self._commanded_heading.get(acting_cs) if acting_cs else None

        if acting_cs:
            self._apply_action(acting_cs, int(action))

        self._los_this_step = False
        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        self._check_los_now()
        self._step_count += 1

        self._process_exits()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, int(action), pre_cmd_hdg)

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
                    self._ep_stats['actions'], minlength=8).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """
        Rebuild the urgency matrix, then select the aircraft with the highest
        total urgency burden (sum across all pairs) as the focus aircraft.
        Applies a hysteresis lock to avoid premature switching mid-resolution.
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

        # primary: highest single-pair urgency; tiebreak: highest total load
        if pair_max.max() > 0:
            tied = np.where(pair_max == pair_max.max())[0]
            winner = tied[int(np.argmax(total_load[tied]))]
            best_cs = active[winner]
        else:
            best_cs = self._drift_fallback(active)

        # hysteresis: keep current focus unless it is resolved or an emergency arises
        if self._focus_cs in active and best_cs != self._focus_cs:
            focus_pair_max = pair_max[active.index(self._focus_cs)]
            focus_resolved = (self._steps_since_urgency.get(self._focus_cs, clear_steps)
                              >= clear_steps)
            emergency      = pair_max.max() >= CONFIG['focus_emergency_u']
            # drift case: also require minimum hold time before switching
            drift_locked   = (focus_pair_max == 0 and
                              self._focus_hold_steps < clear_steps)
            if (focus_pair_max > 0 or not focus_resolved or drift_locked) and not emergency:
                self._focus_hold_steps += 1
                return self._focus_cs

        # switching to a new aircraft: reset hold counter
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
                positions[cs] = latlon_to_nm(
                    CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])

        clear_nm    = CONFIG['return_clear_nm']
        best_cs     = None
        best_score  = -1.0
        focus_score = 0.0

        for cs in sorted(active):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._destination_ll:
                continue
            dest_bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            hdg_err = wrap_to_180(dest_bearing - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
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

    def _compute_reward(self, acting_cs, action_idx, pre_cmd_hdg=None):
        # LoS: heavy binary penalty when separation is violated this step
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        # Conflict: worst predicted conflict score across all intruders
        r_conflict = -CONFIG['w_conflict'] * self._conflict_score(acting_cs)

        # Drift: per-step cost for holding the aircraft off its planned route
        r_drift = 0.0
        if acting_cs and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                dest_bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[acting_cs]])
                cmd_hdg = self._commanded_heading.get(acting_cs, bs.traf.hdg[idx])
                hdg_err = wrap_to_180(dest_bearing - cmd_hdg)
                r_drift = -CONFIG['w_drift'] * (1.0 - math.cos(math.radians(hdg_err))) / 2.0

        # Workload: one-time cost per instruction (turns: fixed; fly-direct: scales with deviation)
        r_work = 0.0
        if acting_cs:
            if action_idx == 7 and pre_cmd_hdg is not None:
                idx = bs.traf.id2idx(acting_cs)
                if idx >= 0 and acting_cs in self._destination_ll:
                    dest_bearing, _ = geo.kwikqdrdist(
                        bs.traf.lat[idx], bs.traf.lon[idx],
                        *[float(v) for v in self._destination_ll[acting_cs]])
                    deviation = abs(wrap_to_180(float(dest_bearing) - pre_cmd_hdg))
                    r_work = -CONFIG['w_work'] * min(deviation / 60.0, 1.0)
            elif action_idx != 7:   # guard: ACT_COST[7] is None (computed dynamically above)
                r_work = -CONFIG['w_work'] * ACT_COST[action_idx]

        return float(r_los + r_conflict + r_drift + r_work)

    def _conflict_score(self, cs):
        """
        Returns max over all intruders of:
          (1 - tcpa/t_warn) * (1 - dcpa/sep)
        gated by dcpa < sep (only true collision courses score).
        Active LoS contributes the maximum score of 1.
        """
        if cs is None:
            return 0.0
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0

        pos_own  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        own_spd  = _speed_nms(idx)
        own_ve   = own_spd * math.sin(math.radians(bs.traf.hdg[idx]))
        own_vn   = own_spd * math.cos(math.radians(bs.traf.hdg[idx]))
        sep      = CONFIG['sep_nm']
        t_warn   = CONFIG['t_warn']

        worst_score = 0.0
        for other_cs in self._active_callsigns:
            int_idx = bs.traf.id2idx(other_cs)
            if other_cs == cs or int_idx < 0:
                continue

            pos_int  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
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
            if tcpa < 0 or tcpa > CONFIG['lookahead_s']:
                continue

            cpa_east  = d_east  + tcpa * dv_east
            cpa_north = d_north + tcpa * dv_north
            dcpa_sq   = cpa_east**2 + cpa_north**2
            if dcpa_sq >= sep**2:
                continue   # miss distance too large, not a conflict

            dcpa  = math.sqrt(dcpa_sq)
            score = max(0.0, 1.0 - tcpa / t_warn) * max(0.0, 1.0 - dcpa / sep)
            worst_score = max(worst_score, score)

        return worst_score

    def _check_los_now(self):
        """Set _los_this_step if any pair of active aircraft is within sep_nm."""
        active  = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        sep_sq  = CONFIG['sep_nm'] ** 2

        for ii in range(len(active)):
            idx_i   = bs.traf.id2idx(active[ii])
            pos_i   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx_i], bs.traf.lon[idx_i])
            for jj in range(ii + 1, len(active)):
                idx_j   = bs.traf.id2idx(active[jj])
                pos_j   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx_j], bs.traf.lon[idx_j])
                d_east  = pos_j[0] - pos_i[0]
                d_north = pos_j[1] - pos_i[1]
                if d_east**2 + d_north**2 < sep_sq:
                    self._los_this_step = True
                    return

    # -- Action ----------------------------------------------------------------

    def _apply_action(self, cs, action_idx):
        if action_idx == 3:
            return   # hold is a true no-op: no instruction reaches the simulator

        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        if action_idx in TURN_DELTAS:
            # consecutive: stack delta onto current commanded heading
            current_cmd = self._commanded_heading.get(cs, bs.traf.hdg[idx])
            self._commanded_heading[cs] = (current_cmd + TURN_DELTAS[action_idx]) % 360
        elif action_idx == 7:
            # fly direct: reset commanded heading to current bearing to destination
            dest_bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            self._commanded_heading[cs] = float(dest_bearing) % 360

        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # -- Observation -----------------------------------------------------------

    def _get_observation(self):
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)
        own_pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])

        # heading-frame basis vectors (forward = own_hdg, lateral = right of own_hdg)
        sin_hdg = math.sin(math.radians(own_hdg))
        cos_hdg = math.cos(math.radians(own_hdg))

        own_spd  = _speed_nms(idx)
        own_ve   = own_spd * math.sin(math.radians(own_hdg))
        own_vn   = own_spd * math.cos(math.radians(own_hdg))

        dest_bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        hdg_err      = wrap_to_180(dest_bearing - cmd_hdg)
        turn_progress = wrap_to_180(cmd_hdg - own_hdg) / 180.0   # [-1, 1]

        obs = [
            math.sin(math.radians(hdg_err)),   # sin/cos encoding avoids +-180 discontinuity
            math.cos(math.radians(hdg_err)),
            turn_progress,
        ]

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

            int_pos  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            d_east   = int_pos[0] - own_pos[0]
            d_north  = int_pos[1] - own_pos[1]
            dist_nm  = math.sqrt(d_east**2 + d_north**2)

            int_spd  = _speed_nms(int_idx)
            int_ve   = int_spd * math.sin(math.radians(bs.traf.hdg[int_idx]))
            int_vn   = int_spd * math.cos(math.radians(bs.traf.hdg[int_idx]))
            dv_east  = int_ve - own_ve
            dv_north = int_vn - own_vn

            # project relative position into ego heading frame
            ego_lat = d_east * cos_hdg - d_north * sin_hdg   # lateral (right)
            ego_fwd = d_east * sin_hdg + d_north * cos_hdg   # forward

            r_norm     = min(1.0, dist_nm / D_WARN)
            theta_norm = math.atan2(ego_lat, ego_fwd) / math.pi   # 0=ahead, +-1=behind

            # CPA polar features — filled only when dcpa < sep (true conflict course)
            if dist_nm < sep:
                # active LoS: report current position as the CPA
                tcpa_n      = 0.0
                cpa_r_n     = min(1.0, dist_nm / sep)
                cpa_theta_n = math.atan2(ego_lat, ego_fwd) / math.pi
            else:
                rel_spd_sq = dv_east**2 + dv_north**2
                range_rate = d_east * dv_east + d_north * dv_north   # r·v; negative = converging
                tcpa       = (-range_rate / rel_spd_sq) if rel_spd_sq > 1e-12 else -1.0

                if 0 < tcpa <= t_warn:
                    cpa_east  = d_east  + tcpa * dv_east
                    cpa_north = d_north + tcpa * dv_north
                    if cpa_east**2 + cpa_north**2 < sep**2:
                        dcpa         = math.sqrt(cpa_east**2 + cpa_north**2)
                        cpa_ego_lat  = cpa_east  * cos_hdg - cpa_north * sin_hdg
                        cpa_ego_fwd  = cpa_east  * sin_hdg + cpa_north * cos_hdg
                        tcpa_n       = tcpa / t_warn
                        cpa_r_n      = min(1.0, dcpa / sep)
                        cpa_theta_n  = math.atan2(cpa_ego_lat, cpa_ego_fwd) / math.pi
                    else:
                        tcpa_n, cpa_r_n, cpa_theta_n = 1.0, 0.0, 0.0
                else:
                    tcpa_n, cpa_r_n, cpa_theta_n = 1.0, 0.0, 0.0

            # urgency for this pair from the pre-computed matrix
            urgency = 0.0
            if urgency_row is not None and other_cs in self._urgency_cs_list:
                other_row_idx = self._urgency_cs_list.index(other_cs)
                if other_row_idx < len(urgency_row):
                    urgency = float(urgency_row[other_row_idx])

            intruders.append((urgency, dist_nm, r_norm, theta_norm, cpa_r_n, cpa_theta_n, tcpa_n, other_cs))

        # fill slots: urgent pairs first (descending urgency), then nearest (ascending distance)
        urgent  = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0])
        nearest = sorted(intruders, key=lambda r: r[1])

        selected, seen = [], set()
        for rec in urgent:
            if len(selected) >= N_NBR:
                break
            selected.append(rec)
            seen.add(rec[7])
        for rec in nearest:
            if len(selected) >= N_NBR:
                break
            if rec[7] not in seen:
                selected.append(rec)
                seen.add(rec[7])

        for slot_k in range(N_NBR):
            if slot_k < len(selected):
                _, _, r_n, theta_n, cpa_r, cpa_th, t_cpa, _ = selected[slot_k]
                obs += [r_n, theta_n, cpa_r, cpa_th, t_cpa]
            else:
                obs += [1.0, 0.0, 0.0, 0.0, 1.0]   # empty slot sentinel

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
            self._commanded_heading.pop(cs, None)
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

    def _generate_replacement(self, slot):
        """Try to place a new aircraft that clears all currently active aircraft."""
        occupied = [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]
        min_spawn_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        n_ac          = self.n_aircraft

        for _ in range(CONFIG['max_placement_tries']):
            ac = _place_one(self._polygon_shape, random.randint(0, n_ac - 1), n_ac)
            clear_of_all = all(
                geo.kwikdist(float(ac['sp_ll'][0]), float(ac['sp_ll'][1]),
                             float(lat), float(lon)) >= min_spawn_sep
                for lat, lon in occupied
            )
            if clear_of_all:
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
        self._commanded_heading[cs]   = float(ac['heading'])
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
