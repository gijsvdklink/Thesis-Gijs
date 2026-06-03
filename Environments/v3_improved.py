"""
v3_improved — ATCO conflict-resolution environment.

One agent issues one instruction per step to the highest-threat aircraft.
Focus selection uses hard-commitment logic: once a conflict is active
(urgency > 0) the focus never switches away until urgency has been zero for
focus_clear_steps consecutive steps.  This keeps every (obs, action, reward,
next-obs) tuple on the same aircraft, avoiding cross-aircraft credit
assignment confusion.  Exception: if any other aircraft reaches the emergency
urgency threshold (focus_emergency_u = 0.8, tcpa ≈ 120 s), the commitment is
overridden to prevent a developing critical conflict going unaddressed.

Observation  25 floats  (ego-centric from focus aircraft)
  [0]     sin(Δψ_dest)         heading error to destination  [-1, 1]
  [1]     cos(Δψ_dest)                                        [-1, 1]
  [2]     turn_progress        (commanded−actual)/30          [-1, 1]
  [3]     time_to_exit         t_to_boundary / lookahead_s    [0, 1]
  [4]     speed_offset         (cmd_mach − mach_min) / mach_range [0, 1]
  [5:25]  4 intruders × 5     sorted urgency desc, CPA dist asc;
                               diverging pairs sentinel-sorted to back:
            x_norm     lateral displacement / d_warn (75 NM)  [-1, 1]
            y_norm     forward displacement / d_warn (75 NM)  [-1, 1]
            cpa_x      lateral CPA offset / sep_nm            [-1, 1]
            cpa_y      forward CPA offset / sep_nm            [-1, 1]
            tcpa_norm  time to CPA / lookahead_s              [0, 1]
          empty/diverging slot: (1, 1, 0, 1, 1)

Reward  symmetric event-based (same aircraft throughout a conflict sequence)
  +w_resolve                   conflict resolved this step (urgency 0→0 or >0→0)
  −w_resolve                   new conflict entered  (urgency 0→>0)
  −w_resolve                   ongoing conflict      (urgency >0→>0)
  −w_los                       LoS occurred during the 30 s window (additive)
  ±w_nav × Δcos(Δψ_dest)       small: back on track positive, drifting negative
  −w_work × act_cost           1.0 heading / 0.5 speed / 0 direct+hold

Action  (Discrete 10)
  0 −30°   1 −15°   2 direct [free]   3 +15°   4 +30°   5 hold [free]
  6 M−0.04    7 M−0.02    8 M+0.02    9 M+0.04
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
from shapely.geometry import LineString, Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Aircraft & sector
    'ac_type':               'A320',
    'ac_speed':              450.0,           # kts TAS ≈ M0.78 at FL350 — used for episode length
    'ac_mach':               0.78,            # nominal cruise Mach (vcasormach: < 2.0 → Mach)
    'ac_mach_min':           0.70,            # lower bound for speed actions
    'ac_mach_max':           0.82,            # upper bound (A320 MMO)
    'altitude':              350,             # FL350
    'center_ll':             (52.3, 5.3),     # Dutch upper airspace
    'n_aircraft':            lambda: random.randint(2, 15),   # uniform over [2, 15]
    'density_km2':           lambda: random.uniform(5_000.0, 15_000.0),
    # area is derived: n_aircraft × density_km2  (ensures uniform joint distribution)
    'sep_nm':                5.0,             # ICAO separation standard
    'buffer_nm':             5.0,
    'dest_dist_factor':      2.0,
    # Polygon
    'n_vertices':            lambda: random.randint(5, 7),
    'min_circularity':       0.65,
    'max_placement_tries':   50,
    # Aircraft placement jitter
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.30, 0.30),
    # Simulation
    'sim_dt':                0.5,             # BlueSky timestep (s)
    'action_freq':           10,              # RL step = 5 s simulated time
    'lookahead_s':           900.0,           # 15-min conflict lookahead
    't_warn':                600.0,           # urgency ramp starts at tcpa = 10 min
    'crossings_per_episode': 2.5,             # ≈ 60 min simulated per episode
    'spawn_delay_s':         (0, 0),          # immediate replacement keeps density constant
    # Observation
    'n_neighbours':          4,
    # Focus selection — hard commitment with emergency override
    'focus_clear_steps':     5,               # steps at u=0 before focus may switch
    'focus_emergency_u':     0.8,             # another aircraft above this urgency can preempt
    'drift_switch_margin':   0.05,            # min drift advantage to switch focus (≈ 18° extra)
    # Reward weights
    'w_resolve':             1.00,            # ±1 conflict resolved / new or ongoing conflict
    'w_los':                 3.00,            # additive penalty when LoS occurred during step
    'w_nav':                 0.30,            # small ± delta cos(heading error)
    'w_work':                0.05,            # workload per instruction
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR   = CONFIG['n_neighbours']  # 4
OBS_DIM = 5 + N_NBR * 5          # 5 own + 20 intruder = 25

# Normalization: warning-horizon distance and lookahead
D_WARN = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0  # 75 NM (10 min at cruise)

# Heading actions: offset from direct-to-destination; None = hold current heading
HEADING_OFFSETS = [-30, -15, 0, 15, 30, None]  # action indices 0–5
# Speed actions: Mach delta from current commanded Mach; action indices 6–8
SPEED_DELTAS    = [-0.04, -0.02, 0.02, 0.04]

# Workload cost per action (for w_work penalty)
ACT_COST = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]

_bs_initialized = False

__all__ = ['AirspaceEnv', 'CONFIG', 'NM_TO_KM', 'latlon_to_nm', 'wrap_to_180', 'OBS_DIM']

# ── Coordinate helpers ────────────────────────────────────────────────────────

def latlon_to_nm(center_ll, lat, lon):
    clat, clon = center_ll
    return np.array([
        (lon - clon) * 60.0 * math.cos(math.radians(clat)),
        (lat - clat) * 60.0,
    ])

def nm_to_latlon(center_ll, x_nm, y_nm):
    clat, clon = center_ll
    return (clat + y_nm / 60.0,
            clon + x_nm / (60.0 * math.cos(math.radians(clat))))

def wrap_to_180(a):
    return (a + 180) % 360 - 180

def _spd_nms(i):
    """Aircraft TAS in NM/s from BlueSky state."""
    return float(bs.traf.tas[i]) / 1852.0

# ── Sector polygon ────────────────────────────────────────────────────────────

def _make_polygon(area_km2):
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw    = ShapelyPolygon(random_convex_polygon(CONFIG['n_vertices']()))
        scaled = shapely_scale(raw,
                               xfact=math.sqrt(target_nm2 / raw.area),
                               yfact=math.sqrt(target_nm2 / raw.area),
                               origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= CONFIG['min_circularity']:
            break
    cx, cy = scaled.centroid.x, scaled.centroid.y
    return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])

# ── Aircraft placement ────────────────────────────────────────────────────────

def _place_one(polygon, sector, n_sectors):
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx-minx)**2 + (maxy-miny)**2) * CONFIG['dest_dist_factor']
    t_spawn = (sector + CONFIG['spawn_jitter']()) / n_sectors
    t_ref   = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
    sp      = polygon.exterior.interpolate(t_spawn, normalized=True)
    rp      = polygon.exterior.interpolate(t_ref,   normalized=True)
    sp_ll   = nm_to_latlon(CONFIG['center_ll'], sp.x, sp.y)
    rp_ll   = nm_to_latlon(CONFIG['center_ll'], rp.x, rp.y)
    hdg, _  = geo.kwikqdrdist(*[float(v) for v in (*sp_ll, *rp_ll)])
    dlat, dlon = geo.qdrpos(float(sp_ll[0]), float(sp_ll[1]), hdg, dest_dist)
    return {'sp_ll': sp_ll, 'dest_ll': (dlat, dlon), 'heading': hdg}

# ── Pair urgency (focus selector + reward input) ──────────────────────────────

def _pair_urgency(i, j):
    """
    Uses actual aircraft speeds from BlueSky state.
    [1, 10]  dist < sep_nm  (active LoS)
    (0, 1]   tcpa within lookahead and CPA < sep_nm, ramping from t_warn
    0        diverging or will miss
    """
    nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    dx, dy = nm2[0]-nm1[0], nm2[1]-nm1[1]
    d2     = dx*dx + dy*dy
    sep    = CONFIG['sep_nm']

    if d2 < sep*sep:
        return 1.0 + 9.0 * (1.0 - math.sqrt(d2) / sep)

    spd1 = _spd_nms(i)
    spd2 = _spd_nms(j)
    vn1  = spd1 * math.cos(math.radians(bs.traf.hdg[i]))
    ve1  = spd1 * math.sin(math.radians(bs.traf.hdg[i]))
    vn2  = spd2 * math.cos(math.radians(bs.traf.hdg[j]))
    ve2  = spd2 * math.sin(math.radians(bs.traf.hdg[j]))
    dvn, dve = vn2-vn1, ve2-ve1
    rv2  = dvn*dvn + dve*dve
    if rv2 < 1e-12:
        return 0.0

    dot  = dx*dve + dy*dvn
    tcpa = -dot / rv2
    if tcpa < 0 or tcpa > CONFIG['lookahead_s']:
        return 0.0

    if max(0.0, d2 - dot*dot/rv2) >= sep*sep:
        return 0.0

    t_warn = CONFIG['t_warn']
    return min(1.0, max(0.0, (t_warn - tcpa) / t_warn))

# ── BlueSky screen stub ───────────────────────────────────────────────────────

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass

# ── Environment ───────────────────────────────────────────────────────────────

class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(10)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self.n_aircraft           = 0
        self._slots               = []
        self._active_callsigns    = set()
        self._destination_ll      = {}
        self._commanded_heading   = {}
        self._commanded_speed     = {}
        self._steps_since_urgency = {}
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
        self._prev_in_conflict    = {}
        self._prev_cos_diff       = {}
        self._los_this_step       = False
        self._first_step_on_focus = False  # grace step when focus switches to new aircraft

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        n           = CONFIG['n_aircraft']()
        density_km2 = CONFIG['density_km2']()
        area_km2    = float(n * density_km2)

        poly = _make_polygon(area_km2)
        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        diam_nm  = math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
        step_s   = CONFIG['action_freq'] * CONFIG['sim_dt']
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * diam_nm / CONFIG['ac_speed'] * 3600 / step_s
        ))
        self.n_aircraft           = n
        self._slots               = [None] * n
        self._active_callsigns    = set()
        self._destination_ll      = {}
        self._commanded_heading   = {}
        self._commanded_speed     = {}
        self._steps_since_urgency = {}
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._ep_stats            = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': []}
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
        self._prev_in_conflict    = {}
        self._prev_cos_diff       = {}
        self._los_this_step       = False

        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._flat_latlon())
        bs.stack.stack('ASAS OFF')

        min_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        for slot in range(n):
            for _ in range(CONFIG['max_placement_tries']):
                ac = _place_one(self._polygon_shape, slot, n)
                occupied = [
                    (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
                    for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
                ]
                if all(geo.kwikdist(float(ac['sp_ll'][0]), float(ac['sp_ll'][1]),
                                    float(la), float(lo)) >= min_sep
                       for la, lo in occupied):
                    self._spawn_aircraft(slot, ac)
                    break

        self._focus_cs = self._select_focus_aircraft()
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()
        acting_cs = self._focus_cs

        if acting_cs:
            self._apply_action(acting_cs, int(action))

        self._los_this_step = False
        for k in range(CONFIG['action_freq']):
            bs.sim.step()
            if acting_cs and not self._los_this_step and k == CONFIG['action_freq'] - 1:
                self._check_los_now(acting_cs)
        self._step_count += 1

        self._process_exits()
        prev_focus     = acting_cs
        self._focus_cs = self._select_focus_aircraft()
        if self._focus_cs != prev_focus:
            self._first_step_on_focus = True
        reward         = self._compute_reward(acting_cs, int(action))

        U     = self._urgency_matrix
        n_los = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
        self._ep_stats['reward']  += reward
        self._ep_stats['steps']   += 1
        self._ep_stats['actions'].append(int(action))
        if n_los > 0:
            self._ep_stats['los'] += 1

        truncated = self._step_count >= self._max_steps
        info = {'los_pairs': n_los, 'focus_cs': self._focus_cs,
                'n_aircraft': self.n_aircraft}
        if truncated:
            s = max(self._ep_stats['steps'], 1)
            info.update({
                'mean_episode_reward': self._ep_stats['reward'] / s,
                'ep_los_steps':        self._ep_stats['los'],
                'ep_length':           self._ep_stats['steps'],
                'action_distribution': np.bincount(
                    self._ep_stats['actions'], minlength=10).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # ── Focus selection — commitment-based ────────────────────────────────────

    def _select_focus_aircraft(self):
        cs_list = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not cs_list:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return None

        n      = len(cs_list)
        bs_idx = [bs.traf.id2idx(cs) for cs in cs_list]
        U      = np.zeros((n, n))
        for ii in range(n):
            for jj in range(ii+1, n):
                u = _pair_urgency(bs_idx[ii], bs_idx[jj])
                U[ii, jj] = U[jj, ii] = u

        self._urgency_matrix  = U
        self._urgency_cs_list = cs_list

        row_max = U.max(axis=1)
        clear_steps = CONFIG['focus_clear_steps']
        for i, cs in enumerate(cs_list):
            if row_max[i] > 0:
                self._steps_since_urgency[cs] = 0
            else:
                self._steps_since_urgency[cs] = self._steps_since_urgency.get(cs, clear_steps) + 1

        best_cs = (cs_list[int(np.argmax(row_max))]
                   if row_max.max() > 0 else self._drift_fallback(cs_list))

        # Hard commitment: no switch while in active conflict or cooling down,
        # UNLESS another aircraft crosses the emergency urgency threshold.
        if self._focus_cs in cs_list and best_cs != self._focus_cs:
            cur_u          = row_max[cs_list.index(self._focus_cs)]
            focus_resolved = (self._steps_since_urgency.get(self._focus_cs, clear_steps)
                              >= clear_steps)
            emergency      = row_max.max() >= CONFIG['focus_emergency_u']
            if (cur_u > 0 or not focus_resolved) and not emergency:
                return self._focus_cs

        return best_cs

    def _drift_fallback(self, cs_list):
        """
        When no conflicts active: focus on most off-route aircraft.

        Only switches away from the current focus if another aircraft's drift
        exceeds the current focus drift by at least drift_switch_margin.
        This prevents rapid focus-switching when all aircraft are well-aligned
        and drift differences are negligible floating-point noise.
        """
        best_cs, best_drift = None, -1.0
        cur_drift = 0.0

        for cs in sorted(cs_list):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._destination_ll:
                continue
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            diff  = wrap_to_180(bearing - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
            drift = (1 - math.cos(math.radians(diff))) / 2
            if cs == self._focus_cs:
                cur_drift = drift
            if drift > best_drift:
                best_drift, best_cs = drift, cs

        # Hysteresis: keep current focus unless candidate is clearly more off-route
        margin = CONFIG['drift_switch_margin']
        if (self._focus_cs in cs_list
                and best_cs != self._focus_cs
                and best_drift <= cur_drift + margin):
            return self._focus_cs

        return best_cs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx):
        is_in_conflict = self._in_conflict(acting_cs)

        # Grace step on first step after a focus switch: sync state, skip penalty.
        # The agent has not yet observed the new aircraft's geometry or acted on it.
        if self._first_step_on_focus:
            self._first_step_on_focus = False
            self._prev_in_conflict[acting_cs] = is_in_conflict
            r_conflict = 0.0
        else:
            was_in_conflict = self._prev_in_conflict.get(acting_cs, False)
            self._prev_in_conflict[acting_cs] = is_in_conflict

            if was_in_conflict and not is_in_conflict:
                r_conflict = CONFIG['w_resolve']    # resolved
            elif not was_in_conflict and is_in_conflict:
                r_conflict = -CONFIG['w_resolve']   # new conflict
            elif was_in_conflict and is_in_conflict:
                r_conflict = -CONFIG['w_resolve']   # ongoing conflict
            else:
                r_conflict = 0.0                    # clean → clean

        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        r_nav = 0.0
        if acting_cs and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[acting_cs]])
                cos_now  = math.cos(math.radians(wrap_to_180(bearing - bs.traf.hdg[idx])))
                cos_prev = self._prev_cos_diff.get(acting_cs, cos_now)
                self._prev_cos_diff[acting_cs] = cos_now
                r_nav = CONFIG['w_nav'] * (cos_now - cos_prev)

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_conflict + r_los + r_nav + r_work)

    def _in_conflict(self, cs):
        """True if cs has urgency > 0 with any other aircraft (predicted conflict)."""
        if cs is None or cs not in self._urgency_cs_list:
            return False
        i = self._urgency_cs_list.index(cs)
        return i < self._urgency_matrix.shape[0] and bool(self._urgency_matrix[i].max() > 0)

    def _check_los_now(self, cs):
        """Check current separation of cs against all others; sets _los_this_step if < sep_nm."""
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        sep2 = CONFIG['sep_nm'] ** 2
        for other in self._active_callsigns:
            if other == cs:
                continue
            oi = bs.traf.id2idx(other)
            if oi < 0:
                continue
            nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = nm2[0] - nm1[0], nm2[1] - nm1[1]
            if dx * dx + dy * dy < sep2:
                self._los_this_step = True
                return

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        if action_idx < 6:
            # Heading action
            offset = HEADING_OFFSETS[action_idx]
            if offset is None:
                # Hold: re-issue commanded heading
                bs.stack.stack(
                    f'HDG {cs} {self._commanded_heading.get(cs, bs.traf.hdg[idx]):.1f}')
                return
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            if offset == 0:
                # Direct: guard against issuing while mid-turn
                if abs(wrap_to_180(
                        self._commanded_heading.get(cs, bs.traf.hdg[idx])
                        - bs.traf.hdg[idx])) > 5.0:
                    return
            self._commanded_heading[cs] = (float(bearing) + offset) % 360
            bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

        else:
            # Speed action — commanded and issued in Mach
            delta    = SPEED_DELTAS[action_idx - 6]
            cur_mach = self._commanded_speed.get(cs, CONFIG['ac_mach'])
            new_mach = float(np.clip(cur_mach + delta,
                                     CONFIG['ac_mach_min'], CONFIG['ac_mach_max']))
            self._commanded_speed[cs] = new_mach
            bs.stack.stack(f'SPD {cs} {new_mach:.3f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx      = bs.traf.id2idx(cs)
        own_hdg  = bs.traf.hdg[idx]
        cmd_hdg  = self._commanded_heading.get(cs, own_hdg)
        own_nm   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        sep      = CONFIG['sep_nm']
        look     = CONFIG['lookahead_s']
        sin_h    = math.sin(math.radians(own_hdg))
        cos_h    = math.cos(math.radians(own_hdg))
        spd_own  = _spd_nms(idx)
        vn_o     = spd_own * math.cos(math.radians(own_hdg))
        ve_o     = spd_own * math.sin(math.radians(own_hdg))

        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        diff = wrap_to_180(bearing - cmd_hdg)

        turn_prog    = max(-1.0, min(1.0, wrap_to_180(cmd_hdg - own_hdg) / 30.0))
        t_exit       = self._time_to_exit(idx)
        cmd_mach     = self._commanded_speed.get(cs, CONFIG['ac_mach'])
        speed_offset = ((cmd_mach - CONFIG['ac_mach_min'])
                        / (CONFIG['ac_mach_max'] - CONFIG['ac_mach_min']))

        obs = [
            math.sin(math.radians(diff)),  # [0] sin(Δψ_dest)
            math.cos(math.radians(diff)),  # [1] cos(Δψ_dest)
            turn_prog,                     # [2] mid-turn state
            t_exit,                        # [3] time to sector boundary
            speed_offset,                  # [4] speed offset
        ]

        # Urgency row for focus aircraft (for sorting)
        f_row = None
        if cs in self._urgency_cs_list:
            fi = self._urgency_cs_list.index(cs)
            if fi < self._urgency_matrix.shape[0]:
                f_row = self._urgency_matrix[fi]

        intruders = []
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue

            int_nm   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy   = int_nm[0] - own_nm[0], int_nm[1] - own_nm[1]
            spd_int  = _spd_nms(oi)
            vn_i     = spd_int * math.cos(math.radians(bs.traf.hdg[oi]))
            ve_i     = spd_int * math.sin(math.radians(bs.traf.hdg[oi]))
            dvn, dve = vn_i - vn_o, ve_i - ve_o
            rv2      = dvn*dvn + dve*dve

            # CPA geometry
            diverging = True
            cpa_dist  = 3.0 * sep   # sentinel: sort diverging pairs to back
            tcpa_raw  = -1.0
            if rv2 > 1e-12:
                dot      = dx*dve + dy*dvn
                tcpa_raw = -dot / rv2
                if tcpa_raw > 0:
                    diverging = False
                    dx_cpa    = dx + tcpa_raw * dve
                    dy_cpa    = dy + tcpa_raw * dvn
                    cpa_dist  = math.sqrt(dx_cpa*dx_cpa + dy_cpa*dy_cpa)
            if diverging:
                dx_cpa, dy_cpa = dx, dy  # CPA features use current position as sentinel

            # Urgency (for primary sort key)
            urgency = 0.0
            if f_row is not None and other in self._urgency_cs_list:
                oi_u = self._urgency_cs_list.index(other)
                if oi_u < len(f_row):
                    urgency = float(f_row[oi_u])

            # Features in ownship heading frame
            x_norm  = max(-1.0, min(1.0, (dx     * cos_h - dy     * sin_h) / D_WARN))
            y_norm  = max(-1.0, min(1.0, (dx     * sin_h + dy     * cos_h) / D_WARN))
            cpa_x   = max(-1.0, min(1.0, (dx_cpa * cos_h - dy_cpa * sin_h) / sep))
            cpa_y   = max(-1.0, min(1.0, (dx_cpa * sin_h + dy_cpa * cos_h) / sep))
            tcpa_n  = 0.0 if diverging else min(1.0, tcpa_raw / look)

            # Sort: urgency desc (primary), cpa_dist asc (secondary)
            intruders.append((-urgency, cpa_dist, x_norm, y_norm, cpa_x, cpa_y, tcpa_n))

        intruders.sort(key=lambda r: (r[0], r[1]))
        for k in range(N_NBR):
            if k < len(intruders):
                _, _, xn, yn, cx, cy, tn = intruders[k]
                obs += [xn, yn, cx, cy, tn]
            else:
                obs += [1.0, 1.0, 0.0, 1.0, 1.0]  # empty slot sentinel

        return np.array(obs, dtype=np.float32)

    def _time_to_exit(self, idx):
        nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        hdg = bs.traf.hdg[idx]
        end = nm + 400.0 * np.array([math.sin(math.radians(hdg)),
                                      math.cos(math.radians(hdg))])
        ray   = LineString([(float(nm[0]), float(nm[1])), (float(end[0]), float(end[1]))])
        isect = self._polygon_shape.exterior.intersection(ray)
        if isect.is_empty:
            return 1.0
        pts = list(isect.geoms) if hasattr(isect, 'geoms') else [isect]
        d   = min(math.hypot(p.x - nm[0], p.y - nm[1]) for p in pts)
        return min(d / max(_spd_nms(idx) * CONFIG['lookahead_s'], 1.0), 1.0)

    # ── Exits ─────────────────────────────────────────────────────────────────

    def _process_exits(self):
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._destination_ll.pop(cs, None)
            self._commanded_heading.pop(cs, None)
            self._commanded_speed.pop(cs, None)
            self._steps_since_urgency.pop(cs, None)
            self._prev_in_conflict.pop(cs, None)
            self._prev_cos_diff.pop(cs, None)
            if slot not in self._pending_spawns:
                delay = random.randint(*self._spawn_delay_range)
                self._pending_spawns[slot] = delay

    def _find_exited(self):
        out = []
        for cs in list(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                out.append(cs); continue
            inside = bs.tools.areafilter.checkInside(
                'SECTOR',
                np.array([bs.traf.lat[idx]]),
                np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]),
            )
            if not inside[0]:
                out.append(cs)
        return out

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _process_pending_spawns(self):
        ready = sorted(s for s, t in list(self._pending_spawns.items()) if t <= 1)
        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
            else:
                self._pending_spawns[slot] = 5  # sector too dense: retry in 5 steps
        for slot in list(self._pending_spawns):
            self._pending_spawns[slot] -= 1

    def _generate_replacement(self, slot):
        occupied = [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]
        min_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        n       = self.n_aircraft
        for _ in range(CONFIG['max_placement_tries']):
            ac = _place_one(self._polygon_shape, random.randint(0, n - 1), n)
            if all(geo.kwikdist(float(ac['sp_ll'][0]), float(ac['sp_ll'][1]),
                                float(la), float(lo)) >= min_sep
                   for la, lo in occupied):
                return ac
        return None

    def _spawn_aircraft(self, slot, ac):
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        mach = CONFIG['ac_mach']
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['sp_ll'][0]), aclon=float(ac['sp_ll'][1]),
                    achdg=float(ac['heading']), acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)  # FL350 = 350 * 30.48 = 10668 m
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]      = ac['dest_ll']
        self._commanded_heading[cs]   = float(ac['heading'])
        self._commanded_speed[cs]     = mach  # stored as Mach throughout
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._prev_in_conflict[cs]    = False
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _flat_latlon(self):
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
