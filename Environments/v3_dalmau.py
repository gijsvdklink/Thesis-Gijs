"""
v3_dalmau — ATCO conflict-resolution environment.

Reward is purely negative (no positive components):
  −w_los      × 1[LoS]                                      heavy:  separation violation during step
  −w_conflict × (1−dcpa/sep) × (1−tcpa/t_warn)             medium: linear DCPA×TCPA conflict score
  −w_drift    × (1−cos(ψ_dest−ψ_cmd))/2                    medium: commanded heading deviation from route
  −w_work     × act_cost                                    small:  instruction workload
    turn 0/6 (±60°): cost 1.5    turn 1/5 (±45°): cost 1.0    turn 2/4 (±30°): cost 0.5
    hold (3):         cost 0.0   fly-direct (7): cost |Δψ_cmd→dest| / 60°

Observation space (23 floats, ego-centric from focus aircraft):
  [0]   sin(Δψ_dest)   heading error to destination   [-1, 1]
  [1]   cos(Δψ_dest)                                   [-1, 1]
  [2]   turn_progress  (cmd_hdg − actual) / 60         [-1, 1]
  [3:23] 4 intruders × 5 (urgency-desc, dist-asc):
           x_norm   lateral  displacement / D_WARN     [-1, 1]
           y_norm   forward  displacement / D_WARN     [-1, 1]
           cpa_x    lateral  CPA offset   / sep_nm     [-1, 1]
           cpa_y    forward  CPA offset   / sep_nm     [-1, 1]
           tcpa_n   time to CPA / t_warn               [0,  1]
         empty/diverging slot: (1, 1, 0, 1, 1)

Action space (Discrete 8) — ATC-styled hybrid:
  0  δ = −60°   ψ_cmd = ψ_dest − 60°   (cost 1.5)               turn instruction (one-shot)
  1  δ = −45°   ψ_cmd = ψ_dest − 45°   (cost 1.0)               turn instruction (one-shot)
  2  δ = −30°   ψ_cmd = ψ_dest − 30°   (cost 0.5)               turn instruction (one-shot)
  3  HOLD        ψ_cmd unchanged          (free)                   maintain current commanded heading
  4  δ = +30°   ψ_cmd = ψ_dest + 30°   (cost 0.5)               turn instruction (one-shot)
  5  δ = +45°   ψ_cmd = ψ_dest + 45°   (cost 1.0)               turn instruction (one-shot)
  6  δ = +60°   ψ_cmd = ψ_dest + 60°   (cost 1.5)               turn instruction (one-shot)
  7  FLY DIRECT ψ_cmd = ψ_dest          (cost |Δψ|/60°, 0–1.0)  return to direct track

Intended pattern: issue one turn (0–2 or 4–6) → hold (3) until conflict resolves → fly direct (7).
Hold is the only truly free action.
Fly-direct cost scales with the commanded deviation being corrected: free when already on track,
up to 1.0 × w_work when correcting a full 60° deviation.  This discourages premature fly-direct
(aborting a turn before the conflict clears) while making it cheap once the aircraft is nearly
back on course.
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
    'ac_speed':              450.0,
    'ac_mach':               0.78,
    'ac_mach_min':           0.70,
    'ac_mach_max':           0.82,
    'altitude':              350,
    'center_ll':             (52.3, 5.3),
    'n_aircraft':            lambda: random.randint(10, 15),
    'density_km2':           lambda: random.uniform(5_000.0, 15_000.0),
    'sep_nm':                5.0,
    'buffer_nm':             10.0,
    'dest_dist_factor':      2.0,
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
    'crossings_per_episode': 2.5,
    'spawn_delay_s':         (0, 0),
    # Observation
    'n_neighbours':          4,
    # Focus selection
    'focus_clear_steps':     5,
    'focus_emergency_u':     0.8,
    'drift_switch_margin':   0.05,
    # Reward weights
    'w_los':                 10.00,           # heavy: separation violation
    'w_conflict':            3.00,            # medium: imminence × miss-distance of worst conflict
    'w_drift':               0.50,            # accumulates during hold, motivates return to track via action 5
    'w_work':                0.50,            # charged once per turn instruction; hold and direct are free
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR      = CONFIG['n_neighbours']
OBS_DIM    = 3 + N_NBR * 5       # 3 own + 20 intruder = 23

D_WARN  = CONFIG['t_warn'] * CONFIG['ac_speed'] / 3600.0  # 75 NM
V_NOM = CONFIG['ac_speed'] / 3600.0                        # nominal speed (NM/s), used for CPA computation

# Turn offsets for actions 0–2, 4–6 (indexed directly); action 3=hold, action 7=fly-direct
TURN_DELTAS = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}

# Workload cost: turns cost, hold is free; fly-direct (index 7) cost is computed dynamically
ACT_COST = [1.5, 1.0, 0.5, 0.0, 0.5, 1.0, 1.5, None]

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

# ── Pair urgency ──────────────────────────────────────────────────────────────

def _pair_urgency(i, j):
    nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    dx, dy = nm2[0]-nm1[0], nm2[1]-nm1[1]
    d2     = dx*dx + dy*dy
    sep    = CONFIG['sep_nm']

    if d2 < sep*sep:
        return 1.0 + 9.0 * (1.0 - math.sqrt(d2) / sep)

    spd1 = _spd_nms(i); spd2 = _spd_nms(j)
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

    return min(1.0, max(0.0, (CONFIG['t_warn'] - tcpa) / CONFIG['t_warn']))

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
        self._commanded_heading   = {}
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
        self._los_this_step       = False

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
        self._steps_since_urgency = {}
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._ep_stats            = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': []}
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
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
        acting_cs   = self._focus_cs
        pre_cmd_hdg = self._commanded_heading.get(acting_cs) if acting_cs else None

        if acting_cs:
            self._apply_action(acting_cs, int(action))

        self._los_this_step = False
        for k in range(CONFIG['action_freq']):
            bs.sim.step()
        self._check_los_now()
        self._step_count += 1

        self._process_exits()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, int(action), pre_cmd_hdg)

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
                    self._ep_stats['actions'], minlength=8).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # ── Focus selection ───────────────────────────────────────────────────────

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

        if self._focus_cs in cs_list and best_cs != self._focus_cs:
            cur_u          = row_max[cs_list.index(self._focus_cs)]
            focus_resolved = (self._steps_since_urgency.get(self._focus_cs, clear_steps)
                              >= clear_steps)
            emergency      = row_max.max() >= CONFIG['focus_emergency_u']
            if (cur_u > 0 or not focus_resolved) and not emergency:
                return self._focus_cs

        return best_cs

    def _drift_fallback(self, cs_list):
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
        margin = CONFIG['drift_switch_margin']
        if (self._focus_cs in cs_list
                and best_cs != self._focus_cs
                and best_drift <= cur_drift + margin):
            return self._focus_cs
        return best_cs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx, pre_cmd_hdg=None):
        # LoS: heavy binary penalty when separation is violated
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        # Conflict: imminence of the worst predicted conflict (dcpa < sep required)
        # Score = max(0, 1 − tcpa/T_warn) — ranges 0 (far) to 1 (imminent)
        r_conflict = -CONFIG['w_conflict'] * self._conflict_score(acting_cs)

        # Drift: commanded heading deviation from destination bearing
        r_drift = 0.0
        if acting_cs and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[acting_cs]])
                cmd_hdg = self._commanded_heading.get(acting_cs, bs.traf.hdg[idx])
                cos_err = math.cos(math.radians(wrap_to_180(bearing - cmd_hdg)))
                r_drift = -CONFIG['w_drift'] * (1.0 - cos_err) / 2.0

        # Workload: turns have fixed cost; hold is free; fly-direct scales with
        # the commanded deviation being corrected (|Δψ| / 60°), so aborting a
        # full 60° turn early costs as much as the turn itself.
        r_work = 0.0
        if acting_cs:
            if action_idx == 7 and pre_cmd_hdg is not None:
                idx = bs.traf.id2idx(acting_cs)
                if idx >= 0 and acting_cs in self._destination_ll:
                    bearing, _ = geo.kwikqdrdist(
                        bs.traf.lat[idx], bs.traf.lon[idx],
                        *[float(v) for v in self._destination_ll[acting_cs]])
                    deviation = abs(wrap_to_180(float(bearing) - pre_cmd_hdg))
                    r_work = -CONFIG['w_work'] * min(deviation / 60.0, 1.0)
            else:
                r_work = -CONFIG['w_work'] * ACT_COST[action_idx]

        return float(r_los + r_conflict + r_drift + r_work)

    def _conflict_score(self, cs):
        """
        max over intruders of (1 − tcpa/T_warn) × (1 − dcpa/sep), gated by dcpa < sep.
        Returns 0 for diverging pairs or when dcpa ≥ sep (no predicted LoS).
        Aircraft already in LoS contribute score = 1 (maximum imminence).
        """
        if cs is None:
            return 0.0
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0

        nm1    = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        spd1   = _spd_nms(idx)
        vn1    = spd1 * math.cos(math.radians(bs.traf.hdg[idx]))
        ve1    = spd1 * math.sin(math.radians(bs.traf.hdg[idx]))
        sep    = CONFIG['sep_nm']
        t_warn = CONFIG['t_warn']
        look   = CONFIG['lookahead_s']

        max_score = 0.0
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue

            nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = nm2[0] - nm1[0], nm2[1] - nm1[1]
            d_now  = math.sqrt(dx*dx + dy*dy)

            if d_now < sep:
                max_score = 1.0   # in LoS → maximum imminence
                continue

            spd2     = _spd_nms(oi)
            vn2      = spd2 * math.cos(math.radians(bs.traf.hdg[oi]))
            ve2      = spd2 * math.sin(math.radians(bs.traf.hdg[oi]))
            dvn, dve = vn2 - vn1, ve2 - ve1
            rv2      = dvn*dvn + dve*dve
            if rv2 < 1e-12:
                continue

            dot  = dx*dve + dy*dvn
            tcpa = -dot / rv2
            if tcpa < 0 or tcpa > look:
                continue

            dx_c   = dx + tcpa * dve
            dy_c   = dy + tcpa * dvn
            dcpa2  = dx_c*dx_c + dy_c*dy_c
            if dcpa2 >= sep*sep:
                continue   # dcpa ≥ sep, no predicted LoS

            dcpa      = math.sqrt(dcpa2)
            score     = max(0.0, 1.0 - tcpa / t_warn) * max(0.0, 1.0 - dcpa / sep)
            max_score = max(max_score, score)

        return max_score

    def _check_los_now(self):
        cs_list = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        sep2 = CONFIG['sep_nm'] ** 2
        for ii in range(len(cs_list)):
            idx = bs.traf.id2idx(cs_list[ii])
            nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
            for jj in range(ii + 1, len(cs_list)):
                oi  = bs.traf.id2idx(cs_list[jj])
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
        if action_idx in TURN_DELTAS:
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            self._commanded_heading[cs] = (float(bearing) + TURN_DELTAS[action_idx]) % 360
        elif action_idx == 7:  # fly direct
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            self._commanded_heading[cs] = float(bearing) % 360
        # action 3 = hold: ψ_cmd unchanged, just re-issue to BlueSky
        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)
        own_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        sin_h   = math.sin(math.radians(own_hdg))
        cos_h   = math.cos(math.radians(own_hdg))
        spd_own = _spd_nms(idx)
        vn_o    = spd_own * math.cos(math.radians(own_hdg))
        ve_o    = spd_own * math.sin(math.radians(own_hdg))

        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        diff = wrap_to_180(bearing - cmd_hdg)

        turn_prog = max(-1.0, min(1.0, wrap_to_180(cmd_hdg - own_hdg) / 60.0))

        obs = [
            math.sin(math.radians(diff)),
            math.cos(math.radians(diff)),
            turn_prog,
        ]

        f_row = None
        if cs in self._urgency_cs_list:
            fi = self._urgency_cs_list.index(cs)
            if fi < self._urgency_matrix.shape[0]:
                f_row = self._urgency_matrix[fi]

        # Each record: (urgency, d_now, rel_x, rel_y, tcpa_n, cpa_x_n, cpa_y_n, callsign)
        # All positional features in the ego-centric heading frame (forward = ψ_act).
        # tcpa_n  = tcpa/t_warn ∈ [0,1]: 0 = imminent; 1 = beyond warning horizon or safe miss.
        # cpa_x_n = x_CPA/sep, cpa_y_n = y_CPA/sep: non-zero only when dcpa < sep.
        sep    = CONFIG['sep_nm']
        t_warn = CONFIG['t_warn']

        intruders = []
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue

            int_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy  = int_nm[0] - own_nm[0], int_nm[1] - own_nm[1]
            d_now   = math.sqrt(dx*dx + dy*dy)

            spd_int = _spd_nms(oi)
            vn_i    = spd_int * math.cos(math.radians(bs.traf.hdg[oi]))
            ve_i    = spd_int * math.sin(math.radians(bs.traf.hdg[oi]))
            dve     = ve_i - ve_o
            dvn     = vn_i - vn_o

            rel_x = max(-1.0, min(1.0, (dx * cos_h - dy * sin_h) / D_WARN))
            rel_y = max(-1.0, min(1.0, (dx * sin_h + dy * cos_h) / D_WARN))

            # tcpa / cpa coords — only for predicted conflicts (dcpa < sep), matching the reward gate
            if d_now < sep:
                # active LoS: CPA is current relative position
                tcpa_n  = 0.0
                cpa_x_n = max(-1.0, min(1.0, (dx  * cos_h - dy  * sin_h) / sep))
                cpa_y_n = max(-1.0, min(1.0, (dx  * sin_h + dy  * cos_h) / sep))
            else:
                rv2 = dvn*dvn + dve*dve
                tcpa_raw = (-(dx*dve + dy*dvn) / rv2) if rv2 > 1e-12 else -1.0
                if 0 < tcpa_raw <= t_warn:
                    dx_c  = dx + tcpa_raw * dve
                    dy_c  = dy + tcpa_raw * dvn
                    if dx_c*dx_c + dy_c*dy_c < sep*sep:   # dcpa < sep: true collision course
                        tcpa_n  = tcpa_raw / t_warn
                        cpa_x_n = max(-1.0, min(1.0, (dx_c * cos_h - dy_c * sin_h) / sep))
                        cpa_y_n = max(-1.0, min(1.0, (dx_c * sin_h + dy_c * cos_h) / sep))
                    else:
                        tcpa_n, cpa_x_n, cpa_y_n = 1.0, 0.0, 0.0
                else:
                    tcpa_n, cpa_x_n, cpa_y_n = 1.0, 0.0, 0.0

            urgency = 0.0
            if f_row is not None and other in self._urgency_cs_list:
                oi_u = self._urgency_cs_list.index(other)
                if oi_u < len(f_row):
                    urgency = float(f_row[oi_u])

            intruders.append((urgency, d_now, rel_x, rel_y, cpa_x_n, cpa_y_n, tcpa_n, other))

        by_urgency  = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0])
        by_distance = sorted(intruders, key=lambda r: r[1])

        selected = []
        seen     = set()
        for rec in by_urgency:
            if len(selected) >= N_NBR:
                break
            selected.append(rec)
            seen.add(rec[7])
        for rec in by_distance:
            if len(selected) >= N_NBR:
                break
            if rec[7] not in seen:
                selected.append(rec)
                seen.add(rec[7])

        for k in range(N_NBR):
            if k < len(selected):
                _, _, rx, ry, cx, cy, tn, _ = selected[k]
                obs += [rx, ry, cx, cy, tn]
            else:
                obs += [1.0, 1.0, 0.0, 1.0, 1.0]

        return np.array(obs, dtype=np.float32)

    # ── Exits ─────────────────────────────────────────────────────────────────

    def _process_exits(self):
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                bs.traf.delete(idx)
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._destination_ll.pop(cs, None)
            self._commanded_heading.pop(cs, None)
            self._steps_since_urgency.pop(cs, None)
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
        requeued = set()
        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
            else:
                self._pending_spawns[slot] = 5
                requeued.add(slot)
        for slot in list(self._pending_spawns):
            if slot not in requeued:
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
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]      = ac['dest_ll']
        self._commanded_heading[cs]   = float(ac['heading'])
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _flat_latlon(self):
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
