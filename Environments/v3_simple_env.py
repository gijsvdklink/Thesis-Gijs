"""
v3_simple_env — standalone single-agent ATCO conflict-resolution environment.

Sector  : randomised convex polygon, ≈80 000 km² (Dutch upper airspace scale).
Agent   : single ATCO issuing one heading instruction per step to the most
          urgent aircraft (focus aircraft), selected by a pure-geometry urgency
          matrix — no BlueSky CD module used anywhere.

Urgency matrix  u[i,j]  (rebuilt from scratch every step):
  10.0               dist(i,j) < sep_nm          — active LoS
  1/max(tCPA,1)      dCPA < sep_nm, 0≤tCPA≤H     — predicted conflict
  0                  otherwise (including diverging pairs, tCPA < 0)

Action space  (Discrete 6) — incremental offset applied to commanded heading:
  0 → −20°   1 → −10°   2 → hold (0°)   3 → +10°   4 → +20°   5 → snap-to-WP

  Deviations accumulate in _commanded_heading until corrected or snapped back.
  Actions 2 (hold) and 5 (snap-to-WP) are free; all others carry a small penalty.

Observation  (36 floats, ego-centric from focus aircraft):
  Ownship      : cos(Δψ_dest), sin(Δψ_dest)
  8 neighbours : rel_fwd, rel_right, tLOS_norm, dCPA_norm  (nearest-first, zero-padded)
  Global       : n_los_pairs/n_pairs,  n_conf_pairs/n_pairs

Reward  (global ATCO perspective):
  pen_los       -w × n_LoS_pairs          geometry-based, all aircraft every step
  pen_conflict  -w / max(tLOS_s, 1)       focus aircraft conflicts only
  pen_drift     -w × (1−cosΔψ)/2          focus aircraft commanded heading error
  pen_action    -w for ±10°/±20°           hold and snap-to-WP are free
  pen_wrong_exit on exit heading >90° off destination (any aircraft)
"""

import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.tools import geo
from polygenerator import random_convex_polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    'ac_type':               'A320',
    'ac_speed':              450.0,           # kts
    'altitude':              350,             # FL350
    'center_ll':             (52.3, 5.3),     # central Netherlands
    'area_km2':              lambda: 80_000.0,
    'density_km2':           lambda: 8_000.0, # → ~10 aircraft
    'max_agents':            12,
    'n_neighbours':          8,
    'n_vertices':            lambda: random.randint(5, 7),
    'min_circularity':       0.65,
    'sep_nm':                5.0,
    'buffer_nm':             2.0,
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.15, 0.15),
    'dest_dist_factor':      2.0,
    'max_placement_tries':   50,
    'spawn_delay_s':         (120, 300),
    'seed':                  None,
    'sim_dt':                0.5,             # BlueSky timestep (s)
    'action_freq':           10,              # sim steps per RL step → 5 s/step
    'crossings_per_episode': 3,
    'spatial_scale':         150.0,           # NM — normalises ego-centric positions
    'lookahead_s':           600.0,           # conflict scan horizon (10 min)
    'pen_los':              -50.0,
    'pen_conflict':          -3.0,
    'pen_drift':             -3.0,
    'pen_action':            -0.1,
    'pen_wrong_exit':       -20.0,
}

# ── Derived constants ─────────────────────────────────────────────────────────

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NEIGHBOURS = CONFIG['n_neighbours']           # 8
OBS_DIM      = 2 + N_NEIGHBOURS * 4 + 2        # 36

HEADING_DELTAS = [-20, -10, 0, 10, 20, None]   # None = snap back to destination bearing

_bs_initialized = False

__all__ = ['AirspaceEnv', 'SimpleAirspaceEnv', 'CONFIG', 'NM_TO_KM', 'latlon_to_nm']


# ── Coordinate helpers ────────────────────────────────────────────────────────

def nm_to_latlon(center_ll, x_nm, y_nm):
    clat, clon = center_ll
    return (clat + y_nm / 60.0,
            clon + x_nm / (60.0 * math.cos(math.radians(clat))))


def latlon_to_nm(center_ll, lat, lon):
    clat, clon = center_ll
    return np.array([
        (lon - clon) * 60.0 * math.cos(math.radians(clat)),
        (lat - clat) * 60.0,
    ])


def wrap_to_180(angle_deg):
    return (angle_deg + 180) % 360 - 180


# ── Polygon generation ────────────────────────────────────────────────────────

def make_polygon(area_km2, config):
    target_nm2 = area_km2 * KM_TO_NM ** 2
    while True:
        raw    = ShapelyPolygon(random_convex_polygon(config['n_vertices']()))
        scaled = shapely_scale(raw,
                               xfact=math.sqrt(target_nm2 / raw.area),
                               yfact=math.sqrt(target_nm2 / raw.area),
                               origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= config['min_circularity']:
            cx, cy = scaled.centroid.x, scaled.centroid.y
            return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])


# ── Aircraft placement ────────────────────────────────────────────────────────

def _try_place_one(polygon, sector, n_sectors, dest_dist_nm, config):
    t_spawn  = (sector + config['spawn_jitter']()) / n_sectors
    t_ref    = (t_spawn + 0.5 + config['ref_jitter']()) % 1.0
    spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
    ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)
    spawn_ll = nm_to_latlon(config['center_ll'], spawn_pt.x, spawn_pt.y)
    ref_ll   = nm_to_latlon(config['center_ll'], ref_pt.x,   ref_pt.y)
    heading, _ = geo.kwikqdrdist(*[float(v) for v in (*spawn_ll, *ref_ll)])
    dest_lat, dest_lon = geo.qdrpos(float(spawn_ll[0]), float(spawn_ll[1]),
                                    heading, dest_dist_nm)
    return {
        'spawn_ll': spawn_ll,
        'spawn_nm': np.array([spawn_pt.x, spawn_pt.y]),
        'dest_ll':  (dest_lat, dest_lon),
        'dest_nm':  latlon_to_nm(config['center_ll'], dest_lat, dest_lon),
        'heading':  heading,
    }


def _is_too_close(spawn_ll, placed, min_dist_nm):
    return any(
        geo.kwikdist(float(spawn_ll[0]), float(spawn_ll[1]),
                     float(ac['spawn_ll'][0]), float(ac['spawn_ll'][1])) < min_dist_nm
        for ac in placed
    )


def place_aircraft(polygon, n_aircraft, config):
    min_dist_nm  = config['sep_nm'] + config['buffer_nm']
    max_tries    = config.get('max_placement_tries', 50)
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist_nm = math.sqrt((maxx-minx)**2 + (maxy-miny)**2) * config['dest_dist_factor']
    placed = []
    for sector in range(n_aircraft):
        for _ in range(max_tries):
            candidate = _try_place_one(polygon, sector, n_aircraft, dest_dist_nm, config)
            if not _is_too_close(candidate['spawn_ll'], placed, min_dist_nm):
                placed.append(candidate)
                break
        else:
            return None
    return placed


# ── Urgency (pure geometry — identical to make_gifs._pair_urgency) ────────────

def _pair_urgency(idx1, idx2, sep_nm, horizon):
    """
    Urgency score for one aircraft pair.
      10.0              dist < sep_nm           (active LoS)
      1/max(tCPA,1)     dCPA < sep_nm, 0≤tCPA≤H (predicted conflict)
      0                 diverging (tCPA<0) or will miss
    """
    nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx1], bs.traf.lon[idx1])
    nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx2], bs.traf.lon[idx2])

    dx = nm2[0] - nm1[0]
    dy = nm2[1] - nm1[1]
    dist_sq = dx*dx + dy*dy

    if dist_sq < sep_nm * sep_nm:
        return 10.0

    spd1 = bs.traf.gs[idx1] / 1852.0
    spd2 = bs.traf.gs[idx2] / 1852.0
    vn1  = spd1 * math.cos(math.radians(bs.traf.hdg[idx1]))
    ve1  = spd1 * math.sin(math.radians(bs.traf.hdg[idx1]))
    vn2  = spd2 * math.cos(math.radians(bs.traf.hdg[idx2]))
    ve2  = spd2 * math.sin(math.radians(bs.traf.hdg[idx2]))

    dvn = vn2 - vn1
    dve = ve2 - ve1
    rel_v_sq = dvn*dvn + dve*dve

    if rel_v_sq < 1e-12:
        return 0.0

    dot_rv = dx*dve + dy*dvn
    tcpa   = -dot_rv / rel_v_sq

    if tcpa < 0 or tcpa > horizon:
        return 0.0

    dcpa_sq = max(0.0, dist_sq - dot_rv*dot_rv / rel_v_sq)
    if dcpa_sq >= sep_nm * sep_nm:
        return 0.0

    return 1.0 / max(tcpa, 1.0)


# ── Observation helpers ───────────────────────────────────────────────────────

def _aircraft_vel_nm_s(idx):
    spd = bs.traf.gs[idx] / 1852.0
    hdg = math.radians(bs.traf.hdg[idx])
    return spd * math.cos(hdg), spd * math.sin(hdg)


def _relative_ego(own_nm, int_nm, own_hdg_deg, scale):
    dx    = int_nm[0] - own_nm[0]
    dy    = int_nm[1] - own_nm[1]
    sin_h = math.sin(math.radians(own_hdg_deg))
    cos_h = math.cos(math.radians(own_hdg_deg))
    fwd   = ( dx*sin_h + dy*cos_h) / scale
    right = ( dx*cos_h - dy*sin_h) / scale
    return fwd, right


def _compute_tLOS(dx_nm, dy_nm, dvn, dve, sep_nm):
    """
    Analytical entry/exit times of the separation cylinder.
    Returns (t_entry, t_exit) or (None, None) if no conflict.
    t_entry < 0 means pair is already inside (active LoS).
    """
    rel_v_sq = dvn**2 + dve**2
    r_sq     = dx_nm**2 + dy_nm**2
    if rel_v_sq < 1e-12:
        if r_sq < sep_nm**2:
            return (-float('inf'), float('inf'))
        return (None, None)
    rdotv        = dx_nm*dve + dy_nm*dvn
    discriminant = rdotv**2 - rel_v_sq*(r_sq - sep_nm**2)
    if discriminant < 0.0:
        return (None, None)
    sqrt_d = math.sqrt(discriminant)
    t1     = (-rdotv - sqrt_d) / rel_v_sq
    t2     = (-rdotv + sqrt_d) / rel_v_sq
    if t2 <= 0.0:
        return (None, None)
    return (t1, t2)


def _tcpa_dcpa_norm(dx_nm, dy_nm, dvn, dve, separation_nm):
    rel_v_sq = dvn**2 + dve**2
    if rel_v_sq < 1e-12:
        return 1.0, min(separation_nm / CONFIG['sep_nm'], 3.0)
    dot_pv = dx_nm*dve + dy_nm*dvn
    tcpa   = max(0.0, -dot_pv / rel_v_sq)
    dcpa   = math.sqrt(max(0.0, dx_nm**2 + dy_nm**2 - dot_pv**2 / rel_v_sq))
    return min(tcpa / CONFIG['lookahead_s'], 1.0), min(dcpa / CONFIG['sep_nm'], 3.0)


def _intruder_obs(own_idx, int_idx):
    own_lat, own_lon = bs.traf.lat[own_idx], bs.traf.lon[own_idx]
    int_lat, int_lon = bs.traf.lat[int_idx],  bs.traf.lon[int_idx]
    own_nm           = latlon_to_nm(CONFIG['center_ll'], own_lat, own_lon)
    int_nm           = latlon_to_nm(CONFIG['center_ll'], int_lat, int_lon)
    _, sep_nm        = geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
    rel_fwd, rel_right = _relative_ego(own_nm, int_nm, bs.traf.hdg[own_idx],
                                        CONFIG['spatial_scale'])
    vn_own, ve_own   = _aircraft_vel_nm_s(own_idx)
    vn_int, ve_int   = _aircraft_vel_nm_s(int_idx)
    dx  = int_nm[0] - own_nm[0]
    dy  = int_nm[1] - own_nm[1]
    dvn = vn_int - vn_own
    dve = ve_int - ve_own
    t_entry, _   = _compute_tLOS(dx, dy, dvn, dve, CONFIG['sep_nm'])
    tLOS_norm    = (min(max(t_entry, 0.0), CONFIG['lookahead_s']) / CONFIG['lookahead_s']
                    if t_entry is not None else 1.0)
    _, dcpa_norm = _tcpa_dcpa_norm(dx, dy, dvn, dve, sep_nm)
    return sep_nm, [rel_fwd, rel_right, tLOS_norm, dcpa_norm]


# ── BlueSky screen stub ───────────────────────────────────────────────────────

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass  # noqa: ARG002


# ── Environment ───────────────────────────────────────────────────────────────

class AirspaceEnv(gym.Env):
    """
    Single-agent ATCO environment.  The agent issues one heading instruction
    per step to the focus aircraft (highest urgency by pure-geometry matrix).
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized

        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self.n_aircraft         = 0
        self._slots             = []
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._commanded_heading = {}
        self._next_callsign_id  = 0
        self._step_count        = 0
        self._max_steps         = 0
        self._focus_cs          = None
        self.polygon            = None
        self._polygon_shape     = None
        self._pending_spawns    = {}
        self._spawn_delay_range = (24, 60)
        self._ep_stats          = self._blank_stats()

        # Exposed for the visualiser / GIF renderer
        self._urgency_matrix    = np.zeros((0, 0))
        self._urgency_cs_list   = []

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):  # noqa: ARG002
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)
        bs.traf.reset()

        area_km2    = CONFIG['area_km2']()
        density_km2 = CONFIG['density_km2']()
        n_ac        = int(np.clip(round(area_km2 / density_km2), 2, CONFIG['max_agents']))
        polygon     = make_polygon(area_km2, CONFIG)

        minx, miny, maxx, maxy = polygon.bounds
        diameter_nm     = math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
        crossing_time_s = diameter_nm / CONFIG['ac_speed'] * 3600
        steps_per_wave  = crossing_time_s / (CONFIG['action_freq'] * CONFIG['sim_dt'])
        self._max_steps = max(50, round(CONFIG['crossings_per_episode'] * steps_per_wave))

        self._polygon_shape     = polygon
        self.polygon            = np.array(polygon.exterior.coords[:-1])
        self.n_aircraft         = n_ac
        self._slots             = [None] * n_ac
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._commanded_heading = {}
        self._next_callsign_id  = 0
        self._step_count        = 0
        self._ep_stats          = self._blank_stats()
        self._urgency_matrix    = np.zeros((0, 0))
        self._urgency_cs_list   = []

        step_duration_s = CONFIG['action_freq'] * CONFIG['sim_dt']
        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        self._register_airspace()
        bs.stack.stack('ASAS OFF')   # urgency/LoS computed from pure geometry

        first = self._generate_replacement()
        if first is not None:
            self._spawn_aircraft(0, first)
        cumulative = 0
        for slot_i in range(1, n_ac):
            cumulative += random.randint(delay_min, delay_max)
            self._pending_spawns[slot_i] = cumulative

        self._focus_cs = self._select_focus_aircraft()
        return self._get_focus_observation(), {}

    def step(self, action):
        self._process_pending_spawns()
        acting_cs = self._focus_cs

        if acting_cs is not None:
            self._apply_action(acting_cs, int(action))

        half = CONFIG['action_freq'] // 2
        for _ in range(half):
            bs.sim.step()
        self._focus_cs = self._select_focus_aircraft()
        for _ in range(CONFIG['action_freq'] - half):
            bs.sim.step()
        self._step_count += 1

        exit_reward = self._process_exits()
        reward      = self._compute_reward(acting_cs, int(action)) + exit_reward

        self._ep_stats['total_reward'] += reward
        self._ep_stats['steps']        += 1
        self._ep_stats['actions'].append(int(action))

        self._focus_cs = self._select_focus_aircraft()

        terminated = False
        truncated  = self._step_count >= self._max_steps

        # Count LoS steps from urgency matrix (geometry-based, consistent)
        n_los = int((self._urgency_matrix == 10.0).sum()) // 2 if self._urgency_matrix.size > 0 else 0
        if n_los > 0:
            self._ep_stats['los_steps'] += 1

        info = {
            'los_pairs':  n_los,
            'focus_cs':   self._focus_cs,
            'n_aircraft': self.n_aircraft,
        }
        if truncated:
            info.update({
                'mean_episode_reward': self._ep_stats['total_reward'] / max(self._ep_stats['steps'], 1),
                'ep_los_steps':        self._ep_stats['los_steps'],
                'ep_length':           self._ep_stats['steps'],
                'n_aircraft':          self.n_aircraft,
                'action_distribution': np.bincount(self._ep_stats['actions'], minlength=6).tolist(),
            })

        return self._get_focus_observation(), reward, terminated, truncated, info

    # ── Focus aircraft selection ──────────────────────────────────────────────

    def _select_focus_aircraft(self):
        cs_list = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not cs_list:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return None

        n      = len(cs_list)
        bs_idx = [bs.traf.id2idx(cs) for cs in cs_list]
        U      = np.zeros((n, n))

        sep_nm  = float(CONFIG['sep_nm'])
        horizon = float(CONFIG['lookahead_s'])

        for ii in range(n):
            for jj in range(ii + 1, n):
                score = _pair_urgency(bs_idx[ii], bs_idx[jj], sep_nm, horizon)
                U[ii, jj] = U[jj, ii] = score

        self._urgency_matrix  = U
        self._urgency_cs_list = cs_list

        row_max = U.max(axis=1)
        if row_max.max() == 0.0:
            return self._drift_fallback(cs_list)
        return cs_list[int(np.argmax(row_max))]

    def _drift_fallback(self, cs_list):
        """No conflicts — select the aircraft most off-course by commanded heading."""
        best_cs, best_drift = None, -1.0
        for cs in sorted(cs_list):   # sorted: deterministic, set iteration order is arbitrary
            idx     = bs.traf.id2idx(cs)
            dest_ll = self._destination_ll.get(cs)
            if idx < 0 or dest_ll is None:
                continue
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                float(dest_ll[0]), float(dest_ll[1]),
            )
            # Use commanded heading: actual heading lags by several steps at jet turn rates
            cmd_hdg = self._commanded_heading.get(cs, bs.traf.hdg[idx])
            diff    = wrap_to_180(bearing - cmd_hdg)
            drift   = (1 - math.cos(math.radians(diff))) / 2
            if drift > best_drift:
                best_drift, best_cs = drift, cs

        # Hysteresis: keep current focus unless another aircraft is clearly more off-course.
        # Prevents rapid oscillation when two aircraft have nearly equal drift.
        _SWITCH_MARGIN = 0.02   # ~13° commanded-heading difference before switching
        if self._focus_cs in cs_list and best_cs != self._focus_cs:
            curr_idx    = bs.traf.id2idx(self._focus_cs)
            curr_dest   = self._destination_ll.get(self._focus_cs)
            if curr_idx >= 0 and curr_dest is not None:
                curr_bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[curr_idx], bs.traf.lon[curr_idx],
                    float(curr_dest[0]), float(curr_dest[1]),
                )
                curr_cmd   = self._commanded_heading.get(self._focus_cs, bs.traf.hdg[curr_idx])
                curr_diff  = wrap_to_180(curr_bearing - curr_cmd)
                curr_drift = (1 - math.cos(math.radians(curr_diff))) / 2
                if best_drift - curr_drift < _SWITCH_MARGIN:
                    return self._focus_cs

        return best_cs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx):
        reward = 0.0

        # Global LoS penalty — geometry-based, consistent with urgency matrix
        n_los  = int((self._urgency_matrix == 10.0).sum()) // 2 if self._urgency_matrix.size > 0 else 0
        reward += CONFIG['pen_los'] * n_los

        if acting_cs is not None and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                own_nm         = latlon_to_nm(CONFIG['center_ll'],
                                              bs.traf.lat[idx], bs.traf.lon[idx])
                vn_own, ve_own = _aircraft_vel_nm_s(idx)

                for other_cs in self._active_callsigns:
                    if other_cs == acting_cs:
                        continue
                    other_idx = bs.traf.id2idx(other_cs)
                    if other_idx < 0:
                        continue
                    int_nm         = latlon_to_nm(CONFIG['center_ll'],
                                                  bs.traf.lat[other_idx], bs.traf.lon[other_idx])
                    vn_int, ve_int = _aircraft_vel_nm_s(other_idx)
                    dx = int_nm[0] - own_nm[0]
                    dy = int_nm[1] - own_nm[1]
                    t_entry, _ = _compute_tLOS(dx, dy, vn_int-vn_own, ve_int-ve_own,
                                               CONFIG['sep_nm'])
                    if t_entry is not None:
                        tLOS_s  = max(t_entry, 0.0)
                        reward += CONFIG['pen_conflict'] / max(tLOS_s, 1.0)

                dest_bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    float(self._destination_ll[acting_cs][0]),
                    float(self._destination_ll[acting_cs][1]),
                )
                # Use commanded heading so the penalty is felt immediately on the step
                # the action is taken, not 20-30 s later when the aircraft finishes turning.
                cmd_hdg      = self._commanded_heading.get(acting_cs, bs.traf.hdg[idx])
                bearing_diff = wrap_to_180(dest_bearing - cmd_hdg)
                drift_factor = (1.0 - math.cos(math.radians(bearing_diff))) / 2.0
                reward      += CONFIG['pen_drift'] * drift_factor

        if action_idx not in (2, 5):   # hold (0°) and snap-to-WP are free
            reward += CONFIG['pen_action']

        return float(reward)

    # ── Exits ─────────────────────────────────────────────────────────────────

    def _process_exits(self):
        total = 0.0
        for cs in self._find_exited_aircraft():
            slot_i = self._slots.index(cs)
            total += self._exit_penalty(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot_i] = None
            self._destination_ll.pop(cs, None)
            self._commanded_heading.pop(cs, None)
            if slot_i not in self._pending_spawns:
                delay = random.randint(*self._spawn_delay_range)
                self._pending_spawns[slot_i] = delay
        return total

    def _find_exited_aircraft(self):
        exited = []
        for cs in list(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs)
                continue
            inside = bs.tools.areafilter.checkInside(
                'AIRSPACE',
                np.array([bs.traf.lat[idx]]),
                np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]),
            )
            if not inside[0]:
                exited.append(cs)
        return exited

    def _exit_penalty(self, cs):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        dest_bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            float(self._destination_ll[cs][0]),
            float(self._destination_ll[cs][1]),
        )
        bearing_diff = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
        return CONFIG['pen_wrong_exit'] if abs(bearing_diff) > 90.0 else 0.0

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _process_pending_spawns(self):
        ready = sorted(slot_i for slot_i, t in self._pending_spawns.items() if t <= 1)
        for slot_i in ready:
            del self._pending_spawns[slot_i]
            candidate = self._generate_replacement()
            if candidate is not None:
                self._spawn_aircraft(slot_i, candidate)
        for slot_i in self._pending_spawns:
            self._pending_spawns[slot_i] -= 1

    def _generate_replacement(self):
        occupied     = self._active_positions()
        min_dist_nm  = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        minx, miny, maxx, maxy = self._polygon_shape.bounds
        dest_dist_nm = (math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
                        * CONFIG['dest_dist_factor'])
        for _ in range(CONFIG['max_placement_tries']):
            candidate = _try_place_one(self._polygon_shape, 0, 1, dest_dist_nm, CONFIG)
            slat = float(candidate['spawn_ll'][0])
            slon = float(candidate['spawn_ll'][1])
            nearest = min(
                (geo.kwikqdrdist(slat, slon, lat, lon)[1] for lat, lon in occupied),
                default=float('inf'),
            )
            if nearest >= min_dist_nm:
                return candidate
        return None

    def _active_positions(self):
        return [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]

    def _spawn_aircraft(self, slot_idx, aircraft):
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(aircraft['spawn_ll'][0]),
                    aclon=float(aircraft['spawn_ll'][1]),
                    achdg=float(aircraft['heading']),
                    acspd=CONFIG['ac_speed'],
                    acalt=CONFIG['altitude'])
        bs.stack.stack(f'SPD {cs} {int(CONFIG["ac_speed"])}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]    = aircraft['dest_ll']
        self._commanded_heading[cs] = float(aircraft['heading'])
        self._slots[slot_idx]       = cs
        self._active_callsigns.add(cs)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        delta = HEADING_DELTAS[action_idx]
        if delta is None:
            # Snap back to current bearing toward destination
            dest_bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                float(self._destination_ll[cs][0]),
                float(self._destination_ll[cs][1]),
            )
            self._commanded_heading[cs] = float(dest_bearing)
        else:
            self._commanded_heading[cs] = (
                self._commanded_heading.get(cs, bs.traf.hdg[idx]) + delta
            ) % 360
        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_focus_observation(self):
        if self._focus_cs is None or bs.traf.id2idx(self._focus_cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return self._get_agent_observation(self._focus_cs)

    def _get_agent_observation(self, cs):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        own_lat = bs.traf.lat[own_idx]
        own_lon = bs.traf.lon[own_idx]
        own_hdg = bs.traf.hdg[own_idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)

        dest_bearing, _ = geo.kwikqdrdist(
            own_lat, own_lon,
            float(self._destination_ll[cs][0]),
            float(self._destination_ll[cs][1]),
        )
        bearing_diff = wrap_to_180(dest_bearing - cmd_hdg)
        obs = [math.cos(math.radians(bearing_diff)),
               math.sin(math.radians(bearing_diff))]

        neighbours = sorted(
            [_intruder_obs(own_idx, bs.traf.id2idx(other))
             for other in self._active_callsigns
             if other != cs and bs.traf.id2idx(other) >= 0],
            key=lambda x: x[0],
        )
        for k in range(N_NEIGHBOURS):
            obs += neighbours[k][1] if k < len(neighbours) else [0.0, 0.0, 1.0, 3.0]

        # Global sector summary — derived from urgency matrix (no BS CD dependency)
        n = len(self._active_callsigns)
        n_pairs = max(n * (n - 1) // 2, 1)
        U = self._urgency_matrix
        if U.size > 0:
            n_los  = int((U == 10.0).sum()) // 2
            n_conf = int((U > 0).sum()) // 2
        else:
            n_los = n_conf = 0
        obs += [n_los / n_pairs, n_conf / n_pairs]

        return np.array(obs, dtype=np.float32)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _register_airspace(self):
        flat_latlon = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            flat_latlon += [float(lat), float(lon)]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat_latlon)

    def _blank_stats(self):
        return {'total_reward': 0.0, 'los_steps': 0, 'steps': 0, 'actions': []}


# Backwards-compatible alias used by training and visualisation scripts
SimpleAirspaceEnv = AirspaceEnv
