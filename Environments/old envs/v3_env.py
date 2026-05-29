"""
v3_env — Single-agent ATCO conflict-resolution environment.
Scenario: realistic Dutch upper-airspace sector (≈80 000 km², centered on the Netherlands).

The agent plays the role of the ATCO: at each step the most urgent aircraft is selected
deterministically (tLOS reaction-margin urgency) and the agent issues one heading instruction
to it.  All other aircraft maintain their current commanded heading.

Action space (Discrete 6):
  0 → -20°   1 → -10°   2 → hold   3 → +10°   4 → +20°   5 → back to WP

Observation (36 floats) — ego-centric from focus aircraft + global sector summary:
  Ownship      : cos(bearing_to_dest - heading), sin(bearing_to_dest - heading)
  8 neighbours : rel_fwd, rel_right, tLOS_norm, dcpa_norm  (nearest-first, zero-padded)
  Global       : n_los_pairs / n_pairs,  n_conf_pairs / n_pairs

Reward — global ATCO perspective:
  pen_los      : -w × number of LoS pairs   (all aircraft, every step)
  pen_conflict : -w / max(tLOS_s, 1)        (focus aircraft conflicts only)
  pen_drift    : -w × (1-cos Δψ)/2          (focus aircraft only)
  pen_action   : -w for ±10°/±20° changes   (hold and back-to-WP are free)
  pen_wrong_exit: on exit if heading >90° from destination (any aircraft)
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
    # ── Aircraft & airspace ───────────────────────────────────────────────────
    'ac_type':             'A320',
    'ac_speed':            450.0,           # kts  — typical A320 cruise at FL350
    'altitude':            350,             # FL350
    'center_ll':           (52.3, 5.3),     # central Netherlands (≈ Eindhoven)
    'area_km2':            lambda: 80_000.0,  # ~Amsterdam ACC upper sub-sector
    'density_km2':         lambda: 8_000.0,   # 80000/8000 → 10 aircraft
    'max_agents':          12,              # cap handles density variation
    'n_neighbours':        8,              # nearest intruders included in obs
    'n_vertices':          lambda: random.randint(5, 7),   # compact sector shape
    'min_circularity':     0.65,
    'sep_nm':              5.0,             # ICAO horizontal separation standard
    'buffer_nm':           2.0,             # extra clearance at spawn
    'spawn_jitter':        lambda: random.uniform(0.1, 0.9),
    'ref_jitter':          lambda: random.uniform(-0.15, 0.15),
    'dest_dist_factor':    2.0,
    'max_placement_tries': 50,
    'spawn_delay_s':       (120, 300),      # simulated seconds between aircraft entries
    'seed':                None,
    # ── Simulation timing ────────────────────────────────────────────────────
    'sim_dt':              0.5,             # BlueSky timestep (s)
    'action_freq':         10,              # sim steps per RL step → 5 s/step
    'crossings_per_episode': 5,            # episode ≈ 5 × sector-crossing time
    # ── Observation ──────────────────────────────────────────────────────────
    'spatial_scale':       150.0,           # NM — normalises ego-centric positions
    'lookahead_s':         600.0,           # conflict scan horizon (10 min)
    # ── Urgency ──────────────────────────────────────────────────────────────
    'turn_rate_deg_s':     3.0,            # A320 standard-rate bank ≈ 3 °/s
    'urgency_fwd_steps':   4,              # RL steps ahead for developing-conflict scan
    'urgency_fwd_weight':  0.3,            # discount on developing conflicts
    # ── Reward ───────────────────────────────────────────────────────────────
    'pen_los':            -50.0,           # per LoS pair per step (global)
    'pen_conflict':        -3.0,           # per conflict of focus ac, scaled by 1/tLOS
    'pen_drift':           -1.0,           # heading deviation from destination
    'pen_action':          -0.1,           # for ±10°/±20° heading changes
    'pen_wrong_exit':     -20.0,           # exit heading >90° from destination
}

# ── Derived constants ─────────────────────────────────────────────────────────

NM_TO_KM  = 1.852
KM_TO_NM  = 1.0 / NM_TO_KM

N_NEIGHBOURS = CONFIG['n_neighbours']                  # 8
OBS_DIM      = 2 + N_NEIGHBOURS * 4 + 2               # 2 + 32 + 2 = 36

HEADING_DELTAS = [-20, -10, 0, 10, 20, None]   # None = snap back to WP bearing

_bs_initialized = False


# ── Coordinate helpers ────────────────────────────────────────────────────────

def nm_to_latlon(center_ll, x_nm, y_nm):
    clat, clon = center_ll
    return clat + y_nm / 60.0, clon + x_nm / (60.0 * math.cos(math.radians(clat)))


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
    target_area_nm2 = area_km2 * KM_TO_NM ** 2
    while True:
        raw    = ShapelyPolygon(random_convex_polygon(config['n_vertices']()))
        scaled = shapely_scale(raw, xfact=math.sqrt(target_area_nm2 / raw.area),
                               yfact=math.sqrt(target_area_nm2 / raw.area), origin='centroid')
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
    dest_lat, dest_lon = geo.qdrpos(float(spawn_ll[0]), float(spawn_ll[1]), heading, dest_dist_nm)
    return {
        'spawn_ll': spawn_ll,
        'spawn_nm': np.array([spawn_pt.x, spawn_pt.y]),
        'dest_ll':  (dest_lat, dest_lon),
        'dest_nm':  latlon_to_nm(config['center_ll'], dest_lat, dest_lon),
        'heading':  heading,
    }


def _is_too_close(spawn_ll, placed_aircraft, min_dist_nm):
    return any(
        geo.kwikdist(float(spawn_ll[0]), float(spawn_ll[1]),
                     float(ac['spawn_ll'][0]), float(ac['spawn_ll'][1])) < min_dist_nm
        for ac in placed_aircraft
    )


def place_aircraft(polygon, n_aircraft, config):
    min_dist_nm  = config['sep_nm'] + config['buffer_nm']
    max_tries    = config.get('max_placement_tries', 50)
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist_nm = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2) * config['dest_dist_factor']
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
    fwd   = ( dx * sin_h + dy * cos_h) / scale
    right = ( dx * cos_h - dy * sin_h) / scale
    return fwd, right


def _compute_tLOS(dx_nm, dy_nm, dvn, dve, sep_nm):
    """
    Return (t_entry, t_exit) — times when the intruder enters/exits the separation
    cylinder of radius sep_nm around the ownship, assuming constant velocities.

    Solve |r + t·v|² = sep_nm² → quadratic in t.
      r·v = dx*dve + dy*dvn  (East×East + North×North)

    t_entry < 0  →  pair is already inside the zone (active LoS)
    Returns (None, None) when there is no conflict (discriminant < 0 or fully past).
    """
    rel_v_sq = dvn ** 2 + dve ** 2
    r_sq     = dx_nm ** 2 + dy_nm ** 2
    if rel_v_sq < 1e-12:
        if r_sq < sep_nm ** 2:
            return (-float('inf'), float('inf'))   # stationary — permanent LoS
        return (None, None)

    rdotv        = dx_nm * dve + dy_nm * dvn
    discriminant = rdotv ** 2 - rel_v_sq * (r_sq - sep_nm ** 2)
    if discriminant < 0.0:
        return (None, None)

    sqrt_d = math.sqrt(discriminant)
    t1     = (-rdotv - sqrt_d) / rel_v_sq   # entry (earlier)
    t2     = (-rdotv + sqrt_d) / rel_v_sq   # exit  (later)
    if t2 <= 0.0:
        return (None, None)   # conflict is entirely in the past (pair diverging)
    return (t1, t2)


def _tcpa_dcpa_norm(dx_nm, dy_nm, dvn, dve, separation_nm):
    rel_v_sq = dvn ** 2 + dve ** 2
    if rel_v_sq < 1e-12:
        return 1.0, min(separation_nm / CONFIG['sep_nm'], 3.0)
    dot_pv = dx_nm * dve + dy_nm * dvn
    tcpa   = max(0.0, -dot_pv / rel_v_sq)
    dcpa   = math.sqrt(max(0.0, dx_nm ** 2 + dy_nm ** 2 - dot_pv ** 2 / rel_v_sq))
    return min(tcpa / CONFIG['lookahead_s'], 1.0), min(dcpa / CONFIG['sep_nm'], 3.0)


def _intruder_obs(own_idx, int_idx):
    own_lat, own_lon   = bs.traf.lat[own_idx], bs.traf.lon[own_idx]
    int_lat, int_lon   = bs.traf.lat[int_idx],  bs.traf.lon[int_idx]
    own_nm             = latlon_to_nm(CONFIG['center_ll'], own_lat, own_lon)
    int_nm             = latlon_to_nm(CONFIG['center_ll'], int_lat, int_lon)
    _, separation_nm   = geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
    rel_fwd, rel_right = _relative_ego(own_nm, int_nm, bs.traf.hdg[own_idx], CONFIG['spatial_scale'])
    vn_own, ve_own     = _aircraft_vel_nm_s(own_idx)
    vn_int, ve_int     = _aircraft_vel_nm_s(int_idx)
    dx  = int_nm[0] - own_nm[0]
    dy  = int_nm[1] - own_nm[1]
    dvn = vn_int - vn_own
    dve = ve_int - ve_own
    # tLOS_norm: 0 = already in LoS, 1 = no conflict within horizon
    t_entry, _   = _compute_tLOS(dx, dy, dvn, dve, CONFIG['sep_nm'])
    tLOS_norm    = (min(max(t_entry, 0.0), CONFIG['lookahead_s']) / CONFIG['lookahead_s']
                    if t_entry is not None else 1.0)
    _, dcpa_norm = _tcpa_dcpa_norm(dx, dy, dvn, dve, separation_nm)
    return separation_nm, [rel_fwd, rel_right, tLOS_norm, dcpa_norm]


# ── BlueSky screen stub ───────────────────────────────────────────────────────

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass


# ── Environment ───────────────────────────────────────────────────────────────

class AirspaceEnv(gym.Env):
    """
    Single-agent ATCO environment.

    The agent issues one heading instruction per step to the aircraft
    with the highest conflict urgency.  Urgency falls back to drift
    when no aircraft is in a predicted conflict.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(6)

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
        self._pending_spawns    = {}   # {slot_i: steps_remaining_until_spawn}
        self._spawn_delay_range = (24, 60)
        self._ep_stats          = self._blank_stats()

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
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

        # Compute episode length from how long aircraft take to cross the sector
        minx, miny, maxx, maxy = polygon.bounds
        diameter_nm     = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
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

        self._pending_spawns = {}
        step_duration_s = CONFIG['action_freq'] * CONFIG['sim_dt']
        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s))
        self._spawn_delay_range = (delay_min, delay_max)

        self._register_airspace()
        bs.stack.stack('ASAS ON')

        # Spawn the first aircraft immediately; queue the rest with staggered
        # cumulative delays so they enter one by one during the episode.
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
        acting_cs = self._focus_cs   # snapshot — may exit during sim advance

        if acting_cs is not None:
            self._apply_action(acting_cs, int(action))

        # Advance sim in two halves, updating focus between them so fast-developing
        # conflicts are reflected in the observation within the same step.
        half = CONFIG['action_freq'] // 2
        for _ in range(half):
            bs.sim.step()
        self._focus_cs = self._select_focus_aircraft()
        for _ in range(CONFIG['action_freq'] - half):
            bs.sim.step()
        self._step_count += 1

        exit_reward        = self._process_exits()
        reward             = self._compute_reward(acting_cs, int(action)) + exit_reward

        self._ep_stats['total_reward'] += reward
        self._ep_stats['steps']        += 1
        self._ep_stats['actions'].append(int(action))
        if len(bs.traf.cd.lospairs) > 0:
            self._ep_stats['los_steps'] += 1

        self._focus_cs = self._select_focus_aircraft()

        terminated = False
        truncated  = self._step_count >= self._max_steps

        info = {
            'los_pairs': list(bs.traf.cd.lospairs),
            'focus_cs':  self._focus_cs,
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
        """
        Select the aircraft with highest urgency.
        Urgency = sum 1/TCPA over all predicted conflict pairs.
        Falls back to highest drift when no conflicts exist.
        """
        best_cs      = None
        best_urgency = -1.0

        for cs in self._active_callsigns:
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue
            u = self._aircraft_urgency(cs, idx)
            if u > best_urgency:
                best_urgency = u
                best_cs      = cs

        return best_cs

    def _aircraft_urgency(self, cs, idx):
        """
        Urgency score for focus-aircraft selection.

        Per conflicting pair:
          tLOS     — analytical time to separation-zone entry (< 0 = already in LoS)
          t_react  — per-pair reaction time: arcsin((R-dCPA)/range) / turn_rate
                     Head-on (dCPA≈0) needs a large deflection → high t_react.
                     Near-miss (dCPA≈R) needs almost nothing → small t_react.
          margin   — tLOS − t_react  (runway before action window closes)

          Active LoS (t_entry < 0):  score = 1 / max(t_exit, 1)
              t_exit = remaining time in zone. Large → converging (urgent).
              Small  → diverging, self-resolving (less urgent).
          Future conflict (t_entry ≥ 0): score = 1 / max(margin, 1)

        Lead  : max single-pair score (imminence dominates).
        Boost : additive, capped at +0.15 for 4+ conflicts (leverage only
                separates aircraft of comparable imminence, never overrides it).
        Fwd   : projected-position scan (FWD_S ahead) catches developing conflicts
                not yet visible in the current conflict set.
        Drift : tiny fallback when no conflict exists at all.
        """
        TURN_RATE = CONFIG['turn_rate_deg_s']
        STEP_S    = CONFIG['action_freq'] * CONFIG['sim_dt']
        FWD_S     = CONFIG['urgency_fwd_steps'] * STEP_S
        FWD_W     = CONFIG['urgency_fwd_weight']
        HORIZON   = CONFIG['lookahead_s']
        sep_std   = CONFIG['sep_nm']

        own_nm         = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        vn_own, ve_own = _aircraft_vel_nm_s(idx)

        best_score  = 0.0
        n_conflicts = 0
        fwd_score   = 0.0

        for other_cs in self._active_callsigns:
            if other_cs == cs:
                continue
            other_idx = bs.traf.id2idx(other_cs)
            if other_idx < 0:
                continue

            int_nm         = latlon_to_nm(CONFIG['center_ll'],
                                          bs.traf.lat[other_idx], bs.traf.lon[other_idx])
            vn_int, ve_int = _aircraft_vel_nm_s(other_idx)
            dx  = int_nm[0] - own_nm[0]
            dy  = int_nm[1] - own_nm[1]
            dvn = vn_int - vn_own
            dve = ve_int - ve_own

            t_entry, t_exit = _compute_tLOS(dx, dy, dvn, dve, sep_std)

            if t_entry is not None and (t_entry < 0.0 or t_entry <= HORIZON):
                # ── Variable t_react from dCPA ─────────────────────────────────
                r_sq     = dx ** 2 + dy ** 2
                rel_v_sq = dvn ** 2 + dve ** 2
                if rel_v_sq > 1e-12:
                    rdotv = dx * dve + dy * dvn
                    dcpa  = math.sqrt(max(0.0, r_sq - rdotv ** 2 / rel_v_sq))
                else:
                    dcpa = math.sqrt(r_sq)
                range_nm = math.sqrt(max(r_sq, 1e-6))
                defl_rad = math.asin(min(1.0, max(0.0, (sep_std - dcpa) / range_nm)))
                t_react  = max(5.0, math.degrees(defl_rad) / TURN_RATE)

                if t_entry < 0.0:
                    # Active LoS — score by remaining time inside zone
                    pair_score = 1.0 / max(t_exit, 1.0)
                else:
                    # Future conflict — score by reaction margin
                    margin     = t_entry - t_react
                    pair_score = 1.0 / max(margin, 1.0)

                best_score   = max(best_score, pair_score)
                n_conflicts += 1

            else:
                # ── Forward-looking scan ───────────────────────────────────────
                dx_f   = dx + dve * FWD_S
                dy_f   = dy + dvn * FWD_S
                t_f, _ = _compute_tLOS(dx_f, dy_f, dvn, dve, sep_std)
                if t_f is not None:
                    # dCPA and t_react from projected geometry
                    r_sq_f   = dx_f ** 2 + dy_f ** 2
                    rel_v_sq = dvn ** 2 + dve ** 2
                    if rel_v_sq > 1e-12:
                        rdotv_f = dx_f * dve + dy_f * dvn
                        dcpa_f  = math.sqrt(max(0.0, r_sq_f - rdotv_f ** 2 / rel_v_sq))
                    else:
                        dcpa_f = math.sqrt(r_sq_f)
                    range_f   = math.sqrt(max(r_sq_f, 1e-6))
                    defl_f    = math.asin(min(1.0, max(0.0, (sep_std - dcpa_f) / range_f)))
                    t_react_f = max(5.0, math.degrees(defl_f) / TURN_RATE)
                    abs_tLOS  = FWD_S + max(t_f, 0.0)
                    margin_f  = abs_tLOS - t_react_f
                    fwd_score += FWD_W / max(margin_f, 1.0)

        # ── Compose ───────────────────────────────────────────────────────────
        if best_score > 0.0:
            boost   = min(n_conflicts - 1, 3) * 0.05   # additive, capped at +0.15
            urgency = best_score + boost + fwd_score
        else:
            urgency = fwd_score

        if urgency == 0.0:
            dest_bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                float(self._destination_ll[cs][0]), float(self._destination_ll[cs][1])
            )
            bearing_diff = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
            urgency = (1.0 - math.cos(math.radians(bearing_diff))) / 2.0 * 0.001

        return urgency

    def _register_airspace(self):
        flat_latlon = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            flat_latlon += [float(lat), float(lon)]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat_latlon)

    # ── Step helpers ──────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx):
        reward = 0.0

        # Global LoS penalty — ATCO is responsible for the whole sector
        reward += CONFIG['pen_los'] * len(bs.traf.cd.lospairs)

        # Focus aircraft specific penalties
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
                    t_entry_r, _ = _compute_tLOS(
                        dx, dy, vn_int - vn_own, ve_int - ve_own, CONFIG['sep_nm'])
                    if t_entry_r is not None:
                        tLOS_s  = max(t_entry_r, 0.0)
                        reward += CONFIG['pen_conflict'] / max(tLOS_s, 1.0)

                dest_bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    float(self._destination_ll[acting_cs][0]),
                    float(self._destination_ll[acting_cs][1])
                )
                bearing_diff  = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
                drift_factor  = (1.0 - math.cos(math.radians(bearing_diff))) / 2.0
                reward       += CONFIG['pen_drift'] * drift_factor

        if action_idx not in (2, 5):
            reward += CONFIG['pen_action']

        return float(reward)

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

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _process_pending_spawns(self):
        """Decrement per-slot countdown; spawn any slot whose counter hits zero."""
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
        dest_dist_nm = (math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
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
                    aclat=float(aircraft['spawn_ll'][0]), aclon=float(aircraft['spawn_ll'][1]),
                    achdg=float(aircraft['heading']),     acspd=CONFIG['ac_speed'],
                    acalt=CONFIG['altitude'])
        # Explicitly command speed and altitude so the performance model does not
        # override the initial value during the sim steps before RL control begins.
        bs.stack.stack(f'SPD {cs} {int(CONFIG["ac_speed"])}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]    = aircraft['dest_ll']
        self._commanded_heading[cs] = float(aircraft['heading'])
        self._slots[slot_idx]       = cs
        self._active_callsigns.add(cs)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx   = bs.traf.id2idx(cs)
        delta = HEADING_DELTAS[action_idx]
        if delta is None:
            if idx >= 0:
                dest_bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    float(self._destination_ll[cs][0]), float(self._destination_ll[cs][1])
                )
                self._commanded_heading[cs] = float(dest_bearing)
        else:
            self._commanded_heading[cs] += delta
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

        dest_bearing, _ = geo.kwikqdrdist(own_lat, own_lon,
                                           float(self._destination_ll[cs][0]),
                                           float(self._destination_ll[cs][1]))
        bearing_diff = wrap_to_180(dest_bearing - own_hdg)
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

        # Global sector summary — scales to any number of aircraft
        n = len(self._active_callsigns)
        n_pairs     = max(n * (n - 1) / 2, 1)
        n_los       = len(bs.traf.cd.lospairs)
        n_conflicts = len(getattr(bs.traf.cd, 'confpairs', []))
        obs += [n_los / n_pairs, n_conflicts / n_pairs]

        return np.array(obs, dtype=np.float32)

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _exit_penalty(self, cs):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        dest_bearing, _ = geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                           float(self._destination_ll[cs][0]),
                                           float(self._destination_ll[cs][1]))
        bearing_diff = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
        return CONFIG['pen_wrong_exit'] if abs(bearing_diff) > 90.0 else 0.0

    def _find_exited_aircraft(self):
        exited = []
        for cs in list(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs)
                continue
            inside = bs.tools.areafilter.checkInside(
                'AIRSPACE',
                np.array([bs.traf.lat[idx]]), np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]),
            )
            if not inside[0]:
                exited.append(cs)
        return exited

    def _blank_stats(self):
        return {'total_reward': 0.0, 'los_steps': 0, 'steps': 0, 'actions': []}


# ── Pygame visualisation (aircraft fly straight — no policy) ──────────────────

if __name__ == '__main__':
    import sys
    import pygame

    N_EPISODES  = 3
    WINDOW_SIZE = 850
    FPS         = 8

    SLOT_COLORS = [
        (50,  180,  50), (50,  130, 220), (255, 140,   0), (150,  50, 200),
        (0,   200, 200), (220, 200,   0), (220,  30,  30), (160, 160, 160),
    ]
    RED, BLACK, WHITE = (220, 30, 30), (0, 0, 0), (255, 255, 255)
    FOCUS_COLOR       = (255, 255, 0)   # yellow highlight for focus aircraft

    class _View:
        def __init__(self, polygon_nm, w, h):
            pad  = CONFIG['sep_nm'] * NM_TO_KM * 2.0
            km   = polygon_nm * NM_TO_KM
            span = max(km[:, 0].max() - km[:, 0].min() + 2 * pad,
                       km[:, 1].max() - km[:, 1].min() + 2 * pad)
            self._cx = km[:, 0].mean()
            self._cy = km[:, 1].mean()
            self._sc = min(w, h) / span
            self._w  = w
            self._h  = h

        def nm_to_px(self, x, y):
            return (int((x * NM_TO_KM - self._cx) *  self._sc + self._w / 2),
                    int((y * NM_TO_KM - self._cy) * -self._sc + self._h / 2))

        def latlon_to_px(self, lat, lon):
            nm = latlon_to_nm(CONFIG['center_ll'], lat, lon)
            return self.nm_to_px(nm[0], nm[1])

        def nm_to_px_len(self, nm):
            return max(1, int(nm * NM_TO_KM * self._sc))

    def draw_dashed(surface, color, p1, p2, dash=8, gap=5):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        total  = math.hypot(dx, dy)
        if total < 1:
            return
        ux, uy  = dx / total, dy / total
        x, y    = float(p1[0]), float(p1[1])
        drawn   = 0.0
        drawing = True
        while drawn < total:
            seg = min(dash if drawing else gap, total - drawn)
            if drawing:
                pygame.draw.line(surface, color,
                                 (round(x), round(y)),
                                 (round(x + ux * seg), round(y + uy * seg)), 1)
            x += ux * seg; y += uy * seg; drawn += seg; drawing = not drawing

    env = AirspaceEnv()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('v3_env — straight flight (no policy)')
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 13)
    fsmall = pygame.font.SysFont('monospace', 11)

    for episode in range(N_EPISODES):
        obs, _     = env.reset()
        n_agents   = env.n_aircraft
        view       = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_px     = view.nm_to_px_len(CONFIG['sep_nm'] / 2)
        dest_by_cs = {cs: view.latlon_to_px(*env._destination_ll[cs])
                      for cs in env._active_callsigns}

        total_reward = 0.0
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            obs, reward, terminated, truncated, info = env.step(2)  # hold action
            total_reward += reward
            done   = terminated or truncated
            step  += 1
            if info['los_pairs']:
                los_steps += 1

            for cs in env._active_callsigns:
                if cs not in dest_by_cs and cs in env._destination_ll:
                    dest_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)

            los_set    = {cs for pair in bs.traf.cd.lospairs for cs in pair}
            focus_cs   = env._focus_cs

            for slot_i, cs in enumerate(env._slots):
                if cs is None or cs not in env._active_callsigns:
                    continue
                idx = bs.traf.id2idx(cs)
                if idx < 0:
                    continue
                px, py    = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                is_focus  = cs == focus_cs
                is_los    = cs in los_set
                color     = RED if is_los else (FOCUS_COLOR if is_focus else SLOT_COLORS[slot_i % len(SLOT_COLORS)])
                ring_r    = sep_px + 4 if is_focus else sep_px
                pygame.draw.circle(screen, color, (px, py), ring_r, 2 if is_focus else 1)
                if cs in dest_by_cs:
                    draw_dashed(screen, color, (px, py), dest_by_cs[cs])
                pygame.draw.circle(screen, color, (px, py), 5)
                screen.blit(fsmall.render(cs + (' ◄' if is_focus else ''), True, color),
                            (px + 7, py - 7))

            hud = [
                f'Episode {episode + 1}/{N_EPISODES}  ({n_agents} ac)  [straight flight]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Focus: {focus_cs}   served={env._next_callsign_id}   active={len(env._active_callsigns)}',
                f'Total reward={total_reward:.1f}',
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            legend_y = WINDOW_SIZE - 16 - n_agents * 14
            for i in range(n_agents):
                screen.blit(fsmall.render(f'● Slot {i}', True, SLOT_COLORS[i % len(SLOT_COLORS)]),
                            (8, legend_y + i * 14))
            screen.blit(fsmall.render('● YELLOW = focus aircraft', True, FOCUS_COLOR),
                        (8, WINDOW_SIZE - 26))
            screen.blit(fsmall.render('● RED = separation violation', True, RED),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(FPS)

        print(f'Episode {episode + 1:2d}  n_ac={n_agents}  steps={step}  '
              f'total_reward={total_reward:.2f}  LoS-steps={los_steps}  '
              f'served={env._next_callsign_id}')

    pygame.quit()
