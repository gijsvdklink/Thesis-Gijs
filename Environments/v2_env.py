"""
bs_complex_no_delay_v2 — BlueSky MARL conflict-resolution environment.

Fixed aircraft count throughout the episode.  All n_ac aircraft spawn at
reset and every exit is immediately replaced by a fresh aircraft, keeping
the count constant.  Episode ends only when max_steps is reached.

Action space (Discrete 6):
  0 → -20°   1 → -10°   2 → hold   3 → +10°   4 → +20°   5 → back to WP

Observation per agent (18 floats — fixed regardless of n_aircraft):
  Ownship      : cos(bearing_to_dest - heading), sin(bearing_to_dest - heading)
  4 neighbours : rel_fwd, rel_right, tcpa_norm, dcpa_norm  (nearest first, zero-padded)

Reward — penalties only:
  pen_los        : fixed per step when in loss of separation
  pen_conflict   : -w / max(tcpa_s, 1)  per active conflict pair involving ownship
  pen_drift      : -w * (1 - cos(bearing_to_dest - heading)) / 2  per step
                   (0 when aligned, 0.5 at 90 degrees off, 1.0 when pointing away)
  pen_action     : -w for heading change actions only (±10°, ±20°); hold and back-to-WP are free
  pen_wrong_exit : applied on exit when aircraft is heading away from destination

Run directly to visualise with aircraft flying straight (no policy):
  python bs_complex_no_delay_v2.py
"""

import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.tools import geo
from polygenerator import random_convex_polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    'ac_type':             'A320',
    'ac_speed':            350.0,           # knots
    'altitude':            350,             # flight level
    'center_ll':           (52.0, 4.5),     # sector centre (lat, lon)
    'area_km2':            4500, #lambda: random.uniform(4000.0, 40000.0),
    'density_km2':         1500, #lambda: random.uniform(1500.0, 4000.0),
    'max_agents':          10,
    'n_neighbours':        4,
    'n_vertices':          lambda: random.randint(5, 12),
    'min_circularity':     0.70,
    'sep_nm':              5.0,             # separation standard (NM)
    'buffer_nm':           3.0,             # extra buffer for spawn clearance
    'spawn_jitter':        lambda: random.uniform(0.1, 0.9),
    'ref_jitter':          lambda: random.uniform(-0.15, 0.15),
    'dest_dist_factor':    2.0,
    'max_placement_tries': 50,
    'seed':                None,
    'sim_dt':              0.5,             # simulation timestep (seconds)
    'action_freq':         10,              # sim steps per RL step: 10 * 0.5s = 5s
    'crossings_per_episode':             5,              # episode length = crossings_per_episode × avg sector crossing time
    'spatial_scale':       50.0,            # NM — normalises relative positions
    'lookahead_s':         300.0,           # seconds — normalises TCPA
    'pen_los':            -50.0,            # per step in loss of separation
    'pen_conflict':        -3.0,            # * (1 / max(tcpa_s, 1)) per conflict pair
    'pen_drift':           -1.0,            # * (1 - cos(bearing_to_dest - heading)) / 2
    'pen_action':          -0.1,            # per non-zero heading change
    'pen_wrong_exit':     -20.0,            # when exiting heading away from destination
}

# ── Derived constants ─────────────────────────────────────────────────────────

NM_TO_KM  = 1.852
KM_TO_NM  = 1.0 / NM_TO_KM

MAX_AGENTS   = CONFIG['max_agents']
N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 2 + N_NEIGHBOURS * 4        # 18

HEADING_DELTAS = [-20, -10, 0, 10, 20, None]   # None = snap back to WP bearing

_bs_initialized = False


# ── Coordinate helpers ────────────────────────────────────────────────────────

def nm_to_latlon(center_ll, x_nm, y_nm):
    """Local NM offsets (x east, y north) → (lat, lon)."""
    clat, clon = center_ll
    return clat + y_nm / 60.0, clon + x_nm / (60.0 * math.cos(math.radians(clat)))


def latlon_to_nm(center_ll, lat, lon):
    """(lat, lon) → local NM offsets (x east, y north) as numpy array."""
    clat, clon = center_ll
    return np.array([
        (lon - clon) * 60.0 * math.cos(math.radians(clat)),
        (lat - clat) * 60.0,
    ])


def wrap_to_180(angle_deg):
    """Wrap angle to (-180, +180]."""
    return (angle_deg + 180) % 360 - 180


# ── Polygon generation ────────────────────────────────────────────────────────

def make_polygon(area_km2, config):
    """Random convex polygon with given area, centred at NM origin."""
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
    """Try placing one aircraft in the given boundary sector."""
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
    """True if spawn_ll is within min_dist_nm of any already-placed aircraft."""
    return any(
        geo.kwikdist(float(spawn_ll[0]), float(spawn_ll[1]),
                     float(ac['spawn_ll'][0]), float(ac['spawn_ll'][1])) < min_dist_nm
        for ac in placed_aircraft
    )


def place_aircraft(polygon, n_aircraft, config):
    """
    Place n_aircraft on the polygon boundary using equal-arc sectors.
    Returns a list of aircraft dicts, or None if any sector fails.
    """
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
    """Ground speed of aircraft as (v_north, v_east) in NM/s."""
    spd = bs.traf.gs[idx] / 1852.0
    hdg = math.radians(bs.traf.hdg[idx])
    return spd * math.cos(hdg), spd * math.sin(hdg)


def _relative_ego(own_nm, int_nm, own_hdg_deg, scale):
    """
    Intruder relative position in ownship ego frame (forward, right), normalised by scale.
    own_nm / int_nm: [east_nm, north_nm].
    """
    dx    = int_nm[0] - own_nm[0]   # east difference
    dy    = int_nm[1] - own_nm[1]   # north difference
    sin_h = math.sin(math.radians(own_hdg_deg))
    cos_h = math.cos(math.radians(own_hdg_deg))
    fwd   = ( dx * sin_h + dy * cos_h) / scale   # along heading direction
    right = ( dx * cos_h - dy * sin_h) / scale   # perpendicular right
    return fwd, right


def _tcpa_dcpa_norm(dx_nm, dy_nm, dvn, dve, separation_nm):
    """
    Normalised TCPA in [0, 1] and DCPA in [0, 3].
    dx_nm: east separation (NM).  dy_nm: north separation (NM).
    dvn: north relative velocity (NM/s).  dve: east relative velocity (NM/s).
    """
    rel_v_sq = dvn ** 2 + dve ** 2
    if rel_v_sq < 1e-12:
        return 1.0, min(separation_nm / CONFIG['sep_nm'], 3.0)
    dot_pv = dx_nm * dve + dy_nm * dvn   # r . v_rel  (east*east_vel + north*north_vel)
    tcpa   = max(0.0, -dot_pv / rel_v_sq)
    dcpa   = math.sqrt(max(0.0, dx_nm ** 2 + dy_nm ** 2 - dot_pv ** 2 / rel_v_sq))
    return min(tcpa / CONFIG['lookahead_s'], 1.0), min(dcpa / CONFIG['sep_nm'], 3.0)


def _intruder_obs(own_idx, int_idx, spatial_scale):
    """
    Observation features for one intruder from the ownship's perspective.
    Returns (separation_nm, [rel_fwd, rel_right, tcpa_norm, dcpa_norm]).
    """
    own_lat, own_lon = bs.traf.lat[own_idx], bs.traf.lon[own_idx]
    int_lat, int_lon = bs.traf.lat[int_idx],  bs.traf.lon[int_idx]
    own_nm = latlon_to_nm(CONFIG['center_ll'], own_lat, own_lon)
    int_nm = latlon_to_nm(CONFIG['center_ll'], int_lat, int_lon)

    _, separation_nm   = geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
    rel_fwd, rel_right = _relative_ego(own_nm, int_nm, bs.traf.hdg[own_idx], spatial_scale)

    vn_own, ve_own = _aircraft_vel_nm_s(own_idx)
    vn_int, ve_int = _aircraft_vel_nm_s(int_idx)
    dx = int_nm[0] - own_nm[0]   # east
    dy = int_nm[1] - own_nm[1]   # north
    tcpa_norm, dcpa_norm = _tcpa_dcpa_norm(dx, dy, vn_int - vn_own, ve_int - ve_own, separation_nm)

    return separation_nm, [rel_fwd, rel_right, tcpa_norm, dcpa_norm]


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
        self._spatial_scale     = CONFIG['spatial_scale']
        self.polygon            = None
        self._polygon_shape     = None

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

        # Retry until a valid polygon and aircraft placement is found
        while True:
            polygon          = make_polygon(area_km2, CONFIG)
            initial_aircraft = place_aircraft(polygon, n_ac, CONFIG)
            if initial_aircraft is not None:
                break

        # Compute episode length from how long aircraft take to cross the sector
        minx, miny, maxx, maxy = polygon.bounds
        diameter_nm     = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
        crossing_time_s = diameter_nm / CONFIG['ac_speed'] * 3600
        steps_per_wave  = crossing_time_s / (CONFIG['action_freq'] * CONFIG['sim_dt'])
        self._max_steps     = max(50, round(CONFIG['crossings_per_episode'] * steps_per_wave))
        self._spatial_scale = diameter_nm


        self._polygon_shape     = polygon
        self.polygon            = np.array(polygon.exterior.coords[:-1])
        self.n_aircraft         = n_ac
        self._slots             = [None] * n_ac
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._commanded_heading = {}
        self._next_callsign_id  = 0
        self._step_count        = 0

        self._register_airspace()

        for i, ac in enumerate(initial_aircraft):
            self._spawn_aircraft(i, ac)

        bs.stack.stack('ASAS ON')

        return self._get_all_observations(), {}

    def step(self, actions):
        for slot_i, cs in enumerate(self._slots):
            if cs is not None:
                self._apply_action(cs, int(actions[slot_i]))

        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        self._step_count += 1

        rewards = self._compute_rewards(actions)
        self._process_exits(rewards)

        terminated = False
        truncated  = self._step_count >= self._max_steps
        info = {
            'los_pairs':  list(bs.traf.cd.lospairs),
            'active':     list(self._active_callsigns),
            'n_aircraft': self.n_aircraft,
        }
        return self._get_all_observations(), rewards, terminated, truncated, info

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _register_airspace(self):
        """Register the polygon boundary with BlueSky's area filter."""
        flat_latlon = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            flat_latlon += [float(lat), float(lon)]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat_latlon)

    # ── Step helpers ──────────────────────────────────────────────────────────

    def _compute_rewards(self, actions):
        rewards = np.zeros(self.n_aircraft, dtype=np.float32)
        for slot_i, cs in enumerate(self._slots):
            if cs is not None:
                rewards[slot_i] = self._step_reward(cs, int(actions[slot_i]))
        return rewards

    def _process_exits(self, rewards):
        """
        Remove exited aircraft and immediately spawn a replacement in the same slot.
        This keeps the active aircraft count constant throughout the episode.
        """
        for cs in self._find_exited_aircraft():
            slot_i = self._slots.index(cs)
            rewards[slot_i] += self._exit_penalty(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot_i] = None
            replacement = self._generate_replacement()
            if replacement is not None:
                self._spawn_aircraft(slot_i, replacement)

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _generate_replacement(self):
        """
        Generate a fresh aircraft on the boundary, pre-cleared of active traffic.
        Checks against current active positions so the aircraft can spawn immediately.
        """
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
        """Current (lat, lon) for all active aircraft."""
        return [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]

    def _spawn_aircraft(self, slot_idx, aircraft):
        """Create a new aircraft in BlueSky and register it in the given slot."""
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(aircraft['spawn_ll'][0]), aclon=float(aircraft['spawn_ll'][1]),
                    achdg=float(aircraft['heading']),     acspd=CONFIG['ac_speed'],
                    acalt=CONFIG['altitude'])
        self._destination_ll[cs]    = aircraft['dest_ll']
        self._commanded_heading[cs] = float(aircraft['heading'])
        self._slots[slot_idx]       = cs
        self._active_callsigns.add(cs)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx   = bs.traf.id2idx(cs)
        delta = HEADING_DELTAS[action_idx]
        if delta is None:
            # Snap back to current bearing toward destination
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

    def _ownship_obs(self, cs, own_idx):
        """2 floats: direction to destination in ego frame."""
        own_lat, own_lon = bs.traf.lat[own_idx], bs.traf.lon[own_idx]
        own_hdg          = bs.traf.hdg[own_idx]
        dest_ll          = self._destination_ll[cs]

        dest_bearing, _  = geo.kwikqdrdist(own_lat, own_lon, float(dest_ll[0]), float(dest_ll[1]))
        bearing_diff     = wrap_to_180(dest_bearing - own_hdg)

        return [math.cos(math.radians(bearing_diff)),
                math.sin(math.radians(bearing_diff))]

    def _get_agent_observation(self, cs):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        obs = self._ownship_obs(cs, own_idx)

        neighbours = sorted(
            [_intruder_obs(own_idx, bs.traf.id2idx(other), self._spatial_scale)
             for other in self._active_callsigns
             if other != cs and bs.traf.id2idx(other) >= 0],
            key=lambda x: x[0],
        )
        for k in range(N_NEIGHBOURS):
            obs += neighbours[k][1] if k < len(neighbours) else [0.0, 0.0, 1.0, 3.0]

        return np.array(obs, dtype=np.float32)

    def _get_all_observations(self):
        return np.stack([
            self._get_agent_observation(cs) if cs is not None else np.zeros(OBS_DIM, dtype=np.float32)
            for cs in self._slots
        ])

    # ── Reward ────────────────────────────────────────────────────────────────

    def _step_reward(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0

        reward   = 0.0
        own_lat  = bs.traf.lat[idx]
        own_lon  = bs.traf.lon[idx]
        own_nm   = latlon_to_nm(CONFIG['center_ll'], own_lat, own_lon)
        vn_own, ve_own = _aircraft_vel_nm_s(idx)

        # Loss-of-separation penalty
        in_los = cs in {ac for pair in bs.traf.cd.lospairs for ac in pair}
        if in_los:
            reward += CONFIG['pen_los']

        # Conflict penalty: proportional to urgency (1 / tcpa) per conflicting pair
        for other_cs in self._active_callsigns:
            if other_cs == cs:
                continue
            other_idx = bs.traf.id2idx(other_cs)
            if other_idx < 0:
                continue
            int_nm         = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[other_idx], bs.traf.lon[other_idx])
            vn_int, ve_int = _aircraft_vel_nm_s(other_idx)
            dx = int_nm[0] - own_nm[0]
            dy = int_nm[1] - own_nm[1]
            tcpa_norm, dcpa_norm = _tcpa_dcpa_norm(dx, dy, vn_int - vn_own, ve_int - ve_own, CONFIG['sep_nm'])
            if dcpa_norm < 1.0:   # predicted miss distance is within separation standard
                tcpa_s = tcpa_norm * CONFIG['lookahead_s']
                reward += CONFIG['pen_conflict'] / max(tcpa_s, 1.0)

        # Drift penalty: 0 when aligned, 0.5 at 90 degrees, 1.0 when pointing away
        dest_bearing, _ = geo.kwikqdrdist(own_lat, own_lon,
                                           float(self._destination_ll[cs][0]),
                                           float(self._destination_ll[cs][1]))
        bearing_diff        = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
        drift_penalty_factor = (1.0 - math.cos(math.radians(bearing_diff))) / 2.0
        reward += CONFIG['pen_drift'] * drift_penalty_factor

        # Action penalty: any non-zero heading change
        if action_idx not in (2, 5):   # hold (0°) and back-to-WP are free
            reward += CONFIG['pen_action']

        return float(reward)

    def _exit_penalty(self, cs):
        """Heavy penalty when aircraft exits heading more than 90 degrees from destination."""
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        dest_bearing, _ = geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                           float(self._destination_ll[cs][0]),
                                           float(self._destination_ll[cs][1]))
        bearing_diff = wrap_to_180(dest_bearing - bs.traf.hdg[idx])
        return CONFIG['pen_wrong_exit'] if abs(bearing_diff) > 90.0 else 0.0

    def _find_exited_aircraft(self):
        """Callsigns that have left the airspace polygon."""
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


# ── MARL VecEnv wrapper for SB3 ──────────────────────────────────────────────

class MARLVecEnv(VecEnv):
    """Single-process VecEnv presenting MAX_AGENTS virtual environments to SB3."""

    def __init__(self):
        self._env     = AirspaceEnv()
        self._n_slots = MAX_AGENTS
        self._stats   = self._blank_stats()
        super().__init__(MAX_AGENTS, self._env.observation_space, self._env.action_space)

    def reset(self):
        obs, _        = self._env.reset()
        self._n_slots = self._env.n_aircraft
        self._stats   = self._blank_stats()
        return self._pad_obs(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self._env.step(self._actions[:self._n_slots])
        done  = terminated or truncated
        dones = np.full(MAX_AGENTS, done)

        padded_obs     = self._pad_obs(obs)
        padded_rewards = self._pad_rewards(rewards)

        self._stats['rewards'][:self._n_slots] += rewards
        self._stats['los_steps'] += int(len(info['los_pairs']) > 0)
        self._stats['steps']     += 1
        self._stats['actions'].extend(self._actions[:self._n_slots].tolist())

        infos = [{} for _ in range(MAX_AGENTS)]
        if done:
            for i in range(MAX_AGENTS):
                infos[i]['terminal_observation'] = padded_obs[i]
            infos[0].update({
                'mean_episode_reward': float(self._stats['rewards'][:self._n_slots].mean()),
                'ep_los_steps':        self._stats['los_steps'],
                'ep_length':           self._stats['steps'],
                'n_aircraft':          self._n_slots,
                'action_distribution': np.bincount(self._stats['actions'], minlength=6).tolist(),
            })
            obs_new, _    = self._env.reset()
            self._n_slots = self._env.n_aircraft
            padded_obs    = self._pad_obs(obs_new)
            self._stats   = self._blank_stats()
        return padded_obs, padded_rewards, dones, infos

    def close(self): pass
    def get_attr(self, _attr_name, _indices=None):            return [None]  * self.num_envs
    def set_attr(self, _attr_name, _value, _indices=None):    pass
    def env_method(self, _method_name, *_args, **_kwargs):    return [None]  * self.num_envs
    def env_is_wrapped(self, _wrapper_class, _indices=None):  return [False] * self.num_envs
    def seed(self, _seed=None):                               return [None]  * self.num_envs

    def _pad_obs(self, obs):
        out = np.zeros((MAX_AGENTS, OBS_DIM), dtype=np.float32)
        out[:len(obs)] = obs
        return out

    def _pad_rewards(self, rewards):
        out = np.zeros(MAX_AGENTS, dtype=np.float32)
        out[:len(rewards)] = rewards
        return out

    def _blank_stats(self):
        return {'rewards': np.zeros(MAX_AGENTS), 'los_steps': 0, 'steps': 0, 'actions': []}


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
    pygame.display.set_caption('bs_complex_no_delay_v2 — straight flight (no policy)')
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

        total_rewards = np.zeros(n_agents)
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            # Hold action for all slots — aircraft fly straight
            actions = np.full(n_agents, 2, dtype=int)

            obs, rewards, terminated, truncated, info = env.step(actions)
            total_rewards += rewards
            done   = terminated or truncated
            step  += 1
            if info['los_pairs']:
                los_steps += 1

            for cs in env._active_callsigns:
                if cs not in dest_by_cs and cs in env._destination_ll:
                    dest_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)

            los_set = {cs for pair in bs.traf.cd.lospairs for cs in pair}
            for slot_i, cs in enumerate(env._slots):
                if cs is None or cs not in env._active_callsigns:
                    continue
                idx = bs.traf.id2idx(cs)
                if idx < 0:
                    continue
                px, py = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                color  = RED if cs in los_set else SLOT_COLORS[slot_i % len(SLOT_COLORS)]
                pygame.draw.circle(screen, color, (px, py), sep_px, 1)
                if cs in dest_by_cs:
                    draw_dashed(screen, color, (px, py), dest_by_cs[cs])
                pygame.draw.circle(screen, color, (px, py), 5)
                screen.blit(fsmall.render(cs, True, color), (px + 7, py - 7))

            hud = [
                f'Episode {episode + 1}/{N_EPISODES}  ({n_agents} slots)  [straight flight]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Served {env._next_callsign_id}   active={len(env._active_callsigns)}',
                '  '.join(f'S{i}={total_rewards[i]:.0f}' for i in range(n_agents)),
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            legend_y = WINDOW_SIZE - 16 - n_agents * 14
            for i in range(n_agents):
                screen.blit(
                    fsmall.render(f'● Slot {i}', True, SLOT_COLORS[i % len(SLOT_COLORS)]),
                    (8, legend_y + i * 14),
                )
            screen.blit(fsmall.render('● RED = separation violation', True, RED),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(FPS)

        print(f'Episode {episode + 1:2d}  n_ac={n_agents}  steps={step}  '
              f'mean_reward={total_rewards.mean():.2f}  LoS-steps={los_steps}  '
              f'served={env._next_callsign_id}')

    pygame.quit()
