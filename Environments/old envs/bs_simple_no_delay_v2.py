"""
bs_simple_no_delay_v2 — BlueSky MARL conflict-resolution environment (simple).

Fixed airspace (3 000 km²), 3 slots, 3 waves × 3 aircraft = 9 total.
Aircraft enter one-by-one with a staggered initial entry. As each aircraft exits,
the next one from the pool enters its freed slot (individual replacement).

Action space (Discrete 4):
  0 → -10°   1 → +10°   2 → hold   3 → snap back to original heading

Observation per agent (15 floats):
  Ownship      : dist_to_dest, cos_dest_bearing, sin_dest_bearing, cos_drift, sin_drift
  2 neighbours : rel_fwd, rel_right, tcpa_norm, dcpa_norm, dist_norm  (nearest first)

Run directly to visualise a random policy:
  python bs_simple_no_delay_v2.py
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
    'ac_type':            'A320',
    'ac_speed':           150.0,
    'altitude':           350,
    'center_ll':          (52.0, 4.5),
    'n_aircraft':         3,
    'area_km2':           3000.0,
    'n_vertices':         lambda: random.randint(5, 12),
    'min_circularity':    0.70,
    'sep_nm':             5.0,
    'buffer_nm':          2.0,
    'spawn_jitter':       lambda: random.uniform(0.1, 0.9),
    'ref_jitter':         lambda: random.uniform(-0.15, 0.15),
    'dest_dist_factor':   2.0,
    'max_placement_tries': 50,
    'seed':               None,
    'sim_dt':             0.5,
    'action_freq':        10,
    'max_steps':          1500,
    'n_waves':            3,
    'min_entry_gap':      5,
    'max_entry_gap':      20,
    'spatial_scale':      50.0,
    'lookahead_s':        300.0,
    'w_progress':         1.0,
    'w_sep':              6.0,
    'w_action':           0.05,
    'w_time':            -0.05,
    'intrusion_pen':     -15.0,
    'exit_correct':       5.0,
}

# ── Derived constants ─────────────────────────────────────────────────────────

NM_TO_KM  = 1.852
KM_TO_NM  = 1.0 / NM_TO_KM

N_AIRCRAFT  = CONFIG['n_aircraft']
N_WAVES     = CONFIG['n_waves']
N_NEIGHBOURS = N_AIRCRAFT - 1   # 2
OBS_DIM     = 5 + N_NEIGHBOURS * 5   # 15

STEP_DIST_NM          = CONFIG['ac_speed'] * CONFIG['sim_dt'] * CONFIG['action_freq'] / 1852.0
PROGRESS_REWARD_SCALE = CONFIG['w_progress'] / STEP_DIST_NM

HEADING_DELTAS = [-10, 10, 0, None]   # action index → heading change (None = snap back)

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
    """Wrap angle to (−180, +180]."""
    return (angle_deg + 180) % 360 - 180


# ── Polygon generation ────────────────────────────────────────────────────────

def make_polygon(area_km2, config):
    """Random convex polygon with given area, centred at NM origin."""
    target_area_nm2 = area_km2 * KM_TO_NM ** 2
    while True:
        raw     = ShapelyPolygon(random_convex_polygon(config['n_vertices']()))
        scaled  = shapely_scale(raw, xfact=math.sqrt(target_area_nm2 / raw.area),
                                yfact=math.sqrt(target_area_nm2 / raw.area), origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= config['min_circularity']:
            cx, cy = scaled.centroid.x, scaled.centroid.y
            return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])


# ── Aircraft placement ────────────────────────────────────────────────────────

def _try_place_one(polygon, sector, n_sectors, dest_dist_nm, config):
    """
    Try placing one aircraft in the given boundary sector.
    t_spawn and t_ref are normalised arc positions on the polygon perimeter.
    """
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


# ── Observation helpers (module-level pure functions) ─────────────────────────

def _aircraft_vel_nm_s(idx):
    """Ground speed of aircraft as (v_north, v_east) in NM/s."""
    spd = bs.traf.gs[idx] / 1852.0
    hdg = math.radians(bs.traf.hdg[idx])
    return spd * math.cos(hdg), spd * math.sin(hdg)


def _relative_ego(own_nm, int_nm, own_hdg_deg, scale):
    """Intruder relative position in ownship ego frame (forward, right), divided by scale."""
    dx    = int_nm[0] - own_nm[0]
    dy    = int_nm[1] - own_nm[1]
    cos_h = math.cos(math.radians(own_hdg_deg))
    sin_h = math.sin(math.radians(own_hdg_deg))
    return ( dx * cos_h + dy * sin_h) / scale, (-dx * sin_h + dy * cos_h) / scale


def _tcpa_dcpa_norm(dx_nm, dy_nm, dvn, dve, separation_nm):
    """
    Normalised TCPA in [0, 1] and DCPA in [0, 3].
    dx/dy: relative position (NM, north/east). dvn/dve: relative velocity (NM/s).
    """
    rel_v_sq = dvn ** 2 + dve ** 2
    if rel_v_sq < 1e-12:
        return 1.0, min(separation_nm / CONFIG['sep_nm'], 3.0)
    dot_pv = dx_nm * dvn + dy_nm * dve
    tcpa   = max(0.0, -dot_pv / rel_v_sq)
    dcpa   = math.sqrt(max(0.0, dx_nm ** 2 + dy_nm ** 2 - dot_pv ** 2 / rel_v_sq))
    return min(tcpa / CONFIG['lookahead_s'], 1.0), min(dcpa / CONFIG['sep_nm'], 3.0)


def _intruder_obs(own_idx, int_idx):
    """
    Observation features for one intruder from the ownship's perspective.
    Returns (separation_nm, [rel_fwd, rel_right, tcpa_norm, dcpa_norm, dist_norm]).
    """
    own_lat, own_lon = bs.traf.lat[own_idx], bs.traf.lon[own_idx]
    int_lat, int_lon = bs.traf.lat[int_idx],  bs.traf.lon[int_idx]
    own_nm = latlon_to_nm(CONFIG['center_ll'], own_lat, own_lon)
    int_nm = latlon_to_nm(CONFIG['center_ll'], int_lat, int_lon)

    _, separation_nm   = geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
    rel_fwd, rel_right = _relative_ego(own_nm, int_nm, bs.traf.hdg[own_idx], CONFIG['spatial_scale'])

    vn_own, ve_own = _aircraft_vel_nm_s(own_idx)
    vn_int, ve_int = _aircraft_vel_nm_s(int_idx)
    dx, dy = int_nm[0] - own_nm[0], int_nm[1] - own_nm[1]
    tcpa_norm, dcpa_norm = _tcpa_dcpa_norm(dx, dy, vn_int - vn_own, ve_int - ve_own, separation_nm)

    return separation_nm, [rel_fwd, rel_right, tcpa_norm, dcpa_norm, separation_nm / CONFIG['spatial_scale']]


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
        self.action_space      = spaces.Discrete(4)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._slots             = [None] * N_AIRCRAFT
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._original_heading  = {}
        self._commanded_heading = {}
        self._next_callsign_id  = 0
        self._slot_queues       = [[] for _ in range(N_AIRCRAFT)]  # per-slot replacement queues
        self._pending_spawns    = []   # (step_t, aircraft_dict) — time- and sep-gated
        self._step_count        = 0
        self.polygon            = None

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)
        bs.traf.reset()

        polygon = make_polygon(CONFIG['area_km2'], CONFIG)
        waves   = self._generate_all_aircraft(polygon)

        self.polygon            = np.array(polygon.exterior.coords[:-1])
        self._step_count        = 0
        self._slots             = [None] * N_AIRCRAFT
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._original_heading  = {}
        self._commanded_heading = {}
        self._next_callsign_id  = 0
        self._slot_queues       = [[waves[w][i] for w in range(1, N_WAVES)] for i in range(N_AIRCRAFT)]
        self._pending_spawns    = []

        self._register_airspace()

        # Stagger the initial N_AIRCRAFT entries across time
        t = 0
        for ac in waves[0]:
            self._pending_spawns.append((t, ac))
            t += random.randint(CONFIG['min_entry_gap'], CONFIG['max_entry_gap'])
        self._spawn_pending_aircraft()

        bs.stack.stack('ASAS ON')
        return self._get_all_observations(), {}

    def step(self, actions):
        for slot_i, callsign in enumerate(self._slots):
            if callsign is not None:
                self._apply_action(callsign, int(actions[slot_i]))

        prev_distances = self._record_distances()

        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        self._step_count += 1

        rewards = self._compute_rewards(actions, prev_distances)
        self._process_exits(rewards)
        self._spawn_pending_aircraft()

        terminated = (not any(self._slot_queues)
                      and not self._pending_spawns
                      and len(self._active_callsigns) == 0)
        truncated = self._step_count >= CONFIG['max_steps']
        info = {'los_pairs': list(bs.traf.cd.lospairs), 'active': list(self._active_callsigns)}
        return self._get_all_observations(), rewards, terminated, truncated, info

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _generate_all_aircraft(self, polygon):
        """Pre-generate all N_WAVES × N_AIRCRAFT aircraft for this episode."""
        waves = []
        for _ in range(N_WAVES):
            while True:
                wave = place_aircraft(polygon, N_AIRCRAFT, CONFIG)
                if wave is not None:
                    waves.append(wave)
                    break
        return waves

    def _register_airspace(self):
        """Register the polygon boundary with BlueSky's area filter."""
        flat_latlon = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            flat_latlon += [float(lat), float(lon)]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat_latlon)

    # ── Step helpers ──────────────────────────────────────────────────────────

    def _record_distances(self):
        """Distance to destination for each active aircraft, before the sim step."""
        distances = {}
        for cs in self._active_callsigns:
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                _, dist = geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                          float(self._destination_ll[cs][0]),
                                          float(self._destination_ll[cs][1]))
                distances[cs] = dist
        return distances

    def _compute_rewards(self, actions, prev_distances):
        """Per-slot step rewards from progress, separation, intrusion, time, and action cost."""
        rewards = np.zeros(N_AIRCRAFT, dtype=np.float32)
        for slot_i, cs in enumerate(self._slots):
            if cs is not None:
                rewards[slot_i] = self._step_reward(cs, prev_distances.get(cs, 0.0))
                if int(actions[slot_i]) in (0, 1):
                    rewards[slot_i] -= CONFIG['w_action']
        return rewards

    def _process_exits(self, rewards):
        """Apply exit reward, remove exited aircraft, and queue their replacements."""
        for cs in self._find_exited_aircraft():
            slot_i = self._slots.index(cs)
            rewards[slot_i] += self._exit_reward(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot_i] = None
            if self._slot_queues[slot_i]:
                self._pending_spawns.append((self._step_count, self._slot_queues[slot_i].pop(0)))

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_pending_aircraft(self):
        """Spawn pending aircraft that are time-ready and have enough separation."""
        occupied      = self._active_positions()
        still_pending = []

        for t, aircraft in self._pending_spawns:
            if t > self._step_count or not self._clear_to_spawn(aircraft, occupied):
                still_pending.append((t, aircraft))
                continue
            free_slots = [i for i, cs in enumerate(self._slots) if cs is None]
            if free_slots:
                self._spawn_aircraft(free_slots[0], aircraft)
                occupied.append((float(aircraft['spawn_ll'][0]), float(aircraft['spawn_ll'][1])))
            else:
                still_pending.append((t, aircraft))

        self._pending_spawns = still_pending

    def _active_positions(self):
        """Current (lat, lon) list for all active aircraft."""
        return [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]

    def _clear_to_spawn(self, aircraft, occupied_positions):
        """True if the aircraft's spawn point has ≥ 2× sep_nm from all occupied positions."""
        slat, slon = float(aircraft['spawn_ll'][0]), float(aircraft['spawn_ll'][1])
        nearest = min(
            (geo.kwikqdrdist(slat, slon, lat, lon)[1] for lat, lon in occupied_positions),
            default=float('inf'),
        )
        return nearest >= CONFIG['sep_nm'] * 2.0

    def _spawn_aircraft(self, slot_idx, aircraft):
        """Create a new aircraft in BlueSky and register it in the given slot."""
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(aircraft['spawn_ll'][0]), aclon=float(aircraft['spawn_ll'][1]),
                    achdg=float(aircraft['heading']),     acspd=CONFIG['ac_speed'],
                    acalt=CONFIG['altitude'])
        self._destination_ll[cs]    = aircraft['dest_ll']
        self._original_heading[cs]  = float(aircraft['heading'])
        self._commanded_heading[cs] = float(aircraft['heading'])
        self._slots[slot_idx]       = cs
        self._active_callsigns.add(cs)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        delta = HEADING_DELTAS[action_idx]
        if delta is None:
            self._commanded_heading[cs] = self._original_heading[cs]
        else:
            self._commanded_heading[cs] += delta
        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _ownship_obs(self, cs, own_idx):
        """5 floats: destination distance/bearing in ego frame + drift from spawn heading."""
        own_lat, own_lon, own_hdg = bs.traf.lat[own_idx], bs.traf.lon[own_idx], bs.traf.hdg[own_idx]
        dest_ll = self._destination_ll[cs]
        scale   = CONFIG['spatial_scale']

        dest_bearing, dest_dist_nm = geo.kwikqdrdist(
            own_lat, own_lon, float(dest_ll[0]), float(dest_ll[1])
        )
        dest_ego = wrap_to_180(dest_bearing - own_hdg)
        drift    = wrap_to_180(own_hdg - self._original_heading[cs])

        return [dest_dist_nm / scale,
                math.cos(math.radians(dest_ego)), math.sin(math.radians(dest_ego)),
                math.cos(math.radians(drift)),     math.sin(math.radians(drift))]

    def _get_agent_observation(self, cs):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        obs = self._ownship_obs(cs, own_idx)

        neighbours = sorted(
            [_intruder_obs(own_idx, bs.traf.id2idx(other))
             for other in self._active_callsigns
             if other != cs and bs.traf.id2idx(other) >= 0],
            key=lambda x: x[0],   # nearest first
        )
        for k in range(N_NEIGHBOURS):
            obs += neighbours[k][1] if k < len(neighbours) else [0.0, 0.0, 1.0, 3.0, 1.0]

        return np.array(obs, dtype=np.float32)

    def _get_all_observations(self):
        return np.stack([
            self._get_agent_observation(cs) if cs is not None else np.zeros(OBS_DIM, dtype=np.float32)
            for cs in self._slots
        ])

    # ── Reward ────────────────────────────────────────────────────────────────

    def _nearest_separation(self, cs, own_lat, own_lon):
        """Distance (NM) to the nearest other active aircraft."""
        dists = [
            geo.kwikqdrdist(own_lat, own_lon,
                            bs.traf.lat[bs.traf.id2idx(other)],
                            bs.traf.lon[bs.traf.id2idx(other)])[1]
            for other in self._active_callsigns
            if other != cs and bs.traf.id2idx(other) >= 0
        ]
        return min(dists, default=float('inf'))

    def _step_reward(self, cs, prev_dist_nm):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        own_lat, own_lon = bs.traf.lat[idx], bs.traf.lon[idx]
        _, current_dist_nm = geo.kwikqdrdist(own_lat, own_lon,
                                              float(self._destination_ll[cs][0]),
                                              float(self._destination_ll[cs][1]))
        progress  = PROGRESS_REWARD_SCALE * (prev_dist_nm - current_dist_nm)
        violation = max(0.0, 1.0 - self._nearest_separation(cs, own_lat, own_lon) / CONFIG['sep_nm'])
        sep_pen   = -CONFIG['w_sep'] * violation ** 2
        in_los    = cs in {ac for pair in bs.traf.cd.lospairs for ac in pair}
        intrusion = CONFIG['intrusion_pen'] if in_los else 0.0
        return float(progress + sep_pen + intrusion + CONFIG['w_time'])

    def _find_exited_aircraft(self):
        """Callsigns that have left the airspace polygon."""
        exited = []
        for cs in list(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs); continue
            inside = bs.tools.areafilter.checkInside(
                'AIRSPACE',
                np.array([bs.traf.lat[idx]]), np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]),
            )
            if not inside[0]:
                exited.append(cs)
        return exited

    def _exit_reward(self, cs):
        """Bonus for exiting aligned with destination; zero for wrong-direction exits."""
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        dest_bearing, _ = geo.kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                           float(self._destination_ll[cs][0]),
                                           float(self._destination_ll[cs][1]))
        alignment = math.cos(math.radians(wrap_to_180(bs.traf.hdg[idx] - dest_bearing)))
        return float(CONFIG['exit_correct'] * max(0.0, alignment))


# ── MARL VecEnv wrapper for SB3 ──────────────────────────────────────────────

class MARLVecEnv(VecEnv):
    """Single-process VecEnv wrapper. Used for local testing; training uses ParallelMARLVecEnv."""

    def __init__(self):
        self._env   = AirspaceEnv()
        self._stats = self._blank_stats()
        super().__init__(N_AIRCRAFT, self._env.observation_space, self._env.action_space)

    def reset(self):
        obs, _      = self._env.reset()
        self._stats = self._blank_stats()
        return obs

    def step_async(self, actions): self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self._env.step(self._actions)
        done  = terminated or truncated
        dones = np.full(N_AIRCRAFT, done)

        self._stats['rewards']   += rewards
        self._stats['los_steps'] += int(len(info['los_pairs']) > 0)
        self._stats['steps']     += 1
        self._stats['actions'].extend(self._actions.tolist())

        infos = [{} for _ in range(N_AIRCRAFT)]
        if done:
            for i in range(N_AIRCRAFT):
                infos[i]['terminal_observation'] = obs[i]
            infos[0].update({
                'mean_episode_reward': float(self._stats['rewards'].mean()),
                'ep_los_steps':        self._stats['los_steps'],
                'ep_length':           self._stats['steps'],
                'action_distribution': np.bincount(self._stats['actions'], minlength=4).tolist(),
            })
            obs, _ = self._env.reset()
            self._stats = self._blank_stats()
        return obs, rewards, dones, infos

    def close(self): pass
    def get_attr(self, attr_name, indices=None):           return [None]  * self.num_envs
    def set_attr(self, attr_name, value, indices=None):    pass
    def env_method(self, method_name, *args, **kwargs):    return [None]  * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def seed(self, seed=None):                             return [None]  * self.num_envs
    def _blank_stats(self):
        return {'rewards': np.zeros(N_AIRCRAFT), 'los_steps': 0, 'steps': 0, 'actions': []}


# ── Pygame visualisation (random policy) ─────────────────────────────────────

if __name__ == '__main__':
    import sys
    import pygame

    N_EPISODES  = 3
    WINDOW_SIZE = 800
    FPS         = 8
    SLOT_COLORS = [(50, 180, 50), (50, 130, 220), (255, 140, 0)]
    RED, BLACK, WHITE = (220, 30, 30), (0, 0, 0), (255, 255, 255)
    ACTION_LABELS     = ['-10°', '+10°', '0°', 'back']
    N_TOTAL           = N_WAVES * N_AIRCRAFT

    class _View:
        def __init__(self, polygon_nm, w, h):
            pad  = CONFIG['sep_nm'] * NM_TO_KM * 2.0
            km   = polygon_nm * NM_TO_KM
            span = max(km[:, 0].max() - km[:, 0].min() + 2 * pad,
                       km[:, 1].max() - km[:, 1].min() + 2 * pad)
            self._cx = km[:, 0].mean(); self._cy = km[:, 1].mean()
            self._sc = min(w, h) / span; self._w = w; self._h = h

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
        if total < 1: return
        ux, uy  = dx / total, dy / total
        x, y    = float(p1[0]), float(p1[1])
        drawn   = 0.0; drawing = True
        while drawn < total:
            seg = min(dash if drawing else gap, total - drawn)
            if drawing:
                pygame.draw.line(surface, color, (round(x), round(y)),
                                 (round(x + ux * seg), round(y + uy * seg)), 1)
            x += ux * seg; y += uy * seg; drawn += seg; drawing = not drawing

    env = AirspaceEnv()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('bs_simple_no_delay_v2 — random policy')
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 13)
    fsmall = pygame.font.SysFont('monospace', 11)

    for episode in range(N_EPISODES):
        obs, _        = env.reset()
        view          = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px    = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_px        = view.nm_to_px_len(CONFIG['sep_nm'] / 2)
        dest_by_cs    = {cs: view.latlon_to_px(*env._destination_ll[cs]) for cs in env._active_callsigns}
        total_rewards = np.zeros(N_AIRCRAFT)
        last_actions  = [2] * N_AIRCRAFT
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()

            actions = [env.action_space.sample() for _ in range(N_AIRCRAFT)]
            last_actions = list(actions)
            obs, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_rewards += rewards; done = terminated or truncated; step += 1
            if info['los_pairs']: los_steps += 1
            for cs in env._active_callsigns:
                if cs not in dest_by_cs and cs in env._destination_ll:
                    dest_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)
            los_set = {cs for pair in bs.traf.cd.lospairs for cs in pair}
            for slot_i, cs in enumerate(env._slots):
                if cs is None or cs not in env._active_callsigns: continue
                idx = bs.traf.id2idx(cs)
                if idx < 0: continue
                px, py = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                color  = RED if cs in los_set else SLOT_COLORS[slot_i]
                pygame.draw.circle(screen, color, (px, py), sep_px, 1)
                if cs in dest_by_cs: draw_dashed(screen, color, (px, py), dest_by_cs[cs])
                pygame.draw.circle(screen, color, (px, py), 5)
                screen.blit(fsmall.render(f'{cs} {ACTION_LABELS[last_actions[slot_i]]}', True, color), (px+7, py-7))

            hud = [
                f'Episode {episode+1}/{N_EPISODES}  ({N_TOTAL} total aircraft)  [random policy]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Served {env._next_callsign_id}/{N_TOTAL}   queued={sum(len(q) for q in env._slot_queues)}   pending={len(env._pending_spawns)}   active={len(info["active"])}',
                '  '.join(f'R{i}={total_rewards[i]:.1f}' for i in range(N_AIRCRAFT)),
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))
            for i in range(N_AIRCRAFT):
                screen.blit(fsmall.render(f'● Slot {i}', True, SLOT_COLORS[i]), (8, WINDOW_SIZE - 40 + i * 14))
            screen.blit(fsmall.render('● RED = separation violation', True, RED), (8, WINDOW_SIZE - 12))
            pygame.display.flip(); clock.tick(FPS)

        print(f'Episode {episode+1:2d}  steps={step}  mean_reward={total_rewards.mean():.2f}  LoS-steps={los_steps}')
    pygame.quit()
