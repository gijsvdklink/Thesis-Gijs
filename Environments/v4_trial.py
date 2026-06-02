"""
v4_trial — single-ownship, LiDAR-observation conflict-avoidance environment.

Mirrors the observation and reward design of navigation-mappo-rl, adapted for
a BlueSky airspace scenario with a single controlled agent.

Observation  5 ego features + 36 rays × 2 LiDAR channels  (77 floats per frame)
  [0]   sin(Δψ_dest)     heading error to destination               [-1, 1]
  [1]   cos(Δψ_dest)                                                [-1, 1]
  [2]   speed_ratio      cmd_mach / ac_mach                         [0, ~1]
  [3]   turn_progress    (cmd_hdg − actual_hdg) / 30                [-1, 1]
  [4]   time_to_exit     normalised time to sector boundary          [0, 1]
  [5:41]   aircraft LiDAR — 36 ego-centric rays, ray 0 = dead ahead, clockwise.
           0.0 = intruder touching ownship, 1.0 = clear / beyond range
  [41:77]  boundary LiDAR — same 36 rays cast against sector polygon.
           0.0 = at boundary, 1.0 = boundary beyond lidar_range_nm

Temporal context is provided by VecFrameStack(4) in the training script,
stacking 4 consecutive frames → 308-dim observation fed to the network.
This matches the history_length = 4 used in navigation-mappo-rl.

Action  Discrete(9) — ATCO-style clearances
  0 −30°   1 −15°   2 direct [free]   3 +15°   4 +30°   5 hold [free]
  6 M−0.04   7 M−0.02   8 M+0.02

Reward  (navigation-mappo-rl style, adapted for airspace)
  +w_nav   × cos(Δψ_actual) × speed_ratio   navigation progress
  −w_safe  × exp(−min_ac_lidar_nm / sep)    proximity penalty
  −w_alive                                   alive cost (encourages efficiency)
  −w_work  × ACT_COST[action]               ATCO workload penalty
  −w_exit                                    wrong-direction sector exit (sparse)
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
    'ac_type':               'A320',
    'ac_speed':              450.0,
    'ac_mach':               0.78,
    'ac_mach_min':           0.70,
    'ac_mach_max':           0.82,
    'altitude':              350,
    'center_ll':             (52.3, 5.3),
    'area_km2':              lambda: random.uniform(10_000.0, 80_000.0),
    'density_km2':           lambda: random.uniform(5_000.0, 15_000.0),
    'max_agents':            15,
    'sep_nm':                5.0,
    'buffer_nm':             5.0,
    'dest_dist_factor':      2.0,
    'n_vertices':            lambda: random.randint(5, 7),
    'min_circularity':       0.65,
    'max_placement_tries':   50,
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.30, 0.30),
    'sim_dt':                0.5,
    'action_freq':           60,
    'lookahead_s':           900.0,
    'crossings_per_episode': 2.5,
    'spawn_delay_s':         (0, 0),
    # LiDAR — 2 channels: aircraft + sector boundary
    'n_lidar_rays':          36,
    'lidar_range_nm':        30.0,   # 6× sep_nm
    'lidar_detect_nm':       5.0,    # ray hit half-width = sep_nm
    # Reward weights
    'w_safe':                1.00,
    'w_nav':                 0.20,
    'w_alive':               0.05,
    'w_work':                0.05,
    'w_exit':                0.50,
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_RAYS    = CONFIG['n_lidar_rays']
N_CHANNELS = 2                       # aircraft + boundary
OBS_DIM   = 5 + N_RAYS * N_CHANNELS  # 77

HEADING_OFFSETS = [-30, -15, 0, 15, 30, None]
SPEED_DELTAS    = [-0.04, -0.02, 0.02]
ACT_COST        = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5]

OWNSHIP_SLOT = 0

_bs_initialized = False

__all__ = ['AirspaceEnv', 'CONFIG', 'NM_TO_KM', 'latlon_to_nm', 'wrap_to_180', 'OBS_DIM']

# ── Coordinate helpers ─────────────────────────────────────────────────────────

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

# ── Sector polygon ─────────────────────────────────────────────────────────────

def _make_polygon():
    target_nm2 = CONFIG['area_km2']() * KM_TO_NM ** 2
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

# ── Aircraft placement ─────────────────────────────────────────────────────────

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
        self.action_space      = spaces.Discrete(9)

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
        self._commanded_speed   = {}
        self._ownship_cs        = None
        self._next_callsign_id  = 0
        self._step_count        = 0
        self._max_steps         = 0
        self._pending_spawns    = {}
        self._spawn_delay_range = (1, 1)
        self._ep_stats          = {}
        self.polygon            = None
        self._polygon_shape     = None
        self._last_ac_lidar     = np.ones(N_RAYS, dtype=np.float32)
        self._last_bnd_lidar    = np.ones(N_RAYS, dtype=np.float32)

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

        poly = _make_polygon()
        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        diam_nm  = math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
        step_s   = CONFIG['action_freq'] * CONFIG['sim_dt']
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * diam_nm / CONFIG['ac_speed'] * 3600 / step_s
        ))

        n = int(np.clip(
            round(CONFIG['area_km2']() / CONFIG['density_km2']()), 2, CONFIG['max_agents']
        ))
        self.n_aircraft         = n
        self._slots             = [None] * n
        self._active_callsigns  = set()
        self._destination_ll    = {}
        self._commanded_heading = {}
        self._commanded_speed   = {}
        self._ownship_cs        = None
        self._next_callsign_id  = 0
        self._step_count        = 0
        self._ep_stats          = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': []}
        self._last_ac_lidar     = np.ones(N_RAYS, dtype=np.float32)
        self._last_bnd_lidar    = np.ones(N_RAYS, dtype=np.float32)

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

        self._ownship_cs = self._slots[OWNSHIP_SLOT]
        own_idx = bs.traf.id2idx(self._ownship_cs) if self._ownship_cs else -1
        if own_idx >= 0:
            self._last_ac_lidar  = self._compute_ac_lidar(self._ownship_cs, own_idx)
            self._last_bnd_lidar = self._compute_bnd_lidar(own_idx)
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()

        if self._ownship_cs and bs.traf.id2idx(self._ownship_cs) >= 0:
            self._apply_action(self._ownship_cs, int(action))

        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        self._step_count += 1

        exit_r = self._process_exits()

        if self._slots[OWNSHIP_SLOT] is not None:
            self._ownship_cs = self._slots[OWNSHIP_SLOT]

        own_idx = bs.traf.id2idx(self._ownship_cs) if self._ownship_cs else -1
        if own_idx >= 0:
            self._last_ac_lidar  = self._compute_ac_lidar(self._ownship_cs, own_idx)
            self._last_bnd_lidar = self._compute_bnd_lidar(own_idx)
        else:
            self._last_ac_lidar  = np.ones(N_RAYS, dtype=np.float32)
            self._last_bnd_lidar = np.ones(N_RAYS, dtype=np.float32)

        reward = self._compute_reward(self._ownship_cs, int(action)) + exit_r
        n_los  = self._count_los()

        self._ep_stats['reward']  += reward
        self._ep_stats['steps']   += 1
        self._ep_stats['actions'].append(int(action))
        if n_los > 0:
            self._ep_stats['los'] += 1

        truncated = self._step_count >= self._max_steps
        info = {'los_pairs': n_los, 'n_aircraft': self.n_aircraft}
        if truncated:
            s = max(self._ep_stats['steps'], 1)
            info.update({
                'mean_episode_reward': self._ep_stats['reward'] / s,
                'ep_los_steps':        self._ep_stats['los'],
                'ep_length':           self._ep_stats['steps'],
                'action_distribution': np.bincount(
                    self._ep_stats['actions'], minlength=9).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # ── LiDAR — aircraft channel ───────────────────────────────────────────────

    def _compute_ac_lidar(self, cs, idx):
        """
        36 ego-centric rays; each returns normalised distance to nearest
        intruder aircraft.  Intruders modelled as circles of radius lidar_detect_nm.
        0.0 = intruder at ownship position, 1.0 = clear / beyond lidar_range_nm.
        """
        hdg_rad = math.radians(bs.traf.hdg[idx])
        sin_h   = math.sin(hdg_rad)
        cos_h   = math.cos(hdg_rad)
        own_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        max_r   = CONFIG['lidar_range_nm']
        det_r   = CONFIG['lidar_detect_nm']
        det_r2  = det_r * det_r
        readings = np.ones(N_RAYS, dtype=np.float32)

        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue
            int_nm = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx_e = float(int_nm[0] - own_nm[0])
            dx_n = float(int_nm[1] - own_nm[1])
            if dx_e*dx_e + dx_n*dx_n > (max_r + det_r) ** 2:
                continue
            # Ego frame: px = right, py = forward
            px = dx_e * cos_h - dx_n * sin_h
            py = dx_e * sin_h + dx_n * cos_h
            for r in range(N_RAYS):
                theta = math.radians(r * 360.0 / N_RAYS)
                rdx   = math.sin(theta)
                rdy   = math.cos(theta)
                t = px * rdx + py * rdy
                if t < 0:
                    continue
                perp2 = max(0.0, px*px + py*py - t*t)
                if perp2 < det_r2:
                    hit  = max(0.0, t - math.sqrt(max(0.0, det_r2 - perp2)))
                    norm = min(hit / max_r, 1.0)
                    if norm < readings[r]:
                        readings[r] = norm
        return readings

    # ── LiDAR — boundary channel ───────────────────────────────────────────────

    def _compute_bnd_lidar(self, idx):
        """
        36 rays cast against the sector polygon boundary.
        0.0 = ownship is at the boundary, 1.0 = boundary beyond lidar_range_nm.
        Rays are in absolute NM heading (not ego-relative) to decouple boundary
        geometry from ownship orientation; the ego frame is recovered by the
        same ordering as the aircraft channel (ray 0 = dead ahead, clockwise).
        """
        own_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        own_hdg = bs.traf.hdg[idx]
        max_r   = CONFIG['lidar_range_nm']
        readings = np.ones(N_RAYS, dtype=np.float32)

        for r in range(N_RAYS):
            # Absolute heading: ray 0 = dead ahead of ownship, clockwise
            alpha = math.radians(own_hdg + r * 360.0 / N_RAYS)
            ray_e = math.sin(alpha)
            ray_n = math.cos(alpha)
            end   = (float(own_nm[0] + max_r * ray_e),
                     float(own_nm[1] + max_r * ray_n))
            ray_line = LineString([(float(own_nm[0]), float(own_nm[1])), end])
            isect    = self._polygon_shape.exterior.intersection(ray_line)
            if isect.is_empty:
                continue
            pts = list(isect.geoms) if hasattr(isect, 'geoms') else [isect]
            d   = min(math.hypot(p.x - own_nm[0], p.y - own_nm[1]) for p in pts)
            readings[r] = min(d / max_r, 1.0)
        return readings

    # ── LoS counter ───────────────────────────────────────────────────────────

    def _count_los(self):
        if not self._ownship_cs:
            return 0
        own_idx = bs.traf.id2idx(self._ownship_cs)
        if own_idx < 0:
            return 0
        own_nm = latlon_to_nm(CONFIG['center_ll'],
                               bs.traf.lat[own_idx], bs.traf.lon[own_idx])
        sep2  = CONFIG['sep_nm'] ** 2
        count = 0
        for cs in self._active_callsigns:
            if cs == self._ownship_cs:
                continue
            oi = bs.traf.id2idx(cs)
            if oi < 0:
                continue
            nm = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = float(nm[0] - own_nm[0]), float(nm[1] - own_nm[1])
            if dx*dx + dy*dy < sep2:
                count += 1
        return count

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, cs, action_idx):
        # Navigation progress: cos(heading error) × speed ratio (nav-mappo style)
        r_nav = 0.0
        if cs and cs in self._destination_ll:
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[cs]])
                heading_err = wrap_to_180(bearing - bs.traf.hdg[idx])
                speed_ratio = self._commanded_speed.get(cs, CONFIG['ac_mach']) / CONFIG['ac_mach']
                r_nav = CONFIG['w_nav'] * math.cos(math.radians(heading_err)) * speed_ratio

        # Safety: exponential proximity via minimum aircraft LiDAR reading
        min_ac_nm = float(self._last_ac_lidar.min()) * CONFIG['lidar_range_nm']
        r_safe = -CONFIG['w_safe'] * math.exp(-min_ac_nm / CONFIG['sep_nm'])

        # Alive cost — encourages efficient navigation (from nav-mappo-rl)
        r_alive = -CONFIG['w_alive']

        # Workload: discrete ATCO action cost
        r_work = -CONFIG['w_work'] * ACT_COST[action_idx]

        return float(r_nav + r_safe + r_alive + r_work)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        if action_idx < 6:
            offset = HEADING_OFFSETS[action_idx]
            if offset is None:
                bs.stack.stack(
                    f'HDG {cs} {self._commanded_heading.get(cs, bs.traf.hdg[idx]):.1f}')
                return
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            if offset == 0:
                if abs(wrap_to_180(
                        self._commanded_heading.get(cs, bs.traf.hdg[idx])
                        - bs.traf.hdg[idx])) > 5.0:
                    return
            self._commanded_heading[cs] = (float(bearing) + offset) % 360
            bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')
        else:
            delta    = SPEED_DELTAS[action_idx - 6]
            cur_mach = self._commanded_speed.get(cs, CONFIG['ac_mach'])
            new_mach = float(np.clip(cur_mach + delta,
                                     CONFIG['ac_mach_min'], CONFIG['ac_mach_max']))
            self._commanded_speed[cs] = new_mach
            bs.stack.stack(f'MACH {cs} {new_mach:.3f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        cs = self._ownship_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)
        cmd_mach = self._commanded_speed.get(cs, CONFIG['ac_mach'])

        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        diff = wrap_to_180(bearing - cmd_hdg)

        speed_ratio  = cmd_mach / CONFIG['ac_mach']
        turn_prog    = max(-1.0, min(1.0, wrap_to_180(cmd_hdg - own_hdg) / 30.0))
        t_exit       = self._time_to_exit(idx)

        ego = np.array([
            math.sin(math.radians(diff)),
            math.cos(math.radians(diff)),
            speed_ratio,
            turn_prog,
            t_exit,
        ], dtype=np.float32)

        return np.concatenate([ego, self._last_ac_lidar, self._last_bnd_lidar])

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
        r = 0.0
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            if slot == OWNSHIP_SLOT:
                r += self._exit_penalty(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._destination_ll.pop(cs, None)
            self._commanded_heading.pop(cs, None)
            self._commanded_speed.pop(cs, None)
            if slot not in self._pending_spawns:
                delay = random.randint(*self._spawn_delay_range)
                self._pending_spawns[slot] = delay
        return r

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

    def _exit_penalty(self, cs):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        return -CONFIG['w_exit'] \
               if abs(wrap_to_180(bearing - bs.traf.hdg[idx])) > 90 else 0.0

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _process_pending_spawns(self):
        ready = sorted(s for s, t in list(self._pending_spawns.items()) if t <= 1)
        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
            else:
                self._pending_spawns[slot] = 5
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
        cs   = f'AC{self._next_callsign_id:02d}'
        mach = CONFIG['ac_mach']
        self._next_callsign_id += 1
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['sp_ll'][0]), aclon=float(ac['sp_ll'][1]),
                    achdg=float(ac['heading']), acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'MACH {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]    = ac['dest_ll']
        self._commanded_heading[cs] = float(ac['heading'])
        self._commanded_speed[cs]   = mach
        self._slots[slot]           = cs
        self._active_callsigns.add(cs)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _flat_latlon(self):
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
