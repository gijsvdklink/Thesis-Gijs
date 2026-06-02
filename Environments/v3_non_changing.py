"""
v3_non_changing — single-ownship conflict-avoidance environment.

One aircraft (slot 0, callsign OWN) is controlled by the agent.
All other aircraft are uncontrolled traffic: they fly their initial heading
and receive no instructions at any point.  The agent never has to manage
traffic-vs-traffic conflicts; it only needs to keep OWN separated from the
surrounding flow.

This is a deliberate simplification used to test whether a policy can learn
basic separation behaviour before tackling the full multi-aircraft problem.

Observation  21 floats  (ego-centric from OWN — identical structure to v3_improved)
  [0]  sin(Δψ_dest)         [-1, 1]
  [1]  cos(Δψ_dest)         [-1, 1]
  [2]  turn_progress        [-1, 1]
  [3]  time_to_exit         [0, 1]
  [4]  speed_offset         [0, 1]
  [5:21]  4 nearest traffic × (dist, fwd, right, cpa_side)

Reward  attributed solely to OWN (same formula as v3_improved)
  −w_safe  × cpa_penalty
  −w_eff   × drift
  −w_work  × act_cost
  −w_exit  (sparse, on wrong-direction exit of OWN)

Action  Discrete(9) — same as v3_improved, applied to OWN only.
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
# Identical to v3_improved so the two environments are directly comparable.

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
    'action_freq':           20,
    'lookahead_s':           900.0,
    't_warn':                600.0,
    'crossings_per_episode': 2.5,
    'spawn_delay_s':         (0, 0),
    'n_neighbours':          4,
    'w_safe':                1.00,
    'w_exit':                0.50,
    'w_eff':                 0.20,
    'w_work':                0.05,
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR   = CONFIG['n_neighbours']
OBS_DIM = 5 + N_NBR * 4           # 21

HEADING_OFFSETS = [-30, -15, 0, 15, 30, None]
SPEED_DELTAS    = [-0.04, -0.02, 0.02]
ACT_COST        = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5]

OWNSHIP_SLOT = 0   # slot index that is always the controlled aircraft

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

# ── Pair urgency (kept for visualisation only) ────────────────────────────────

def _pair_urgency(i, j):
    nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    dx, dy = nm2[0]-nm1[0], nm2[1]-nm1[1]
    d2     = dx*dx + dy*dy
    sep    = CONFIG['sep_nm']
    if d2 < sep*sep:
        return 1.0 + 9.0 * (1.0 - math.sqrt(d2) / sep)
    spd1 = _spd_nms(i);  spd2 = _spd_nms(j)
    vn1 = spd1*math.cos(math.radians(bs.traf.hdg[i]))
    ve1 = spd1*math.sin(math.radians(bs.traf.hdg[i]))
    vn2 = spd2*math.cos(math.radians(bs.traf.hdg[j]))
    ve2 = spd2*math.sin(math.radians(bs.traf.hdg[j]))
    dvn, dve = vn2-vn1, ve2-ve1
    rv2 = dvn*dvn + dve*dve
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
        self._ownship_cs        = None   # callsign of the controlled aircraft (slot 0)
        self._next_callsign_id  = 0
        self._step_count        = 0
        self._max_steps         = 0
        self._pending_spawns    = {}
        self._spawn_delay_range = (1, 1)
        self._ep_stats          = {}
        self.polygon            = None
        self._polygon_shape     = None

        # Kept for GIF visualisation compatibility
        self._urgency_matrix    = np.zeros((0, 0))
        self._urgency_cs_list   = []
        self._focus_cs          = None   # always = _ownship_cs

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
        self._urgency_matrix    = np.zeros((0, 0))
        self._urgency_cs_list   = []

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

        # Slot 0 is always the ownship
        self._ownship_cs = self._slots[OWNSHIP_SLOT]
        self._focus_cs   = self._ownship_cs
        self._update_urgency_matrix()
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()

        # Apply action exclusively to the ownship
        if self._ownship_cs and bs.traf.id2idx(self._ownship_cs) >= 0:
            self._apply_action(self._ownship_cs, int(action))

        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        self._step_count += 1

        exit_r = self._process_exits()

        # Update ownship callsign in case it exited and was replaced
        if self._slots[OWNSHIP_SLOT] is not None:
            self._ownship_cs = self._slots[OWNSHIP_SLOT]
        self._focus_cs = self._ownship_cs

        self._update_urgency_matrix()
        reward = self._compute_reward(self._ownship_cs, int(action)) + exit_r

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
                    self._ep_stats['actions'], minlength=9).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # ── Urgency matrix (visualisation only — not used for control) ────────────

    def _update_urgency_matrix(self):
        cs_list = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not cs_list:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return
        n      = len(cs_list)
        bs_idx = [bs.traf.id2idx(cs) for cs in cs_list]
        U      = np.zeros((n, n))
        for ii in range(n):
            for jj in range(ii+1, n):
                u = _pair_urgency(bs_idx[ii], bs_idx[jj])
                U[ii, jj] = U[jj, ii] = u
        self._urgency_matrix  = U
        self._urgency_cs_list = cs_list

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx):
        cpa_min = self._min_cpa_dist(acting_cs)
        sep2    = 2.0 * CONFIG['sep_nm']
        r_safe  = -CONFIG['w_safe'] * max(0.0, 1.0 - cpa_min / sep2)

        drift_a = 0.0
        if acting_cs and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[acting_cs]])
                diff    = wrap_to_180(
                    bearing - self._commanded_heading.get(acting_cs, bs.traf.hdg[idx]))
                drift_a = (1.0 - math.cos(math.radians(diff))) / 2.0

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_safe - CONFIG['w_eff'] * drift_a + r_work)

    def _min_cpa_dist(self, cs):
        if cs is None:
            return float('inf')
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return float('inf')
        nm1  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        spd1 = _spd_nms(idx)
        vn1  = spd1 * math.cos(math.radians(bs.traf.hdg[idx]))
        ve1  = spd1 * math.sin(math.radians(bs.traf.hdg[idx]))
        min_d = float('inf')
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue
            nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = nm2[0]-nm1[0], nm2[1]-nm1[1]
            d_now  = math.sqrt(dx*dx + dy*dy)
            spd2   = _spd_nms(oi)
            vn2    = spd2 * math.cos(math.radians(bs.traf.hdg[oi]))
            ve2    = spd2 * math.sin(math.radians(bs.traf.hdg[oi]))
            dvn, dve = vn2-vn1, ve2-ve1
            rv2    = dvn*dvn + dve*dve
            if rv2 < 1e-12:
                cpa_d = d_now
            else:
                dot  = dx*dve + dy*dvn
                tcpa = -dot / rv2
                if tcpa <= 0 or tcpa > CONFIG['lookahead_s']:
                    cpa_d = d_now
                else:
                    dx_c  = dx + tcpa * dve
                    dy_c  = dy + tcpa * dvn
                    cpa_d = math.sqrt(dx_c*dx_c + dy_c*dy_c)
            min_d = min(min_d, cpa_d)
        return min_d

    # ── Action (applied to ownship only) ──────────────────────────────────────

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

    # ── Observation (always from ownship) ─────────────────────────────────────

    def _get_observation(self):
        cs = self._ownship_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)
        own_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        sep     = CONFIG['sep_nm']
        sin_h   = math.sin(math.radians(own_hdg))
        cos_h   = math.cos(math.radians(own_hdg))
        spd_own = _spd_nms(idx)
        vn_o    = spd_own * math.cos(math.radians(own_hdg))
        ve_o    = spd_own * math.sin(math.radians(own_hdg))

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
            math.sin(math.radians(diff)),
            math.cos(math.radians(diff)),
            turn_prog,
            t_exit,
            speed_offset,
        ]

        intruders = []
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue
            int_nm = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = int_nm[0]-own_nm[0], int_nm[1]-own_nm[1]
            dist   = math.sqrt(dx*dx + dy*dy)
            rel_bearing = math.degrees(math.atan2(dx*cos_h - dy*sin_h,
                                                   dx*sin_h + dy*cos_h))
            fwd   = math.cos(math.radians(rel_bearing))
            right = math.sin(math.radians(rel_bearing))
            spd_int = _spd_nms(oi)
            vn_i    = spd_int * math.cos(math.radians(bs.traf.hdg[oi]))
            ve_i    = spd_int * math.sin(math.radians(bs.traf.hdg[oi]))
            dvn, dve = vn_i-vn_o, ve_i-ve_o
            rv2      = dvn*dvn + dve*dve
            cpa_side = 0.0
            cpa_dist = dist
            if rv2 > 1e-12:
                dot      = dx*dve + dy*dvn
                tcpa_raw = -dot / rv2
                tcpa     = max(0.0, tcpa_raw)
                dx_cpa   = dx + tcpa*dve
                dy_cpa   = dy + tcpa*dvn
                cpa_side = max(-1.0, min(1.0, (dx_cpa*cos_h - dy_cpa*sin_h) / sep))
                if tcpa_raw > 0:
                    cpa_dist = math.sqrt(dx_cpa*dx_cpa + dy_cpa*dy_cpa)
            dist_norm = min(dist / (3.0 * sep), 1.0)
            intruders.append((cpa_dist, dist_norm, fwd, right, cpa_side))

        intruders.sort(key=lambda x: x[0])
        for k in range(N_NBR):
            if k < len(intruders):
                _, dn, fw, ri, cs_ = intruders[k]
                obs += [dn, fw, ri, cs_]
            else:
                obs += [1.0, 0.0, 0.0, 0.0]

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
        r = 0.0
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            # Exit penalty only for the ownship
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
