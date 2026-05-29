"""
v3_simple_env — ATCO conflict-resolution environment.

One agent issues one heading instruction per step to the most urgent aircraft.
Urgency, observation and reward are all built from pure geometry.

Observation  16 floats
  [0,1]   cos/sin( dest_bearing − cmd_heading )
  [2]     clearance_timer  (0 = just had conflict, 1 = clear ≥ 30 steps)
  [3:14]  3 nearest intruders × 4:  rel_fwd, rel_right, tLOS_inv, urgency_k
  [15]    sector urgency density  min(Σu/10, 1)

Reward  (pure penalties)
  w_urgency × Σu          safety     — penalises all conflict imminence
  w_drift   × drift       efficiency — penalises heading deviation from destination
  pen_action              workload   — cost per resolution turn issued
  pen_wrong_exit          exit       — wrong-direction sector exit

Action  (Discrete 6)  absolute offsets from direct-to-destination bearing
  0 −30°   1 −15°   2 0° direct [free]   3 +15°   4 +30°   5 hold [free]
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

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Aircraft & sector
    'ac_type':              'A320',
    'ac_speed':             450.0,           # kts
    'altitude':             350,             # FL350
    'center_ll':            (52.3, 5.3),     # Dutch upper airspace
    'area_km2':             lambda: 80_000.0,
    'density_km2':          lambda: 8_000.0, # → ~10 aircraft
    'max_agents':           12,
    'sep_nm':               5.0,             # ICAO separation standard
    'buffer_nm':            2.0,
    'dest_dist_factor':     2.0,
    # Polygon
    'n_vertices':           lambda: random.randint(5, 7),
    'min_circularity':      0.65,
    'max_placement_tries':  50,
    # Aircraft placement jitter — each aircraft gets its own boundary sector,
    # with randomised entry point and reference point so headings cross realistically
    'spawn_jitter':         lambda: random.uniform(0.1, 0.9),
    'ref_jitter':           lambda: random.uniform(-0.15, 0.15),
    # Simulation
    'sim_dt':               0.5,             # BlueSky timestep (s)
    'action_freq':          20,              # RL steps → 10 s/decision
    'lookahead_s':          600.0,
    'crossings_per_episode': 3,
    'spawn_delay_s':        (120, 300),
    # Observation
    'n_neighbours':         3,
    'spatial_scale':        150.0,           # NM — position normalisation
    'clearance_horizon':    30,              # steps until clearance_timer = 1.0
    # Reward
    'w_urgency':           -3.0,
    'w_drift':             -1.0,
    'pen_action':          -0.1,
    'pen_wrong_exit':      -20.0,
    'seed':                 None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

N_NBR    = CONFIG['n_neighbours']
OBS_DIM  = 3 + N_NBR * 4 + 1       # = 16
# Ownship [3]: Δψ/180, |Δψ|/30, clearance_timer
# Intruder [4]: dist_norm, bearing_sin, cpa_right_norm, urgency_k
# Global  [1]: urgency_density

HEADING_OFFSETS = [-30, -15, 0, 15, 30, None]

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

# ── Sector polygon ────────────────────────────────────────────────────────────

def _make_polygon():
    """Random convex polygon scaled to the configured area, filtered by circularity."""
    target_nm2 = CONFIG['area_km2']() * KM_TO_NM ** 2
    while True:
        raw    = ShapelyPolygon(random_convex_polygon(CONFIG['n_vertices']()))
        scaled = shapely_scale(raw,
                               xfact=math.sqrt(target_nm2 / raw.area),
                               yfact=math.sqrt(target_nm2 / raw.area),
                               origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= CONFIG['min_circularity']:
            cx, cy = scaled.centroid.x, scaled.centroid.y
            return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])

# ── Aircraft placement ────────────────────────────────────────────────────────

def _place_one(polygon, sector, n_sectors):
    """
    Place one aircraft in its boundary sector with jitter.

    The polygon perimeter is divided into n_sectors equal slices.  Aircraft
    slot `sector` spawns within its slice (jittered) and heads toward a
    reference point on the far side of the polygon, then continues to a
    destination beyond the sector.  This gives realistic crossing traffic:
    each aircraft enters from a different direction with a different heading.
    """
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

# ── Urgency (pure geometry, smooth LoS boundary) ──────────────────────────────

def _pair_urgency(i, j):
    """
    [1, 10] dist < sep_nm  (active LoS, smooth)
    (0, 1]  predicted conflict within lookahead horizon
    0       diverging or will miss
    """
    nm1 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    nm2 = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    dx, dy  = nm2[0]-nm1[0], nm2[1]-nm1[1]
    d2      = dx*dx + dy*dy
    sep     = CONFIG['sep_nm']

    if d2 < sep*sep:
        return 1.0 + 9.0*(1.0 - math.sqrt(d2)/sep)

    spd = CONFIG['ac_speed'] / 1852.0
    vn1 = spd*math.cos(math.radians(bs.traf.hdg[i]))
    ve1 = spd*math.sin(math.radians(bs.traf.hdg[i]))
    vn2 = spd*math.cos(math.radians(bs.traf.hdg[j]))
    ve2 = spd*math.sin(math.radians(bs.traf.hdg[j]))
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

    return 1.0 / max(tcpa, 1.0)

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

        self.n_aircraft            = 0
        self._slots                = []
        self._active_callsigns     = set()
        self._destination_ll       = {}
        self._commanded_heading    = {}
        self._steps_since_urgency  = {}
        self._next_callsign_id     = 0
        self._step_count           = 0
        self._max_steps            = 0
        self._focus_cs             = None
        self._pending_spawns       = {}
        self._spawn_delay_range    = (24, 60)
        self._ep_stats             = {}
        self.polygon               = None
        self._polygon_shape        = None

        # Exposed for visualiser / make_gifs
        self._urgency_matrix       = np.zeros((0, 0))
        self._urgency_cs_list      = []

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)

        bs.traf.reset()

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
        self.n_aircraft            = n
        self._slots                = [None] * n
        self._active_callsigns     = set()
        self._destination_ll       = {}
        self._commanded_heading    = {}
        self._steps_since_urgency  = {}
        self._next_callsign_id     = 0
        self._step_count           = 0
        self._ep_stats             = {'reward': 0.0, 'steps': 0, 'los': 0, 'actions': []}
        self._urgency_matrix       = np.zeros((0, 0))
        self._urgency_cs_list      = []

        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._flat_latlon())
        bs.stack.stack('ASAS OFF')

        # Spawn first aircraft immediately; stagger the rest across the episode start.
        # Each slot is assigned its own perimeter sector so aircraft enter from
        # different directions — this is what creates realistic crossing traffic.
        ac0 = _place_one(self._polygon_shape, 0, n)
        self._spawn_aircraft(0, ac0)
        cumulative = 0
        for slot in range(1, n):
            cumulative += random.randint(delay_min, delay_max)
            self._pending_spawns[slot] = cumulative

        self._focus_cs = self._select_focus_aircraft()
        return self._get_observation(), {}

    def step(self, action):
        self._process_pending_spawns()
        acting_cs = self._focus_cs

        if acting_cs:
            self._apply_action(acting_cs, int(action))

        half = CONFIG['action_freq'] // 2
        for _ in range(half):
            bs.sim.step()
        self._focus_cs = self._select_focus_aircraft()
        for _ in range(CONFIG['action_freq'] - half):
            bs.sim.step()
        self._step_count += 1

        exit_r         = self._process_exits()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, int(action)) + exit_r

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
                    self._ep_stats['actions'], minlength=6).tolist(),
            })
        return self._get_observation(), reward, False, truncated, info

    # ── Focus & urgency ───────────────────────────────────────────────────────

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

        # Update clearance timers
        row_max = U.max(axis=1)
        for i, cs in enumerate(cs_list):
            if row_max[i] > 0:
                self._steps_since_urgency[cs] = 0
            else:
                self._steps_since_urgency[cs] = self._steps_since_urgency.get(cs, 30) + 1

        if row_max.max() > 0:
            return cs_list[int(np.argmax(row_max))]
        return self._drift_fallback(cs_list)

    def _drift_fallback(self, cs_list):
        best_cs, best_drift = None, -1.0
        for cs in sorted(cs_list):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._destination_ll:
                continue
            bearing, _ = geo.kwikqdrdist(
                bs.traf.lat[idx], bs.traf.lon[idx],
                *[float(v) for v in self._destination_ll[cs]])
            diff  = wrap_to_180(bearing - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
            drift = (1 - math.cos(math.radians(diff))) / 2
            if drift > best_drift:
                best_drift, best_cs = drift, cs
        return best_cs

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, acting_cs, action_idx):
        U       = self._urgency_matrix
        urg_sum = float(U.sum()) / 2.0 if U.size > 0 else 0.0
        r       = CONFIG['w_urgency'] * urg_sum

        if acting_cs and acting_cs in self._destination_ll:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                bearing, _ = geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    *[float(v) for v in self._destination_ll[acting_cs]])
                diff = wrap_to_180(
                    bearing - self._commanded_heading.get(acting_cs, bs.traf.hdg[idx]))
                r += CONFIG['w_drift'] * (1 - math.cos(math.radians(diff))) / 2

        if action_idx not in (2, 5):
            r += CONFIG['pen_action']

        return float(r)

    # ── Action ────────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        offset = HEADING_OFFSETS[action_idx]
        if offset is None:
            # Explicit hold: re-issue the current commanded heading so BlueSky
            # always receives a command this step (no silent no-op).
            bs.stack.stack(f'HDG {cs} {self._commanded_heading.get(cs, bs.traf.hdg[idx]):.1f}')
            return

        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])

        if offset == 0:
            # Guard: don't issue "direct" while mid-turn
            if abs(wrap_to_180(
                    self._commanded_heading.get(cs, bs.traf.hdg[idx]) - bs.traf.hdg[idx]
               )) > 5.0:
                return

        self._commanded_heading[cs] = (float(bearing) + offset) % 360
        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        """
        16-float observation:
          [0]    Δψ/180            signed heading error to destination  (-1 left, +1 right)
          [1]    |Δψ|/30           magnitude of deviation (0=direct, 1=30°+ off)
          [2]    clearance_timer   0=just had conflict, 1=clear ≥30 steps
          [3:14] 3 intruders × 4:
                   dist_norm       current distance / (3·sep_nm)  [0,1]
                   bearing_sin     sin of bearing to intruder      [-1,1]  (left/right NOW)
                   cpa_right_norm  lateral offset at CPA / sep_nm  [-1,1]  (left/right at CPA)
                   urgency_k       pair urgency, min(u/10, 1)       [0,1]
          [15]   Σu/10             sector urgency density
        """
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        cmd_hdg = self._commanded_heading.get(cs, own_hdg)
        own_nm  = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        spd     = CONFIG['ac_speed'] / 3600.0   # NM/s
        sep     = CONFIG['sep_nm']
        sin_h   = math.sin(math.radians(own_hdg))
        cos_h   = math.cos(math.radians(own_hdg))

        bearing, _ = geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            *[float(v) for v in self._destination_ll[cs]])
        diff = wrap_to_180(bearing - cmd_hdg)

        obs = [
            diff / 180.0,                                                          # Δψ/180
            min(abs(diff) / 30.0, 1.0),                                            # |Δψ|/30
            min(self._steps_since_urgency.get(cs, 30) / CONFIG['clearance_horizon'], 1.0),
        ]

        # Urgency lookup
        U       = self._urgency_matrix
        urg_idx = {c: i for i, c in enumerate(self._urgency_cs_list)}
        own_ui  = urg_idx.get(cs, -1)
        vn_o    = spd * math.cos(math.radians(own_hdg))
        ve_o    = spd * math.sin(math.radians(own_hdg))

        intruders = []
        for other in self._active_callsigns:
            oi = bs.traf.id2idx(other)
            if other == cs or oi < 0:
                continue
            int_nm = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[oi], bs.traf.lon[oi])
            dx, dy = int_nm[0]-own_nm[0], int_nm[1]-own_nm[1]
            dist   = math.sqrt(dx*dx + dy*dy)

            # Which side is the intruder NOW (positive = right of own heading)
            rel_right   = dx*cos_h - dy*sin_h
            bearing_sin = rel_right / max(dist, 1e-3)

            # Which side does the intruder PASS at CPA
            vn_i = spd * math.cos(math.radians(bs.traf.hdg[oi]))
            ve_i = spd * math.sin(math.radians(bs.traf.hdg[oi]))
            dvn, dve = vn_i-vn_o, ve_i-ve_o
            rv2  = dvn*dvn + dve*dve
            cpa_right_norm = 0.0
            if rv2 > 1e-12:
                dot  = dx*dve + dy*dvn
                tcpa = max(0.0, -dot / rv2)
                dx_cpa     = dx + tcpa*dve        # relative position at CPA
                dy_cpa     = dy + tcpa*dvn
                cpa_right  = dx_cpa*cos_h - dy_cpa*sin_h   # rightward in ego frame
                cpa_right_norm = max(-1.0, min(1.0, cpa_right / sep))

            # Pair urgency
            int_ui = urg_idx.get(other, -1)
            u_k = float(U[own_ui, int_ui]) \
                  if (own_ui >= 0 and int_ui >= 0 and U.size > 0) else 0.0

            dist_norm = min(dist / (3.0 * sep), 1.0)
            intruders.append((dist, dist_norm, bearing_sin, cpa_right_norm, min(u_k/10.0, 1.0)))

        intruders.sort()
        for k in range(N_NBR):
            if k < len(intruders):
                _, dn, bs_, cr, uk = intruders[k]
                obs += [dn, bs_, cr, uk]
            else:
                obs += [1.0, 0.0, 0.0, 0.0]   # distant, centred, no CPA offset, no urgency

        urg_sum = float(U.sum()) / 2.0 if U.size > 0 else 0.0
        obs.append(min(urg_sum / 10.0, 1.0))

        return np.array(obs, dtype=np.float32)

    # ── Exits ─────────────────────────────────────────────────────────────────

    def _process_exits(self):
        r = 0.0
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            r   += self._exit_penalty(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._destination_ll.pop(cs, None)
            self._commanded_heading.pop(cs, None)
            self._steps_since_urgency.pop(cs, None)
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
        return CONFIG['pen_wrong_exit'] \
               if abs(wrap_to_180(bearing - bs.traf.hdg[idx])) > 90 else 0.0

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _process_pending_spawns(self):
        for slot in sorted(s for s, t in self._pending_spawns.items() if t <= 1):
            del self._pending_spawns[slot]
            # Re-enter from the same perimeter sector the slot was originally assigned,
            # keeping traffic patterns consistent across the episode.
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
        for slot in self._pending_spawns:
            self._pending_spawns[slot] -= 1

    def _generate_replacement(self, slot):
        """Try up to max_placement_tries times to find a clear spawn for this slot."""
        occupied = [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0
        ]
        min_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        n       = self.n_aircraft
        for _ in range(CONFIG['max_placement_tries']):
            ac = _place_one(self._polygon_shape, slot, n)
            if all(geo.kwikdist(float(ac['sp_ll'][0]), float(ac['sp_ll'][1]),
                                float(la), float(lo)) >= min_sep
                   for la, lo in occupied):
                return ac
        return None

    def _spawn_aircraft(self, slot, ac):
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['sp_ll'][0]), aclon=float(ac['sp_ll'][1]),
                    achdg=float(ac['heading']), acspd=CONFIG['ac_speed'],
                    acalt=CONFIG['altitude'])
        bs.stack.stack(f'SPD {cs} {int(CONFIG["ac_speed"])}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs]      = ac['dest_ll']
        self._commanded_heading[cs]   = float(ac['heading'])
        self._steps_since_urgency[cs] = 30
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _flat_latlon(self):
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
