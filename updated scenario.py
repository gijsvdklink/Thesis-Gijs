"""
ATC Scenario — sector and traffic generator.

Generates a random convex sector, populates it with A320 aircraft at FL350,
and advances the BlueSky simulation.  Exited aircraft are replaced to maintain
constant traffic density throughout an episode.

Intended as a standalone scenario layer that can be wrapped by any RL
environment or used directly for analysis / visualisation.

Usage
-----
    from scenario import Scenario, CONFIG

    sc = Scenario()
    sc.reset()

    for _ in range(200):
        info = sc.step()
        print(info['n_conflicts'], info['n_los'])
        for ac in sc.aircraft_states():
            print(ac['callsign'], ac['lat'], ac['lon'])

Configuration
-------------
All parameters live in CONFIG.  Override before constructing Scenario:

    from scenario import CONFIG
    CONFIG['n_aircraft'] = lambda: 5          # fixed aircraft count
    sc = Scenario()
"""

import math
import random

import numpy as np

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.tools import geo
from bluesky.stack.stackbase import Stack as _BsStack
from polygenerator import random_convex_polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Aircraft
    'ac_type':               'A320',
    'ac_speed':              450.0,           # kts TAS ≈ M0.78 at FL350
    'ac_mach':               0.78,            # cruise Mach (BlueSky SPD command)
    'altitude':              350,             # FL350
    # Sector
    'center_ll':             (52.3, 5.3),     # sector centre (lat, lon) — Dutch upper airspace
    'n_aircraft':            lambda: random.randint(2, 15),   # uniform discrete [2, 15]
    'density_km2':           lambda: random.uniform(5_000.0, 15_000.0),
    # area is derived: n_aircraft × density_km2  (ensures uniform joint distribution)
    'sep_nm':                5.0,             # ICAO separation standard (NM)
    'buffer_nm':             5.0,             # extra clearance above sep for spawning
    'dest_dist_factor':      2.0,             # destination placed this × diameter beyond boundary
    # Polygon shape
    'n_vertices':            lambda: random.randint(5, 7),
    'min_circularity':       0.65,            # isoperimetric ratio filter
    # Aircraft placement jitter
    'spawn_jitter':          lambda: random.uniform(0.1, 0.9),
    'ref_jitter':            lambda: random.uniform(-0.30, 0.30),
    'max_placement_tries':   50,
    # Simulation timing
    'sim_dt':                0.5,             # BlueSky timestep (s)
    'step_s':                30.0,            # simulated seconds per scenario step
    'lookahead_s':           900.0,           # conflict detection lookahead (15 min)
    't_warn':                600.0,           # urgency ramp start — 10 min before CPA
    'crossings_per_episode': 2.5,             # episode ≈ 2.5 sector crossings ≈ 60 min
    'spawn_delay_s':         (0, 0),          # delay before replacing an exited aircraft
    # Reproducibility
    'seed':                  None,
}

NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

_bs_initialized = False

# ── Coordinate helpers ────────────────────────────────────────────────────────

def latlon_to_nm(center_ll, lat, lon):
    """Convert (lat, lon) to local NM coordinates centred on center_ll."""
    clat, clon = center_ll
    return np.array([
        (lon - clon) * 60.0 * math.cos(math.radians(clat)),
        (lat - clat) * 60.0,
    ])

def nm_to_latlon(center_ll, x_nm, y_nm):
    """Inverse of latlon_to_nm."""
    clat, clon = center_ll
    return (clat + y_nm / 60.0,
            clon + x_nm / (60.0 * math.cos(math.radians(clat))))

def wrap_to_180(a):
    """Wrap angle (degrees) to (−180, +180]."""
    return (a + 180) % 360 - 180

def _tas_nms(i):
    """True airspeed of aircraft i in NM/s."""
    return float(bs.traf.tas[i]) / 1852.0

# ── Sector polygon ────────────────────────────────────────────────────────────

def make_polygon(area_km2):
    """
    Generate a random convex sector polygon of the requested area.

    Parameters
    ----------
    area_km2 : float
        Target sector area in km².

    Returns
    -------
    ShapelyPolygon
        Convex polygon centred at the NM origin, with vertices in NM.
    """
    target_nm2 = area_km2 * KM_TO_NM ** 2
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

def _place_one(polygon, sector, n_sectors, dest_dist_nm):
    """
    Place one aircraft in perimeter arc `sector` of `n_sectors`.

    The aircraft spawns on the boundary and flies toward a reference point
    on the roughly opposite arc, then continues to a destination well
    beyond the sector boundary.

    Returns a dict with keys: spawn_ll, dest_ll, heading.
    """
    t_spawn  = (sector + CONFIG['spawn_jitter']()) / n_sectors
    t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
    sp       = polygon.exterior.interpolate(t_spawn, normalized=True)
    rp       = polygon.exterior.interpolate(t_ref,   normalized=True)
    sp_ll    = nm_to_latlon(CONFIG['center_ll'], sp.x, sp.y)
    rp_ll    = nm_to_latlon(CONFIG['center_ll'], rp.x, rp.y)
    hdg, _   = geo.kwikqdrdist(*[float(v) for v in (*sp_ll, *rp_ll)])
    dlat, dlon = geo.qdrpos(float(sp_ll[0]), float(sp_ll[1]), hdg, dest_dist_nm)
    return {'spawn_ll': sp_ll, 'dest_ll': (dlat, dlon), 'heading': hdg}

def place_aircraft(polygon, n_aircraft):
    """
    Place n_aircraft on the polygon boundary using equal-arc sectors.

    Each aircraft gets its own arc of the perimeter.  Placement is retried
    up to max_placement_tries times per slot to satisfy minimum separation.

    Returns a list of aircraft dicts, or None if any slot fails.
    """
    min_sep_nm   = CONFIG['sep_nm'] + CONFIG['buffer_nm']
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist_nm = (math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
                    * CONFIG['dest_dist_factor'])
    placed = []
    for sector in range(n_aircraft):
        for _ in range(CONFIG['max_placement_tries']):
            ac = _place_one(polygon, sector, n_aircraft, dest_dist_nm)
            too_close = any(
                geo.kwikdist(float(ac['spawn_ll'][0]), float(ac['spawn_ll'][1]),
                             float(p['spawn_ll'][0]),  float(p['spawn_ll'][1])) < min_sep_nm
                for p in placed
            )
            if not too_close:
                placed.append(ac)
                break
        else:
            return None
    return placed

# ── Conflict geometry ─────────────────────────────────────────────────────────

def pair_urgency(i, j):
    """
    Urgency score for BlueSky aircraft pair (i, j).

    Combines intrusion depth (for active LoS) and time pressure (for
    predicted conflicts) into a single score:

        [1, 10]   active LoS  (current distance < sep_nm)
        (0,  1]   predicted conflict within lookahead, ramping from t_warn
        0         diverging or no conflict predicted

    The ramp is linear from 0 at t_warn to 1 at tcpa = 0, so urgency
    reaches 1 exactly when the LoS branch takes over.
    """
    nm_i = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[i], bs.traf.lon[i])
    nm_j = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[j], bs.traf.lon[j])
    dx, dy = nm_j[0]-nm_i[0], nm_j[1]-nm_i[1]
    d2     = dx*dx + dy*dy
    sep    = CONFIG['sep_nm']

    # Active LoS: score in [1, 10], higher = deeper intrusion
    if d2 < sep*sep:
        return 1.0 + 9.0 * (1.0 - math.sqrt(d2) / sep)

    # Compute relative velocity
    spd_i = _tas_nms(i);  spd_j = _tas_nms(j)
    vn_i  = spd_i * math.cos(math.radians(bs.traf.hdg[i]))
    ve_i  = spd_i * math.sin(math.radians(bs.traf.hdg[i]))
    vn_j  = spd_j * math.cos(math.radians(bs.traf.hdg[j]))
    ve_j  = spd_j * math.sin(math.radians(bs.traf.hdg[j]))
    dvn, dve = vn_j-vn_i, ve_j-ve_i
    rv2      = dvn*dvn + dve*dve
    if rv2 < 1e-12:
        return 0.0  # identical velocities — no relative motion

    # Time to CPA
    dot  = dx*dve + dy*dvn
    tcpa = -dot / rv2
    if tcpa < 0 or tcpa > CONFIG['lookahead_s']:
        return 0.0  # diverging or beyond lookahead

    # Miss distance at CPA
    if max(0.0, d2 - dot*dot/rv2) >= sep*sep:
        return 0.0  # will not violate separation

    return min(1.0, max(0.0, (CONFIG['t_warn'] - tcpa) / CONFIG['t_warn']))

# ── BlueSky screen stub ───────────────────────────────────────────────────────

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0): pass

# ── Scenario ──────────────────────────────────────────────────────────────────

class Scenario:
    """
    Self-contained airspace scenario.

    Manages sector generation, aircraft spawning and replacement, simulation
    stepping, and conflict monitoring.  Does not contain any RL-specific logic
    (no observations, rewards, or actions).

    Attributes
    ----------
    polygon : np.ndarray, shape (V, 2)
        Sector boundary vertices in local NM coordinates (x_east, y_north).
    n_aircraft : int
        Number of aircraft slots (maintained constant throughout episode).
    urgency_matrix : np.ndarray, shape (N, N)
        Pairwise urgency scores for all active aircraft (see pair_urgency).
    urgency_cs_list : list[str]
        Callsign order matching the rows/columns of urgency_matrix.
    step_count : int
        Number of steps elapsed since last reset.
    max_steps : int
        Episode length in steps (derived from sector size at reset).
    """

    def __init__(self):
        global _bs_initialized
        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._n_substeps       = round(CONFIG['step_s'] / CONFIG['sim_dt'])
        self._slots            = []
        self._active_callsigns = set()
        self._destination_ll   = {}
        self._next_id          = 0
        self._pending_spawns   = {}
        self._spawn_delay_range = (1, 1)
        self._polygon_shape    = None

        self.polygon           = None
        self.n_aircraft        = 0
        self.urgency_matrix    = np.zeros((0, 0))
        self.urgency_cs_list   = []
        self.step_count        = 0
        self.max_steps         = 0

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def active_callsigns(self):
        """Frozenset of callsigns currently inside the sector."""
        return frozenset(self._active_callsigns)

    @property
    def destination_ll(self):
        """Dict mapping callsign → (lat, lon) destination."""
        return dict(self._destination_ll)

    def reset(self, seed=None):
        """
        Initialise a new episode.

        Generates a fresh random sector, places aircraft, and resets all
        counters.  Safe to call multiple times.
        """
        effective_seed = seed if seed is not None else CONFIG['seed']
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        n_ac        = CONFIG['n_aircraft']()
        density_km2 = CONFIG['density_km2']()
        area_km2    = float(n_ac * density_km2)

        # Retry until polygon + placement both succeed
        while True:
            polygon  = make_polygon(area_km2)
            aircraft = place_aircraft(polygon, n_ac)
            if aircraft is not None:
                break

        # Episode length: time for crossings_per_episode sector crossings
        minx, miny, maxx, maxy = polygon.bounds
        diam_nm = math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
        self.max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * diam_nm / CONFIG['ac_speed'] * 3600
            / CONFIG['step_s']
        ))

        self._polygon_shape    = polygon
        self.polygon           = np.array(polygon.exterior.coords[:-1])
        self.n_aircraft        = n_ac
        self._slots            = [None] * n_ac
        self._active_callsigns = set()
        self._destination_ll   = {}
        self._next_id          = 0
        self.step_count        = 0
        self.urgency_matrix    = np.zeros((0, 0))
        self.urgency_cs_list   = []

        step_s    = CONFIG['step_s']
        delay_min = max(1, round(CONFIG['spawn_delay_s'][0] / step_s))
        delay_max = max(1, round(CONFIG['spawn_delay_s'][1] / step_s))
        self._spawn_delay_range = (delay_min, delay_max)
        self._pending_spawns    = {}

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._sector_latlon())
        bs.stack.stack('ASAS OFF')

        for i, ac in enumerate(aircraft):
            self._spawn(i, ac)

        self._update_urgency()

    def step(self):
        """
        Advance the scenario by one step (30 s simulated).

        Processes pending replacement spawns, advances the BlueSky simulation,
        handles exits, and recomputes the urgency matrix.

        Returns
        -------
        dict with keys:
            'n_los'       : number of aircraft pairs currently in LoS
            'n_conflicts' : number of pairs with urgency > 0
            'truncated'   : True if episode length has been reached
        """
        self._process_pending_spawns()

        for _ in range(self._n_substeps):
            bs.sim.step()
        self.step_count += 1

        self._process_exits()
        self._update_urgency()

        U = self.urgency_matrix
        n_los       = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
        n_conflicts = int((U > 0.0).sum()) // 2 if U.size > 0 else 0

        return {
            'n_los':       n_los,
            'n_conflicts': n_conflicts,
            'truncated':   self.step_count >= self.max_steps,
        }

    def aircraft_states(self):
        """
        Return current state of all active aircraft.

        Returns
        -------
        list of dicts, each containing:
            callsign, lat, lon, hdg, tas_kt, mach, alt_ft, dest_ll
        """
        states = []
        for cs in sorted(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue
            states.append({
                'callsign': cs,
                'lat':      float(bs.traf.lat[idx]),
                'lon':      float(bs.traf.lon[idx]),
                'hdg':      float(bs.traf.hdg[idx]),
                'tas_kt':   float(bs.traf.tas[idx]) / 0.5144,
                'mach':     float(bs.traf.M[idx]),
                'alt_ft':   float(bs.traf.alt[idx]) / 0.3048,
                'dest_ll':  self._destination_ll.get(cs),
            })
        return states

    # ── Urgency ───────────────────────────────────────────────────────────────

    def _update_urgency(self):
        """Recompute the full N×N urgency matrix for all active aircraft."""
        cs_list = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not cs_list:
            self.urgency_matrix  = np.zeros((0, 0))
            self.urgency_cs_list = []
            return

        n      = len(cs_list)
        bs_idx = [bs.traf.id2idx(cs) for cs in cs_list]
        U      = np.zeros((n, n))
        for ii in range(n):
            for jj in range(ii+1, n):
                u = pair_urgency(bs_idx[ii], bs_idx[jj])
                U[ii, jj] = U[jj, ii] = u

        self.urgency_matrix  = U
        self.urgency_cs_list = cs_list

    # ── Exits and replacements ────────────────────────────────────────────────

    def _process_exits(self):
        """Remove aircraft that have left the sector and schedule replacements."""
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active_callsigns.discard(cs)
            self._destination_ll.pop(cs, None)
            self._slots[slot] = None
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = random.randint(*self._spawn_delay_range)

    def _find_exited(self):
        """Return callsigns that are no longer inside the sector polygon."""
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

    def _process_pending_spawns(self):
        """Spawn replacements whose countdown has elapsed; decrement others."""
        ready = sorted(slot for slot, t in self._pending_spawns.items() if t <= 1)
        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement()
            if ac is not None:
                self._spawn(slot, ac)
            else:
                self._pending_spawns[slot] = 5  # sector too dense — retry in 5 steps
        for slot in list(self._pending_spawns):
            self._pending_spawns[slot] -= 1

    def _generate_replacement(self):
        """
        Find a valid spawn position on the sector boundary.

        Tries up to max_placement_tries random positions and returns the first
        one that is at least sep_nm + buffer_nm clear of all active aircraft.
        Returns None if no valid position is found.
        """
        occupied    = [(bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
                       for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        min_sep_nm  = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        minx, miny, maxx, maxy = self._polygon_shape.bounds
        dest_dist_nm = (math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
                        * CONFIG['dest_dist_factor'])

        for _ in range(CONFIG['max_placement_tries']):
            ac   = _place_one(self._polygon_shape, random.randint(0, self.n_aircraft - 1),
                               self.n_aircraft, dest_dist_nm)
            slat = float(ac['spawn_ll'][0])
            slon = float(ac['spawn_ll'][1])
            if all(geo.kwikdist(slat, slon, float(la), float(lo)) >= min_sep_nm
                   for la, lo in occupied):
                return ac
        return None

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn(self, slot, ac):
        """Create an aircraft in BlueSky and register it in the given slot."""
        cs   = f'AC{self._next_id:02d}'
        self._next_id += 1
        mach = CONFIG['ac_mach']
        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['spawn_ll'][0]), aclon=float(ac['spawn_ll'][1]),
                    achdg=float(ac['heading']),     acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')
        self._destination_ll[cs] = ac['dest_ll']
        self._slots[slot]        = cs
        self._active_callsigns.add(cs)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _sector_latlon(self):
        """Flatten polygon vertices to the [lat, lon, lat, lon, ...] list BlueSky expects."""
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
