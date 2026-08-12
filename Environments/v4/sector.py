"""
Sector generation and aircraft routing.

make_sector_polygon builds a random, reasonably round convex sector of a target area.
plan_entry_route picks an entry point and an exit (reference) point on the boundary and
derives the straight route between them. All geometry is in the flat NM frame, so a held
route heading reads as exactly zero drift.

Both take an explicit random.Random instance. The environment owns those generators and
keeps the scenario stream separate from the response-delay stream -- see config.py.
"""

import math
import random

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale
from polygenerator import random_convex_polygon

from .config import CONFIG, KM_TO_NM
from .geometry import nm_to_latlon


def _circularity(polygon):
    """4*pi*area / perimeter^2 -- 1.0 for a circle, lower for elongated shapes."""
    return 4 * math.pi * polygon.area / polygon.length ** 2


def _random_convex_polygon(n_vertices, rng):
    """polygenerator's random_convex_polygon, driven by `rng` instead of global state.

    The library reads Python's global `random` stream and offers no way to inject a
    generator, so the global stream is seeded from `rng` for the duration of the call
    and put back exactly as it was afterwards. The result is reproducible from `rng`
    alone, and no other user of the global stream notices.
    """
    saved_state = random.getstate()
    random.seed(rng.randrange(2 ** 32))
    try:
        return random_convex_polygon(n_vertices)
    finally:
        random.setstate(saved_state)


def make_sector_polygon(area_km2, rng):
    """A random convex polygon of the requested area, centred at the origin (NM frame).

    Retries until the shape is reasonably round (circularity >= min_circularity). Slivers
    are rejected because every route across one would be a short clip of a corner rather
    than a genuine crossing.
    """
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw   = ShapelyPolygon(_random_convex_polygon(CONFIG['n_vertices'](rng), rng))
        scale = math.sqrt(target_nm2 / raw.area)
        scaled = shapely_scale(raw, xfact=scale, yfact=scale, origin='centroid')
        if _circularity(scaled) >= CONFIG['min_circularity']:
            break

    cx, cy = scaled.centroid.x, scaled.centroid.y
    return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])


def plan_entry_route(polygon, sector, n_sectors, rng):
    """Plan one aircraft's crossing: where it enters, where it should leave, and on what
    heading. Returns lat/lon for spawn, destination and reference point plus the heading.

    Entry and exit points sit at evenly spaced, jittered positions along the boundary, so
    the traffic is spread around the sector rather than clustered. The exit starts half a
    perimeter from the entry and is jittered by +-0.5, which makes crossing directions
    fully random.

    The destination is placed dest_dist_factor sector-diameters BEYOND the exit point.
    Being that far away, the bearing to it barely changes as the aircraft flies, so simply
    holding a heading keeps the aircraft on route and "drift" stays well defined.

    Because the exit jitter spans a full half-perimeter, it can land close to the entry and
    produce a near-zero chord. Such an aircraft would leave within a step or two without
    ever really flying, polluting the arrival statistics, so short chords are resampled.

    The returned 'ref_ll' is where this aircraft would cross the boundary if it flew the
    whole way without ever turning. env._score_arrival keeps it and measures the exit
    deviation against it, which is why it is returned even though the simulation itself
    never uses it.
    """
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2) * CONFIG['dest_dist_factor']
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        t_spawn  = (sector + CONFIG['spawn_jitter'](rng)) / n_sectors
        t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter'](rng)) % 1.0
        spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
        ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)
        if math.hypot(ref_pt.x - spawn_pt.x, ref_pt.y - spawn_pt.y) >= min_chord:
            break

    route_hdg = math.degrees(math.atan2(ref_pt.x - spawn_pt.x,
                                        ref_pt.y - spawn_pt.y)) % 360.0
    dest_e = spawn_pt.x + dest_dist * math.sin(math.radians(route_hdg))
    dest_n = spawn_pt.y + dest_dist * math.cos(math.radians(route_hdg))

    center = CONFIG['center_ll']
    return {
        'sp_ll':   nm_to_latlon(center, spawn_pt.x, spawn_pt.y),
        'dest_ll': nm_to_latlon(center, dest_e,     dest_n),
        'ref_ll':  nm_to_latlon(center, ref_pt.x,   ref_pt.y),
        'heading': route_hdg,
    }
