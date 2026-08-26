"""Sector generation and aircraft routing in the flat NM frame; both entry points take an explicit rng."""

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
    """polygenerator's random_convex_polygon, driven by `rng` instead of the global random stream."""
    saved_state = random.getstate()
    random.seed(rng.randrange(2 ** 32))
    try:
        return random_convex_polygon(n_vertices)
    finally:
        random.setstate(saved_state)


def make_sector_polygon(area_km2, rng):
    """A random convex polygon of the requested area at the origin, retried until reasonably round."""
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
    """Plan one crossing: where it enters, its INITIAL HEADING, and the exit point it would reach unturned."""
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        t_spawn  = (sector + CONFIG['spawn_jitter'](rng)) / n_sectors
        t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter'](rng)) % 1.0
        spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
        ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)
        if math.hypot(ref_pt.x - spawn_pt.x, ref_pt.y - spawn_pt.y) >= min_chord:
            break

    initial_hdg = math.degrees(math.atan2(ref_pt.x - spawn_pt.x,
                                          ref_pt.y - spawn_pt.y)) % 360.0

    center = CONFIG['center_ll']
    return {
        'sp_ll':   nm_to_latlon(center, spawn_pt.x, spawn_pt.y),
        'heading': initial_hdg,
    }
