# Sector generation and entry routing, in the flat NM frame.

import math
import random

import numpy as np

from shapely.geometry import LineString, Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale
from polygenerator import random_convex_polygon

from .config import CONFIG, KM_TO_NM


def _circularity(polygon):
    return 4 * math.pi * polygon.area / polygon.length ** 2


def _random_convex_polygon(n_vertices, rng):
    saved_state = random.getstate()
    random.seed(rng.randrange(2 ** 32))
    try:
        return random_convex_polygon(n_vertices)
    finally:
        random.setstate(saved_state)


def make_sector_polygon(area_km2, rng):
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


def exit_point(polygon, start_nm, heading_deg):
    east, north = float(start_nm[0]), float(start_nm[1])
    h = math.radians(heading_deg)

    minx, miny, maxx, maxy = polygon.bounds
    reach = 2.0 * math.hypot(maxx - minx, maxy - miny)   # past any point of the sector
    ray   = LineString([(east, north),
                        (east + reach * math.sin(h), north + reach * math.cos(h))])

    chord = ray.intersection(polygon)
    if chord.is_empty or chord.geom_type != 'LineString':
        return np.array([east, north])

    far = max(chord.coords, key=lambda c: math.hypot(c[0] - east, c[1] - north))
    return np.array(far)


def plan_entry_route(polygon, rng):
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        entry   = polygon.exterior.interpolate(rng.uniform(0.0, 1.0), normalized=True)
        heading = rng.uniform(0.0, 360.0)
        start   = np.array([entry.x, entry.y])

        leaves = exit_point(polygon, start, heading)
        if math.hypot(leaves[0] - start[0], leaves[1] - start[1]) >= min_chord:
            return {'pos_nm': start, 'heading': heading}
    return None
