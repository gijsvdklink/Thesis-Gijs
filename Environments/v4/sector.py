"""
Sector generation and aircraft placement.

make_polygon builds a random, reasonably round convex sector of a target area.
place_aircraft picks an entry point and an exit (reference) point on the boundary and
derives the straight route between them. All geometry is in the flat NM frame, so a held
route heading reads as exactly zero drift.
"""

import math

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale
from polygenerator import random_convex_polygon

from .config import CONFIG, KM_TO_NM
from .geometry import nm_to_latlon


def make_polygon(area_km2):
    """A random convex polygon of the requested area, centred at the origin (NM frame)."""
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw   = ShapelyPolygon(random_convex_polygon(CONFIG['n_vertices']()))
        scale = math.sqrt(target_nm2 / raw.area)
        scaled = shapely_scale(raw, xfact=scale, yfact=scale, origin='centroid')
        if 4 * math.pi * scaled.area / scaled.length ** 2 >= CONFIG['min_circularity']:
            break
    cx, cy = scaled.centroid.x, scaled.centroid.y
    return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])


def place_aircraft(polygon, sector, n_sectors):
    """Place one aircraft on the polygon boundary and derive its route.

    The entry and exit (reference) points sit at evenly spaced, jittered arc positions.
    The route heading is the spawn->reference bearing; the destination is a far point along
    that heading. Returns lat/lon for spawn, destination and reference plus the route heading.

    ref_jitter spans [-0.5, 0.5], so t_ref can land arbitrarily close to t_spawn and produce a
    near-zero chord. Such an aircraft exits within a step or two without ever really flying,
    which pollutes the arrival statistics. Resample until the chord clears min_chord_nm.
    """
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2) * CONFIG['dest_dist_factor']
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        t_spawn  = (sector + CONFIG['spawn_jitter']()) / n_sectors
        t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
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
