# Coordinate transforms in a flat east/north NM frame centred on the sector, valid at the equator.

import math
import numpy as np

NMS_PER_MS = 1.0 / 1852.0   # m/s -> NM/s


def latlon_to_nm(center_ll, lat, lon):
    """(lat, lon) -> (east_nm, north_nm) relative to center_ll."""
    ref_lat, ref_lon = center_ll
    east_nm  = (lon - ref_lon) * 60.0 * math.cos(math.radians(ref_lat))
    north_nm = (lat - ref_lat) * 60.0
    return np.array([east_nm, north_nm])


def nm_to_latlon(center_ll, east_nm, north_nm):
    """(east_nm, north_nm) offsets -> (lat, lon)."""
    ref_lat, ref_lon = center_ll
    return (ref_lat + north_nm / 60.0,
            ref_lon + east_nm / (60.0 * math.cos(math.radians(ref_lat))))


def wrap_to_180(angle_deg):
    """Wrap an angle in degrees to (-180, 180]."""
    return (angle_deg + 180) % 360 - 180


def point_ahead(from_ll, heading_deg, distance_nm):
    """(lat, lon) reached by flying `distance_nm` from `from_ll` on a constant TRUE heading."""
    heading  = math.radians(heading_deg)
    lat      = from_ll[0] + distance_nm * math.cos(heading) / 60.0
    mean_lat = math.radians((from_ll[0] + lat) / 2.0)
    lon      = from_ll[1] + distance_nm * math.sin(heading) / (60.0 * math.cos(mean_lat))
    return lat, lon


def heading_to_velocity(speed, heading_deg):
    """Speed + heading (deg) -> (east, north) velocity components."""
    h = math.radians(heading_deg)
    return speed * math.sin(h), speed * math.cos(h)
