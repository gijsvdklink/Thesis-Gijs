# The flat east/north NM frame the controller reasons in, and the pair separation maths.

import math

import numpy as np

from bluesky.tools.aero import nm as _M_PER_NM     # 1852.0
from bluesky.tools.geo import qdrpos

from .config import CONFIG

NMS_PER_MS = 1.0 / _M_PER_NM   # m/s -> NM/s


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


def point_ahead(from_ll, heading_deg, distance_nm):
    """(lat, lon) reached by flying `distance_nm` from `from_ll` on a constant TRUE heading."""
    lat, lon = qdrpos(from_ll[0], from_ll[1], heading_deg, distance_nm)
    return float(lat), float(lon)


def heading_to_velocity(speed, heading_deg):
    """Speed + heading (deg) -> (east, north) velocity components."""
    h = math.radians(heading_deg)
    return speed * math.sin(h), speed * math.cos(h)


_TINY = 1e-12


def pairwise(pos, vel):
    """(dist_sq, range_rate, rel_spd_sq, t_los) for every pair; t_los is +inf when they never intrude."""
    d  = pos[None, :, :] - pos[:, None, :]        # d[i, j] = pos[j] - pos[i]
    dv = vel[None, :, :] - vel[:, None, :]

    dist_sq    = np.einsum('ijk,ijk->ij', d, d)
    rel_spd_sq = np.einsum('ijk,ijk->ij', dv, dv)
    range_rate = np.einsum('ijk,ijk->ij', d, dv)

    safe_rel = np.where(rel_spd_sq < _TINY, 1.0, rel_spd_sq)
    tcpa     = -range_rate / safe_rel
    dcpa_sq  = np.maximum(0.0, dist_sq - range_rate ** 2 / safe_rel)

    sep_sq   = CONFIG['sep_nm'] ** 2
    intrudes = (rel_spd_sq >= _TINY) & (tcpa >= 0) & (dcpa_sq < sep_sq)
    t_los    = np.where(intrudes,
                        tcpa - np.sqrt(np.maximum(0.0, sep_sq - dcpa_sq) / safe_rel),
                        np.inf)
    return dist_sq, range_rate, rel_spd_sq, t_los
