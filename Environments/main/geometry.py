# The flat east/north NM frame the controller reasons in, and the pair separation maths.

import math

import numpy as np

from bluesky.tools.aero import nm as _M_PER_NM     # 1852.0
from bluesky.tools.geo import qdrpos

from .config import CONFIG

NMS_PER_MS = 1.0 / _M_PER_NM   # m/s -> NM/s


def latlon_to_nm(center_ll, lat, lon):
    ref_lat, ref_lon = center_ll
    east  = (np.asarray(lon) - ref_lon) * 60.0 * math.cos(math.radians(ref_lat))
    north = (np.asarray(lat) - ref_lat) * 60.0
    return np.stack([east, north], axis=-1)


def nm_to_latlon(center_ll, east_nm, north_nm):
    ref_lat, ref_lon = center_ll
    return (ref_lat + north_nm / 60.0,
            ref_lon + east_nm / (60.0 * math.cos(math.radians(ref_lat))))


def point_ahead(from_ll, heading_deg, distance_nm):
    lat, lon = qdrpos(from_ll[0], from_ll[1], heading_deg, distance_nm)
    return float(lat), float(lon)


def heading_to_velocity(speed, heading_deg):
    h = math.radians(heading_deg)
    return speed * math.sin(h), speed * math.cos(h)


_TINY = 1e-12


def cpa(rel_pos, rel_vel):
    dist_sq    = np.einsum('...k,...k->...', rel_pos, rel_pos)
    rel_spd_sq = np.einsum('...k,...k->...', rel_vel, rel_vel)
    range_rate = np.einsum('...k,...k->...', rel_pos, rel_vel)

    moving   = rel_spd_sq >= _TINY
    safe_rel = np.where(moving, rel_spd_sq, 1.0)
    return dist_sq, -range_rate / safe_rel, dist_sq - range_rate ** 2 / safe_rel, safe_rel, moving


def pairwise(pos, vel):
    rel_pos = pos[None, :, :] - pos[:, None, :]     # rel_pos[i, j] = pos[j] - pos[i]
    rel_vel = vel[None, :, :] - vel[:, None, :]

    dist_sq, tcpa, dcpa_sq, safe_rel, moving = cpa(rel_pos, rel_vel)
    dcpa_sq = np.maximum(0.0, dcpa_sq)              # against round-off; dcpa is never really negative

    sep_sq   = CONFIG['sep_nm'] ** 2
    intrudes = moving & (tcpa >= 0) & (dcpa_sq < sep_sq)
    t_los    = np.where(intrudes,
                        tcpa - np.sqrt(np.maximum(0.0, sep_sq - dcpa_sq) / safe_rel),
                        np.inf)
    return dist_sq, t_los
