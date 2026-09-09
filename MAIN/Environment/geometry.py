# The flat east/north NM frame the controller reasons in, and the pair separation maths.

import math

import numpy as np

from bluesky.tools.aero import nm as _M_PER_NM     # 1852.0
from bluesky.tools.geo import qdrpos
from bluesky.tools.misc import degto180

from .config import CONFIG, NO_CONFLICT_S

NMS_PER_MS = 1.0 / _M_PER_NM   # m/s -> NM/s

# Drift small enough to count as back on route, in the units heading_drift returns. Taken from
# the on-route tolerance so the ranking and the on-route KPI agree on what "on route" means.
ON_ROUTE_DRIFT = 1 - math.cos(math.radians(CONFIG['on_route_hdg_tol_deg']))


def heading_drift(initial_hdg, actual_hdg):
    # 0 when on the assigned heading, 2 when reversed. The focus tie-break and the drift
    # penalty are the same measure, so they are the same function.
    return 1 - math.cos(math.radians(degto180(initial_hdg - actual_hdg)))


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


def time_to_los(tcpa, dcpa_sq, safe_rel, moving):
    # Seconds until the pair is first within sep_nm, and inf when they never close inside it.
    # Written once here: the urgency ranking and the spawn test must agree on what a predicted
    # loss of separation is.
    sep_sq   = CONFIG['sep_nm'] ** 2
    intrudes = moving & (tcpa >= 0) & (dcpa_sq < sep_sq)
    return np.where(intrudes,
                    tcpa - np.sqrt(np.maximum(0.0, sep_sq - dcpa_sq) / safe_rel),
                    np.inf)


def pairwise(pos, vel):
    rel_pos = pos[None, :, :] - pos[:, None, :]     # rel_pos[i, j] = pos[j] - pos[i]
    rel_vel = vel[None, :, :] - vel[:, None, :]

    dist_sq, tcpa, dcpa_sq, safe_rel, moving = cpa(rel_pos, rel_vel)
    dcpa_sq = np.maximum(0.0, dcpa_sq)              # against round-off; dcpa is never really negative

    return dist_sq, time_to_los(tcpa, dcpa_sq, safe_rel, moving)


def urgency_matrix(pos, vel):
    # The urgency matrix: 0 outside the horizon, ramping to 1 at the moment of intrusion, then
    # 1 to 10 once separation is actually lost. Returns (urgency, t_los), t_los capped at the
    # horizon -- an unbounded one would dominate the VecNormalize variance.
    n = len(pos)
    if n < 2:
        return np.zeros((n, n)), np.full((n, n), NO_CONFLICT_S)

    sep, t_warn   = CONFIG['sep_nm'], CONFIG['t_warn']
    dist_sq, t_los = pairwise(pos, vel)

    urgency = np.where(t_los <= t_warn, (t_warn - np.clip(t_los, 0.0, t_warn)) / t_warn, 0.0)
    in_los  = dist_sq < sep * sep
    urgency = np.where(in_los, 1.0 + 9.0 * (1.0 - np.sqrt(dist_sq) / sep), urgency)
    np.fill_diagonal(urgency, 0.0)

    return urgency, np.clip(t_los, 0.0, NO_CONFLICT_S)
