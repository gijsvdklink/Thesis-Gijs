# Separation geometry: time-to-LoS, pairwise urgency (which drives focus selection and
# intruder ordering), the "safe to return" check, and the live LoS check.
#
# The pairwise functions take POSITION AND VELOCITY ARRAYS and work on the whole traffic
# picture at once. With 15-30 aircraft that is 100-450 pairs every step, so doing it in
# numpy rather than a Python loop is where most of the environment's CPU time was going.

import math

import numpy as np
import bluesky as bs

from .config import CONFIG
from .geometry import NMS_PER_MS

_TINY = 1e-12


def traffic_states(indices):
    """Positions (NM, east/north) and velocities (NM/s) for BlueSky indices, as (n, 2) arrays."""
    idx = np.asarray(indices, dtype=int)
    ref_lat, ref_lon = CONFIG['center_ll']

    east  = (bs.traf.lon[idx] - ref_lon) * 60.0 * math.cos(math.radians(ref_lat))
    north = (bs.traf.lat[idx] - ref_lat) * 60.0
    speed = bs.traf.tas[idx] * NMS_PER_MS
    hdg   = np.radians(bs.traf.hdg[idx])

    pos = np.stack([east, north], axis=1)
    vel = np.stack([speed * np.sin(hdg), speed * np.cos(hdg)], axis=1)
    return pos, vel


def time_to_loss_of_separation(dist_sq, range_rate, rel_spd_sq, sep):
    """Seconds until two aircraft first lose separation, or None if they never do.

    range_rate = r.v (negative = converging). This is EARLIER than time-to-CPA: a pair
    enters the protected circle before closest approach.
        t_los = tcpa - sqrt((sep^2 - dcpa^2) / |v|^2)
    """
    if rel_spd_sq < _TINY:
        return None
    tcpa = -range_rate / rel_spd_sq
    if tcpa < 0:
        return None                                    # diverging
    # Clamped: the subtraction can round marginally negative for collinear pairs.
    dcpa_sq = max(0.0, dist_sq - range_rate ** 2 / rel_spd_sq)
    if dcpa_sq >= sep * sep:
        return None                                    # miss distance too large
    return tcpa - math.sqrt((sep * sep - dcpa_sq) / rel_spd_sq)


def _pairwise(pos, vel):
    """(dist_sq, range_rate, rel_spd_sq, t_los) for every pair, as (n, n) arrays.

    t_los is +inf for pairs that never intrude, so a single comparison against the
    horizon covers both "no conflict" and "too far off".
    """
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


def urgency_matrix(pos, vel):
    """Symmetric pairwise urgency over the traffic picture.

      > 1     active LoS    (1 at the separation boundary -> 10 at zero distance)
      0..1    predicted LoS (0 at t_warn -> 1 at intrusion now)
      0       safe / diverging

    The LoS branch starts exactly where the predicted branch tops out, so an active loss
    always outranks every predicted one.
    """
    n = len(pos)
    if n < 2:
        return np.zeros((n, n))

    sep, t_warn = CONFIG['sep_nm'], CONFIG['t_warn']
    dist_sq, _, _, t_los = _pairwise(pos, vel)

    urgency = np.where(t_los <= t_warn, (t_warn - np.clip(t_los, 0.0, t_warn)) / t_warn, 0.0)
    in_los  = dist_sq < sep * sep
    urgency = np.where(in_los, 1.0 + 9.0 * (1.0 - np.sqrt(dist_sq) / sep), urgency)

    np.fill_diagonal(urgency, 0.0)
    return urgency


def route_return_blocked(pos, route_vel):
    """Per aircraft, 1.0 if turning back onto its route is NOT free.

    Every aircraft is put on its ROUTE heading (which is robust to in-progress avoidance
    manoeuvres) and the resulting corridors are checked against each other. Blocked means
    already in LoS with someone, or losing separation with them within t_warn.
    """
    n = len(pos)
    if n < 2:
        return np.zeros(n)

    dist_sq, _, _, t_los = _pairwise(pos, route_vel)
    blocked = (dist_sq < CONFIG['sep_nm'] ** 2) | (t_los <= CONFIG['t_warn'])
    np.fill_diagonal(blocked, False)
    return blocked.any(axis=1).astype(float)
