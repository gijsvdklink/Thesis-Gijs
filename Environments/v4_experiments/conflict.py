"""
Separation geometry: time-to-loss-of-separation, pairwise urgency (which drives focus
selection and intruder ordering), the "safe to return" check, and the live LoS check.

All functions are pure (no env state): they read the live BlueSky traffic by callsign
or index and take any extra state (active callsigns, route headings) as arguments.
"""

import math
import bluesky as bs

from .config import CONFIG
from .geometry import (aircraft_position_nm, aircraft_speed_nms, aircraft_state,
                       heading_to_velocity)


def time_to_los(dist_sq, range_rate, rel_spd_sq, sep):
    """Seconds until two aircraft first LOSE SEPARATION (distance < sep), from their
    current relative state. range_rate = r.v (negative = converging), rel_spd_sq = |v|^2.

    Returns the LoS-entry time (>= 0), or None if they never intrude (diverging, parallel,
    or miss distance >= sep). This is EARLIER than time-to-CPA: a pair enters the protected
    circle before closest approach, so it is the honest "time until intrusion".
        t_los = tcpa - sqrt((sep^2 - dcpa^2) / |v|^2)
    """
    if rel_spd_sq < 1e-12:
        return None
    tcpa = -range_rate / rel_spd_sq
    if tcpa < 0:
        return None                                    # diverging
    dcpa_sq = max(0.0, dist_sq - range_rate ** 2 / rel_spd_sq)
    if dcpa_sq >= sep * sep:
        return None                                    # miss distance too large: never intrudes
    return tcpa - math.sqrt((sep * sep - dcpa_sq) / rel_spd_sq)


def urgency_from_state(pos_i, vel_i, pos_j, vel_j):
    """Urgency of the conflict between two aircraft given their planar state (NM, NM/s).

    Returns:
      > 1     active LoS    (1 at the sep boundary -> 10 at zero distance)
      0..1    predicted LoS (0 at t_warn -> 1 at t_los = 0, i.e. intrusion now)
      0       safe / diverging
    """
    d_east  = pos_j[0] - pos_i[0]
    d_north = pos_j[1] - pos_i[1]
    dist_sq = d_east ** 2 + d_north ** 2
    sep     = CONFIG['sep_nm']

    if dist_sq < sep ** 2:
        return 1.0 + 9.0 * (1.0 - math.sqrt(dist_sq) / sep)

    dv_east    = vel_j[0] - vel_i[0]
    dv_north   = vel_j[1] - vel_i[1]
    rel_spd_sq = dv_east ** 2 + dv_north ** 2
    range_rate = d_east * dv_east + d_north * dv_north   # r.v; negative = converging

    t_los = time_to_los(dist_sq, range_rate, rel_spd_sq, sep)
    if t_los is None or t_los > CONFIG['lookahead_s']:
        return 0.0
    return min(1.0, max(0.0, (CONFIG['t_warn'] - t_los) / CONFIG['t_warn']))


def pair_urgency(idx_i, idx_j):
    """Urgency between the two BlueSky aircraft at indices idx_i and idx_j."""
    pos_i, vel_i = aircraft_state(idx_i)
    pos_j, vel_j = aircraft_state(idx_j)
    return urgency_from_state(pos_i, vel_i, pos_j, vel_j)


def return_blocked(cs, active_callsigns, route_hdg):
    """Binary {0, 1}: 1 if returning aircraft cs to its route is NOT free.

    Puts the ownship on its route heading and checks it against every other aircraft on
    ITS route heading (using route headings, not transient ones, keeps this robust to
    in-progress avoidance manoeuvres). Returns 1 if already in LoS with anyone, or if any
    route-corridor pair would lose separation within the warning horizon t_warn.
    """
    idx = bs.traf.id2idx(cs)
    if idx < 0:
        return 0.0

    sep = CONFIG['sep_nm']
    own_pos = aircraft_position_nm(idx)
    own_ve, own_vn = heading_to_velocity(aircraft_speed_nms(idx), route_hdg[cs])

    for other in active_callsigns:
        j = bs.traf.id2idx(other)
        if other == cs or j < 0:
            continue

        pos_j   = aircraft_position_nm(j)
        d_east  = pos_j[0] - own_pos[0]
        d_north = pos_j[1] - own_pos[1]
        dist_sq = d_east ** 2 + d_north ** 2
        if dist_sq < sep * sep:
            return 1.0                                 # already in LoS

        int_ve, int_vn = heading_to_velocity(aircraft_speed_nms(j),
                                             route_hdg.get(other, bs.traf.hdg[j]))
        dv_east, dv_north = int_ve - own_ve, int_vn - own_vn
        rel_spd_sq = dv_east ** 2 + dv_north ** 2
        range_rate = d_east * dv_east + d_north * dv_north

        t_los = time_to_los(dist_sq, range_rate, rel_spd_sq, sep)
        if t_los is not None and t_los <= CONFIG['t_warn']:
            return 1.0                                 # route corridors lose sep within t_warn
    return 0.0


def any_los(active_callsigns):
    """True if any pair of currently-flying aircraft is within sep_nm of each other."""
    flying = [cs for cs in active_callsigns if bs.traf.id2idx(cs) >= 0]
    sep_sq = CONFIG['sep_nm'] ** 2
    for a in range(len(flying)):
        pos_a = aircraft_position_nm(bs.traf.id2idx(flying[a]))
        for b in range(a + 1, len(flying)):
            pos_b = aircraft_position_nm(bs.traf.id2idx(flying[b]))
            if (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2 < sep_sq:
                return True
    return False
