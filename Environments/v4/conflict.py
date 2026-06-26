"""
Separation geometry: time-to-loss-of-separation, pairwise urgency, the conflict
score that drives the reward, the "safe to return" check, and the live LoS check.

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


def conflict_score(cs, active_callsigns, hdg_deg=None, infinite_horizon=False):
    """Worst conflict score for aircraft cs against all others:
        max over intruders of (1 - t_los/t_warn) * (1 - dcpa/sep)
    gated to converging pairs whose miss distance is inside sep. Active LoS scores 1.

    hdg_deg overrides the ownship heading (pass the route heading to ask "would flying
    direct now create a conflict?"). infinite_horizon=True drops the time term and scores
    by miss distance alone (used for the "safe to return" signal).
    """
    idx = bs.traf.id2idx(cs) if cs is not None else -1
    if idx < 0:
        return 0.0

    own_hdg = bs.traf.hdg[idx] if hdg_deg is None else hdg_deg
    own_pos = aircraft_position_nm(idx)
    own_ve, own_vn = heading_to_velocity(aircraft_speed_nms(idx), own_hdg)
    sep, t_warn = CONFIG['sep_nm'], CONFIG['t_warn']

    worst = 0.0
    for other in active_callsigns:
        j = bs.traf.id2idx(other)
        if other == cs or j < 0:
            continue

        pos_j   = aircraft_position_nm(j)
        d_east  = pos_j[0] - own_pos[0]
        d_north = pos_j[1] - own_pos[1]
        dist_sq = d_east ** 2 + d_north ** 2
        if math.sqrt(dist_sq) < sep:
            worst = 1.0                                # active LoS -> max score
            continue

        int_ve, int_vn = heading_to_velocity(aircraft_speed_nms(j), bs.traf.hdg[j])
        dv_east, dv_north = int_ve - own_ve, int_vn - own_vn
        rel_spd_sq = dv_east ** 2 + dv_north ** 2
        if rel_spd_sq < 1e-12:
            continue
        range_rate = d_east * dv_east + d_north * dv_north

        t_los = time_to_los(dist_sq, range_rate, rel_spd_sq, sep)
        if t_los is None:
            continue                                   # diverging or never intrudes
        if not infinite_horizon and t_los > CONFIG['lookahead_s']:
            continue

        dcpa = math.sqrt(max(0.0, dist_sq - range_rate ** 2 / rel_spd_sq))
        miss = max(0.0, 1.0 - dcpa / sep)
        if infinite_horizon:
            score = miss
        else:
            score = max(0.0, 1.0 - max(0.0, t_los) / t_warn) * miss
        worst = max(worst, score)

    return worst


def return_blocked(cs, active_callsigns, route_hdg):
    """Binary {0, 1}: 1 if returning aircraft cs to its route is NOT free.

    Route-vs-route CPA: ownship on its own route heading vs every intruder on ITS route
    heading, from current positions. Using each intruder's route heading (not its transient
    one) keeps this robust to in-progress avoidance manoeuvres. Returns 1 if already in LoS
    with anyone, or if any route corridor pair would breach separation.
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
        if d_east ** 2 + d_north ** 2 < sep * sep:
            return 1.0                                 # already in LoS

        int_hdg = route_hdg.get(other, bs.traf.hdg[j])
        int_ve, int_vn = heading_to_velocity(aircraft_speed_nms(j), int_hdg)
        dv_east, dv_north = int_ve - own_ve, int_vn - own_vn
        rel_spd_sq = dv_east ** 2 + dv_north ** 2
        if rel_spd_sq < 1e-12:
            continue
        tcpa = -(d_east * dv_east + d_north * dv_north) / rel_spd_sq
        if tcpa < 0:
            continue                                   # route corridors diverge
        cpa_east  = d_east + tcpa * dv_east
        cpa_north = d_north + tcpa * dv_north
        if cpa_east ** 2 + cpa_north ** 2 < sep * sep:
            return 1.0                                 # route corridors conflict
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
