# The CR tool: what the ATCO is advised by, and the only thing here that learns.
#
# Two parts, as the thesis describes it:
#   1. an urgency ranking that scores every predicted conflict and picks the focus ship
#   2. the PPO model, which sees that one aircraft and returns one action
#
# The model itself is Stable-Baselines3 and lives outside the environment; this class is
# everything around it -- the ranking, the focus ship it applies to, and turning the action it
# returns into an advisory for the ATCO.

import math

import numpy as np

from bluesky.tools.misc import degto180

from .config import (CONFIG, NO_CONFLICT_S, TURN_DELTAS, SPEED_ACTIONS,
                     RETURN_TO_ROUTE_ACTION)
from .geometry import pairwise


def heading_drift(initial_hdg, actual_hdg):
    # 0 when on the assigned heading, 2 when reversed. The focus tie-break and the drift
    # penalty are the same measure, so they are the same function.
    return 1 - math.cos(math.radians(degto180(initial_hdg - actual_hdg)))


# Drift small enough to count as back on route, in the units heading_drift returns. Taken from
# the arrival tolerance so the ranking and the arrival KPI agree on what "on route" means.
ON_ROUTE_DRIFT = 1 - math.cos(math.radians(CONFIG['arrival_hdg_tol_deg']))


class CRTool:

    def __init__(self):
        self.focus_cs   = None            # the aircraft being advised
        self.hold_steps = 0               # steps it has held the focus
        self.urgency    = np.zeros((0, 0))
        self.t_los      = np.zeros((0, 0))   # capped at the horizon, for the observation

    def rank(self, pos, vel):
        # The urgency matrix: 0 outside the horizon, ramping to 1 at the moment of intrusion,
        # then 1 to 10 once separation is actually lost.
        n = len(pos)
        if n < 2:
            self.t_los   = np.full((n, n), NO_CONFLICT_S)
            self.urgency = np.zeros((n, n))
            return self.urgency

        sep, t_warn = CONFIG['sep_nm'], CONFIG['t_warn']
        dist_sq, t_los = pairwise(pos, vel)

        # An unbounded t_los would dominate the VecNormalize variance, so it is capped here
        # rather than recomputed per pair in the observation.
        self.t_los = np.clip(t_los, 0.0, NO_CONFLICT_S)

        urgency = np.where(t_los <= t_warn, (t_warn - np.clip(t_los, 0.0, t_warn)) / t_warn, 0.0)
        in_los  = dist_sq < sep * sep
        urgency = np.where(in_los, 1.0 + 9.0 * (1.0 - np.sqrt(dist_sq) / sep), urgency)

        np.fill_diagonal(urgency, 0.0)
        self.urgency = urgency
        return urgency

    def select_focus(self, flying, row_of, aircraft, hdg):
        # Pick the aircraft to advise, and report the spell just ended as (spells, steps) so the
        # caller can record it. Returns (focus_cs, closed_spell_steps or None).
        if not flying:
            return None, None

        worst = self.urgency.max(axis=1)
        # An aircraft needs attention while it is in conflict, and equally while it is still off
        # the heading it was given: the two branches below select on exactly those two grounds,
        # so both have to keep the clock at zero. Off-route is the same 5 degrees the arrival
        # KPI calls on-route, rather than a second threshold that could drift away from it.
        for i, cs in enumerate(flying):
            ac = aircraft[cs]
            needs_attention = (worst[i] > 0
                               or heading_drift(ac.initial_hdg, hdg[i]) > ON_ROUTE_DRIFT)
            ac.steps_since_attention = 0 if needs_attention else ac.steps_since_attention + 1

        incumbent = self.focus_cs
        # Switching every time the ranking shifts would make the problem unlearnable, so the
        # incumbent is held until it has needed nothing for focus_clear_steps...
        held = (incumbent in row_of
                and aircraft[incumbent].steps_since_attention < CONFIG['focus_clear_steps'])
        # ...unless something elsewhere is urgent enough to interrupt.
        emergency = worst.max() >= CONFIG['focus_emergency_u']

        if held and not emergency:
            best_cs = incumbent
        elif worst.max() > 0:
            best_cs = flying[int(np.argmax(worst))]
        else:
            # Nothing in conflict anywhere: attend to whoever is furthest off their route.
            best_cs = max(flying, key=lambda cs: heading_drift(aircraft[cs].initial_hdg,
                                                               hdg[row_of[cs]]))

        closed = None
        if best_cs != incumbent:
            if incumbent is not None:
                closed = self.hold_steps + 1
            self.hold_steps = 0
        else:
            self.hold_steps += 1

        self.focus_cs = best_cs
        return best_cs, closed

    def advisory(self, action_idx, ac):
        # One action from the model -> what the ATCO is asked to pass on. Turns accumulate on the
        # last EXECUTED instruction, so the offset is re-derived rather than remembered.
        advisory = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            mach = ac.commanded_mach + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            advisory['target_mach'] = min(CONFIG['ac_mach_max'],
                                          max(CONFIG['ac_mach_min'], mach))

        elif action_idx in TURN_DELTAS:
            offset = degto180(ac.commanded_hdg - ac.initial_hdg) + TURN_DELTAS[action_idx]
            advisory['target_hdg'] = (ac.initial_hdg + offset) % 360

        elif action_idx == RETURN_TO_ROUTE_ACTION:
            advisory['target_hdg'] = ac.initial_hdg % 360

        return advisory
