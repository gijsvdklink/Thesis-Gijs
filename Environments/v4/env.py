# v4 ATCO conflict-resolution environment. One aircraft (the focus) is controlled per
# step. delays.py decides when the pilot acts on an instruction. reset() derives three
# independent random streams so the experiment arms share their scenarios.

import math
from random import Random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.stack.stackbase import Stack as _BsStack

from .config import (CONFIG, SEED_STRIDE, STEP_DURATION_S, OBS_DIM, N_ACTIONS, N_NEIGHBOURS,
                     CRUISE_SPD_NMS, NMS_TO_KT, KT_PER_MACH, EMPTY_RANGE_NM, NO_CONFLICT_S,
                     TURN_DELTAS, SPEED_ACTIONS, HOLD_ACTION, RETURN_TO_ROUTE_ACTION, ACT_COST)
from .delays import DELAY_MODES, ResponseDelay
from .geometry import latlon_to_nm, nm_to_latlon, wrap_to_180, heading_to_velocity
from .conflict import (traffic_states, urgency_matrix, any_loss_of_separation,
                       route_return_blocked, time_to_loss_of_separation)
from .sector import make_sector_polygon, plan_entry_route

# Offsets that keep the three per-episode streams from shadowing one another.
_TRAFFIC_SEED_XOR = 0xA5A5_5A5A
_DELAY_SEED_XOR   = 0x5A5A_A5A5

_bs_initialized = False


class _ScreenDummy(ScreenIO):
    """Silences BlueSky's screen output (we run headless)."""
    def echo(self, text='', flags=0):
        pass


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # Empty intruder slot: unreachably far, stationary, no predicted LoS.
    _EMPTY_SLOT = [EMPTY_RANGE_NM, 0.0, 0.0, 0.0, NO_CONFLICT_S]

    def __init__(self, delay_mode=None, seed=None):
        super().__init__()
        global _bs_initialized

        # Per-instance rather than a CONFIG edit: SubprocVecEnv workers do not inherit
        # parent-process CONFIG changes under spawn, so this travels via env_kwargs.
        self.delay_mode = delay_mode if delay_mode is not None else CONFIG['delay_mode']
        if self.delay_mode not in DELAY_MODES:
            raise ValueError(f'unknown delay_mode {self.delay_mode!r}; expected {DELAY_MODES}')

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        self._seed_base     = int(seed if seed is not None else CONFIG['seed'])
        self._episode_index = 0

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._new_episode_generators()
        self._reset_episode_state()

    def _reset_episode_state(self):
        """Per-episode state, reset to clean defaults."""
        self.n_aircraft           = 0
        self.polygon              = None
        self._polygon_shape       = None
        self._slots               = []
        self._active_callsigns    = set()
        self._focus_cs            = None
        self._focus_hold_steps    = 0
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._max_steps           = 0
        self._pending_spawns      = {}   # slot -> steps until the slot is refilled
        self._spawn_delay_range   = (1, 1)
        self._los_this_step       = False
        self._last_intruder_cs    = [None] * N_NEIGHBOURS   # callsign per obs slot (viz only)
        self._sim_time_s          = 0.0  # simulated seconds since reset; the delay clock

        # Traffic picture, gathered once per step by _refresh_traffic_view.
        self._urgency_cs_list = []                    # row order of the arrays below
        self._row_of          = {}                    # callsign -> row
        self._pos             = np.zeros((0, 2))      # NM, east/north
        self._vel             = np.zeros((0, 2))      # NM/s
        self._hdg             = np.zeros(0)           # deg
        self._urgency_matrix  = np.zeros((0, 0))
        self._return_blocked  = np.zeros(0)           # 1 = cannot turn back onto route

        # -- per-aircraft state, all keyed by callsign --
        self._destination_ll      = {}   # far point along the route (visualisation only)
        self._exit_ref_nm         = {}   # where a straight flight would leave the sector
        self._route_hdg           = {}   # live bearing (deg) to destination
        self._commanded_heading   = {}   # last EXECUTED heading instruction
        self._commanded_mach      = {}   # last EXECUTED speed instruction
        self._returning_to_route  = {}   # return-to-route mode latched on
        self._steps_since_urgency = {}   # steps since this aircraft last had a conflict
        self._pending_cmd         = {}   # issued-but-not-executed instruction
        self._executed_count      = {}   # instructions executed while holding the focus
        self._manoeuvred          = {}   # aircraft that have had an instruction EXECUTED

        # Registry so _forget_aircraft clears every trace of a departed aircraft.
        self._per_aircraft_state = [
            self._destination_ll, self._exit_ref_nm, self._route_hdg,
            self._commanded_heading, self._commanded_mach, self._returning_to_route,
            self._steps_since_urgency, self._pending_cmd, self._executed_count,
            self._manoeuvred,
        ]

        self._prev_los_pairs = set()
        self._ep_stats = {
            'reward': 0.0, 'steps': 0, 'actions': [],
            'los_steps': 0,        # steps with at least one pair in LoS
            'los_events': 0,       # distinct intrusions (entries, not steps)
            'flight_s': 0.0,       # airborne time flown by all aircraft
            'exits': 0,            # aircraft that left having actually flown
            # Route keeping is scored over MANOEUVRED aircraft only. Roughly 80% of traffic
            # crosses untouched and exits perfectly on route, which would otherwise swamp
            # both numbers and make them read as "how many aircraft were ignored".
            'manoeuvred_exits': 0,
            'on_route': 0,         # ...of which left within the heading tolerance
            'deviation_nm': 0.0,   # ...summed distance from the no-turn exit point
            'delay_next': 0, 'delay_sum_s': 0.0, 'delay_drawn': 0,
            'focus_spells': 0, 'focus_spell_steps': 0,
            'discarded': 0,        # instructions replaced before the pilot could fly them
        }

    # -- Gym interface ---------------------------------------------------------

    def _new_episode_generators(self):
        """Three independent streams for this episode, derived from one seed.

        Workers start at seed + rank, so worker w episode k lands on the same scenario
        in every experiment arm.
        """
        self.episode_seed = self._seed_base + self._episode_index * SEED_STRIDE
        self._episode_index += 1

        self.scenario_rng   = Random(self.episode_seed)
        self.traffic_rng    = Random(self.episode_seed ^ _TRAFFIC_SEED_XOR)
        self.delay_rng      = Random(self.episode_seed ^ _DELAY_SEED_XOR)
        self.response_delay = ResponseDelay(self.delay_mode, self.delay_rng)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_base, self._episode_index = int(seed), 0
        self._new_episode_generators()

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._reset_episode_state()
        self._build_sector()
        self._spawn_initial_traffic()

        self._refresh_traffic_view()
        self._focus_cs = self._select_focus_aircraft()
        return self._build_observation(), {}

    def step(self, action):
        action = int(action)
        self._spawn_due_aircraft()
        self._update_route_headings()

        # The aircraft that HELD the focus when this action was chosen. Reward is charged
        # to it, not to whichever aircraft the focus moves to at the end of the step.
        acting_cs = self._focus_cs
        if acting_cs:
            self._issue_instruction(acting_cs, action)
        self._update_return_to_route_headings()

        self._advance_simulation()

        # LoS is measured over everyone still airborne, before exits are retired.
        self._los_this_step = any_loss_of_separation(self._airborne_positions())
        self._step_count += 1

        self._remove_exited_aircraft()
        self._refresh_traffic_view()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, action)
        self._record_step_stats(action, reward)

        n_los_pairs = len(self._pairs_in_loss_of_separation())
        truncated   = self._step_count >= self._max_steps
        info = {'los_pairs': n_los_pairs, 'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(self._episode_summary())
        return self._build_observation(), reward, False, truncated, info

    def _advance_simulation(self):
        """One RL step of simulated time, releasing instructions as they fall due.

        The agent observes every 5 s but the pilot may act on any second, so the queue is
        flushed at 1 s resolution.
        """
        for _ in range(CONFIG['action_freq']):
            self._execute_due_instructions()
            bs.sim.step()
            self._sim_time_s += CONFIG['sim_dt']

    # -- Traffic picture -------------------------------------------------------

    def _airborne_indices(self):
        """(callsigns, BlueSky indices) of everything still airborne, in callsign order.

        Sorted, not set order: Python randomises string hashing per process, and this
        order decides urgency tie-breaks and intruder slot assignment.
        """
        flying, indices = [], []
        for cs in sorted(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                flying.append(cs)
                indices.append(idx)
        return flying, indices

    def _airborne_positions(self):
        """Positions only -- used for the LoS check before exits are retired."""
        _, indices = self._airborne_indices()
        return traffic_states(indices)[0] if indices else np.zeros((0, 2))

    def _refresh_traffic_view(self):
        """Gather the whole traffic picture once per step.

        Everything downstream (focus selection, the reward, the observation) reads these
        arrays instead of querying BlueSky per aircraft.
        """
        flying, indices = self._airborne_indices()
        self._urgency_cs_list = flying
        self._row_of          = {cs: i for i, cs in enumerate(flying)}

        if not flying:
            self._pos = self._vel = np.zeros((0, 2))
            self._hdg = self._return_blocked = np.zeros(0)
            self._urgency_matrix = np.zeros((0, 0))
            return

        self._pos, self._vel = traffic_states(indices)
        self._hdg            = bs.traf.hdg[np.asarray(indices, dtype=int)]
        self._urgency_matrix = urgency_matrix(self._pos, self._vel)

        # Whether each aircraft could turn back onto its route: same geometry, but with
        # everyone flying their ROUTE heading instead of their current one.
        speed      = np.hypot(self._vel[:, 0], self._vel[:, 1])
        route_hdg  = np.radians([self._route_hdg.get(cs, h) for cs, h in zip(flying, self._hdg)])
        route_vel  = np.stack([speed * np.sin(route_hdg), speed * np.cos(route_hdg)], axis=1)
        self._return_blocked = route_return_blocked(self._pos, route_vel)

    def _pairs_in_loss_of_separation(self):
        """Callsign pairs currently in LoS (urgency > 1 marks an active loss)."""
        if not self._urgency_matrix.size:
            return set()
        rows, cols = np.where(self._urgency_matrix > 1.0)
        return {(self._urgency_cs_list[i], self._urgency_cs_list[j])
                for i, j in zip(rows, cols) if i < j}

    # -- Statistics ------------------------------------------------------------

    def _record_step_stats(self, action, reward):
        """Accumulate the per-step counters that feed _episode_summary."""
        s = self._ep_stats
        s['reward'] += reward
        s['steps']  += 1
        s['actions'].append(action)
        if self._los_this_step:
            s['los_steps'] += 1

        # Airborne time flown this step, over all aircraft: the denominator that makes
        # LoS counts comparable between episodes of different size and density.
        s['flight_s'] += len(self._urgency_cs_list) * STEP_DURATION_S

        # Count LoS ENTRIES, not steps-in-LoS: one long intrusion is one event.
        los_pairs = self._pairs_in_loss_of_separation()
        s['los_events'] += len(los_pairs - self._prev_los_pairs)
        self._prev_los_pairs = los_pairs

    def _episode_summary(self):
        """End-of-episode metrics, logged by the training callbacks."""
        s = self._ep_stats
        flight_hours = max(s['flight_s'] / 3600.0, 1e-9)
        drawn        = max(s['delay_drawn'], 1)
        handled      = s['manoeuvred_exits']

        return {
            'mean_episode_reward': s['reward'] / max(s['steps'], 1),
            'ep_reward_total':     s['reward'],
            'ep_length':           s['steps'],
            'ep_los_steps':        s['los_steps'],
            'ep_los_events':       s['los_events'],
            'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),

            # Traffic-normalised safety: episodes differ in aircraft count, sector size and
            # length, so raw LoS counts are not comparable between them.
            'ep_flight_hours':      s['flight_s'] / 3600.0,
            'ep_los_events_per_fh': s['los_events'] / flight_hours,

            # Route keeping, over MANOEUVRED aircraft only -- what the intervention cost
            # the flights that actually received one. With no manoeuvres there is nothing
            # to score, so deviation is 0 and every aircraft left on route.
            'ep_exit_deviation_nm': s['deviation_nm'] / handled if handled else 0.0,
            'ep_arrival_rate':      s['on_route'] / handled if handled else 1.0,
            'ep_manoeuvred_exits':  handled,
            'ep_exits':             s['exits'],

            # Diagnostics -- not logged to TensorBoard; Validation/delay_diagnostics.py
            # reports them.
            'ep_delay_mean_s':     s['delay_sum_s'] / drawn,
            'ep_delay_next_frac':  s['delay_next'] / drawn,
            'ep_focus_hold_steps': s['focus_spell_steps'] / max(s['focus_spells'], 1),
            'ep_discarded':        s['discarded'],
        }

    # -- Scenario setup --------------------------------------------------------

    def _build_sector(self):
        """Draw this episode's sector and derive the episode length from it.

        Count and density are sampled independently; the area follows from them.
        """
        n_ac     = CONFIG['n_aircraft'](self.scenario_rng)
        area_km2 = float(n_ac / CONFIG['rho'](self.scenario_rng))
        poly     = make_sector_polygon(area_km2, self.scenario_rng)

        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm = math.hypot(maxx - minx, maxy - miny)

        # crossings_per_episode traversals at cruise: ~2100-4800 steps, 3-6.6 h.
        crossing_time_s = sector_diam_nm / CONFIG['ac_speed'] * 3600
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * crossing_time_s / STEP_DURATION_S))

        self.n_aircraft = n_ac
        self._slots     = [None] * n_ac
        self._spawn_delay_range = (max(1, round(CONFIG['spawn_delay_s'][0] / STEP_DURATION_S)),
                                   max(1, round(CONFIG['spawn_delay_s'][1] / STEP_DURATION_S)))

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._polygon_latlon_pairs())
        bs.stack.stack('ASAS OFF')

    def _spawn_initial_traffic(self):
        """Fill every slot at reset; queue and retry any that cannot be placed safely."""
        for slot in range(self.n_aircraft):
            for _ in range(CONFIG['max_placement_tries']):
                route = plan_entry_route(self._polygon_shape, slot, self.n_aircraft,
                                         self.scenario_rng)
                if self._spawn_is_safe(route):
                    self._create_aircraft(slot, route)
                    break
            else:
                self._pending_spawns[slot] = 5   # could not place now: retry via the queue

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """Pick the focus aircraft from the current urgency matrix.

        Highest worst-pair urgency wins (tiebreak: total urgency burden), with hysteresis
        that keeps the current focus while it is still active. An emergency overrides the
        hysteresis; a conflict-free sector falls back to the most-drifted aircraft.
        """
        flying = self._urgency_cs_list
        if not flying:
            return None

        worst_pair = self._urgency_matrix.max(axis=1)
        total_load = self._urgency_matrix.sum(axis=1)

        clear_steps = CONFIG['focus_clear_steps']
        for i, cs in enumerate(flying):
            if worst_pair[i] > 0:
                self._steps_since_urgency[cs] = 0
            else:
                self._steps_since_urgency[cs] = self._steps_since_urgency.get(cs, clear_steps) + 1

        focus_idx      = self._row_of.get(self._focus_cs, -1)
        focus_urgency  = worst_pair[focus_idx] if focus_idx >= 0 else 0.0
        focus_resolved = self._steps_since_urgency.get(self._focus_cs, clear_steps) >= clear_steps
        emergency      = worst_pair.max() >= CONFIG['focus_emergency_u']
        drift_locked   = focus_urgency == 0 and self._focus_hold_steps < clear_steps

        if worst_pair.max() > 0:
            tied = np.where(worst_pair >= worst_pair.max() - 1e-9)[0]
            if focus_idx >= 0 and focus_idx in tied:
                best_cs = self._focus_cs
            else:
                best_cs = flying[tied[int(np.argmax(total_load[tied]))]]
        else:
            best_cs = self._select_drifter_to_recover(flying)

        # Hysteresis: hold the current focus while it is still active, unless it is fully
        # resolved or an emergency forces a switch.
        if focus_idx >= 0 and best_cs != self._focus_cs:
            if (focus_urgency > 0 or not focus_resolved or drift_locked) and not emergency:
                best_cs = self._focus_cs

        if best_cs != self._focus_cs:
            if self._focus_cs is not None:      # close out the spell, for the diagnostic
                self._ep_stats['focus_spells']      += 1
                self._ep_stats['focus_spell_steps'] += self._focus_hold_steps + 1
            self._focus_hold_steps = 0
            self._executed_count[best_cs] = 0   # new ownship: this pilot is not engaged yet
        else:
            self._focus_hold_steps += 1
        return best_cs

    def _heading_drift(self, cs):
        """1 - cos(commanded heading - route heading): 0 on route, 2 reversed."""
        row     = self._row_of.get(cs)
        flying  = self._hdg[row] if row is not None else 0.0
        hdg_err = wrap_to_180(self._route_hdg[cs] - self._commanded_heading.get(cs, flying))
        return 1 - math.cos(math.radians(hdg_err))

    def _select_drifter_to_recover(self, flying):
        """Most-drifted aircraft that is free to turn back, with a hysteresis margin.

        Focusing a drifter that cannot return only burns steps, so blocked ones are
        filtered out -- unless every drifter is blocked, in which case the filter drops.
        """
        drift = {cs: self._heading_drift(cs) for cs in flying if cs in self._route_hdg}

        free = [cs for cs in drift
                if drift[cs] > 0 and not self._return_blocked[self._row_of[cs]]]
        candidates = free or list(drift)
        if not candidates:
            return self._focus_cs

        best_cs     = max(candidates, key=lambda cs: drift[cs])
        focus_score = drift.get(self._focus_cs, 0.0)

        # Stay put unless the challenger is clearly more drifted.
        if (self._focus_cs in drift and best_cs != self._focus_cs
                and drift[best_cs] <= focus_score + CONFIG['drift_switch_margin']):
            return self._focus_cs
        return best_cs

    # -- Route tracking --------------------------------------------------------

    def _update_route_headings(self):
        """Route heading = live bearing to the (far) destination, so it self-corrects."""
        center = CONFIG['center_ll']
        for cs in self._active_callsigns:
            idx  = bs.traf.id2idx(cs)
            dest = self._destination_ll.get(cs)
            if idx < 0 or dest is None:
                continue
            dest_nm = latlon_to_nm(center, float(dest[0]), float(dest[1]))
            pos     = latlon_to_nm(center, bs.traf.lat[idx], bs.traf.lon[idx])
            self._route_hdg[cs] = math.degrees(
                math.atan2(dest_nm[0] - pos[0], dest_nm[1] - pos[1])) % 360.0

    def _update_return_to_route_headings(self):
        """Re-aim returning aircraft each step, so they track the route not a frozen bearing."""
        for cs, on in self._returning_to_route.items():
            if on and bs.traf.id2idx(cs) >= 0 and cs in self._route_hdg:
                self._commanded_heading[cs] = self._route_hdg[cs]
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # -- Reward ----------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        """Purely negative: separation loss, drift off route, and the radio call itself.

        Drift uses the COMMANDED heading, which only moves at execution, so a turn the
        pilot has not made yet earns no credit.
        """
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        r_drift = 0.0
        if acting_cs and acting_cs in self._route_hdg and acting_cs in self._row_of:
            r_drift = -CONFIG['w_drift'] * self._heading_drift(acting_cs)

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_drift + r_work)

    # -- Instructions: issue now, execute after the pilot's delay ---------------

    def _build_instruction(self, cs, action_idx):
        """Action index -> instruction payload.

        The delta is measured from the aircraft's last EXECUTED state, never from an
        instruction still waiting to be flown. Re-issuing therefore REPLACES the
        outstanding instruction instead of stacking onto it: +30 chosen three times in a
        row is one 30 deg turn, and +30 followed by -30 is a 30 deg turn to the left.

        Return-to-route carries no heading; it is resolved at execution, since the route
        bearing moves while the pilot delays.
        """
        cmd = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            base = self._commanded_mach.get(cs, CONFIG['ac_mach'])
            mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            cmd['target_mach'] = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))

        elif action_idx in TURN_DELTAS:
            base = self._commanded_heading.get(cs, self._hdg[self._row_of[cs]])
            cmd['target_hdg'] = (base + TURN_DELTAS[action_idx]) % 360

        elif action_idx == RETURN_TO_ROUTE_ACTION:
            cmd['return_to_route'] = True

        return cmd

    def _issue_instruction(self, cs, action_idx):
        """Queue the instruction. Nothing reaches BlueSky until _execute_due_instructions."""
        if action_idx == HOLD_ACTION or cs not in self._row_of:
            return   # hold is a true no-op: no instruction is transmitted at all

        pending = self._pending_cmd.get(cs)
        cmd     = self._build_instruction(cs, action_idx)

        if pending is not None:
            # Amendment: new payload, inherited clock -- re-issuing cannot restart the
            # delay. The instruction it replaces is DISCARDED, never flown, and the agent
            # was charged for that radio call all the same.
            cmd['execute_at_s'] = pending['execute_at_s']
            cmd['issued_at_s']  = pending['issued_at_s']
            self._ep_stats['discarded'] += 1
        else:
            # Engaged = an advisory executed while this aircraft holds the focus.
            engaged = self._executed_count.get(cs, 0) > 0
            delay   = self.response_delay.draw(engaged)
            cmd['execute_at_s'] = self._sim_time_s + delay
            cmd['issued_at_s']  = self._sim_time_s

            self._ep_stats['delay_next']  += int(engaged)
            self._ep_stats['delay_sum_s'] += delay
            self._ep_stats['delay_drawn'] += 1

        self._pending_cmd[cs] = cmd

    def _execute_due_instructions(self):
        """Execute due instructions -- the only path to BlueSky and the commanded state."""
        for cs, cmd in list(self._pending_cmd.items()):
            if cmd['execute_at_s'] > self._sim_time_s:
                continue
            del self._pending_cmd[cs]

            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue                     # aircraft left before the pilot acted

            if 'target_mach' in cmd:
                self._commanded_mach[cs] = cmd['target_mach']
                bs.stack.stack(f'SPD {cs} {cmd["target_mach"]:.3f}')
            else:
                returning = bool(cmd.get('return_to_route'))
                self._returning_to_route[cs] = returning
                self._commanded_heading[cs] = (self._route_hdg.get(cs, bs.traf.hdg[idx])
                                               if returning else cmd['target_hdg'])
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

            # Only an EXECUTED advisory counts towards engagement, and only an executed
            # one means this aircraft was actually manoeuvred (which the route-keeping
            # metrics are scored over).
            self._executed_count[cs] = self._executed_count.get(cs, 0) + 1
            self._manoeuvred[cs] = True

    # -- Observation -----------------------------------------------------------

    def _build_observation(self):
        """ACAS Xu state of the focus aircraft, in raw units (NM, kt, s, rad)."""
        cs = self._focus_cs
        if cs is None or cs not in self._row_of:
            # No controllable aircraft: on route, nominal speed, clear, nothing pending.
            self._last_intruder_cs = [None] * N_NEIGHBOURS
            return np.array([0.0, CONFIG['ac_speed'], 0.0, CONFIG['ac_speed'], 0.0, 0.0, 0.0]
                            + self._EMPTY_SLOT * N_NEIGHBOURS, dtype=np.float32)

        obs = self._ownship_features(cs) + self._intruder_features(cs)
        return np.array(obs, dtype=np.float32)

    def _ownship_features(self, cs):
        """The 7 ownship features. dpsi is what the aircraft is doing, a_cmd what was asked."""
        row       = self._row_of[cs]
        own_hdg   = self._hdg[row]
        route_hdg = self._route_hdg[cs]
        cmd_hdg   = self._commanded_heading.get(cs, own_hdg)

        dpsi_act = math.radians(wrap_to_180(own_hdg - route_hdg))   # actual heading error
        a_cmd    = math.radians(wrap_to_180(cmd_hdg - route_hdg))   # commanded heading error
        v_own    = math.hypot(self._vel[row, 0], self._vel[row, 1]) * NMS_TO_KT
        v_cmd    = self._commanded_mach.get(cs, CONFIG['ac_mach']) * KT_PER_MACH

        # Measured from the ORIGINAL issue: an amendment inherits both the deadline and
        # this clock, so wait_s always runs against the deadline the pilot is working to.
        pending = self._pending_cmd.get(cs)
        wait_s  = self._sim_time_s - pending['issued_at_s'] if pending else 0.0

        return [dpsi_act,
                v_own,
                a_cmd,
                v_cmd,
                float(self._return_blocked[row]),   # 1 = returning is BLOCKED
                1.0 if pending else 0.0,            # constant 0 when delay_mode='none'
                wait_s]

    def _intruder_features(self, cs):
        """Feature blocks for the intruder slots, ego-centric to the ownship."""
        own_row = self._row_of[cs]
        sep     = CONFIG['sep_nm']
        own_pos, own_vel = self._pos[own_row], self._vel[own_row]
        own_hdg = self._hdg[own_row]
        sin_own = math.sin(math.radians(own_hdg))
        cos_own = math.cos(math.radians(own_hdg))
        urgency_row = self._urgency_matrix[own_row]

        intruders = []
        for j, other in enumerate(self._urgency_cs_list):
            if j == own_row:
                continue

            d_east, d_north = self._pos[j] - own_pos
            dist_nm = math.hypot(d_east, d_north)

            ego_right = d_east * cos_own - d_north * sin_own
            ego_ahead = d_east * sin_own + d_north * cos_own
            theta = math.atan2(ego_right, ego_ahead)
            psi   = math.radians(wrap_to_180(self._hdg[j] - own_hdg))
            v_int = math.hypot(self._vel[j, 0], self._vel[j, 1]) * NMS_TO_KT

            if dist_nm < sep:
                tlos = 0.0                             # already inside the protected zone
            else:
                dv_east, dv_north = self._vel[j] - own_vel
                rel_spd_sq = dv_east ** 2 + dv_north ** 2
                range_rate = d_east * dv_east + d_north * dv_north
                t_los = time_to_loss_of_separation(dist_nm ** 2, range_rate, rel_spd_sq, sep)
                # Capped at the horizon: t_los is unbounded for near-parallel pairs, which
                # would dominate the VecNormalize variance. Doubles as "no conflict".
                tlos = NO_CONFLICT_S if t_los is None else min(max(0.0, t_los), NO_CONFLICT_S)

            intruders.append((float(urgency_row[j]), dist_nm,
                              [dist_nm, theta, psi, v_int, tlos], other))

        return self._fill_intruder_slots(intruders)

    def _fill_intruder_slots(self, intruders):
        """Lay the first N_NEIGHBOURS unique candidates into slots: urgent first, then nearest."""
        ordered = (sorted((r for r in intruders if r[0] > 0), key=lambda r: -r[0])
                   + sorted(intruders, key=lambda r: r[1]))

        selected, seen = [], set()
        for rec in ordered:
            if len(selected) >= N_NEIGHBOURS:
                break
            if rec[3] not in seen:
                selected.append(rec)
                seen.add(rec[3])

        self._last_intruder_cs = [selected[k][3] if k < len(selected) else None
                                  for k in range(N_NEIGHBOURS)]
        features = []
        for k in range(N_NEIGHBOURS):
            features += selected[k][2] if k < len(selected) else self._EMPTY_SLOT
        return features

    # -- Spawning & exits ------------------------------------------------------

    def _spawn_due_aircraft(self):
        """Spawn aircraft whose countdown reached 1; decrement the rest."""
        ready    = sorted(slot for slot, t in self._pending_spawns.items() if t <= 1)
        requeued = set()
        for slot in ready:
            del self._pending_spawns[slot]
            route = self._plan_safe_entry()
            if route is not None:
                self._create_aircraft(slot, route)
            else:
                self._pending_spawns[slot] = 5   # retry in 5 steps
                requeued.add(slot)
        for slot in list(self._pending_spawns):
            if slot not in requeued:
                self._pending_spawns[slot] -= 1

    def _spawn_is_safe(self, route):
        """Admit a spawn only if it clears a static buffer and predicts no LoS within t_warn.

        Conflicts should emerge from geometry evolving, never from spawning into one.
        """
        _, indices = self._airborne_indices()
        if not indices:
            return True

        pos, vel = traffic_states(indices)
        cand_pos = latlon_to_nm(CONFIG['center_ll'],
                                float(route['sp_ll'][0]), float(route['sp_ll'][1]))
        cand_vel = np.array(heading_to_velocity(CRUISE_SPD_NMS, route['heading']))

        d       = pos - cand_pos
        dist_sq = np.einsum('ij,ij->i', d, d)
        if (dist_sq < (CONFIG['sep_nm'] + CONFIG['buffer_nm']) ** 2).any():
            return False                                    # static buffer

        dv         = vel - cand_vel
        rel_spd_sq = np.einsum('ij,ij->i', dv, dv)
        range_rate = np.einsum('ij,ij->i', d, dv)
        safe_rel   = np.where(rel_spd_sq < 1e-12, 1.0, rel_spd_sq)
        tcpa       = -range_rate / safe_rel
        dcpa_sq    = dist_sq - range_rate ** 2 / safe_rel

        predicted_los = ((rel_spd_sq >= 1e-12) & (tcpa >= 0) & (tcpa <= CONFIG['t_warn'])
                         & (dcpa_sq < CONFIG['sep_nm'] ** 2))
        return not bool(predicted_los.any())

    def _plan_safe_entry(self):
        """Plan a route for a replacement aircraft that passes the spawn safety test."""
        for _ in range(CONFIG['max_placement_tries']):
            slot  = self.traffic_rng.randint(0, self.n_aircraft - 1)
            route = plan_entry_route(self._polygon_shape, slot, self.n_aircraft,
                                     self.traffic_rng)
            if self._spawn_is_safe(route):
                return route
        return None

    def _create_aircraft(self, slot, route):
        """Put a new aircraft into BlueSky and initialise its per-aircraft state."""
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        mach = CONFIG['ac_mach']

        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(route['sp_ll'][0]), aclon=float(route['sp_ll'][1]),
                    achdg=float(route['heading']), acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')

        self._destination_ll[cs]      = route['dest_ll']
        self._exit_ref_nm[cs]         = latlon_to_nm(CONFIG['center_ll'],
                                                     float(route['ref_ll'][0]),
                                                     float(route['ref_ll'][1]))
        self._route_hdg[cs]           = float(route['heading'])
        self._commanded_heading[cs]   = float(route['heading'])
        self._commanded_mach[cs]      = CONFIG['ac_mach']
        self._returning_to_route[cs]  = False
        self._executed_count[cs]      = 0
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    def _forget_aircraft(self, cs):
        """Drop every per-aircraft entry for an aircraft that has left."""
        for state in self._per_aircraft_state:
            state.pop(cs, None)

    def _remove_exited_aircraft(self):
        """Retire aircraft that have left the sector, score them, and queue a respawn."""
        for cs in self._exited_callsigns():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                self._score_arrival(cs, idx)
                bs.traf.delete(idx)
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            self._forget_aircraft(cs)
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = self.traffic_rng.randint(*self._spawn_delay_range)

    def _exited_callsigns(self):
        """Callsigns that have left the sector. Sorted: each exit draws from traffic_rng."""
        exited = []
        for cs in sorted(self._active_callsigns):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs)
                continue
            inside = bs.tools.areafilter.checkInside(
                'SECTOR', np.array([bs.traf.lat[idx]]), np.array([bs.traf.lon[idx]]),
                np.array([CONFIG['altitude'] * 30.48]))
            if not inside[0]:
                exited.append(cs)
        return exited

    def _score_arrival(self, cs, idx):
        """Score one exiting aircraft: on-route heading, and how far off it left.

        Metrics only. Aircraft that were never manoeuvred are skipped -- they crossed
        untouched and would score a free perfect arrival, which says nothing about the
        policy. That filter also subsumes the old minimum-lifetime rule: aircraft spawn on
        the boundary and can register as outside within a step or two, but reaching the
        focus and having an advisory execute takes far longer, so none of them are scored.
        """
        self._ep_stats['exits'] += 1
        if cs not in self._manoeuvred:
            return

        self._ep_stats['manoeuvred_exits'] += 1

        # Exit deviation: distance between where it actually crossed the boundary and
        # where it would have crossed had it never turned.
        exit_ref = self._exit_ref_nm.get(cs)
        if exit_ref is not None:
            pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
            self._ep_stats['deviation_nm'] += math.hypot(pos[0] - exit_ref[0],
                                                         pos[1] - exit_ref[1])

        route_hdg = self._route_hdg.get(cs)
        if route_hdg is None:
            return
        if abs(wrap_to_180(float(bs.traf.hdg[idx]) - route_hdg)) <= CONFIG['arrival_hdg_tol_deg']:
            self._ep_stats['on_route'] += 1

    # -- Misc ------------------------------------------------------------------

    def _polygon_latlon_pairs(self):
        """Polygon vertices flattened to [lat, lon, lat, lon, ...] for BlueSky POLY."""
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
