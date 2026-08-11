"""
v4 -- ATCO conflict-resolution environment (multi-aircraft, ACAS Xu observation).

One aircraft (the focus / ownship) is controlled per step; focus follows the worst
conflict, falling back to the most-drifted aircraft that is free to return to its route
when the sector is clear.

Observation (27 floats), ego-centric from the focus aircraft, in RAW PHYSICAL UNITS.
Nothing is divided by a hand-picked constant and nothing is clipped to [0, 1]; the
features go out as NM, kt, seconds and radians, and VecNormalize(norm_obs=True) does
the standardising during training. Feeding these raw values to a policy trained with
VecNormalize therefore REQUIRES loading the matching vecnorm.pkl.
  ownship (7): dpsi (actual heading error to route, rad), v_own (TAS, kt),
               a_cmd (commanded heading error, rad), v_cmd (commanded TAS, kt),
               retn_conf (1 = returning to route is blocked),
               pending (1 = an issued instruction has not been executed yet),
               wait_s (s since that instruction was issued; 0 when nothing is pending)
  per intruder (4 x 5): dist (range, NM), theta (bearing, rad), psi (rel heading, rad),
               v_int (TAS, kt), tlos (time-to-LoS, s; NO_CONFLICT_S when the pair never
               loses separation or is beyond the conflict horizon)

Action (Discrete 10): turn -+60/45/30 (stack on commanded heading), hold (no-op),
  return-to-route (persistent), speed up/down (step commanded Mach).

ACTION-RESPONSE DELAY (delay_mode = none | deterministic | probabilistic)
  An instruction is ISSUED when the agent selects it and EXECUTED when the pilot acts on
  it, delay_s later. Only execution reaches BlueSky and only execution updates
  _commanded_heading / _commanded_mach, so the a_cmd observation and the drift reward
  never credit a turn that has not started. The queue is checked every simulated second
  (inside the action_freq loop), so 15 s means 15 s and not "3 RL steps".
  Amendments: re-issuing while an instruction is outstanding replaces the payload but
  INHERITS the original deadline -- the clock belongs to the pilot's engagement, not to
  the message. The reduced delay_next_s applies only once an advisory has actually
  EXECUTED while this aircraft holds the focus; the counter resets the moment the focus
  moves to another aircraft, so a returning aircraft starts from delay_first_s again.

Reward (purely negative): -w_los*1[LoS] - w_drift*(1-cos(dpsi)) - w_work*ACT_COST[action].
Workload is charged at ISSUE time: the radio call happens whether or not the pilot complies.
"""

import math
import random
from collections import namedtuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.stack.stackbase import Stack as _BsStack

from .config import (CONFIG, OBS_DIM, N_ACTIONS, N_NEIGHBOURS, CRUISE_SPD_NMS,
                     NMS_TO_KT, KT_PER_MACH, EMPTY_RANGE_NM, NO_CONFLICT_S,
                     TURN_DELTAS, SPEED_ACTIONS, HOLD_ACTION, RETURN_TO_ROUTE_ACTION, ACT_COST)
from .geometry import (latlon_to_nm, nm_to_latlon, wrap_to_180, aircraft_speed_nms,
                       aircraft_position_nm, aircraft_position_and_velocity,
                       heading_to_velocity)
from .conflict import (route_return_blocked, any_loss_of_separation, urgency_between,
                       time_to_loss_of_separation)
from .sector import make_sector_polygon, plan_entry_route

# The focus aircraft's own state, resolved once per observation and then shared by the
# ownship features and every intruder calculation.
OwnshipView = namedtuple('OwnshipView', 'idx hdg pos spd vel_east vel_north sin_hdg cos_hdg')

_bs_initialized = False


class _ScreenDummy(ScreenIO):
    """Silences BlueSky's screen output (we run headless)."""
    def echo(self, text='', flags=0):
        pass


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # Empty-intruder-slot sentinel, in the same raw units as a real slot:
    # unreachably far, stationary, and with no predicted LoS.
    _EMPTY_SLOT = [EMPTY_RANGE_NM, 0.0, 0.0, 0.0, NO_CONFLICT_S]

    def __init__(self, delay_mode=None):
        super().__init__()
        global _bs_initialized
        # Per-instance, not a CONFIG mutation: SubprocVecEnv workers do not inherit a
        # parent-process CONFIG edit under spawn, so the mode must travel via env_kwargs.
        self.delay_mode = delay_mode if delay_mode is not None else CONFIG['delay_mode']
        if self.delay_mode not in ('none', 'deterministic', 'probabilistic'):
            raise ValueError(f'unknown delay_mode {self.delay_mode!r}')

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
        bs.scr = _ScreenDummy()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._reset_episode_state()

    def _reset_episode_state(self):
        """Reset all per-episode state to clean defaults (shared by __init__ and reset)."""
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
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []   # row/column order of _urgency_matrix
        self._los_this_step       = False
        self._last_intruder_cs    = [None] * N_NEIGHBOURS   # callsign per obs slot (viz only)
        self._sim_time_s          = 0.0  # simulated seconds since reset; the delay clock

        # -- per-aircraft state, all keyed by callsign --
        self._destination_ll      = {}   # far point along the route (visualisation only)
        self._spawn_step          = {}   # step at spawn (to drop aircraft that never flew)
        self._route_hdg           = {}   # live bearing (deg) to destination, updated each step
        self._commanded_heading   = {}   # last EXECUTED heading instruction
        self._commanded_mach      = {}   # last EXECUTED speed instruction
        self._returning_to_route  = {}   # return-to-route mode latched on
        self._steps_since_urgency = {}   # steps since this aircraft last had a conflict
        self._pending_cmd         = {}   # issued-but-not-executed instruction
        self._executed_count      = {}   # instructions executed while holding the focus

        # One registry so that removing an aircraft clears every trace of it. Adding a
        # per-aircraft dict above and forgetting to clear it below was the obvious bug
        # waiting to happen, so the cleanup reads from this list instead.
        self._per_aircraft_state = [
            self._destination_ll, self._spawn_step, self._route_hdg,
            self._commanded_heading, self._commanded_mach, self._returning_to_route,
            self._steps_since_urgency, self._pending_cmd, self._executed_count,
        ]

        self._prev_los_pairs = set()
        self._ep_stats       = {
            'reward': 0.0, 'steps': 0, 'actions': [],
            'los_steps': 0,    # steps with at least one pair in LoS
            'los_events': 0,   # distinct intrusions (entries, not steps)
            'exits': 0,        # aircraft that left the sector having actually flown
            'on_route': 0,     # ...of which left without drift
        }

    # -- Gym interface ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        effective_seed = seed if seed is not None else CONFIG['seed']
        super().reset(seed=effective_seed)
        if effective_seed is not None:
            random.seed(effective_seed)
            np.random.seed(effective_seed)

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._reset_episode_state()
        self._build_sector()
        self._spawn_initial_traffic()

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
        self._update_return_to_route_headings()   # re-aim returning aircraft before propagating

        self._advance_simulation()

        self._los_this_step = any_loss_of_separation(self._active_callsigns)
        self._step_count += 1

        self._remove_exited_aircraft()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, action)
        self._record_step_stats(action, reward)

        n_los_pairs = int((self._urgency_matrix > 1.0).sum()) // 2 if self._urgency_matrix.size else 0
        truncated   = self._step_count >= self._max_steps
        info = {'los_pairs': n_los_pairs, 'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(self._episode_summary())
        return self._build_observation(), reward, False, truncated, info

    def _advance_simulation(self):
        """Propagate one RL step of simulated time, releasing instructions as their
        deadlines pass. The agent observes once per RL step (action_freq x sim_dt = 5 s),
        but the pilot may act on any simulated second, so the queue is flushed at 1 s
        resolution -- that is what makes a 15 s delay mean 15 s.
        """
        for _ in range(CONFIG['action_freq']):
            self._execute_due_instructions()
            bs.sim.step()
            self._sim_time_s += CONFIG['sim_dt']

    def _record_step_stats(self, action, reward):
        """Accumulate the per-step counters that feed _episode_summary."""
        self._ep_stats['reward'] += reward
        self._ep_stats['steps']  += 1
        self._ep_stats['actions'].append(action)
        if self._los_this_step:
            self._ep_stats['los_steps'] += 1

        # Count LoS ENTRIES, not steps-in-LoS: a single long intrusion is one event.
        los_pairs = self._pairs_in_loss_of_separation()
        self._ep_stats['los_events'] += len(los_pairs - self._prev_los_pairs)
        self._prev_los_pairs = los_pairs

    def _episode_summary(self):
        """End-of-episode metrics (logged by the training callbacks)."""
        s = self._ep_stats
        return {
            'mean_episode_reward': s['reward'] / max(s['steps'], 1),
            'ep_reward_total':     s['reward'],
            'ep_length':           s['steps'],
            'ep_los_steps':        s['los_steps'],
            'ep_los_events':       s['los_events'],
            'ep_arrival_rate':     s['on_route'] / max(s['exits'], 1),
            'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),
        }

    # -- Scenario setup --------------------------------------------------------

    def _build_sector(self):
        """Draw this episode's sector and derive the episode length from it.

        Aircraft count and density are sampled independently and the AREA follows from
        them, so episodes differ in size but present comparable traffic density.
        """
        n_ac     = CONFIG['n_aircraft']()
        area_km2 = float(n_ac / CONFIG['rho']())
        poly = make_sector_polygon(area_km2)
        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm  = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
        step_duration_s = CONFIG['action_freq'] * CONFIG['sim_dt']

        # Episode length scales with the sector: crossings_per_episode traversals at cruise.
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * sector_diam_nm / CONFIG['ac_speed'] * 3600 / step_duration_s
        ))
        self.n_aircraft = n_ac
        self._slots     = [None] * n_ac
        self._spawn_delay_range = (max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s)),
                                   max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s)))

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._polygon_latlon_pairs())
        bs.stack.stack('ASAS OFF')

    def _spawn_initial_traffic(self):
        """Fill every slot at reset. A slot that cannot be placed safely right now is
        queued and retried, rather than forcing an unsafe spawn."""
        for slot in range(self.n_aircraft):
            for _ in range(CONFIG['max_placement_tries']):
                route = plan_entry_route(self._polygon_shape, slot, self.n_aircraft)
                if self._spawn_is_safe(route):   # 15 NM buffer (sep + buffer) to all traffic
                    self._create_aircraft(slot, route)
                    break
            else:
                self._pending_spawns[slot] = 5   # could not place now: retry via the queue

    # -- Focus selection -------------------------------------------------------

    def _build_urgency_matrix(self, flying):
        """Symmetric matrix of pairwise urgency over the flying aircraft, plus the
        per-aircraft summaries the selection rules use.

        worst_pair (row max) is the selection criterion; total_load (row sum) breaks ties,
        so among equally urgent aircraft the one juggling the most conflicts wins.
        """
        indices = [bs.traf.id2idx(cs) for cs in flying]
        urgency = np.zeros((len(flying), len(flying)))
        for i in range(len(flying)):
            for j in range(i + 1, len(flying)):
                urgency[i, j] = urgency[j, i] = urgency_between(indices[i], indices[j])
        self._urgency_matrix  = urgency
        self._urgency_cs_list = flying
        return urgency.max(axis=1), urgency.sum(axis=1)

    def _select_focus_aircraft(self):
        """Rebuild the urgency matrix and pick the focus aircraft.

        Highest worst-pair urgency wins (tiebreak: total urgency burden), with hysteresis
        that keeps the current focus while it is still active. An emergency (urgency >=
        focus_emergency_u) overrides hysteresis; a conflict-free sector falls back to the
        most-drifted aircraft that is free to return to its route.
        """
        # SORTED, not raw set order: callsigns live in a set, and Python randomises string
        # hashing per process, so an unsorted iteration would order `flying` differently in
        # every run. That order decides urgency tie-breaks below, which would make training
        # runs irreproducible even at a fixed seed.
        flying = sorted(cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0)
        if not flying:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return None

        worst_pair, total_load = self._build_urgency_matrix(flying)

        clear_steps = CONFIG['focus_clear_steps']
        for i, cs in enumerate(flying):
            if worst_pair[i] > 0:
                self._steps_since_urgency[cs] = 0
            else:
                self._steps_since_urgency[cs] = self._steps_since_urgency.get(cs, clear_steps) + 1

        focus_idx      = flying.index(self._focus_cs) if self._focus_cs in flying else -1
        focus_urgency  = worst_pair[focus_idx] if focus_idx >= 0 else 0.0
        focus_resolved = self._steps_since_urgency.get(self._focus_cs, clear_steps) >= clear_steps
        emergency      = worst_pair.max() >= CONFIG['focus_emergency_u']
        drift_locked   = focus_urgency == 0 and self._focus_hold_steps < clear_steps

        # Candidate: highest worst-pair urgency (tiebreak total_load); ties stay with focus
        if worst_pair.max() > 0:
            tied = np.where(worst_pair >= worst_pair.max() - 1e-9)[0]
            if focus_idx >= 0 and np.any(tied == focus_idx):
                best_cs = self._focus_cs
            else:
                best_cs = flying[tied[int(np.argmax(total_load[tied]))]]
        else:
            best_cs = self._select_drifter_to_recover(flying)

        # Hysteresis: keep the current focus while it is still active, unless fully
        # resolved or an emergency forces a switch.
        if focus_idx >= 0 and best_cs != self._focus_cs:
            keep = (focus_urgency > 0 or not focus_resolved or drift_locked) and not emergency
            if keep:
                best_cs = self._focus_cs

        if best_cs != self._focus_cs:
            self._focus_hold_steps = 0
            # New ownship: this pilot is not engaged yet, so their next advisory is a
            # "first" one and earns the full delay_first_s.
            self._executed_count[best_cs] = 0
        else:
            self._focus_hold_steps += 1
        return best_cs

    def _heading_drift(self, cs, idx):
        """1 - cos of the gap between the commanded heading and the route heading:
        0 when on route, 2 when reversed. The same shape the drift reward uses.

        The factor 1/2 that used to normalise this to [0, 1] now lives in w_drift, so the
        reward is unchanged. Anything else comparing this against an absolute constant --
        drift_switch_margin in _select_drifter_to_recover -- is on the [0, 2] scale."""
        hdg_err = wrap_to_180(self._route_hdg[cs] - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
        return 1 - math.cos(math.radians(hdg_err))

    def _select_drifter_to_recover(self, flying):
        """Pick the most-drifted aircraft that can safely be sent back to its route.

        Score = heading drift, restricted to aircraft whose route heading is currently
        free (route_return_blocked == 0) -- the same predicate the observation reports as
        retn_conf. A drifter that cannot turn back offers the agent nothing to act on, so
        focusing it only burns steps. If every drifted aircraft is blocked the filter is
        dropped, so the rule always returns a focus. A hysteresis margin keeps the current
        focus unless another aircraft scores clearly higher.
        """
        drift = {}
        for cs in sorted(flying):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._route_hdg:
                continue
            drift[cs] = self._heading_drift(cs, idx)

        free = [cs for cs in drift
                if drift[cs] > 0 and not route_return_blocked(cs, flying, self._route_hdg)]
        candidates = free or list(drift)

        best_cs, best_score, focus_score = None, -1.0, 0.0
        for cs in candidates:
            score = drift[cs]
            if cs == self._focus_cs:
                focus_score = score
            if score > best_score:
                best_score, best_cs = score, cs

        margin = CONFIG['drift_switch_margin']
        if (self._focus_cs in flying and best_cs != self._focus_cs
                and best_score <= focus_score + margin):
            return self._focus_cs
        return best_cs

    def _pairs_in_loss_of_separation(self):
        """Callsign pairs currently in loss of separation (urgency > 1 marks an active LoS)."""
        m = self._urgency_matrix
        if not m.size:
            return set()
        rows, cols = np.where(m > 1.0)
        return {(self._urgency_cs_list[i], self._urgency_cs_list[j])
                for i, j in zip(rows, cols) if i < j}

    # -- Route tracking --------------------------------------------------------

    def _update_route_headings(self):
        """Recompute each aircraft's route heading as the live bearing from its current
        position to its destination. Using the far destination keeps the bearing stable
        while naturally correcting for any drift accumulated during avoidance manoeuvres."""
        center = CONFIG['center_ll']
        for cs in self._active_callsigns:
            idx  = bs.traf.id2idx(cs)
            dest = self._destination_ll.get(cs)
            if idx < 0 or dest is None:
                continue
            dest_nm = latlon_to_nm(center, float(dest[0]), float(dest[1]))
            pos     = aircraft_position_nm(idx)
            self._route_hdg[cs] = math.degrees(
                math.atan2(dest_nm[0] - pos[0], dest_nm[1] - pos[1])) % 360.0

    def _update_return_to_route_headings(self):
        """Re-issue the current route heading for every returning aircraft each step, so
        it keeps tracking the (slowly moving) route rather than a frozen bearing."""
        for cs, on in self._returning_to_route.items():
            idx = bs.traf.id2idx(cs)
            if on and idx >= 0 and cs in self._route_hdg:
                self._commanded_heading[cs] = self._route_hdg[cs]
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # -- Reward ----------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        """Purely negative: a separation loss, being off route, and the radio call itself.

        Drift is measured on the COMMANDED heading, which only changes at execution, so a
        turn the pilot has not made yet earns no credit. Workload is charged at issue time
        because the radio call happens whether or not the pilot complies.
        """
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        r_drift = 0.0
        if acting_cs and acting_cs in self._route_hdg:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                r_drift = -CONFIG['w_drift'] * self._heading_drift(acting_cs, idx)

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_drift + r_work)

    # -- Instructions: issue now, execute after the pilot's delay ---------------

    def _draw_response_delay(self, executed_n):
        """Seconds until the pilot acts. executed_n = advisories already EXECUTED while this
        aircraft has held the focus, so 0 means the pilot is not engaged yet."""
        if self.delay_mode == 'none':
            return 0.0
        mean = CONFIG['delay_first_s'] if executed_n == 0 else CONFIG['delay_next_s']
        if self.delay_mode == 'deterministic':
            return mean
        # Log-normal parameterised on the MEAN: E[X] = exp(mu + sigma^2/2), so mu below
        # makes E[X] land exactly on `mean` rather than on the median.
        sigma = CONFIG['delay_sigma']
        mu    = math.log(mean) - sigma ** 2 / 2.0
        return min(random.lognormvariate(mu, sigma), CONFIG['delay_max_s'])

    def _build_instruction(self, cs, idx, action_idx, pending):
        """Translate an action index into the instruction payload to be flown later.

        Turns and speed changes stack onto the controller's last STATED intent -- the
        outstanding target if one exists (the pilot has not acted on it yet), otherwise the
        flying heading. Return-to-route carries no heading: it is resolved at EXECUTION time,
        because the route bearing drifts while the pilot delays.
        """
        cmd = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            base = pending['target_mach'] if pending and 'target_mach' in pending \
                   else self._commanded_mach.get(cs, CONFIG['ac_mach'])
            mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            cmd['target_mach'] = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))
        elif action_idx in TURN_DELTAS:
            base = pending['target_hdg'] if pending and 'target_hdg' in pending \
                   else self._commanded_heading.get(cs, bs.traf.hdg[idx])
            cmd['target_hdg'] = (base + TURN_DELTAS[action_idx]) % 360
        elif action_idx == RETURN_TO_ROUTE_ACTION:
            cmd['return_to_route'] = True

        return cmd

    def _issue_instruction(self, cs, action_idx):
        """Register the agent's instruction. Nothing reaches BlueSky here -- the aircraft
        only responds in _execute_due_instructions, delay_s later."""
        if action_idx == HOLD_ACTION:
            return   # hold: true no-op, no instruction is transmitted at all

        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        pending = self._pending_cmd.get(cs)
        cmd     = self._build_instruction(cs, idx, action_idx, pending)

        if pending is not None:
            # Amendment: the deadline belongs to the pilot's engagement, not to the message.
            # Replace the payload, inherit the clock, draw no new delay, and do not touch
            # _executed_count -- nothing has been executed yet.
            cmd['execute_at_s'] = pending['execute_at_s']
            cmd['issued_at_s']  = pending['issued_at_s']
        else:
            # _executed_count is reset in _select_focus_aircraft whenever the ownship
            # changes, so "engagement" lasts exactly as long as this aircraft holds focus.
            delay = self._draw_response_delay(self._executed_count.get(cs, 0))
            cmd['execute_at_s'] = self._sim_time_s + delay
            cmd['issued_at_s']  = self._sim_time_s

        self._pending_cmd[cs] = cmd

    def _execute_due_instructions(self):
        """Execute every instruction whose deadline has arrived. This is the only place a
        heading/speed instruction reaches BlueSky or updates the commanded state."""
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
                if cmd.get('return_to_route'):
                    self._returning_to_route[cs]       = True
                    self._commanded_heading[cs] = self._route_hdg.get(cs, bs.traf.hdg[idx])
                else:
                    self._returning_to_route[cs]       = False
                    self._commanded_heading[cs] = cmd['target_hdg']
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

            # Only an EXECUTED advisory earns the shorter delay_next_s.
            self._executed_count[cs] = self._executed_count.get(cs, 0) + 1

    # -- Observation -----------------------------------------------------------

    def _build_observation(self):
        """ACAS Xu states for the focus aircraft against its nearest/most-urgent intruders,
        reported in raw physical units (NM, kt, s, rad) -- see the module docstring."""
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            # no controllable aircraft: on-route, nominal speed, conflict-free, nothing pending
            self._last_intruder_cs = [None] * N_NEIGHBOURS
            return np.array([0.0, CONFIG['ac_speed'], 0.0, CONFIG['ac_speed'], 0.0, 0.0, 0.0]
                            + self._EMPTY_SLOT * N_NEIGHBOURS, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        own_spd = aircraft_speed_nms(idx)
        own_ve, own_vn = heading_to_velocity(own_spd, own_hdg)
        own = OwnshipView(idx=idx, hdg=own_hdg, pos=aircraft_position_nm(idx), spd=own_spd,
                          vel_east=own_ve, vel_north=own_vn,
                          sin_hdg=math.sin(math.radians(own_hdg)),
                          cos_hdg=math.cos(math.radians(own_hdg)))

        obs = self._ownship_features(cs, own)
        obs += self._intruder_features(cs, own)
        return np.array(obs, dtype=np.float32)

    def _ownship_features(self, cs, own):
        """The 7 focus-ship features. dpsi is what the aircraft is actually doing, a_cmd is
        what the controller asked for -- under a delay these differ, and the gap between
        them IS the outstanding instruction."""
        route_hdg = self._route_hdg[cs]
        cmd_hdg   = self._commanded_heading.get(cs, own.hdg)
        dpsi_act  = math.radians(wrap_to_180(own.hdg - route_hdg))   # actual heading error to route
        a_cmd     = math.radians(wrap_to_180(cmd_hdg - route_hdg))   # commanded heading error
        v_cmd     = self._commanded_mach.get(cs, CONFIG['ac_mach']) * KT_PER_MACH

        retn_conf = route_return_blocked(cs, self._active_callsigns, self._route_hdg)

        # How long the outstanding instruction has been waiting. Under a probabilistic
        # delay this is what lets the controller reason about the draw: the longer a
        # pilot has been silent, the sooner the response is due. Measured from the
        # ORIGINAL issue -- an amendment inherits both the deadline and this clock, so
        # wait_s always runs against the deadline the pilot is actually working to.
        pending_cmd = self._pending_cmd.get(cs)
        wait_s      = self._sim_time_s - pending_cmd['issued_at_s'] if pending_cmd else 0.0

        return [dpsi_act,
                own.spd * NMS_TO_KT,
                a_cmd,
                v_cmd,
                retn_conf,      # 1 = returning to route is BLOCKED (not "safe to return")
                # Whether an instruction is outstanding -- the honest observable: under a
                # probabilistic delay the controller knows the pilot has not acted yet, but
                # not when they will. Both are constant 0.0 when delay_mode='none'.
                1.0 if pending_cmd else 0.0,
                wait_s]

    def _focus_urgency_row(self, cs):
        """This aircraft's row of the urgency matrix (built in _select_focus_aircraft),
        used to prioritise which intruders get an observation slot. None if absent."""
        if cs in self._urgency_cs_list:
            row = self._urgency_cs_list.index(cs)
            if row < self._urgency_matrix.shape[0]:
                return self._urgency_matrix[row]
        return None

    def _intruder_features(self, cs, own):
        """Feature blocks for the N_NEIGHBOURS intruder slots, ego-centric to the ownship.

        Slots are filled by urgency first (worst threat in slot 0) and then by proximity,
        so each slot carries a stable meaning instead of being an arbitrary list. Unused
        slots take the far-away sentinel.
        """
        sep         = CONFIG['sep_nm']
        urgency_row = self._focus_urgency_row(cs)
        intruders   = []

        # Sorted for the same reason as `flying`: equal-distance candidates keep their
        # input order through the stable sort below, so slot assignment must not depend on
        # set iteration order.
        for other in sorted(self._active_callsigns):
            j = bs.traf.id2idx(other)
            if other == cs or j < 0:
                continue

            int_pos = aircraft_position_nm(j)
            d_east, d_north = int_pos[0] - own.pos[0], int_pos[1] - own.pos[1]
            dist_nm = math.sqrt(d_east ** 2 + d_north ** 2)
            int_hdg = bs.traf.hdg[j]
            int_spd = aircraft_speed_nms(j)

            ego_lat = d_east * own.cos_hdg - d_north * own.sin_hdg   # + = right of ownship heading
            ego_fwd = d_east * own.sin_hdg + d_north * own.cos_hdg   # + = ahead

            theta = math.atan2(ego_lat, ego_fwd)
            psi   = math.radians(wrap_to_180(int_hdg - own.hdg))
            v_int = int_spd * NMS_TO_KT

            if dist_nm < sep:
                tlos = 0.0                             # already inside the protected zone
            else:
                int_ve, int_vn = heading_to_velocity(int_spd, int_hdg)
                dv_east, dv_north = int_ve - own.vel_east, int_vn - own.vel_north
                rel_spd_sq = dv_east ** 2 + dv_north ** 2
                range_rate = d_east * dv_east + d_north * dv_north
                t_los = time_to_loss_of_separation(dist_nm ** 2, range_rate, rel_spd_sq, sep)
                # Capped at the planning horizon: t_los is unbounded for a near-parallel
                # pair, and letting those values through would dominate the VecNormalize
                # running variance. The cap doubles as the "no conflict" value.
                tlos = NO_CONFLICT_S if t_los is None else min(max(0.0, t_los), NO_CONFLICT_S)

            pair_u = 0.0
            if urgency_row is not None and other in self._urgency_cs_list:
                col = self._urgency_cs_list.index(other)
                if col < len(urgency_row):
                    pair_u = float(urgency_row[col])

            intruders.append((pair_u, dist_nm, [dist_nm, theta, psi, v_int, tlos], other))

        return self._fill_intruder_slots(intruders)

    def _fill_intruder_slots(self, intruders):
        """Order candidates -- urgent pairs first (descending urgency), then nearest
        (ascending distance) -- and lay the first N_NEIGHBOURS unique ones into slots."""
        ordered = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0]) \
                  + sorted(intruders, key=lambda r: r[1])
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
        ready    = sorted(slot for slot, t in list(self._pending_spawns.items()) if t <= 1)
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
        """Admit a candidate spawn only if it clears (1) a static buffer to all traffic and
        (2) a conflict-free entry: flying its route at cruise it must not reach CPA < sep
        against any active aircraft (current trajectory) within t_warn.

        Conflicts should emerge from geometry evolving, never from spawning into one.
        """
        pos_c = latlon_to_nm(CONFIG['center_ll'], float(route['sp_ll'][0]), float(route['sp_ll'][1]))
        min_spawn_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        sep, horizon  = CONFIG['sep_nm'], CONFIG['t_warn']
        cand_ve, cand_vn = heading_to_velocity(CRUISE_SPD_NMS, route['heading'])

        for cs in self._active_callsigns:
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue
            pos_o, vel_o = aircraft_position_and_velocity(idx)
            d_east, d_north = pos_o[0] - pos_c[0], pos_o[1] - pos_c[1]
            if math.hypot(d_east, d_north) < min_spawn_sep:
                return False                                # (1) static buffer
            if not CONFIG.get('spawn_conflict_free', True):
                continue

            dv_east, dv_north = vel_o[0] - cand_ve, vel_o[1] - cand_vn
            rel_sq = dv_east ** 2 + dv_north ** 2
            if rel_sq < 1e-12:
                continue
            tcpa = -(d_east * dv_east + d_north * dv_north) / rel_sq
            if tcpa < 0 or tcpa > horizon:
                continue                                    # diverging or beyond t_warn
            cpa_e, cpa_n = d_east + tcpa * dv_east, d_north + tcpa * dv_north
            if cpa_e ** 2 + cpa_n ** 2 < sep * sep:
                return False                                # (2) predicted LoS within t_warn
        return True

    def _plan_safe_entry(self):
        """Plan a route for a replacement aircraft that passes the spawn safety test."""
        for _ in range(CONFIG['max_placement_tries']):
            route = plan_entry_route(self._polygon_shape,
                                     random.randint(0, self.n_aircraft - 1), self.n_aircraft)
            if self._spawn_is_safe(route):
                return route
        return None

    def _create_aircraft(self, slot, route):
        """Put a new aircraft into BlueSky and initialise all of its per-aircraft state."""
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
        self._spawn_step[cs]          = self._step_count
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
        """Remove aircraft that have left the sector, scoring on-target arrivals, and
        queue their slot for a respawn."""
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
                self._pending_spawns[slot] = random.randint(*self._spawn_delay_range)

    def _exited_callsigns(self):
        """Callsigns that have left the sector or are no longer in BlueSky.

        Sorted: each exit draws a respawn delay from the RNG, so the order aircraft are
        retired in decides the order random numbers are consumed."""
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

    # -- Arrival scoring -------------------------------------------------------

    def _score_arrival(self, cs, idx):
        """Count one exiting aircraft, and whether it left WITHOUT DRIFT -- flying within
        arrival_hdg_tol_deg of its route heading rather than still sitting in an avoidance
        deviation. METRIC ONLY: this never enters the reward.

        Aircraft spawn ON the sector boundary and can register as outside within a step or
        two, before they have really flown. Those are excluded, or they would count as
        free perfect arrivals and inflate the rate.
        """
        life = self._step_count - self._spawn_step.get(cs, 0)
        if life <= CONFIG['arrival_min_life_steps']:
            return

        self._ep_stats['exits'] += 1
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
