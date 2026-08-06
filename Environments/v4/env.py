"""
v4 -- ATCO conflict-resolution environment (multi-aircraft, ACAS Xu observation).

One aircraft (the focus / ownship) is controlled per step; focus follows the worst
conflict, falling back to the most drift-x-clearance aircraft when the sector is clear.

Observation (26 floats), ego-centric from the focus aircraft, in RAW PHYSICAL UNITS.
Nothing is divided by a hand-picked constant and nothing is clipped to [0, 1]; the
features go out as NM, kt, seconds and radians, and VecNormalize(norm_obs=True) does
the standardising during training. Feeding these raw values to a policy trained with
VecNormalize therefore REQUIRES loading the matching vecnorm.pkl.
  ownship (6): dpsi (actual heading error to route, rad), v_own (TAS, kt),
               a_cmd (commanded heading error, rad), v_cmd (commanded TAS, kt),
               retn_conf (1 = returning to route is blocked),
               pending (1 = an issued instruction has not been executed yet)
  per intruder (4 x 5): dist (range, NM), theta (bearing, rad), psi (rel heading, rad),
               v_int (TAS, kt), tlos (time-to-LoS, s; NO_CONFLICT_S when the pair never
               loses separation or is beyond the conflict horizon)

Action (Discrete 10): turn -+60/45/30 (stack on commanded heading), hold (no-op),
  fly-direct (return to route, persistent), speed up/down (step commanded Mach).

ACTION-RESPONSE DELAY (delay_mode = none | deterministic | probabilistic)
  An action is ISSUED when the agent selects it and EXECUTED when the pilot acts on it,
  delay_s later. Only execution reaches BlueSky and only execution updates
  _commanded_heading / _commanded_mach, so the a_cmd observation and the drift reward
  never credit a turn that has not started. The queue is checked every simulated second
  (inside the action_freq loop), so 12.5 s means 12.5 s and not "2 or 3 RL steps".
  Amendments: re-issuing while an instruction is outstanding replaces the payload but
  INHERITS the original deadline -- the clock belongs to the pilot's engagement, not to
  the message. The reduced delay_next_s applies only once an advisory has actually
  EXECUTED in the current urgency cycle.

Reward (purely negative): -w_los*1[LoS] - w_drift*(1-cos(dpsi))/2 - w_work*ACT_COST[action].
Workload is charged at ISSUE time: the radio call happens whether or not the pilot complies.
"""

import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.tools import geo
from bluesky.stack.stackbase import Stack as _BsStack

from .config import (CONFIG, OBS_DIM, N_ACTIONS, N_NEIGHBOURS, CRUISE_SPD_NMS,
                     NMS_TO_KT, KT_PER_MACH, EMPTY_RANGE_NM, NO_CONFLICT_S,
                     TURN_DELTAS, SPEED_ACTIONS, ACT_COST)
from .geometry import (latlon_to_nm, nm_to_latlon, wrap_to_180, aircraft_speed_nms,
                       aircraft_position_nm, aircraft_state, heading_to_velocity)
from .conflict import (return_blocked, any_los, pair_urgency, time_to_los)
from .sector import make_polygon, place_aircraft

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

        self._clear_episode_state()

    def _clear_episode_state(self):
        """Reset all per-episode state to clean defaults (shared by __init__ and reset)."""
        self.n_aircraft           = 0
        self.polygon              = None
        self._polygon_shape       = None
        self._slots               = []
        self._active_callsigns    = set()
        self._destination_ll      = {}   # far point along the route (visualisation only)
        self._ref_ll              = {}   # exit reference point per callsign
        self._route_hdg           = {}   # live bearing (deg) to destination, updated each step
        self._commanded_heading   = {}
        self._commanded_mach      = {}
        self._direct_mode         = {}   # fly-direct (back-to-route) active per callsign
        self._steps_since_urgency = {}
        self._focus_hold_steps    = 0
        self._focus_cs            = None
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._max_steps           = 0
        self._pending_spawns      = {}
        self._spawn_delay_range   = (1, 1)
        self._urgency_matrix      = np.zeros((0, 0))
        self._urgency_cs_list     = []
        self._los_this_step       = False
        self._last_intruder_cs    = [None] * N_NEIGHBOURS   # callsign per obs slot (viz only)

        # -- action-response delay --
        self._sim_time_s     = 0.0   # simulated seconds since reset; the delay clock
        self._pending_cmd    = {}    # cs -> issued-but-not-executed instruction
        self._executed_count = {}    # cs -> advisories EXECUTED in the current urgency cycle

        # -- arrival scoring --
        self._spawn_step = {}        # cs -> step at spawn (to drop aircraft that never flew)
        self._spawn_ll   = {}        # cs -> entry point (for the cross-track flavor)

        self._prev_los_pairs = set()
        self._ep_stats       = {
            'reward': 0.0, 'steps': 0, 'los': 0, 'actions': [], 'exits': 0, 'arrivals': 0,
            # arrival flavors, counted over 'flown' aircraft only
            'flown': 0, 'arr_on_route': 0, 'arr_xtrack': 0, 'arr_ref': 0,
            # safety
            'los_events': 0, 'ac_steps': 0,
            # delay / strategy
            'delays': [], 'amendments': 0, 'amend_lead_s': [], 'executed': 0,
            'pending_steps': 0, 'tlos_at_issue': [], 'turn_mag': [],
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

        self._clear_episode_state()

        n_ac     = CONFIG['n_aircraft']()
        area_km2 = float(n_ac / CONFIG['rho']())
        poly = make_polygon(area_km2)
        self._polygon_shape = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm  = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
        step_duration_s = CONFIG['action_freq'] * CONFIG['sim_dt']
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * sector_diam_nm / CONFIG['ac_speed'] * 3600 / step_duration_s
        ))
        self.n_aircraft = n_ac
        self._slots     = [None] * n_ac
        self._spawn_delay_range = (max(1, round(CONFIG['spawn_delay_s'][0] / step_duration_s)),
                                   max(1, round(CONFIG['spawn_delay_s'][1] / step_duration_s)))

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._flat_latlon())
        bs.stack.stack('ASAS OFF')

        for slot in range(n_ac):
            for _ in range(CONFIG['max_placement_tries']):
                ac = place_aircraft(self._polygon_shape, slot, n_ac)
                if self._spawn_ok(ac):       # 15 NM buffer (sep + buffer) to all traffic
                    self._spawn_aircraft(slot, ac)
                    break
            else:
                self._pending_spawns[slot] = 5   # could not place now: retry via the queue

        self._focus_cs = self._select_focus_aircraft()
        return self._get_observation(), {}

    def step(self, action):
        action = int(action)
        self._process_pending_spawns()
        self._refresh_route_headings()
        acting_cs = self._focus_cs

        if acting_cs:
            self._issue_action(acting_cs, action)
        self._update_direct_headings()   # re-aim fly-direct aircraft before propagating

        # The agent observes once per RL step (action_freq x sim_dt = 5 s), but the pilot
        # may act on any simulated second, so the queue is flushed at 1 s resolution.
        for _ in range(CONFIG['action_freq']):
            self._flush_due_commands()
            bs.sim.step()
            self._sim_time_s += CONFIG['sim_dt']

        self._los_this_step = any_los(self._active_callsigns)
        self._step_count += 1

        self._process_exits()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, action)

        self._ep_stats['reward'] += reward
        self._ep_stats['steps']  += 1
        self._ep_stats['actions'].append(action)
        if self._los_this_step:
            self._ep_stats['los'] += 1
        if self._focus_cs in self._pending_cmd:
            self._ep_stats['pending_steps'] += 1
        self._ep_stats['ac_steps'] += len(self._urgency_cs_list)

        # Count LoS ENTRIES, not steps-in-LoS: a single long intrusion is one event.
        los_pairs = self._current_los_pairs()
        self._ep_stats['los_events'] += len(los_pairs - self._prev_los_pairs)
        self._prev_los_pairs = los_pairs

        n_los_pairs = int((self._urgency_matrix > 1.0).sum()) // 2 if self._urgency_matrix.size else 0
        truncated   = self._step_count >= self._max_steps
        info = {'los_pairs': n_los_pairs, 'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(self._episode_summary())
        return self._get_observation(), reward, False, truncated, info

    def _episode_summary(self):
        """End-of-episode metrics (logged by the training callbacks)."""
        s       = self._ep_stats
        n_steps = max(s['steps'], 1)
        flown   = max(s['flown'], 1)
        mean    = lambda xs: float(np.mean(xs)) if xs else 0.0

        # Aircraft-hours actually flown this episode: every aircraft alive for one RL step
        # contributes action_freq * sim_dt seconds.
        flight_hours = s['ac_steps'] * CONFIG['action_freq'] * CONFIG['sim_dt'] / 3600.0

        return {
            'mean_episode_reward': s['reward'] / n_steps,
            'ep_reward_total':     s['reward'],
            'ep_length':           s['steps'],

            # -- safety --
            'ep_los_steps':        s['los'],
            'ep_los_events':       s['los_events'],
            'ep_flight_hours':     flight_hours,
            'ep_los_per_fh':       s['los_events'] / max(flight_hours, 1e-9),

            # -- arrival, three flavors over aircraft that actually flew --
            'ep_exits':            s['exits'],
            'ep_flown':            s['flown'],
            'ep_arrivals':         s['arrivals'],
            'ep_arrival_rate':     s['arrivals'] / max(s['exits'], 1),   # legacy, all exits
            'ep_arr_on_route':     s['arr_on_route'] / flown,
            'ep_arr_xtrack':       s['arr_xtrack'] / flown,
            'ep_arr_ref':          s['arr_ref'] / flown,

            # -- delay realisation --
            'ep_mean_delay_s':     mean(s['delays']),
            'ep_executed':         s['executed'],
            'ep_pending_frac':     s['pending_steps'] / n_steps,

            # -- strategy --
            'ep_tlos_at_issue':    mean(s['tlos_at_issue']),
            'ep_turn_magnitude':   mean(s['turn_mag']),
            'ep_amendments':       s['amendments'],
            'ep_amend_lead_s':     mean(s['amend_lead_s']),

            'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),
        }

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """Rebuild the urgency matrix and pick the focus aircraft.

        Highest worst-pair urgency wins (tiebreak: total urgency burden), with hysteresis
        that keeps the current focus while it is still active. An emergency (urgency >=
        focus_emergency_u) overrides hysteresis; a conflict-free sector falls back to the
        most-drifted aircraft that is free to return to its route.
        """
        flying = [cs for cs in self._active_callsigns if bs.traf.id2idx(cs) >= 0]
        if not flying:
            self._urgency_matrix  = np.zeros((0, 0))
            self._urgency_cs_list = []
            return None

        indices = [bs.traf.id2idx(cs) for cs in flying]
        urgency = np.zeros((len(flying), len(flying)))
        for i in range(len(flying)):
            for j in range(i + 1, len(flying)):
                urgency[i, j] = urgency[j, i] = pair_urgency(indices[i], indices[j])
        self._urgency_matrix  = urgency
        self._urgency_cs_list = flying

        worst_pair = urgency.max(axis=1)   # worst single-pair urgency per aircraft
        total_load = urgency.sum(axis=1)   # total urgency burden per aircraft

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
            best_cs = self._drift_fallback(flying)

        # Hysteresis: keep the current focus while it is still active, unless fully
        # resolved or an emergency forces a switch.
        if focus_idx >= 0 and best_cs != self._focus_cs:
            keep = (focus_urgency > 0 or not focus_resolved or drift_locked) and not emergency
            if keep:
                best_cs = self._focus_cs

        self._focus_hold_steps = 0 if best_cs != self._focus_cs else self._focus_hold_steps + 1
        return best_cs

    def _drift_fallback(self, flying):
        """Pick the most-drifted aircraft that can safely be sent back to its route.

        Score = heading drift, restricted to aircraft whose route heading is currently
        free (return_blocked == 0) -- the same predicate the observation reports as
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
            hdg_err = wrap_to_180(self._route_hdg[cs] - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
            drift[cs] = (1 - math.cos(math.radians(hdg_err))) / 2

        free = [cs for cs in drift
                if drift[cs] > 0 and not return_blocked(cs, flying, self._route_hdg)]
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

    # -- Reward ----------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

        r_drift = 0.0
        if acting_cs and acting_cs in self._route_hdg:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                cmd_hdg = self._commanded_heading.get(acting_cs, bs.traf.hdg[idx])
                hdg_err = wrap_to_180(self._route_hdg[acting_cs] - cmd_hdg)
                r_drift = -CONFIG['w_drift'] * (1.0 - math.cos(math.radians(hdg_err))) / 2.0

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_drift + r_work)

    # -- Actions ---------------------------------------------------------------

    def _draw_delay(self, executed_n):
        """Seconds until the pilot acts. executed_n = advisories already EXECUTED in this
        urgency cycle, so 0 means this is the cycle's first advisory."""
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

    def _issue_action(self, cs, action_idx):
        """Register the agent's instruction. Nothing reaches BlueSky here -- the aircraft
        only responds in _flush_due_commands, delay_s later."""
        if action_idx == 3:
            return   # hold: true no-op, no instruction is transmitted at all

        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        pending = self._pending_cmd.get(cs)
        cmd     = {'action': action_idx}

        # A turn stacks onto the controller's last STATED intent: the outstanding target if
        # one exists (the pilot has not acted on it yet), otherwise the flying heading.
        if action_idx in SPEED_ACTIONS:
            base = pending['target_mach'] if pending and 'target_mach' in pending \
                   else self._commanded_mach.get(cs, CONFIG['ac_mach'])
            mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            cmd['target_mach'] = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))
        elif action_idx in TURN_DELTAS:
            base = pending['target_hdg'] if pending and 'target_hdg' in pending \
                   else self._commanded_heading.get(cs, bs.traf.hdg[idx])
            cmd['target_hdg'] = (base + TURN_DELTAS[action_idx]) % 360
            self._ep_stats['turn_mag'].append(abs(TURN_DELTAS[action_idx]))
        elif action_idx == 7:
            # Resolve the route heading at EXECUTION time: it drifts while the pilot delays.
            cmd['direct'] = True

        if pending is not None:
            # Amendment: the deadline belongs to the pilot's engagement, not to the message.
            # Replace the payload, inherit the clock, draw no new delay, and do not touch
            # _executed_count -- nothing has been executed yet.
            cmd['execute_at_s'] = pending['execute_at_s']
            cmd['issued_at_s']  = pending['issued_at_s']
            self._ep_stats['amendments'] += 1
            self._ep_stats['amend_lead_s'].append(pending['execute_at_s'] - self._sim_time_s)
        else:
            clear = CONFIG['focus_clear_steps']
            if self._steps_since_urgency.get(cs, clear) >= clear:
                self._executed_count[cs] = 0        # fresh urgency cycle -> next is a "first"
            delay = self._draw_delay(self._executed_count.get(cs, 0))
            cmd['execute_at_s'] = self._sim_time_s + delay
            cmd['issued_at_s']  = self._sim_time_s
            self._ep_stats['delays'].append(delay)

        tlos = self._focus_tlos(cs)
        if tlos is not None:
            self._ep_stats['tlos_at_issue'].append(tlos)

        self._pending_cmd[cs] = cmd

    def _flush_due_commands(self):
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
                if cmd.get('direct'):
                    self._direct_mode[cs]       = True
                    self._commanded_heading[cs] = self._route_hdg.get(cs, bs.traf.hdg[idx])
                else:
                    self._direct_mode[cs]       = False
                    self._commanded_heading[cs] = cmd['target_hdg']
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

            # Only an EXECUTED advisory earns the shorter delay_next_s for this cycle.
            self._executed_count[cs] = self._executed_count.get(cs, 0) + 1
            self._ep_stats['executed'] += 1

    def _focus_tlos(self, cs):
        """Time-to-LoS (s) of the aircraft's worst pair, inverted from its urgency, or None
        when it has no conflict. Used to measure how early the agent acts."""
        if cs not in self._urgency_cs_list or not self._urgency_matrix.size:
            return None
        row = self._urgency_cs_list.index(cs)
        if row >= self._urgency_matrix.shape[0]:
            return None
        u = float(self._urgency_matrix[row].max())
        if u <= 0:
            return None                      # conflict-free: not an anticipation datapoint
        return CONFIG['t_warn'] * max(0.0, 1.0 - min(u, 1.0))

    def _current_los_pairs(self):
        """Callsign pairs currently in loss of separation (urgency > 1 marks an active LoS)."""
        m = self._urgency_matrix
        if not m.size:
            return set()
        rows, cols = np.where(m > 1.0)
        return {(self._urgency_cs_list[i], self._urgency_cs_list[j])
                for i, j in zip(rows, cols) if i < j}

    def _refresh_route_headings(self):
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

    def _update_direct_headings(self):
        """Re-issue the fixed route heading for every fly-direct aircraft each step."""
        for cs, on in self._direct_mode.items():
            idx = bs.traf.id2idx(cs)
            if on and idx >= 0 and cs in self._route_hdg:
                self._commanded_heading[cs] = self._route_hdg[cs]
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # -- Observation -----------------------------------------------------------

    def _get_observation(self):
        """ACAS Xu states for the focus aircraft against its nearest/most-urgent intruders,
        reported in raw physical units (NM, kt, s, rad) -- see the module docstring."""
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            # no controllable aircraft: on-route, nominal speed, conflict-free, nothing pending
            self._last_intruder_cs = [None] * N_NEIGHBOURS
            return np.array([0.0, CONFIG['ac_speed'], 0.0, CONFIG['ac_speed'], 0.0, 0.0]
                            + self._EMPTY_SLOT * N_NEIGHBOURS, dtype=np.float32)

        idx     = bs.traf.id2idx(cs)
        own_hdg = bs.traf.hdg[idx]
        own_pos = aircraft_position_nm(idx)
        own_spd = aircraft_speed_nms(idx)
        own_ve, own_vn = heading_to_velocity(own_spd, own_hdg)
        sin_hdg, cos_hdg = math.sin(math.radians(own_hdg)), math.cos(math.radians(own_hdg))

        route_hdg = self._route_hdg[cs]
        cmd_hdg   = self._commanded_heading.get(cs, own_hdg)
        dpsi_act  = math.radians(wrap_to_180(own_hdg - route_hdg))   # actual heading error to route
        a_cmd     = math.radians(wrap_to_180(cmd_hdg - route_hdg))   # commanded heading error
        v_cmd     = self._commanded_mach.get(cs, CONFIG['ac_mach']) * KT_PER_MACH

        # pre-computed urgency row for this aircraft (built in _select_focus_aircraft);
        # used to prioritise which intruders fill the observation slots below.
        urgency_row = None
        if cs in self._urgency_cs_list:
            row = self._urgency_cs_list.index(cs)
            if row < self._urgency_matrix.shape[0]:
                urgency_row = self._urgency_matrix[row]

        retn_conf = return_blocked(cs, self._active_callsigns, self._route_hdg)

        obs = [dpsi_act,
               own_spd * NMS_TO_KT,
               a_cmd,
               v_cmd,
               retn_conf,      # retn_conf ("safe to return"); dummied to 0.0 when ablated
               # Whether an instruction is outstanding -- the honest observable: under a
               # probabilistic delay the controller knows the pilot has not acted yet, but
               # not when they will. Constant 0.0 when delay_mode='none'.
               1.0 if cs in self._pending_cmd else 0.0]

        sep = CONFIG['sep_nm']
        intruders = []
        for other in self._active_callsigns:
            j = bs.traf.id2idx(other)
            if other == cs or j < 0:
                continue

            int_pos = aircraft_position_nm(j)
            d_east, d_north = int_pos[0] - own_pos[0], int_pos[1] - own_pos[1]
            dist_nm = math.sqrt(d_east ** 2 + d_north ** 2)
            int_hdg = bs.traf.hdg[j]
            int_spd = aircraft_speed_nms(j)

            ego_lat = d_east * cos_hdg - d_north * sin_hdg   # + = right of ownship heading
            ego_fwd = d_east * sin_hdg + d_north * cos_hdg   # + = ahead

            theta = math.atan2(ego_lat, ego_fwd)
            psi   = math.radians(wrap_to_180(int_hdg - own_hdg))
            v_int = int_spd * NMS_TO_KT

            if dist_nm < sep:
                tlos = 0.0                             # already inside the protected zone
            else:
                int_ve, int_vn = heading_to_velocity(int_spd, int_hdg)
                dv_east, dv_north = int_ve - own_ve, int_vn - own_vn
                rel_spd_sq = dv_east ** 2 + dv_north ** 2
                range_rate = d_east * dv_east + d_north * dv_north
                t_los = time_to_los(dist_nm ** 2, range_rate, rel_spd_sq, sep)
                # No upper clip beyond the planning horizon: t_los is unbounded for a
                # near-parallel pair, and letting those values through would dominate
                # the VecNormalize running variance.
                tlos = NO_CONFLICT_S if t_los is None else min(max(0.0, t_los), NO_CONFLICT_S)

            pair_u = 0.0
            if urgency_row is not None and other in self._urgency_cs_list:
                col = self._urgency_cs_list.index(other)
                if col < len(urgency_row):
                    pair_u = float(urgency_row[col])

            intruders.append((pair_u, dist_nm, [dist_nm, theta, psi, v_int, tlos], other))

        # fill slots: urgent pairs first (descending urgency), then nearest (ascending distance)
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
        for k in range(N_NEIGHBOURS):
            obs += selected[k][2] if k < len(selected) else self._EMPTY_SLOT
        return np.array(obs, dtype=np.float32)

    # -- Spawning & exits ------------------------------------------------------

    def _process_pending_spawns(self):
        """Spawn aircraft whose countdown reached 1; decrement the rest."""
        ready    = sorted(slot for slot, t in list(self._pending_spawns.items()) if t <= 1)
        requeued = set()
        for slot in ready:
            del self._pending_spawns[slot]
            ac = self._generate_replacement(slot)
            if ac is not None:
                self._spawn_aircraft(slot, ac)
            else:
                self._pending_spawns[slot] = 5   # retry in 5 steps
                requeued.add(slot)
        for slot in list(self._pending_spawns):
            if slot not in requeued:
                self._pending_spawns[slot] -= 1

    def _spawn_ok(self, ac):
        """Admit a candidate spawn only if it clears (1) a static buffer to all traffic and
        (2) a conflict-free entry: flying its route at cruise it must not reach CPA < sep
        against any active aircraft (current trajectory) within t_warn."""
        pos_c = latlon_to_nm(CONFIG['center_ll'], float(ac['sp_ll'][0]), float(ac['sp_ll'][1]))
        min_spawn_sep = CONFIG['sep_nm'] + CONFIG['buffer_nm']
        sep, horizon  = CONFIG['sep_nm'], CONFIG['t_warn']
        cand_ve, cand_vn = heading_to_velocity(CRUISE_SPD_NMS, ac['heading'])

        for cs in self._active_callsigns:
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue
            pos_o, vel_o = aircraft_state(idx)
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

    def _generate_replacement(self, slot):
        """Find a placement for a new aircraft that clears the spawn buffer to all traffic."""
        for _ in range(CONFIG['max_placement_tries']):
            ac = place_aircraft(self._polygon_shape, random.randint(0, self.n_aircraft - 1), self.n_aircraft)
            if self._spawn_ok(ac):
                return ac
        return None

    def _spawn_aircraft(self, slot, ac):
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        mach = CONFIG['ac_mach']

        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(ac['sp_ll'][0]), aclon=float(ac['sp_ll'][1]),
                    achdg=float(ac['heading']), acspd=mach,
                    acalt=CONFIG['altitude'] * 30.48)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')

        self._destination_ll[cs]      = ac['dest_ll']
        self._ref_ll[cs]              = ac['ref_ll']
        self._spawn_ll[cs]            = ac['sp_ll']
        self._spawn_step[cs]          = self._step_count
        self._route_hdg[cs]           = float(ac['heading'])
        self._commanded_heading[cs]   = float(ac['heading'])
        self._commanded_mach[cs]      = CONFIG['ac_mach']
        self._direct_mode[cs]         = False
        self._executed_count[cs]      = 0
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    def _cross_track_nm(self, cs, idx):
        """Perpendicular distance (NM) from the aircraft to its intended spawn->reference
        line. Unlike distance-to-reference this does not care WHERE along the route the
        aircraft crossed the boundary, only how far sideways it ended up."""
        sp, rf = self._spawn_ll.get(cs), self._ref_ll.get(cs)
        if sp is None or rf is None:
            return None
        centre = CONFIG['center_ll']
        a = latlon_to_nm(centre, float(sp[0]), float(sp[1]))
        b = latlon_to_nm(centre, float(rf[0]), float(rf[1]))
        p = aircraft_position_nm(idx)
        de, dn = b[0] - a[0], b[1] - a[1]
        chord  = math.hypot(de, dn)
        if chord < 1e-9:
            return None
        return abs((p[0] - a[0]) * dn - (p[1] - a[1]) * de) / chord

    def _score_arrival(self, cs, idx):
        """Score one exiting aircraft under all three arrival flavors (lenient -> strict).
        Aircraft that barely existed never flew a route, so they are excluded entirely."""
        self._ep_stats['exits'] += 1

        ref_ll  = self._ref_ll.get(cs)
        dist_nm = geo.kwikdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]),
                               float(ref_ll[0]), float(ref_ll[1]))
        if dist_nm <= CONFIG['arrival_tol_nm']:
            self._ep_stats['arrivals'] += 1          # legacy metric, kept for continuity

        life = self._step_count - self._spawn_step.get(cs, 0)
        if life <= CONFIG['arrival_min_life_steps']:
            return
        self._ep_stats['flown'] += 1

        # (1) on_route -- is it flying straight at its destination again, i.e. no longer
        #     in an active deviation? Lenient: ignores accumulated lateral displacement.
        route_hdg = self._route_hdg.get(cs)
        if route_hdg is not None:
            if abs(wrap_to_180(float(bs.traf.hdg[idx]) - route_hdg)) <= CONFIG['arrival_hdg_tol_deg']:
                self._ep_stats['arr_on_route'] += 1

        # (2) xtrack -- lateral offset from the intended route line. Catches exactly the
        #     displacement that (1) ignores, without depending on the exit geometry.
        xt = self._cross_track_nm(cs, idx)
        if xt is not None and xt <= CONFIG['arrival_xtrack_nm']:
            self._ep_stats['arr_xtrack'] += 1

        # (3) ref -- strictest: did it actually reach its reference point?
        if dist_nm <= CONFIG['arrival_tol_nm']:
            self._ep_stats['arr_ref'] += 1

    def _process_exits(self):
        """Remove aircraft that have left the sector, scoring on-target arrivals, and
        queue their slot for a respawn."""
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                if self._ref_ll.get(cs) is not None:
                    self._score_arrival(cs, idx)
                bs.traf.delete(idx)
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            for d in (self._destination_ll, self._ref_ll, self._route_hdg, self._commanded_heading,
                      self._commanded_mach, self._direct_mode, self._steps_since_urgency,
                      self._pending_cmd, self._executed_count, self._spawn_step, self._spawn_ll):
                d.pop(cs, None)
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = random.randint(*self._spawn_delay_range)

    def _find_exited(self):
        """Callsigns that have left the sector or are no longer in BlueSky."""
        exited = []
        for cs in list(self._active_callsigns):
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

    # -- Misc ------------------------------------------------------------------

    def _flat_latlon(self):
        """Polygon vertices flattened to [lat, lon, lat, lon, ...] for BlueSky POLY."""
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
