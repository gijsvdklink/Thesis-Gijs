# v4 ATCO conflict-resolution environment: one focus aircraft per step, advisories acted on after a delay drawn per piece of advice, and an INITIAL HEADING per aircraft that every directional quantity is measured against.

import math
from random import Random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shapely.geometry import Point
from shapely.prepared import prep

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.stack.stackbase import Stack as _BsStack

from .config import (CONFIG, TRAIN_SEEDS, SEED_STRIDE, STEP_DURATION_S, OBS_DIM, N_ACTIONS, N_NEIGHBOURS,
                     CRUISE_SPD_NMS, NMS_TO_KT, KT_PER_MACH, EMPTY_RANGE_NM, NO_CONFLICT_S,
                     TURN_DELTAS, SPEED_ACTIONS, HOLD_ACTION, RETURN_TO_ROUTE_ACTION,
                     ACT_COST)
from .delays import DELAY_MODES, ResponseDelay
from .geometry import (latlon_to_nm, nm_to_latlon, wrap_to_180, heading_to_velocity,
                       point_ahead)
from .conflict import (traffic_states, urgency_matrix,
                       route_return_blocked, time_to_loss_of_separation)
from .sector import make_sector_polygon, plan_entry_route

_bs_initialized = False


class _ScreenDummy(ScreenIO):
    """Silences BlueSky's screen output (we run headless)."""
    def echo(self, text='', flags=0):
        pass


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # Empty intruder slot: unreachably far, stationary, no predicted LoS.
    _EMPTY_SLOT = [EMPTY_RANGE_NM, 0.0, 0.0, 0.0, NO_CONFLICT_S]

    def __init__(self, delay_mode=None, seed=None, delay_mean_s=None, seed_pool=TRAIN_SEEDS):
        super().__init__()
        global _bs_initialized

        # Per-instance rather than a CONFIG edit: SubprocVecEnv workers do not inherit CONFIG changes.
        self.delay_mode = delay_mode if delay_mode is not None else CONFIG['delay_mode']
        if self.delay_mode not in DELAY_MODES:
            raise ValueError(f'unknown delay_mode {self.delay_mode!r}; expected {DELAY_MODES}')

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        # Delay magnitude: the MEAN response time in seconds, one number per run.
        self.delay_mean_s = delay_mean_s

        # Which pool this environment's episodes come from, and where in it to start.
        self.seed_pool      = seed_pool
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
        self._polygon_ready       = None
        self._slots               = []
        self._active_callsigns    = set()
        self._focus_cs            = None
        self._focus_hold_steps    = 0
        self._next_callsign_id    = 0
        self._step_count          = 0
        self._max_steps           = 0
        self._pending_spawns      = {}   # slot -> steps until the slot is refilled
        self._los_seconds_this_step = 0   # simulated seconds of this step spent in LoS
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
        self._initial_hdg         = {}   # heading (deg) assigned at spawn; NEVER changes
        self._exit_ref_nm         = {}   # where it would leave the sector if never turned
        self._commanded_heading   = {}   # last EXECUTED heading instruction
        self._commanded_mach      = {}   # last EXECUTED speed instruction
        self._steps_since_urgency = {}   # steps since this aircraft last had a conflict
        self._pending_advisory    = {}   # issued, not yet executed by the pilot
        self._spawn_nm            = {}   # where it entered (NM, east/north)
        self._flown_nm            = {}   # track length flown so far
        self._last_pos_nm         = {}   # position at the previous step, for that track length

        # Registry so _forget_aircraft clears every trace of a departed aircraft.
        self._per_aircraft_state = [
            self._initial_hdg, self._exit_ref_nm,
            self._commanded_heading, self._commanded_mach,
            self._steps_since_urgency, self._pending_advisory,
            self._spawn_nm, self._flown_nm, self._last_pos_nm,
        ]

        self._prev_los_pairs      = set()
        self._prev_conflict_pairs = set()
        self._ep_stats = {
            'reward': 0.0, 'steps': 0, 'actions': [],
            'los_seconds': 0,      # simulated seconds with at least one pair in LoS
            'los_events': 0,       # distinct intrusions (entries, scanned every second)
            'conflicts': 0,        # distinct predicted intrusions within t_warn (entries, per step)
            'flight_s': 0.0,       # airborne time flown by all aircraft
            'exits': 0,            # aircraft that left having actually flown
            'on_route': 0,         # ...of which left within the heading tolerance
            'deviation_nm': 0.0,   # ...summed distance from the no-turn exit point
            'flown_nm': 0.0,       # ...summed track length actually flown
            'route_nm': 0.0,       # ...summed straight-line length of the route they were given
            # Drift summed over aircraft and steps; divided by aircraft-steps for a mean angle.
            'drift_deg_sum': 0.0, 'drift_samples': 0,
            'delay_sum_s': 0.0, 'delay_served': 0,
            'focus_spells': 0, 'focus_spell_steps': 0,
            'discarded': 0,        # advisories replaced before they could be flown
            'repeats': 0,          # the same advice re-selected while it was still standing
            'turns': 0, 'speeds': 0,   # advisories actually transmitted, by kind
        }

    # -- Gym interface ---------------------------------------------------------

    def _new_episode_generators(self):
        """Draw this episode's scenario seed from this environment's pool, then three independent streams from it."""
        low, high = self.seed_pool
        self.episode_seed = low + (self._seed_base
                                   + self._episode_index * SEED_STRIDE) % (high - low)
        self._episode_index += 1

        # Three streams off one seed, so a delay draw never shifts a scenario decision.
        self.scenario_rng   = Random(self.episode_seed)
        self.traffic_rng    = Random(self.episode_seed + 1)
        self.delay_rng      = np.random.default_rng(self.episode_seed + 2)
        self.response_delay = ResponseDelay(
            self.delay_mode, self.delay_rng,
            **({'mean_s': self.delay_mean_s} if self.delay_mean_s is not None else {}))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_base, self._episode_index = int(seed), 0   # restart the sequence
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

        # The aircraft that HELD the focus when this action was chosen; the reward is charged to it.
        acting_cs = self._focus_cs
        if acting_cs:
            self._issue_advisory(acting_cs, action)

        # Separation is scanned every simulated second, so brief intrusions between steps are seen.
        self._advance_simulation()
        self._step_count += 1

        self._remove_exited_aircraft()
        self._refresh_traffic_view()
        self._focus_cs = self._select_focus_aircraft()
        reward         = self._compute_reward(acting_cs, action)
        self._record_step_stats(action, reward)

        truncated = self._step_count >= self._max_steps
        info = {'los_seconds': self._los_seconds_this_step,
                'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(self._episode_summary())
        return self._build_observation(), reward, False, truncated, info

    def _advance_simulation(self):
        """One RL step of simulated time, flushing the advisory queue and scanning separation at 1 s resolution."""
        self._los_seconds_this_step = 0

        for _ in range(CONFIG['action_freq']):
            self._execute_due_advisories()
            bs.sim.step()
            self._sim_time_s += CONFIG['sim_dt']

            pairs = self._scan_separation()
            self._los_seconds_this_step += bool(pairs)
            # Entries only: a pair already in LoS a second ago is the same event.
            self._ep_stats['los_events'] += len(pairs - self._prev_los_pairs)
            self._prev_los_pairs = pairs

    def _scan_separation(self):
        """Callsign pairs closer than the separation minimum at this instant, read from BlueSky directly."""
        flying, indices = self._airborne_indices()
        if len(flying) < 2:
            return set()
        pos     = traffic_states(indices)[0]
        delta   = pos[:, None, :] - pos[None, :, :]
        dist_sq = (delta ** 2).sum(axis=-1)
        rows, cols = np.where(dist_sq < CONFIG['sep_nm'] ** 2)
        return {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}

    # -- Traffic picture -------------------------------------------------------

    def _airborne_indices(self):
        """(callsigns, BlueSky indices) of everything airborne, in sorted callsign order, mapped in one pass."""
        index_of = {cs: i for i, cs in enumerate(bs.traf.id)}
        flying, indices = [], []
        for cs in sorted(self._active_callsigns):
            idx = index_of.get(cs, -1)
            if idx >= 0:
                flying.append(cs)
                indices.append(idx)
        return flying, indices

    def _refresh_traffic_view(self):
        """Gather the whole traffic picture once per step; everything downstream reads these arrays."""
        flying, indices = self._airborne_indices()
        self._urgency_cs_list = flying
        self._row_of          = {cs: i for i, cs in enumerate(flying)}

        if not flying:
            self._pos = self._vel = np.zeros((0, 2))
            self._hdg = self._return_blocked = np.zeros(0)
            self._urgency_matrix = np.zeros((0, 0))
            self._prev_conflict_pairs = set()
            return

        self._pos, self._vel = traffic_states(indices)
        self._hdg            = bs.traf.hdg[np.asarray(indices, dtype=int)]
        self._urgency_matrix = urgency_matrix(self._pos, self._vel)
        self._count_conflicts(flying)

        # Whether each aircraft could turn back onto its route: same geometry, on INITIAL headings.
        speed      = np.hypot(self._vel[:, 0], self._vel[:, 1])
        init_hdg   = np.radians([self._initial_hdg.get(cs, h) for cs, h in zip(flying, self._hdg)])
        route_vel  = np.stack([speed * np.sin(init_hdg), speed * np.cos(init_hdg)], axis=1)
        self._return_blocked = route_return_blocked(self._pos, route_vel)

    # -- Statistics ------------------------------------------------------------

    def _count_conflicts(self, flying):
        """Count pairs that ENTER the conflict horizon: urgency > 0 is a LoS predicted within t_warn."""
        rows, cols = np.where(self._urgency_matrix > 0)
        pairs = {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}
        self._ep_stats['conflicts'] += len(pairs - self._prev_conflict_pairs)
        self._prev_conflict_pairs = pairs

    def _record_step_stats(self, action, reward):
        """Accumulate the per-step counters that feed _episode_summary."""
        s = self._ep_stats
        s['reward'] += reward
        s['steps']  += 1
        s['actions'].append(action)
        s['los_seconds'] += self._los_seconds_this_step

        # Airborne time flown this step: the denominator that makes LoS counts comparable.
        s['flight_s'] += len(self._urgency_cs_list) * STEP_DURATION_S

        # Drift of every airborne aircraft from the heading it was assigned at spawn.
        for cs in self._urgency_cs_list:
            if cs in self._initial_hdg:
                s['drift_deg_sum'] += abs(wrap_to_180(
                    self._hdg[self._row_of[cs]] - self._initial_hdg[cs]))
                s['drift_samples'] += 1

        # Track length flown this step, leg by leg; the total is scored when the aircraft leaves.
        for cs in self._urgency_cs_list:
            self._advance_track(cs, self._pos[self._row_of[cs]])

    def _advance_track(self, cs, position):
        """Add the leg this aircraft just flew to its track length."""
        previous = self._last_pos_nm.get(cs)
        if previous is None:
            return
        self._flown_nm[cs] = self._flown_nm.get(cs, 0.0) + math.hypot(position[0] - previous[0],
                                                                     position[1] - previous[1])
        self._last_pos_nm[cs] = (float(position[0]), float(position[1]))

    def _episode_summary(self):
        """End-of-episode metrics, logged by the training callbacks."""
        s = self._ep_stats
        flight_hours = max(s['flight_s'] / 3600.0, 1e-9)
        served       = max(s['delay_served'], 1)
        exits        = s['exits']

        # Advisories TRANSMITTED, counted in _issue_advisory rather than off the action histogram.
        turns, speeds = s['turns'], s['speeds']

        return {
            'mean_episode_reward': s['reward'] / max(s['steps'], 1),
            'ep_reward_total':     s['reward'],
            'ep_length':           s['steps'],
            'ep_los_seconds':      s['los_seconds'],
            'ep_los_fraction':     s['los_seconds'] / max(s['steps'] * STEP_DURATION_S, 1),
            'ep_los_events':       s['los_events'],
            'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),

            # Traffic-normalised safety: raw LoS counts are not comparable between episodes.
            'ep_flight_hours':      s['flight_s'] / 3600.0,
            'ep_los_events_per_fh': s['los_events'] / flight_hours,
            'ep_conflicts':         s['conflicts'],
            'ep_conflicts_per_fh':  s['conflicts'] / flight_hours,

            # Route keeping over every exit: arrival within arrival_hdg_tol_deg, deviation from the no-turn exit.
            'ep_exit_deviation_nm': s['deviation_nm'] / exits if exits else 0.0,
            # Track flown over the straight route: 1.0 is a perfectly direct crossing.
            'ep_path_ratio':        s['flown_nm'] / s['route_nm'] if s['route_nm'] else 1.0,
            'ep_arrival_rate':      s['on_route'] / exits if exits else 1.0,
            'ep_exits':             exits,

            # Drift from assigned headings and the calls it took; the per-flight-hour rates are the comparable ones.
            'ep_mean_drift_deg':      s['drift_deg_sum'] / max(s['drift_samples'], 1),
            'ep_turns':               turns,
            'ep_speed_changes':       speeds,
            'ep_turns_per_fh':        turns / flight_hours,
            'ep_speed_changes_per_fh': speeds / flight_hours,

            # Diagnostics -- kept in the evaluation CSVs rather than TensorBoard.
            'ep_delay_mean_s':     s['delay_sum_s'] / served,
            'ep_focus_hold_steps': s['focus_spell_steps'] / max(s['focus_spells'], 1),
            'ep_discarded':        s['discarded'],
            # The advice already standing, selected again: charged as workload, but nothing new is assessed.
            'ep_repeats':          s['repeats'],
        }

    # -- Scenario setup --------------------------------------------------------

    def _build_sector(self):
        """Draw this episode's sector and derive the episode length from it."""
        n_ac     = CONFIG['n_aircraft'](self.scenario_rng)
        area_km2 = float(n_ac / CONFIG['rho'](self.scenario_rng))
        poly     = make_sector_polygon(area_km2, self.scenario_rng)

        self._polygon_shape = poly
        # Prepared once per episode: _no_turn_exit_nm runs tens of thousands of containment tests.
        self._polygon_ready = prep(poly)
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm = math.hypot(maxx - minx, maxy - miny)

        # crossings_per_episode traversals at cruise: ~1050-2400 steps, 1.5-3.3 h.
        crossing_time_s = sector_diam_nm / CONFIG['ac_speed'] * 3600
        self._max_steps = max(50, round(
            CONFIG['crossings_per_episode'] * crossing_time_s / STEP_DURATION_S))

        self.n_aircraft = n_ac
        self.rho        = n_ac / area_km2      # recorded per episode by the evaluation
        self._slots     = [None] * n_ac

        bs.tools.areafilter.defineArea('SECTOR', 'POLY', self._polygon_latlon_pairs())
        bs.stack.stack('ASAS OFF')

    def _spawn_initial_traffic(self):
        """Fill every slot at reset; queue and retry any that cannot be placed safely."""
        for slot in range(self.n_aircraft):
            for _ in range(CONFIG['max_placement_tries']):
                route = self._random_start_in_sector()
                if route is not None and self._spawn_is_safe(route):
                    self._create_aircraft(slot, route)
                    break
            else:
                self._pending_spawns[slot] = 5   # could not place now: retry via the queue

    def _random_start_in_sector(self):
        """A random position inside the sector on a random heading, or None if its no-turn exit is too close."""
        minx, miny, maxx, maxy = self._polygon_shape.bounds
        for _ in range(CONFIG['max_placement_tries']):
            east  = self.scenario_rng.uniform(minx, maxx)
            north = self.scenario_rng.uniform(miny, maxy)
            if not self._polygon_ready.contains(Point(east, north)):
                continue                       # the bounding box is not the sector

            heading = self.scenario_rng.uniform(0.0, 360.0)
            sp_ll   = nm_to_latlon(CONFIG['center_ll'], east, north)
            exit_nm = self._no_turn_exit_nm(sp_ll, heading)
            if math.hypot(exit_nm[0] - east, exit_nm[1] - north) >= CONFIG['min_chord_nm']:
                return {'sp_ll': sp_ll, 'heading': heading}
        return None

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """Pick the focus from the urgency matrix: worst pair wins, with hysteresis and an emergency override."""
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

        # Hysteresis: hold the current focus while it is still active, unless resolved or overridden.
        if focus_idx >= 0 and best_cs != self._focus_cs:
            if (focus_urgency > 0 or not focus_resolved or drift_locked) and not emergency:
                best_cs = self._focus_cs

        if best_cs != self._focus_cs:
            if self._focus_cs is not None:      # close out the spell, for the diagnostic
                self._ep_stats['focus_spells']      += 1
                self._ep_stats['focus_spell_steps'] += self._focus_hold_steps + 1
            self._focus_hold_steps = 0
        else:
            self._focus_hold_steps += 1
        return best_cs

    def _heading_drift(self, cs):
        """1 - cos(current heading - initial heading), off the ACTUAL heading: 0 on route, 2 reversed."""
        row = self._row_of.get(cs)
        if row is None:
            return 0.0
        hdg_err = wrap_to_180(self._initial_hdg[cs] - self._hdg[row])
        return 1 - math.cos(math.radians(hdg_err))

    def _select_drifter_to_recover(self, flying):
        """Most-drifted aircraft that is free to turn back, with a hysteresis margin."""
        drift = {cs: self._heading_drift(cs) for cs in flying if cs in self._initial_hdg}

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

    # -- Reward ----------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        """Purely negative: separation loss, drift off the assigned heading, and the cost of the call."""
        r_los = -CONFIG['w_los'] if self._los_seconds_this_step else 0.0

        r_drift = 0.0
        if acting_cs and acting_cs in self._initial_hdg and acting_cs in self._row_of:
            r_drift = -CONFIG['w_drift'] * self._heading_drift(acting_cs)

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_drift + r_work)

    # -- Advisories: issued now, executed after the pilot's response delay -------

    def _current_offset(self, cs):
        """Commanded offset from the initial heading, re-derived from the last EXECUTED advisory."""
        init = self._initial_hdg[cs]
        return wrap_to_180(self._commanded_heading.get(cs, init) - init)

    def _build_advisory(self, cs, action_idx):
        """Action index -> what the pilot is asked to fly; turns accumulate on the last EXECUTED advisory."""
        advisory = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            base = self._commanded_mach.get(cs, CONFIG['ac_mach'])
            mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            advisory['target_mach'] = min(CONFIG['ac_mach_max'],
                                          max(CONFIG['ac_mach_min'], mach))

        elif action_idx in TURN_DELTAS:
            offset = self._current_offset(cs) + TURN_DELTAS[action_idx]
            advisory['target_hdg'] = (self._initial_hdg[cs] + offset) % 360

        elif action_idx == RETURN_TO_ROUTE_ACTION:
            advisory['target_hdg'] = self._initial_hdg[cs] % 360

        return advisory

    def _issue_advisory(self, cs, action_idx):
        """Queue the advisory: CHANGED advice draws a fresh delay and discards what it replaced, while the SAME advice left standing keeps the assessment already running."""
        if action_idx == HOLD_ACTION or cs not in self._row_of:
            return   # hold is a true no-op: no advisory is transmitted at all

        advisory = self._build_advisory(cs, action_idx)
        pending  = self._pending_advisory.get(cs)

        if pending is not None and self._same_advice(pending, advisory):
            self._ep_stats['repeats'] += 1
            return

        if pending is not None:
            self._ep_stats['discarded'] += 1

        # Counted here rather than from the action histogram: these are the advisories transmitted.
        self._ep_stats['speeds' if 'target_mach' in advisory else 'turns'] += 1

        delay_s = self.response_delay.sample_delay_s()
        advisory['issued_at_s']  = self._sim_time_s
        advisory['execute_at_s'] = self._sim_time_s + delay_s
        self._pending_advisory[cs] = advisory

    def _same_advice(self, pending, advisory):
        """Do these two advisories ask for the same thing? Compared by what is COMMANDED, not by action index."""
        for key in ('target_hdg', 'target_mach'):
            if key in pending and key in advisory:
                return abs(pending[key] - advisory[key]) < 1e-9
        return False

    def _execute_due_advisories(self):
        """Fly every advisory whose execution time has come: the only path to BlueSky, called once per second."""
        for cs, advisory in list(self._pending_advisory.items()):
            if self._sim_time_s < advisory['execute_at_s']:
                continue
            del self._pending_advisory[cs]

            # The delay the pilot actually took, rounded up to the second the queue was checked.
            self._ep_stats['delay_sum_s']  += self._sim_time_s - advisory['issued_at_s']
            self._ep_stats['delay_served'] += 1

            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue                     # aircraft left before the pilot acted

            if 'target_mach' in advisory:
                self._commanded_mach[cs] = advisory['target_mach']
                bs.stack.stack(f'SPD {cs} {advisory["target_mach"]:.3f}')
            else:
                # Absolute target fixed at issue time: the initial heading it offsets never moves.
                self._commanded_heading[cs] = advisory['target_hdg']
                bs.stack.stack(f'HDG {cs} {advisory["target_hdg"]:.1f}')

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
        """The 7 ownship features: dpsi what the aircraft is doing, a_cmd what was asked, both off the INITIAL heading."""
        row      = self._row_of[cs]
        own_hdg  = self._hdg[row]
        init_hdg = self._initial_hdg[cs]
        cmd_hdg  = self._commanded_heading.get(cs, own_hdg)

        dpsi_act = math.radians(wrap_to_180(own_hdg - init_hdg))   # drift from assigned heading
        a_cmd    = math.radians(wrap_to_180(cmd_hdg - init_hdg))   # commanded offset
        v_own    = math.hypot(self._vel[row, 0], self._vel[row, 1]) * NMS_TO_KT
        v_cmd    = self._commanded_mach.get(cs, CONFIG['ac_mach']) * KT_PER_MACH

        # Time since the advisory now standing was issued; a replacement restarts it.
        pending = self._pending_advisory.get(cs)
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
                # Capped at the horizon: unbounded t_los would dominate the VecNormalize variance.
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
        """Admit a spawn only if it clears a static buffer and predicts no LoS within t_warn."""
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

        spawn_nm = latlon_to_nm(CONFIG['center_ll'],
                                float(route['sp_ll'][0]), float(route['sp_ll'][1]))

        self._exit_ref_nm[cs]         = self._no_turn_exit_nm(
            (float(route['sp_ll'][0]), float(route['sp_ll'][1])), float(route['heading']))
        self._spawn_nm[cs]            = (float(spawn_nm[0]), float(spawn_nm[1]))
        self._flown_nm[cs]            = 0.0
        self._last_pos_nm[cs]         = self._spawn_nm[cs]
        self._initial_hdg[cs]         = float(route['heading'])
        self._commanded_heading[cs]   = float(route['heading'])
        self._commanded_mach[cs]      = CONFIG['ac_mach']
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
                self._pending_spawns[slot] = 1

    def _exited_callsigns(self):
        """Callsigns that have left the sector, sorted because each exit draws a respawn from traffic_rng."""
        callsigns = sorted(self._active_callsigns)
        index_of  = {cs: i for i, cs in enumerate(bs.traf.id)}
        airborne  = [(cs, index_of[cs]) for cs in callsigns if cs in index_of]

        inside_sector = {}
        if airborne:
            indices = np.array([idx for _, idx in airborne], dtype=int)
            inside  = bs.tools.areafilter.checkInside(
                'SECTOR', bs.traf.lat[indices], bs.traf.lon[indices],
                np.full(len(indices), CONFIG['altitude'] * 30.48))
            inside_sector = {cs: bool(ok) for (cs, _), ok in zip(airborne, inside)}

        # Gone from BlueSky altogether counts as exited, hence the False default.
        return [cs for cs in callsigns if not inside_sector.get(cs, False)]

    def _no_turn_exit_nm(self, spawn_ll, heading_deg):
        """Where this aircraft would leave if never turned: walks the initial heading out in 2 NM steps."""
        previous = latlon_to_nm(CONFIG['center_ll'], *spawn_ll)

        for step in range(1, 1000):                    # 2000 NM, far beyond any sector
            lat, lon = point_ahead(spawn_ll, heading_deg, 2.0 * step)
            point = latlon_to_nm(CONFIG['center_ll'], lat, lon)

            if not self._polygon_ready.contains(Point(point[0], point[1])):
                return (previous + point) / 2.0        # midway across the last step
            previous = point
        return previous

    def _score_arrival(self, cs, idx):
        """Score one exiting aircraft: on-route heading, how far off it left, and how far it flew. Metrics only."""
        self._ep_stats['exits'] += 1
        pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])

        # Track flown against the straight route it was given: spawn point to the no-turn exit.
        self._advance_track(cs, pos)
        spawn, exit_ref = self._spawn_nm.get(cs), self._exit_ref_nm.get(cs)
        if spawn is not None and exit_ref is not None:
            self._ep_stats['flown_nm'] += self._flown_nm.get(cs, 0.0)
            self._ep_stats['route_nm'] += math.hypot(exit_ref[0] - spawn[0],
                                                     exit_ref[1] - spawn[1])

        # Exit deviation: boundary distance between where it left and where it would have left unturned.
        if exit_ref is not None:
            self._ep_stats['deviation_nm'] += math.hypot(pos[0] - exit_ref[0],
                                                         pos[1] - exit_ref[1])

        init_hdg = self._initial_hdg.get(cs)
        if init_hdg is None:
            return
        if abs(wrap_to_180(float(bs.traf.hdg[idx]) - init_hdg)) <= CONFIG['arrival_hdg_tol_deg']:
            self._ep_stats['on_route'] += 1

    # -- Misc ------------------------------------------------------------------

    def _polygon_latlon_pairs(self):
        """Polygon vertices flattened to [lat, lon, lat, lon, ...] for BlueSky POLY."""
        coords = []
        for v in self.polygon:
            lat, lon = nm_to_latlon(CONFIG['center_ll'], v[0], v[1])
            coords += [float(lat), float(lon)]
        return coords
