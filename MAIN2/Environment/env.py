# ATCO conflict-resolution environment: one focus aircraft per step, advisories acted on after a delay drawn per piece of advice, and an INITIAL HEADING per aircraft that every directional quantity is measured against.

import math
from random import Random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import shapely


import bluesky as bs
from bluesky.stack.stackbase import Stack as _BsStack
from bluesky.tools.misc import degto180

from .config import (CONFIG, TRAINING_SCENARIOS, HELD_OUT, STEP_DURATION_S, OBS_DIM,
                     N_ACTIONS, N_NEIGHBOURS,
                     CRUISE_SPD_NMS, NMS_TO_KT, KT_PER_MACH, CRUISE_ALT_M,
                     EMPTY_RANGE_NM, NO_CONFLICT_S,
                     HOLD_ACTION, ACT_COST, TURN_DELTAS, SPEED_ACTIONS,
                     RETURN_TO_INITIAL_HDG_ACTION)
from .atco import DELAY_MODES, ATCO
from .geometry import (latlon_to_nm, nm_to_latlon, heading_to_velocity, cpa, pairwise,
                       time_to_los, heading_drift, urgency_matrix, ON_ROUTE_DRIFT)
from .sector import make_sector_polygon, plan_entry_route, exit_point
from .stats import new_ep_stats, episode_summary
from .traffic import start_bluesky, traffic_states

# The two ways the held instruction can be observed; see AirspaceEnv.__init__.
PENDING_OBS = ('offset', 'target')


# -- The per-aircraft record: what the CONTROLLER knows, not what BlueSky simulates ---


class Aircraft:

    def __init__(self, initial_hdg, no_turn_exit_nm, spawn_pos_nm, prev_pos_nm,
                 commanded_hdg, commanded_mach, steps_since_conflict):
        self.initial_hdg         = initial_hdg          # heading (deg) at spawn; NEVER changes: 84.0
        self.no_turn_exit_nm     = no_turn_exit_nm      # where it would leave if never turned: array([31.8, -19.4])
        self.spawn_pos_nm        = spawn_pos_nm         # where it entered (NM, east/north): (-38.2, 11.6)
        self.prev_pos_nm         = prev_pos_nm          # position at the previous step: (-12.0, 30.9)
        self.commanded_hdg       = commanded_hdg        # last EXECUTED heading instruction: 129.0
        self.commanded_mach      = commanded_mach       # last EXECUTED speed instruction: 0.82
        self.steps_since_conflict = steps_since_conflict  # steps since it was last in conflict: 12
        self.flown_nm            = 0.0                  # track length flown so far: 62.4
        # The last advisory TRANSMITTED for this aircraft, so that repeating the same advice can
        # be told from changing it. None until it has been advised at all.
        self.last_advisory_action = None                # 2, meaning the last advice was turn -30


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # Empty intruder slot: unreachably far, stationary, no predicted LoS.
    _EMPTY_SLOT = [EMPTY_RANGE_NM, 0.0, 0.0, 0.0, NO_CONFLICT_S]

    def __init__(self, delay_mode=None, seed=None, delay_mean_s=None, pending_obs='offset'):
        super().__init__()
        # Per-instance rather than a CONFIG edit: SubprocVecEnv workers do not inherit CONFIG changes.
        self.delay_mode = delay_mode if delay_mode is not None else CONFIG['delay_mode']
        if self.delay_mode not in DELAY_MODES:
            raise ValueError(f'unknown delay_mode {self.delay_mode!r}; expected {DELAY_MODES}')

        # How the instruction the ATCO is holding appears in pend_hdg and pend_spd: 'offset' from
        # what the aircraft already flies (what every trained model so far has seen), or 'target',
        # the held heading and speed themselves in the frame of h_cmd and v_cmd.
        if pending_obs not in PENDING_OBS:
            raise ValueError(f'unknown pending_obs {pending_obs!r}; expected {PENDING_OBS}')
        self.pending_obs = pending_obs

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        # Delay magnitude: the MEAN response time in seconds, one number per run.
        self.delay_mean_s = delay_mean_s

        # The stream this environment draws its training scenarios from.
        self._seed_stream = Random(int(seed if seed is not None else CONFIG['seed']))

        start_bluesky()

        self._new_episode_rngs(None)
        self._reset_episode_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_stream = Random(int(seed))
        self._new_episode_rngs((options or {}).get('scenario_seed'))

        _BsStack.cmdstack.clear()
        bs.traf.reset()
        bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")

        self._reset_episode_state()
        self._build_sector()
        self._spawn_initial_traffic()

        self._refresh_traffic_view()
        self._select_focus()
        return self._build_observation(), {}

    def step(self, action):
        action = int(action)
        self._spawn_due_aircraft()

        # The aircraft that HELD the focus when this action was chosen; the reward is charged to it.
        acting_cs = self.focus_cs
        if acting_cs:
            self._issue_advisory(acting_cs, action)

        # Separation is scanned every simulated second, so brief intrusions between steps are seen.
        self._advance_simulation(acting_cs)
        self._step_count += 1

        self._remove_exited_aircraft()
        self._refresh_traffic_view()
        self._select_focus()
        reward         = self._compute_reward(acting_cs, action)
        self._record_step_stats(action, reward)

        truncated = (self._ep_stats['exits'] >= self._max_exits
                     or self._step_count >= self._max_steps)
        info = {'los_seconds': self._los_seconds_this_step,
                'focus_cs': self.focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(episode_summary(self._ep_stats))
        return self._build_observation(), reward, False, truncated, info

    # -- Focus selection: which single aircraft is advised this step -------------

    def _drift(self, cs):
        # How far this aircraft has strayed from the heading it entered on.
        return heading_drift(self._aircraft[cs].initial_hdg, self._hdg[self._row_of[cs]])

    def _select_focus(self):
        # Pick the aircraft to advise, and close the spell just ended so it can be recorded.
        flying = self._urgency_cs_list
        if not flying:
            return

        worst        = self.urgency.max(axis=1)    # each aircraft's most urgent conflict
        any_conflict = bool((worst > 0).any())

        # Steps since each aircraft was last in conflict; the conflict hold counts on this.
        for i, cs in enumerate(flying):
            ac = self._aircraft[cs]
            ac.steps_since_conflict = 0 if worst[i] > 0 else ac.steps_since_conflict + 1

        # The incumbent is held until it has been clear of conflict for focus_clear_steps, so it
        # is not dropped the instant a pair separates. An urgent conflict anywhere overrides the
        # hold at once; below that threshold the ranking is left alone.
        incumbent     = self.focus_cs
        most_drifting = max(flying, key=self._drift)
        keep = (incumbent in self._row_of
                and self._aircraft[incumbent].steps_since_conflict < CONFIG['focus_clear_steps']
                and worst.max() < CONFIG['focus_emergency_u'])

        if keep:
            best_cs = incumbent
        elif any_conflict:
            # The urgency is symmetric, so both aircraft of the worst pair share the same row
            # maximum. The row sum breaks the tie: the one also caught up in OTHER conflicts is
            # the one worth advising. Without this the tie would fall to BlueSky's row order.
            best_cs = max(flying, key=lambda cs: (worst[self._row_of[cs]],
                                                  self.urgency[self._row_of[cs]].sum()))
        elif self._drift(most_drifting) > ON_ROUTE_DRIFT:
            best_cs = most_drifting
        else:
            # Quiet sector: nothing in conflict and everyone on route. Someone still has to be
            # advised, so the incumbent stays until something happens.
            best_cs = incumbent if incumbent in self._row_of else flying[0]

        if best_cs != incumbent:
            if incumbent is not None:
                self._ep_stats['focus_spells']      += 1
                self._ep_stats['focus_spell_steps'] += self.hold_steps + 1
                # The controller's attention moves with the focus: a half-formed instruction for
                # the aircraft just released is abandoned, not left to fire later. Without this
                # the ATCO holds state for an aircraft the observation no longer describes, so
                # pending reads 0 while an instruction is in fact outstanding.
                if self.atco.cs == incumbent and self.atco.advisory is not None:
                    self._ep_stats['discarded'] += 1
                self.atco.release(incumbent)
            self.hold_steps = 0
        else:
            self.hold_steps += 1

        self.focus_cs = best_cs

    # -- Advisories: issued now, executed after the ATCO's response delay --------

    def _build_advisory(self, action_idx, ac):
        # One action from the policy -> what the ATCO is asked to pass on. Turns accumulate on
        # the last EXECUTED instruction, so the offset is re-derived rather than remembered.
        advisory = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            mach = ac.commanded_mach + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            advisory['target_mach'] = min(CONFIG['ac_mach_max'],
                                          max(CONFIG['ac_mach_min'], mach))

        elif action_idx in TURN_DELTAS:
            offset = degto180(ac.commanded_hdg - ac.initial_hdg) + TURN_DELTAS[action_idx]
            advisory['target_hdg'] = (ac.initial_hdg + offset) % 360

        elif action_idx == RETURN_TO_INITIAL_HDG_ACTION:
            advisory['target_hdg'] = ac.initial_hdg % 360

        return advisory

    def _issue_advisory(self, cs, action_idx):
        if action_idx == HOLD_ACTION or cs not in self._row_of:
            return   # hold is a true no-op: no advisory is transmitted at all

        advisory = self._build_advisory(action_idx, self._aircraft[cs])
        kind     = 'target_mach' if 'target_mach' in advisory else 'target_hdg'

        # Advising what the aircraft already flies is not an instruction, so nothing is
        # transmitted. Advice that merely repeats what the ATCO is holding IS transmitted;
        # the ATCO recognises it and carries on.
        ac        = self._aircraft[cs]
        commanded = ac.commanded_mach if kind == 'target_mach' else ac.commanded_hdg
        if abs(advisory[kind] - commanded) < 1e-9:
            self._ep_stats['repeats'] += 1
            return

        # The ATCO works one instruction at a time, so anything already in hand is dropped --
        # including an instruction for a DIFFERENT aircraft, which is then never flown.
        held = self.atco.advisory
        if not self.atco.receive(cs, advisory, self._sim_time_s):
            self._ep_stats['repeats'] += 1
            return
        if held is not None:
            self._ep_stats['discarded'] += 1

        # Counted here rather than from the action histogram: these are the advisories transmitted.
        self._ep_stats['speeds' if kind == 'target_mach' else 'turns'] += 1
        self._ep_stats['transmitted'][action_idx] += 1

        # The same instruction repeated to the same aircraft is ONE piece of advice, however many
        # times it is sent: turn -30 five times running, or with holds in between, is one. A
        # different action ends the run and starts a new one. Hold transmits nothing, so it never
        # reaches here and cannot break a run.
        if ac.last_advisory_action != action_idx:
            self._ep_stats['speeds_distinct' if kind == 'target_mach' else 'turns_distinct'] += 1
        ac.last_advisory_action = action_idx

    def _still_flying(self, cs):
        return cs in self._aircraft and bs.traf.id2idx(cs) >= 0

    def _execute_due_advisories(self):
        ready = self.atco.act_if_ready(self._sim_time_s, self._still_flying)
        if ready is None:
            return
        cs, advisory = ready

        # Recorded here rather than at issue: a revision can still move the execution time.
        # Only an instruction actually flown is a response time; one deleted because its
        # aircraft left the sector never was one.
        self._ep_stats['delay_sum_s'] += self._sim_time_s - advisory['response_start_s']
        self._ep_stats['delay_acted'] += 1

        ac = self._aircraft[cs]

        if 'target_mach' in advisory:
            ac.commanded_mach = advisory['target_mach']
            bs.stack.stack(f'SPD {cs} {advisory["target_mach"]:.3f}')
        else:
            # Absolute target fixed at issue time: the initial heading it offsets never moves.
            ac.commanded_hdg = advisory['target_hdg']
            bs.stack.stack(f'HDG {cs} {advisory["target_hdg"]:.1f}')

    # -- Observation: what the policy sees ---------------------------------------

    def _build_observation(self):
        cs = self.focus_cs
        if cs is None or cs not in self._row_of:
            # No controllable aircraft: on route, nominal speed, clear, nothing pending.
            self._last_intruder_cs = [None] * N_NEIGHBOURS
            return np.array([0.0, CONFIG['ac_speed'], 0.0, CONFIG['ac_speed'], 0.0, 0.0, 0.0]
                            + self._EMPTY_SLOT * N_NEIGHBOURS, dtype=np.float32)

        obs = self._ownship_features(cs) + self._intruder_features(cs)
        return np.array(obs, dtype=np.float32)

    def _ownship_features(self, cs):
        row      = self._row_of[cs]
        own_hdg  = self._hdg[row]
        ac       = self._aircraft[cs]
        init_hdg = ac.initial_hdg
        cmd_hdg  = ac.commanded_hdg

        dpsi_act = math.radians(degto180(own_hdg - init_hdg))   # drift from assigned heading
        h_cmd    = math.radians(degto180(cmd_hdg - init_hdg))   # commanded offset
        v_own    = math.hypot(self._vel[row, 0], self._vel[row, 1]) * NMS_TO_KT
        v_cmd    = ac.commanded_mach * KT_PER_MACH

        # Time since the advisory now standing was issued; a replacement restarts it.
        pending = self.atco.pending_for(cs)
        wait_s  = self._sim_time_s - pending['issued_at_s'] if pending else 0.0

        # What the controller is holding, as the OFFSET from what the aircraft already flies --
        # the same quantity an action adds, so the policy can see whether the action it is about
        # to choose reproduces the held instruction (a repeat, which lets it mature) or changes
        # it (a revision, which discards it and restarts the response). Without this the two are
        # indistinguishable, and the transition depends on state the policy cannot observe.
        pend_hdg = pend_spd = 0.0
        if pending is not None:
            if 'target_hdg' in pending:
                pend_hdg = math.radians(degto180(pending['target_hdg'] - cmd_hdg))
            else:
                pend_spd = (pending['target_mach'] - ac.commanded_mach) * KT_PER_MACH

        # The same instruction as the heading and speed it sets, measured like h_cmd and v_cmd, so
        # that nothing held reads as h_cmd and v_cmd themselves.
        if self.pending_obs == 'target':
            pend_hdg, pend_spd = h_cmd, v_cmd
            if pending is not None:
                if 'target_hdg' in pending:
                    pend_hdg = math.radians(degto180(pending['target_hdg'] - init_hdg))
                else:
                    pend_spd = pending['target_mach'] * KT_PER_MACH

        return [dpsi_act,
                v_own,
                h_cmd,
                v_cmd,
                float(self._return_blocked[row]),   # 1 = returning is BLOCKED
                1.0 if pending else 0.0,            # constant 0 when delay_mode='none'
                wait_s,
                pend_hdg,                           # 0 when nothing is held
                pend_spd]

    def _intruder_features(self, cs):
        own_row = self._row_of[cs]
        sep     = CONFIG['sep_nm']
        own_pos = self._pos[own_row]
        own_hdg = self._hdg[own_row]
        sin_own = math.sin(math.radians(own_hdg))
        cos_own = math.cos(math.radians(own_hdg))
        urgency_row = self.urgency[own_row]

        intruders = []
        for j, other in enumerate(self._urgency_cs_list):
            if j == own_row:
                continue

            d_east, d_north = self._pos[j] - own_pos
            dist_nm = math.hypot(d_east, d_north)

            ego_right = d_east * cos_own - d_north * sin_own
            ego_ahead = d_east * sin_own + d_north * cos_own
            theta = math.atan2(ego_right, ego_ahead)
            psi   = math.radians(degto180(self._hdg[j] - own_hdg))
            v_int = math.hypot(self._vel[j, 0], self._vel[j, 1]) * NMS_TO_KT

            # 0 inside the protected zone, otherwise the capped prediction from construct_U.
            tlos = 0.0 if dist_nm < sep else float(self.t_los[own_row, j])

            intruders.append((float(urgency_row[j]), dist_nm,
                              [dist_nm, theta, psi, v_int, tlos], other))

        return self._fill_intruder_slots(intruders)

    def _fill_intruder_slots(self, intruders):
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

    # -- Reward ------------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        # One unit per SECOND in which the ADVISED aircraft has lost separation, not one per
        # step in which any was lost: the five seconds inside a step are scanned one by one, so
        # a five-second intrusion is charged five times a one-second one. R_los is therefore
        # -los_seconds, in [-5, 0], and every second of it is attributable to the aircraft the
        # policy was actually given.
        r_los = -CONFIG['w_los'] * self._los_seconds_this_step

        # Summed over the simulated seconds of the step, so w_drift is the cost of one second
        # at that heading offset: one 30-degree turn still equals about 37 s of drifting at
        # 30 degrees.
        r_drift = -CONFIG['w_drift'] * self._drift_sum_this_step

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0

        # Kept apart as well as summed: the three terms answer different questions, and their
        # split is what says whether a policy is paying for safety, for drift or for workload.
        self._ep_stats['reward_los']   += r_los
        self._ep_stats['reward_drift'] += r_drift
        self._ep_stats['reward_work']  += r_work
        return float(r_los + r_drift + r_work)

    # -- Traffic picture: the state everything above reads -----------------------

    def _refresh_traffic_view(self):
        flying, indices = self._airborne_indices()
        self._urgency_cs_list = flying
        self._row_of          = {cs: i for i, cs in enumerate(flying)}

        self._pos, self._vel = traffic_states(indices)
        self._hdg            = bs.traf.hdg[np.asarray(indices, dtype=int)]
        self.urgency, self.t_los = urgency_matrix(self._pos, self._vel)
        self._count_conflicts(flying)

        # Whether each aircraft could return to its initial heading: row i flies its INITIAL
        # heading while every other aircraft keeps its CURRENT velocity. 1.0 = returning is
        # blocked, by a live LoS or one within t_warn.
        speed     = np.hypot(self._vel[:, 0], self._vel[:, 1])
        init_hdg  = np.radians([self._aircraft[cs].initial_hdg for cs in flying])
        route_vel = np.stack([speed * np.sin(init_hdg), speed * np.cos(init_hdg)], axis=1)
        if len(self._pos) < 2:
            self._return_blocked = np.zeros(len(self._pos))
        else:
            route_dist_sq, route_t_los = pairwise(self._pos, self._vel, own_vel=route_vel)
            blocked = ((route_dist_sq < CONFIG['sep_nm'] ** 2)
                       | (route_t_los <= CONFIG['t_warn']))
            np.fill_diagonal(blocked, False)
            self._return_blocked = blocked.any(axis=1).astype(float)

    def _airborne_indices(self, index_of=None):
        if index_of is None:
            index_of = {cs: i for i, cs in enumerate(bs.traf.id)}
        flying, indices = [], []
        for cs in sorted(self._aircraft):
            idx = index_of.get(cs, -1)
            if idx >= 0:
                flying.append(cs)
                indices.append(idx)
        return flying, indices

    # -- Simulation: advancing BlueSky and scanning separation -------------------

    def _advance_simulation(self, acting_cs):
        self._los_seconds_this_step = 0
        self._drift_sum_this_step   = 0.0

        # Nothing is created or deleted inside this loop -- spawns happen before it and exits
        # after -- so BlueSky's row order is fixed and the map is built once instead of five times.
        index_of = {cs: i for i, cs in enumerate(bs.traf.id)}

        for _ in range(CONFIG['action_freq']):
            self._execute_due_advisories()
            bs.sim.step()
            self._sim_time_s += CONFIG['sim_dt']

            # The drift of the advised aircraft is read every simulated second, like the LoS
            # scan, rather than sampled once at the end of the step. A second in which it has
            # already left the sector contributes nothing.
            if acting_cs is not None:
                row = index_of.get(acting_cs)
                if row is not None and acting_cs in self._aircraft:
                    self._drift_sum_this_step += heading_drift(
                        self._aircraft[acting_cs].initial_hdg, float(bs.traf.hdg[row]))

            pairs = self._scan_separation(index_of)

            # The REWARD counts only seconds in which the ADVISED aircraft is the one losing
            # separation. A loss between two aircraft it was never given cannot be explained
            # from its observation -- only four neighbours are encoded -- and could not have
            # been acted on either, since one aircraft is advised at a time. A loss involving
            # the focus ship is always visible: a pair inside sep_nm carries the highest
            # urgency there is, so it always fills the first intruder slot.
            if acting_cs is not None:
                self._los_seconds_this_step += any(acting_cs in pair for pair in pairs)

            # The KPI stays GLOBAL: the reward is a local proxy, the measurement is not.
            # Entries only: a pair already in LoS a second ago is the same event.
            self._ep_stats['los_events'] += len(pairs - self._prev_los_pairs)
            self._prev_los_pairs = pairs

    def _scan_separation(self, index_of=None):
        flying, indices = self._airborne_indices(index_of)
        if len(flying) < 2:
            return set()
        pos     = traffic_states(indices)[0]
        delta   = pos[:, None, :] - pos[None, :, :]
        dist_sq = (delta ** 2).sum(axis=-1)
        rows, cols = np.where(dist_sq < CONFIG['sep_nm'] ** 2)
        return {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}

    # -- Spawning & exits: keeping the sector populated --------------------------

    def _spawn_due_aircraft(self):
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

    def _plan_safe_entry(self):
        for _ in range(CONFIG['max_placement_tries']):
            route = plan_entry_route(self._polygon_shape, self.traffic_rng)
            if route is not None and self._spawn_is_safe(route):
                return route
        return None

    def _spawn_is_safe(self, route):
        _, indices = self._airborne_indices()
        if not indices:
            return True

        # Only the distance counts: an aircraft may appear in a conflict that is already predicted
        # within t_warn, as long as it is sep_nm + buffer_nm clear of all traffic.
        pos, _ = traffic_states(indices)
        dist_sq = ((pos - route['pos_nm']) ** 2).sum(axis=1)
        return not bool((dist_sq < (CONFIG['sep_nm'] + CONFIG['buffer_nm']) ** 2).any())

    def _create_aircraft(self, slot, route):
        cs = f'AC{self._next_callsign_id:02d}'
        self._next_callsign_id += 1
        mach    = CONFIG['ac_mach']
        heading = float(route['heading'])
        lat, lon = nm_to_latlon(CONFIG['center_ll'], *route['pos_nm'])

        bs.traf.cre(cs, actype=CONFIG['ac_type'],
                    aclat=float(lat), aclon=float(lon),
                    achdg=heading, acspd=mach,
                    acalt=CRUISE_ALT_M)
        bs.stack.stack(f'SPD {cs} {mach}')
        bs.stack.stack(f'ALT {cs} FL{CONFIG["altitude"]}')

        entry_nm = (float(route['pos_nm'][0]), float(route['pos_nm'][1]))
        self._aircraft[cs] = Aircraft(
            initial_hdg=heading,
            no_turn_exit_nm=self._no_turn_exit_nm(route['pos_nm'], heading),
            spawn_pos_nm=entry_nm,
            prev_pos_nm=entry_nm,
            commanded_hdg=float(route['heading']),
            commanded_mach=CONFIG['ac_mach'],
            steps_since_conflict=CONFIG['focus_clear_steps'])
        self._slots[slot] = cs

    def _remove_exited_aircraft(self):
        for cs in self._exited_callsigns():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                self._score_exit(cs, idx)
                bs.traf.delete(idx)
            self._slots[slot] = None
            self._aircraft.pop(cs, None)   # one record, so nothing can be left behind
            self.atco.release(cs)           # nothing outstanding for an aircraft that has gone
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = 1

    def _exited_callsigns(self):
        flying, indices = self._airborne_indices()

        inside_sector = {}
        if flying:
            positions = traffic_states(indices)[0]      # NM, the frame the polygon lives in
            # One vectorised predicate for the whole sector rather than a Point object and a
            # separate call per aircraft: this runs every step, for every aircraft.
            inside = shapely.contains_xy(self._polygon_ready, positions[:, 0], positions[:, 1])
            inside_sector = dict(zip(flying, inside))

        # Gone from BlueSky altogether counts as exited, hence the False default.
        return [cs for cs in sorted(self._aircraft) if not inside_sector.get(cs, False)]

    def _no_turn_exit_nm(self, start_nm, heading_deg):
        return exit_point(self._polygon_shape, start_nm, heading_deg)

    # -- Episode setup -----------------------------------------------------------

    def _new_episode_rngs(self, scenario_seed):
        if scenario_seed is None:
            # A training scenario, drawn at random but never one of the 100 held out for the
            # test set: that is what keeps the evaluation honest however long training runs.
            self.episode_seed = self._seed_stream.randrange(TRAINING_SCENARIOS)
            while self.episode_seed in HELD_OUT:
                self.episode_seed = self._seed_stream.randrange(TRAINING_SCENARIOS)
        else:
            self.episode_seed = int(scenario_seed)

        # Three streams spun off the episode seed. Drawn through one master rather than
        # seeded episode_seed+1 / +2, which would make neighbouring scenarios share a stream.
        master              = Random(self.episode_seed)
        self.scenario_rng   = Random(master.getrandbits(64))
        self.traffic_rng    = Random(master.getrandbits(64))
        self.delay_rng      = np.random.default_rng(master.getrandbits(64))
        self.atco           = ATCO(self.delay_mode, self.delay_rng, self.delay_mean_s)

    def _reset_episode_state(self):

        self._next_callsign_id    = 0       # 12, so the next aircraft is 'AC12'
        self._step_count          = 0       # 317
        self._pending_spawns      = {}      # slot -> steps until the slot is refilled: {3: 5, 7: 2}
        self._sim_time_s          = 0.0     # simulated seconds since reset; the delay clock: 1585.0

        # Focus selection: the aircraft being advised, how long it has held the focus, and the
        # pair matrices the ranking and the observation both read.
        self.focus_cs             = None    # 'AC07'
        self.hold_steps           = 0       # 4
        self.urgency              = np.zeros((0, 0))
        self.t_los                = np.zeros((0, 0))

        # One Aircraft record per live aircraft; the keys ARE the active callsigns.
        self._aircraft = {}   # {'AC07': Aircraft(initial_hdg=84.0, commanded_hdg=129.0, ...)}

        # Per-step accumulators, both filled one simulated second at a time in _advance_simulation.
        self._los_seconds_this_step = 0     # 3, out of action_freq
        self._drift_sum_this_step   = 0.0   # summed heading_drift of the advised aircraft

        self._prev_los_pairs      = set()   # pairs in LoS a second ago: {('AC02', 'AC05')}
        self._prev_conflict_pairs = set()   # predicted pairs last step: {('AC02', 'AC05'), ('AC01', 'AC09')}
        self._ep_stats            = new_ep_stats()

    def _build_sector(self):
        n_ac     = CONFIG['n_aircraft'](self.scenario_rng)
        area_km2 = float(n_ac / CONFIG['rho'](self.scenario_rng))
        poly     = make_sector_polygon(area_km2, self.scenario_rng)

        self._polygon_shape = poly
        # Prepared once per episode: _no_turn_exit_nm runs tens of thousands of containment tests.
        shapely.prepare(poly)          # predicates below use the prepared index
        self._polygon_ready = poly
        self.polygon        = np.array(poly.exterior.coords[:-1])

        minx, miny, maxx, maxy = poly.bounds
        sector_diam_nm = math.hypot(maxx - minx, maxy - miny)

        # The episode ends on traffic handled, not on the clock: exits_per_episode aircraft per
        # slot must have left. _max_steps only guards against traffic that never leaves.
        self._max_exits = round(CONFIG['exits_per_episode'] * n_ac)
        crossing_time_s = sector_diam_nm / CONFIG['ac_speed'] * 3600
        self._max_steps = max(50, round(
            CONFIG['max_crossings'] * crossing_time_s / STEP_DURATION_S))

        self.n_aircraft = n_ac
        self.rho        = n_ac / area_km2      # recorded per episode by the evaluation
        self._slots     = [None] * n_ac

        bs.stack.stack('ASAS OFF')

    def _spawn_initial_traffic(self):
        for slot in range(self.n_aircraft):
            for _ in range(CONFIG['max_placement_tries']):
                route = self._random_start_in_sector()
                if route is not None and self._spawn_is_safe(route):
                    self._create_aircraft(slot, route)
                    break
            else:
                self._pending_spawns[slot] = 5   # could not place now: retry via the queue

    def _random_start_in_sector(self):
        minx, miny, maxx, maxy = self._polygon_shape.bounds
        for _ in range(CONFIG['max_placement_tries']):
            east  = self.scenario_rng.uniform(minx, maxx)
            north = self.scenario_rng.uniform(miny, maxy)
            if not shapely.contains_xy(self._polygon_ready, east, north):
                continue                       # the bounding box is not the sector

            heading = self.scenario_rng.uniform(0.0, 360.0)
            start   = np.array([east, north])
            leaves  = exit_point(self._polygon_shape, start, heading)
            if math.hypot(leaves[0] - east, leaves[1] - north) >= CONFIG['min_chord_nm']:
                return {'pos_nm': start, 'heading': heading}
        return None

    # -- Statistics --------------------------------------------------------------

    def _record_step_stats(self, action, reward):
        s = self._ep_stats
        s['reward'] += reward
        s['steps']  += 1
        s['actions'].append(action)
        s['los_seconds'] += self._los_seconds_this_step

        # Airborne time flown this step: the denominator that makes LoS counts comparable.
        s['flight_s'] += len(self._urgency_cs_list) * STEP_DURATION_S

        # Drift of every airborne aircraft from the heading it was assigned at spawn.
        for cs in self._urgency_cs_list:
            ac = self._aircraft.get(cs)
            if ac is not None:
                s['drift_deg_sum'] += abs(degto180(
                    self._hdg[self._row_of[cs]] - ac.initial_hdg))
                s['drift_samples'] += 1

        # Track length flown this step, leg by leg; the total is scored when the aircraft leaves.
        for cs in self._urgency_cs_list:
            self._advance_track(cs, self._pos[self._row_of[cs]])

    def _advance_track(self, cs, position):
        ac = self._aircraft.get(cs)
        if ac is None or ac.prev_pos_nm is None:
            return
        ac.flown_nm += math.hypot(position[0] - ac.prev_pos_nm[0],
                                  position[1] - ac.prev_pos_nm[1])
        ac.prev_pos_nm = (float(position[0]), float(position[1]))

    def _count_conflicts(self, flying):
        rows, cols = np.where(self.urgency > 0)
        pairs = {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}
        self._ep_stats['conflicts'] += len(pairs - self._prev_conflict_pairs)
        self._prev_conflict_pairs = pairs

    def _score_exit(self, cs, idx):
        self._ep_stats['exits'] += 1
        pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])

        # Track flown against the straight route it was given: spawn point to the no-turn exit.
        self._advance_track(cs, pos)
        ac = self._aircraft.get(cs)
        if ac is None:
            return

        spawn, exit_ref = ac.spawn_pos_nm, ac.no_turn_exit_nm
        self._ep_stats['flown_nm'] += ac.flown_nm
        self._ep_stats['route_nm'] += math.hypot(exit_ref[0] - spawn[0],
                                                 exit_ref[1] - spawn[1])

        # Exit deviation: boundary distance between where it left and where it would have left unturned.
        self._ep_stats['deviation_nm'] += math.hypot(pos[0] - exit_ref[0],
                                                     pos[1] - exit_ref[1])

        if abs(degto180(float(bs.traf.hdg[idx]) - ac.initial_hdg)) <= CONFIG['on_route_hdg_tol_deg']:
            self._ep_stats['on_route'] += 1
