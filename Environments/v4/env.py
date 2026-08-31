# v4 ATCO conflict-resolution environment: one focus aircraft per step, advisories acted on after a delay drawn per piece of advice, and an INITIAL HEADING per aircraft that every directional quantity is measured against.

import math
import random
from random import Random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale
from shapely.prepared import prep
from polygenerator import random_convex_polygon

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.stack.stackbase import Stack as _BsStack
from bluesky.tools.aero import nm as _M_PER_NM     # 1852.0
from bluesky.tools.geo import qdrpos
from bluesky.tools.misc import degto180

from .config import (CONFIG, KM_TO_NM, TRAINING_SCENARIOS, STEP_DURATION_S, OBS_DIM,
                     N_ACTIONS, N_NEIGHBOURS,
                     CRUISE_SPD_NMS, NMS_TO_KT, KT_PER_MACH, EMPTY_RANGE_NM, NO_CONFLICT_S,
                     TURN_DELTAS, SPEED_ACTIONS, HOLD_ACTION, RETURN_TO_ROUTE_ACTION,
                     ACT_COST)
from .delays import DELAY_MODES, ResponseDelay


# -- The per-aircraft record: what the CONTROLLER knows, not what BlueSky simulates ---


class Aircraft:

    def __init__(self, initial_hdg, no_turn_exit_nm, spawn_pos_nm, prev_pos_nm,
                 commanded_hdg, commanded_mach, steps_since_urgency):
        self.initial_hdg         = initial_hdg          # heading (deg) at spawn; NEVER changes: 84.0
        self.no_turn_exit_nm     = no_turn_exit_nm      # where it would leave if never turned: array([31.8, -19.4])
        self.spawn_pos_nm        = spawn_pos_nm         # where it entered (NM, east/north): (-38.2, 11.6)
        self.prev_pos_nm         = prev_pos_nm          # position at the previous step: (-12.0, 30.9)
        self.commanded_hdg       = commanded_hdg        # last EXECUTED heading instruction: 129.0
        self.commanded_mach      = commanded_mach       # last EXECUTED speed instruction: 0.82
        self.steps_since_urgency = steps_since_urgency  # steps since it last had a conflict: 12
        self.flown_nm            = 0.0                  # track length flown so far: 62.4
        self.pending_advisory    = None                 # issued, not yet executed: {'action': 5,


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # Empty intruder slot: unreachably far, stationary, no predicted LoS.
    _EMPTY_SLOT = [EMPTY_RANGE_NM, 0.0, 0.0, 0.0, NO_CONFLICT_S]

    def __init__(self, delay_mode=None, seed=None, delay_mean_s=None):
        super().__init__()
        # Per-instance rather than a CONFIG edit: SubprocVecEnv workers do not inherit CONFIG changes.
        self.delay_mode = delay_mode if delay_mode is not None else CONFIG['delay_mode']
        if self.delay_mode not in DELAY_MODES:
            raise ValueError(f'unknown delay_mode {self.delay_mode!r}; expected {DELAY_MODES}')

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
        """seed reseeds the scenario STREAM (training); options={'scenario_seed': n} flies exactly
        scenario n (validation). With neither, the next scenario from the stream."""
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
        self._focus_cs = self.select_focus_ship()
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
        self._focus_cs = self.select_focus_ship()
        reward         = self._compute_reward(acting_cs, action)
        self._record_step_stats(action, reward)

        truncated = self._step_count >= self._max_steps
        info = {'los_seconds': self._los_seconds_this_step,
                'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(episode_summary(self._ep_stats))
        return self._build_observation(), reward, False, truncated, info

    # -- Urgency and focus: the two decisions taken every step ------------------

    def construct_U(self):
        
        pos, vel = self._pos, self._vel
        n = len(pos)
        if n < 2:
            self._t_los = np.full((n, n), NO_CONFLICT_S)
            return np.zeros((n, n))

        sep, t_warn = CONFIG['sep_nm'], CONFIG['t_warn']
        dist_sq, _, _, t_los = pairwise(pos, vel)

        # Kept for the observation: capped at the horizon, since an unbounded t_los would
        # dominate the VecNormalize variance. Recomputing it per pair would repeat this work.
        self._t_los = np.clip(t_los, 0.0, NO_CONFLICT_S)

        urgency = np.where(t_los <= t_warn, (t_warn - np.clip(t_los, 0.0, t_warn)) / t_warn, 0.0)
        in_los  = dist_sq < sep * sep
        urgency = np.where(in_los, 1.0 + 9.0 * (1.0 - np.sqrt(dist_sq) / sep), urgency)

        np.fill_diagonal(urgency, 0.0)
        return urgency

    def select_focus_ship(self):

        flying = self._urgency_cs_list
        if not flying:
            return None

        worst = self._urgency_matrix.max(axis=1)
        for i, cs in enumerate(flying):
            ac = self._aircraft[cs]
            ac.steps_since_urgency = 0 if worst[i] > 0 else ac.steps_since_urgency + 1

        incumbent = self._focus_cs
        emergency = worst.max() >= CONFIG['focus_emergency_u']
        held      = (incumbent in self._row_of
                     and self._aircraft[incumbent].steps_since_urgency < CONFIG['focus_clear_steps'])

        if held and not emergency:
            best_cs = incumbent
        elif worst.max() > 0:
            best_cs = flying[int(np.argmax(worst))]
        else:
            best_cs = max(flying, key=self._heading_drift)

        if best_cs != incumbent:
            if incumbent is not None:           # close out the spell, for the diagnostic
                self._ep_stats['focus_spells']      += 1
                self._ep_stats['focus_spell_steps'] += self._focus_hold_steps + 1
            self._focus_hold_steps = 0
        else:
            self._focus_hold_steps += 1
        return best_cs

    # -- Focus selection helpers -------------------------------------------------

    def _heading_drift(self, cs):
        """1 - cos(current heading - initial heading), off the ACTUAL heading: 0 on route, 2 reversed."""
        row = self._row_of.get(cs)
        if row is None:
            return 0.0
        hdg_err = degto180(self._aircraft[cs].initial_hdg - self._hdg[row])
        return 1 - math.cos(math.radians(hdg_err))

    # -- Advisories: issued now, executed after the pilot's response delay -------

    def _issue_advisory(self, cs, action_idx):
        """Queue the advisory: CHANGED advice draws a fresh delay and discards what it replaced, while the SAME advice left standing keeps the assessment already running."""
        if action_idx == HOLD_ACTION or cs not in self._row_of:
            return   # hold is a true no-op: no advisory is transmitted at all

        advisory = self._build_advisory(cs, action_idx)
        ac       = self._aircraft[cs]
        pending  = ac.pending_advisory

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
        ac.pending_advisory = advisory

    def _build_advisory(self, cs, action_idx):
        """Action index -> what the pilot is asked to fly; turns accumulate on the last EXECUTED advisory."""
        advisory = {'action': action_idx}

        if action_idx in SPEED_ACTIONS:
            base = self._aircraft[cs].commanded_mach
            mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            advisory['target_mach'] = min(CONFIG['ac_mach_max'],
                                          max(CONFIG['ac_mach_min'], mach))

        elif action_idx in TURN_DELTAS:
            offset = self._current_offset(cs) + TURN_DELTAS[action_idx]
            advisory['target_hdg'] = (self._aircraft[cs].initial_hdg + offset) % 360

        elif action_idx == RETURN_TO_ROUTE_ACTION:
            advisory['target_hdg'] = self._aircraft[cs].initial_hdg % 360

        return advisory

    def _current_offset(self, cs):
        """Commanded offset from the initial heading, re-derived from the last EXECUTED advisory."""
        ac = self._aircraft[cs]
        return degto180(ac.commanded_hdg - ac.initial_hdg)

    def _same_advice(self, pending, advisory):
        """Do these two advisories ask for the same thing? Compared by what is COMMANDED, not by action index."""
        for key in ('target_hdg', 'target_mach'):
            if key in pending and key in advisory:
                return abs(pending[key] - advisory[key]) < 1e-9
        return False

    def _execute_due_advisories(self):
        """Fly every advisory whose execution time has come: the only path to BlueSky, called once per second."""
        for cs, ac in list(self._aircraft.items()):
            advisory = ac.pending_advisory
            if advisory is None or self._sim_time_s < advisory['execute_at_s']:
                continue
            ac.pending_advisory = None

            # The delay the pilot actually took, rounded up to the second the queue was checked.
            self._ep_stats['delay_sum_s']  += self._sim_time_s - advisory['issued_at_s']
            self._ep_stats['delay_served'] += 1

            idx = bs.traf.id2idx(cs)
            if idx < 0:
                continue                     # aircraft left before the pilot acted

            if 'target_mach' in advisory:
                ac.commanded_mach = advisory['target_mach']
                bs.stack.stack(f'SPD {cs} {advisory["target_mach"]:.3f}')
            else:
                # Absolute target fixed at issue time: the initial heading it offsets never moves.
                ac.commanded_hdg = advisory['target_hdg']
                bs.stack.stack(f'HDG {cs} {advisory["target_hdg"]:.1f}')

    # -- Observation: what the policy sees ---------------------------------------

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
        ac       = self._aircraft[cs]
        init_hdg = ac.initial_hdg
        cmd_hdg  = ac.commanded_hdg

        dpsi_act = math.radians(degto180(own_hdg - init_hdg))   # drift from assigned heading
        a_cmd    = math.radians(degto180(cmd_hdg - init_hdg))   # commanded offset
        v_own    = math.hypot(self._vel[row, 0], self._vel[row, 1]) * NMS_TO_KT
        v_cmd    = ac.commanded_mach * KT_PER_MACH

        # Time since the advisory now standing was issued; a replacement restarts it.
        pending = ac.pending_advisory
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
            psi   = math.radians(degto180(self._hdg[j] - own_hdg))
            v_int = math.hypot(self._vel[j, 0], self._vel[j, 1]) * NMS_TO_KT

            # 0 inside the protected zone, otherwise the capped prediction from construct_U.
            tlos = 0.0 if dist_nm < sep else float(self._t_los[own_row, j])

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

    # -- Reward ------------------------------------------------------------------

    def _compute_reward(self, acting_cs, action_idx):
        """Purely negative: separation loss, drift off the assigned heading, and the cost of the call."""
        r_los = -CONFIG['w_los'] if self._los_seconds_this_step else 0.0

        r_drift = 0.0
        if acting_cs and acting_cs in self._aircraft and acting_cs in self._row_of:
            r_drift = -CONFIG['w_drift'] * self._heading_drift(acting_cs)

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_drift + r_work)

    # -- Traffic picture: the state everything above reads -----------------------

    def _refresh_traffic_view(self):
        """Gather the whole traffic picture once per step; everything downstream reads these arrays."""
        flying, indices = self._airborne_indices()
        self._urgency_cs_list = flying
        self._row_of          = {cs: i for i, cs in enumerate(flying)}

        self._pos, self._vel = traffic_states(indices)
        self._hdg            = bs.traf.hdg[np.asarray(indices, dtype=int)]
        self._urgency_matrix = self.construct_U()
        self._count_conflicts(flying)

        # Whether each aircraft could turn back onto its route: the same pair geometry, but
        # flown on INITIAL headings. 1.0 = returning is blocked, by a live LoS or one within t_warn.
        speed     = np.hypot(self._vel[:, 0], self._vel[:, 1])
        init_hdg  = np.radians([self._aircraft[cs].initial_hdg for cs in flying])
        route_vel = np.stack([speed * np.sin(init_hdg), speed * np.cos(init_hdg)], axis=1)
        if len(self._pos) < 2:
            self._return_blocked = np.zeros(len(self._pos))
        else:
            route_dist_sq, _, _, route_t_los = pairwise(self._pos, route_vel)
            blocked = ((route_dist_sq < CONFIG['sep_nm'] ** 2)
                       | (route_t_los <= CONFIG['t_warn']))
            np.fill_diagonal(blocked, False)
            self._return_blocked = blocked.any(axis=1).astype(float)

    def _airborne_indices(self):
        """(callsigns, BlueSky indices) of everything airborne, in sorted callsign order, mapped in one pass."""
        index_of = {cs: i for i, cs in enumerate(bs.traf.id)}
        flying, indices = [], []
        for cs in sorted(self._aircraft):
            idx = index_of.get(cs, -1)
            if idx >= 0:
                flying.append(cs)
                indices.append(idx)
        return flying, indices

    # -- Simulation: advancing BlueSky and scanning separation -------------------

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

    # -- Spawning & exits: keeping the sector populated --------------------------

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

    def _plan_safe_entry(self):
        """Plan a route for a replacement aircraft that passes the spawn safety test."""
        for _ in range(CONFIG['max_placement_tries']):
            slot  = self.traffic_rng.randint(0, self.n_aircraft - 1)
            route = plan_entry_route(self._polygon_shape, slot, self.n_aircraft,
                                     self.traffic_rng)
            if self._spawn_is_safe(route):
                return route
        return None

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

        spawn_pos_nm = latlon_to_nm(CONFIG['center_ll'],
                                float(route['sp_ll'][0]), float(route['sp_ll'][1]))

        entry_nm = (float(spawn_pos_nm[0]), float(spawn_pos_nm[1]))
        self._aircraft[cs] = Aircraft(
            initial_hdg=float(route['heading']),
            no_turn_exit_nm=self._no_turn_exit_nm(
                (float(route['sp_ll'][0]), float(route['sp_ll'][1])), float(route['heading'])),
            spawn_pos_nm=entry_nm,
            prev_pos_nm=entry_nm,
            commanded_hdg=float(route['heading']),
            commanded_mach=CONFIG['ac_mach'],
            steps_since_urgency=CONFIG['focus_clear_steps'])
        self._slots[slot] = cs

    def _remove_exited_aircraft(self):
        """Retire aircraft that have left the sector, score them, and queue a respawn."""
        for cs in self._exited_callsigns():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                self._score_arrival(cs, idx)
                bs.traf.delete(idx)
            self._slots[slot] = None
            self._aircraft.pop(cs, None)   # one record, so nothing can be left behind
            if slot not in self._pending_spawns:
                self._pending_spawns[slot] = 1

    def _exited_callsigns(self):
        """Callsigns that have left the sector, sorted because each exit draws a respawn from traffic_rng."""
        callsigns = sorted(self._aircraft)
        index_of  = {cs: i for i, cs in enumerate(bs.traf.id)}
        airborne  = [(cs, index_of[cs]) for cs in callsigns if cs in index_of]

        inside_sector = {}
        if airborne:
            indices = np.array([idx for _, idx in airborne], dtype=int)
            positions = traffic_states(indices)[0]      # NM, the frame the polygon lives in
            inside_sector = {cs: self._polygon_ready.contains(Point(p[0], p[1]))
                             for (cs, _), p in zip(airborne, positions)}

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

    # -- Episode setup -----------------------------------------------------------

    def _new_episode_rngs(self, scenario_seed):
        """Fix this episode's scenario, then take three independent streams off it so a delay draw
        never shifts a scenario decision. scenario_seed None draws the next one from the stream."""
        self.episode_seed   = (self._seed_stream.randrange(TRAINING_SCENARIOS)
                               if scenario_seed is None else int(scenario_seed))

        # Three streams spun off the episode seed. Drawn through one master rather than
        # seeded episode_seed+1 / +2, which would make neighbouring scenarios share a stream.
        master              = Random(self.episode_seed)
        self.scenario_rng   = Random(master.getrandbits(64))
        self.traffic_rng    = Random(master.getrandbits(64))
        self.delay_rng      = np.random.default_rng(master.getrandbits(64))
        self.response_delay = ResponseDelay(
            self.delay_mode, self.delay_rng,
            **({'mean_s': self.delay_mean_s} if self.delay_mean_s is not None else {}))

    def _reset_episode_state(self):

        self._focus_cs            = None    # 'AC02'
        self._focus_hold_steps    = 0       # 4
        self._next_callsign_id    = 0       # 12, so the next aircraft is 'AC12'
        self._step_count          = 0       # 317
        self._pending_spawns      = {}      # slot -> steps until the slot is refilled: {3: 5, 7: 2}
        self._sim_time_s          = 0.0     # simulated seconds since reset; the delay clock: 1585.0

        # One Aircraft record per live aircraft; the keys ARE the active callsigns.
        self._aircraft = {}   # {'AC07': Aircraft(initial_hdg=84.0, commanded_hdg=129.0, ...)}

        self._prev_los_pairs      = set()   # pairs in LoS a second ago: {('AC02', 'AC05')}
        self._prev_conflict_pairs = set()   # predicted pairs last step: {('AC02', 'AC05'), ('AC01', 'AC09')}
        self._ep_stats            = new_ep_stats()

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

    # -- Statistics --------------------------------------------------------------

    def _record_step_stats(self, action, reward):
        """Accumulate the per-step counters that feed stats.episode_summary."""
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
        """Add the leg this aircraft just flew to its track length."""
        ac = self._aircraft.get(cs)
        if ac is None or ac.prev_pos_nm is None:
            return
        ac.flown_nm += math.hypot(position[0] - ac.prev_pos_nm[0],
                                  position[1] - ac.prev_pos_nm[1])
        ac.prev_pos_nm = (float(position[0]), float(position[1]))

    def _count_conflicts(self, flying):
        """Count pairs that ENTER the conflict horizon: urgency > 0 is a LoS predicted within t_warn."""
        rows, cols = np.where(self._urgency_matrix > 0)
        pairs = {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}
        self._ep_stats['conflicts'] += len(pairs - self._prev_conflict_pairs)
        self._prev_conflict_pairs = pairs

    def _score_arrival(self, cs, idx):
        """Score one exiting aircraft: on-route heading, how far off it left, and how far it flew. Metrics only."""
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

        if abs(degto180(float(bs.traf.hdg[idx]) - ac.initial_hdg)) <= CONFIG['arrival_hdg_tol_deg']:
            self._ep_stats['on_route'] += 1


# -- BlueSky: starting it once per process, and reading traffic into our frame ---


_started = False


class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0):
        pass


def start_bluesky():
    """Headless BlueSky, silenced, with the timestep set and the clock running free."""
    global _started
    if not _started:
        bs.init(mode='sim', detached=True)
        _started = True
    bs.scr = _ScreenDummy()
    bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")


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


# -- Geometry: the flat east/north NM frame, and pair separation maths -----------


NMS_PER_MS = 1.0 / _M_PER_NM   # m/s -> NM/s


def latlon_to_nm(center_ll, lat, lon):
    """(lat, lon) -> (east_nm, north_nm) relative to center_ll."""
    ref_lat, ref_lon = center_ll
    east_nm  = (lon - ref_lon) * 60.0 * math.cos(math.radians(ref_lat))
    north_nm = (lat - ref_lat) * 60.0
    return np.array([east_nm, north_nm])


def nm_to_latlon(center_ll, east_nm, north_nm):
    """(east_nm, north_nm) offsets -> (lat, lon)."""
    ref_lat, ref_lon = center_ll
    return (ref_lat + north_nm / 60.0,
            ref_lon + east_nm / (60.0 * math.cos(math.radians(ref_lat))))


def point_ahead(from_ll, heading_deg, distance_nm):
    """(lat, lon) reached by flying `distance_nm` from `from_ll` on a constant TRUE heading."""
    lat, lon = qdrpos(from_ll[0], from_ll[1], heading_deg, distance_nm)
    return float(lat), float(lon)


def heading_to_velocity(speed, heading_deg):
    """Speed + heading (deg) -> (east, north) velocity components."""
    h = math.radians(heading_deg)
    return speed * math.sin(h), speed * math.cos(h)


_TINY = 1e-12


def pairwise(pos, vel):
    """(dist_sq, range_rate, rel_spd_sq, t_los) for every pair; t_los is +inf when they never intrude."""
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


# -- Sector generation and entry routing, in the flat NM frame -------------------


def _circularity(polygon):
    """4*pi*area / perimeter^2 -- 1.0 for a circle, lower for elongated shapes."""
    return 4 * math.pi * polygon.area / polygon.length ** 2


def _random_convex_polygon(n_vertices, rng):
    """polygenerator's random_convex_polygon, driven by `rng` instead of the global random stream."""
    saved_state = random.getstate()
    random.seed(rng.randrange(2 ** 32))
    try:
        return random_convex_polygon(n_vertices)
    finally:
        random.setstate(saved_state)


def make_sector_polygon(area_km2, rng):
    """A random convex polygon of the requested area at the origin, retried until reasonably round."""
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw   = ShapelyPolygon(_random_convex_polygon(CONFIG['n_vertices'](rng), rng))
        scale = math.sqrt(target_nm2 / raw.area)
        scaled = shapely_scale(raw, xfact=scale, yfact=scale, origin='centroid')
        if _circularity(scaled) >= CONFIG['min_circularity']:
            break

    cx, cy = scaled.centroid.x, scaled.centroid.y
    return ShapelyPolygon([(x - cx, y - cy) for x, y in scaled.exterior.coords])


def plan_entry_route(polygon, sector, n_sectors, rng):
    """Plan one crossing: where it enters, its INITIAL HEADING, and the exit point it would reach unturned."""
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        t_spawn  = (sector + CONFIG['spawn_jitter'](rng)) / n_sectors
        t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter'](rng)) % 1.0
        spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
        ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)
        if math.hypot(ref_pt.x - spawn_pt.x, ref_pt.y - spawn_pt.y) >= min_chord:
            break

    initial_hdg = math.degrees(math.atan2(ref_pt.x - spawn_pt.x,
                                          ref_pt.y - spawn_pt.y)) % 360.0

    center = CONFIG['center_ll']
    return {
        'sp_ll':   nm_to_latlon(center, spawn_pt.x, spawn_pt.y),
        'heading': initial_hdg,
    }


# -- Episode metrics: the counters, and the figures derived from them ------------


def new_ep_stats():
    """The per-episode counters, all starting at zero. Comments show an end-of-episode example."""
    return {
        'reward': 0.0,         # -412.7   summed step reward
        'steps': 0,            # 1480     RL steps taken
        'actions': [],         # [3, 3, 5, 7, ...]  one action index per step
        'los_seconds': 0,      # 14       simulated seconds with at least one pair in LoS
        'los_events': 0,       # 3        distinct intrusions (entries, scanned every second)
        'conflicts': 0,        # 47       distinct predicted intrusions within t_warn (entries, per step)
        'flight_s': 0.0,       # 61200.0  airborne time flown by all aircraft
        'exits': 0,            # 21       aircraft that left having actually flown
        'on_route': 0,         # 18       ...of which left within the heading tolerance
        'deviation_nm': 0.0,   # 96.3     ...summed distance from the no-turn exit point
        'flown_nm': 0.0,       # 1742.5   ...summed track length actually flown
        'route_nm': 0.0,       # 1698.0   ...summed straight-line length of the route they were given
        # Drift summed over aircraft and steps; divided by aircraft-steps for a mean angle.
        'drift_deg_sum': 0.0,  # 20450.0  summed |drift| over aircraft-steps
        'drift_samples': 0,    # 17600    aircraft-steps that contributed
        'delay_sum_s': 0.0,    # 2790.0   summed response delay actually served
        'delay_served': 0,     # 93       advisories that reached execution
        'focus_spells': 0,     # 112      times an aircraft became the focus
        'focus_spell_steps': 0,# 1480     steps summed over those spells
        'discarded': 0,        # 19       advisories replaced before they could be flown
        'repeats': 0,          # 240      the same advice re-selected while it was still standing
        'turns': 0,            # 74       turn advisories actually transmitted
        'speeds': 0,           # 38       speed advisories actually transmitted
    }


def episode_summary(stats):
    """End-of-episode metrics, logged by the training callbacks."""
    s = stats
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
