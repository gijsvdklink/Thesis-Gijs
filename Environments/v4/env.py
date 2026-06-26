"""
v4 -- ATCO conflict-resolution environment (multi-aircraft, ACAS Xu observation).

One aircraft (the focus / ownship) is controlled per step; focus follows the worst
conflict, falling back to the most drift-x-clearance aircraft when the sector is clear.

Observation (26 floats), ego-centric from the focus aircraft:
  ownship (6): dpsi (actual heading error to route, rad), v_own (speed / cruise),
               a_cmd (commanded heading error, rad), v_cmd (commanded speed / nominal),
               retn_conf (1 = returning to route is blocked), in_conf (1 = in conflict)
  per intruder (4 x 5): rho (distance / 45 NM), theta (bearing, rad), psi (rel heading,
               rad), v_int (speed / cruise), tau (time-to-LoS / t_warn)

Action (Discrete 10): turn -+60/45/30 (stack on commanded heading), hold (no-op),
  fly-direct (return to route, persistent), speed up/down (step commanded Mach).

Reward (purely negative): -w_los*1[LoS] - w_conflict*conflict_score
  - w_drift*(1-cos(dpsi))/2 - w_work*ACT_COST[action].
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

from .config import (CONFIG, OBS_DIM, N_ACTIONS, N_NEIGHBOURS, WARN_DIST_NM,
                     CRUISE_SPD_NMS, TURN_DELTAS, SPEED_ACTIONS, ACT_COST)
from .geometry import (latlon_to_nm, nm_to_latlon, wrap_to_180, aircraft_speed_nms,
                       aircraft_position_nm, aircraft_state, heading_to_velocity)
from .conflict import (conflict_score, return_blocked, any_los, pair_urgency, time_to_los)
from .sector import make_polygon, place_aircraft

_bs_initialized = False


class _ScreenDummy(ScreenIO):
    """Silences BlueSky's screen output (we run headless)."""
    def echo(self, text='', flags=0):
        pass


class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}

    # ACAS Xu empty-intruder-slot sentinel: rho=1 (far), tau=1 (no imminent LoS)
    _EMPTY_SLOT = [1.0, 0.0, 0.0, 0.0, 1.0]

    def __init__(self, dummy_retn_conf=False, dummy_in_conf=False):
        super().__init__()
        global _bs_initialized
        # Feature-ablation flags: when set, the corresponding ownship observation feature
        # is replaced by a constant (0.0) so it carries no information. The observation
        # dimension is unchanged, keeping the policy architecture identical across runs.
        self.dummy_retn_conf = bool(dummy_retn_conf)   # ablate retn_conf ("safe to return")
        self.dummy_in_conf   = bool(dummy_in_conf)     # ablate in_conf   ("am I in conflict")
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
        self._route_hdg           = {}   # fixed route bearing (deg) per callsign
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
        self._ep_stats            = {'reward': 0.0, 'steps': 0, 'los': 0,
                                     'actions': [], 'exits': 0, 'arrivals': 0}

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
        acting_cs = self._focus_cs

        if acting_cs:
            self._apply_action(acting_cs, action)
        self._update_direct_headings()   # re-aim fly-direct aircraft before propagating

        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
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

        n_los_pairs = int((self._urgency_matrix > 1.0).sum()) // 2 if self._urgency_matrix.size else 0
        truncated   = self._step_count >= self._max_steps
        info = {'los_pairs': n_los_pairs, 'focus_cs': self._focus_cs, 'n_aircraft': self.n_aircraft}
        if truncated:
            info.update(self._episode_summary())
        return self._get_observation(), reward, False, truncated, info

    def _episode_summary(self):
        """End-of-episode metrics (logged by the training callbacks)."""
        s = self._ep_stats
        n_steps = max(s['steps'], 1)
        return {
            'mean_episode_reward': s['reward'] / n_steps,
            'ep_los_steps':        s['los'],
            'ep_length':           s['steps'],
            'ep_exits':            s['exits'],
            'ep_arrival_rate':     s['arrivals'] / max(s['exits'], 1),
            'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),
        }

    # -- Focus selection -------------------------------------------------------

    def _select_focus_aircraft(self):
        """Rebuild the urgency matrix and pick the focus aircraft.

        Highest worst-pair urgency wins (tiebreak: total urgency burden), with hysteresis
        that keeps the current focus while it is still active. An emergency (urgency >=
        focus_emergency_u) overrides hysteresis; a conflict-free sector falls back to the
        best drift-x-clearance drifter.
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
        """Pick the drifted aircraft best placed to be sent back to route.

        Score = drift x clearance, where clearance ramps 0->1 with nearest-neighbour
        distance up to return_clear_nm: drifters in open airspace are prioritised. A
        hysteresis margin keeps the current focus unless another scores clearly higher.
        """
        positions = {cs: aircraft_position_nm(bs.traf.id2idx(cs))
                     for cs in flying if bs.traf.id2idx(cs) >= 0}
        clear_nm = CONFIG['return_clear_nm']

        best_cs, best_score, focus_score = None, -1.0, 0.0
        for cs in sorted(flying):
            idx = bs.traf.id2idx(cs)
            if idx < 0 or cs not in self._route_hdg:
                continue
            hdg_err = wrap_to_180(self._route_hdg[cs] - self._commanded_heading.get(cs, bs.traf.hdg[idx]))
            drift   = (1 - math.cos(math.radians(hdg_err))) / 2
            own     = positions[cs]
            nearest = min((float(np.hypot(*(positions[o] - own))) for o in positions if o != cs),
                          default=clear_nm)
            score   = drift * min(1.0, nearest / clear_nm)

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
        r_los      = -CONFIG['w_los'] if self._los_this_step else 0.0
        r_conflict = -CONFIG['w_conflict'] * conflict_score(acting_cs, self._active_callsigns)

        r_drift = 0.0
        if acting_cs and acting_cs in self._route_hdg:
            idx = bs.traf.id2idx(acting_cs)
            if idx >= 0:
                cmd_hdg = self._commanded_heading.get(acting_cs, bs.traf.hdg[idx])
                hdg_err = wrap_to_180(self._route_hdg[acting_cs] - cmd_hdg)
                r_drift = -CONFIG['w_drift'] * (1.0 - math.cos(math.radians(hdg_err))) / 2.0

        r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
        return float(r_los + r_conflict + r_drift + r_work)

    # -- Actions ---------------------------------------------------------------

    def _apply_action(self, cs, action_idx):
        if action_idx == 3:
            return   # hold: true no-op, no instruction reaches the simulator

        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return

        # Speed: step the commanded Mach within the ATC envelope; leaves heading untouched.
        if action_idx in SPEED_ACTIONS:
            mach = self._commanded_mach.get(cs, CONFIG['ac_mach']) + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
            mach = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))
            self._commanded_mach[cs] = mach
            bs.stack.stack(f'SPD {cs} {mach:.3f}')
            return

        # Heading: a turn stacks onto the commanded heading and cancels fly-direct;
        # fly-direct (7) locks onto the fixed route heading (held by _update_direct_headings).
        if action_idx in TURN_DELTAS:
            current = self._commanded_heading.get(cs, bs.traf.hdg[idx])
            self._commanded_heading[cs] = (current + TURN_DELTAS[action_idx]) % 360
            self._direct_mode[cs] = False
        elif action_idx == 7:
            self._direct_mode[cs] = True
            self._commanded_heading[cs] = self._route_hdg[cs]

        bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    def _update_direct_headings(self):
        """Re-issue the fixed route heading for every fly-direct aircraft each step."""
        for cs, on in self._direct_mode.items():
            idx = bs.traf.id2idx(cs)
            if on and idx >= 0 and cs in self._route_hdg:
                self._commanded_heading[cs] = self._route_hdg[cs]
                bs.stack.stack(f'HDG {cs} {self._commanded_heading[cs]:.1f}')

    # -- Observation -----------------------------------------------------------

    def _get_observation(self):
        """ACAS Xu states for the focus aircraft against its nearest/most-urgent intruders."""
        cs = self._focus_cs
        if cs is None or bs.traf.id2idx(cs) < 0:
            # no controllable aircraft: on-route, nominal speed, conflict-free
            self._last_intruder_cs = [None] * N_NEIGHBOURS
            return np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
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
        v_cmd     = self._commanded_mach.get(cs, CONFIG['ac_mach']) / CONFIG['ac_mach']

        retn_conf = 0.0 if self.dummy_retn_conf else \
            return_blocked(cs, self._active_callsigns, self._route_hdg)
        in_conf = 0.0 if self.dummy_in_conf else \
            (1.0 if conflict_score(cs, self._active_callsigns) > 0.0 else 0.0)

        obs = [dpsi_act,
               own_spd / CRUISE_SPD_NMS,
               a_cmd,
               v_cmd,
               retn_conf,      # retn_conf ("safe to return"); dummied to 0.0 when ablated
               in_conf]        # in_conf   ("am I in conflict"); dummied to 0.0 when ablated

        # pre-computed urgency row for this aircraft (intruder prioritisation)
        urgency_row = None
        if cs in self._urgency_cs_list:
            row = self._urgency_cs_list.index(cs)
            if row < self._urgency_matrix.shape[0]:
                urgency_row = self._urgency_matrix[row]

        sep, t_warn = CONFIG['sep_nm'], CONFIG['t_warn']
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

            rho   = min(1.0, dist_nm / WARN_DIST_NM)
            theta = math.atan2(ego_lat, ego_fwd)
            psi   = math.radians(wrap_to_180(int_hdg - own_hdg))
            v_int = int_spd / CRUISE_SPD_NMS

            if dist_nm < sep:
                tau = 0.0
            else:
                int_ve, int_vn = heading_to_velocity(int_spd, int_hdg)
                dv_east, dv_north = int_ve - own_ve, int_vn - own_vn
                rel_spd_sq = dv_east ** 2 + dv_north ** 2
                range_rate = d_east * dv_east + d_north * dv_north
                t_los = time_to_los(dist_nm ** 2, range_rate, rel_spd_sq, sep)
                tau = min(max(0.0, t_los) / t_warn, 1.0) if t_los is not None else 1.0

            pair_u = 0.0
            if urgency_row is not None and other in self._urgency_cs_list:
                col = self._urgency_cs_list.index(other)
                if col < len(urgency_row):
                    pair_u = float(urgency_row[col])

            intruders.append((pair_u, dist_nm, [rho, theta, psi, v_int, tau], other))

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
        self._route_hdg[cs]           = float(ac['heading'])
        self._commanded_heading[cs]   = float(ac['heading'])
        self._commanded_mach[cs]      = CONFIG['ac_mach']
        self._direct_mode[cs]         = False
        self._steps_since_urgency[cs] = CONFIG['focus_clear_steps']
        self._slots[slot]             = cs
        self._active_callsigns.add(cs)

    def _process_exits(self):
        """Remove aircraft that have left the sector, scoring on-target arrivals, and
        queue their slot for a respawn."""
        for cs in self._find_exited():
            slot = self._slots.index(cs)
            idx  = bs.traf.id2idx(cs)
            if idx >= 0:
                ref_ll = self._ref_ll.get(cs)
                if ref_ll is not None:
                    dist_nm = geo.kwikdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]),
                                           float(ref_ll[0]), float(ref_ll[1]))
                    self._ep_stats['exits'] += 1
                    if dist_nm <= CONFIG['arrival_tol_nm']:
                        self._ep_stats['arrivals'] += 1
                bs.traf.delete(idx)
            self._active_callsigns.discard(cs)
            self._slots[slot] = None
            for d in (self._destination_ll, self._ref_ll, self._route_hdg, self._commanded_heading,
                      self._commanded_mach, self._direct_mode, self._steps_since_urgency):
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
