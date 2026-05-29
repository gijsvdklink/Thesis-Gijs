"""
AirspaceEnv v3 — MARL, stochastic scenario, 4-wave dynamic spawning.

Scenario per episode:
  area_km2    ~ Uniform(1 000, 10 000)
  density     ~ Uniform(750, 1 500) km²/ac
  n_aircraft  = clamp(round(area / density), 2, MAX_AGENTS)

Observation per agent (26 floats — fixed regardless of n_aircraft):
  Ownship : x, y, vx, vy, cos(drift), sin(drift)
  4 nearest neighbours: rel_x, rel_y, rel_vx, rel_vy, dist  × N_NEIGHBOURS
  Missing neighbours padded with zeros.

Action space (Discrete 4):
  0 → −10°   1 → +10°   2 → hold   3 → snap back to original heading

Reward per agent per step:
  R = W_PROGRESS × (dist_prev − dist_now)
    + W_SEP      × min(0, dist_nearest / SEP_NM − 1)
    + INTRUSION_PEN  if LoS
    − W_ACTION       if action ∈ {0, 1}
  Exit toward destination    → +EXIT_CORRECT (+3)
  Exit away from destination → +EXIT_WRONG   (−5)

Episode ends when the 4-wave quota is exhausted and the airspace is empty,
or MAX_STEPS is reached.
"""

import math
import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(__file__)
_BS_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _BS_ROOT)

from improved_scenario import (
    DEFAULT_CONFIG, NM_TO_KM, KM_TO_NM,
    nm_to_latlon, latlon_to_nm,
    make_random_polygon, _place_aircraft, SpawnQueue,
)

# ── Scenario sampling ranges ──────────────────────────────────────────────────
AREA_KM2_MIN    = 1_000
AREA_KM2_MAX    = 10_000
DENSITY_MIN     = 750       # km² per aircraft
DENSITY_MAX     = 1_500
MAX_AGENTS      = 8         # hard cap on n_aircraft per episode

# ── Fixed env constants ───────────────────────────────────────────────────────
N_NEIGHBOURS   = 4          # nearest neighbours in observation (always 4)
N_WAVES        = 4
SPATIAL_SCALE  = 50.0       # NM
ACTION_FREQ    = 10         # BlueSky steps per RL step
MAX_STEPS      = 1_200      # generous ceiling for large airspaces / many waves

AC_SPEED  = DEFAULT_CONFIG['ac_speed']   # m/s
SIM_DT    = DEFAULT_CONFIG['sim_dt']     # s
ALTITUDE  = DEFAULT_CONFIG['altitude']   # FL
AC_TYPE   = DEFAULT_CONFIG['ac_type']
SEP_NM    = DEFAULT_CONFIG['sep_nm']
CENTER_LL = DEFAULT_CONFIG['center_ll']

_CFG = {
    **DEFAULT_CONFIG,
    'n_waves':              N_WAVES,
    'sep_nm':               SEP_NM,
    'buff_nm':              2.0,
    'dest_distance_factor': 3.0,
    'seed':                 None,
}

# ── Reward constants (same as v2 + W_ACTION) ──────────────────────────────────
_STEP_DIST_NM = AC_SPEED * SIM_DT * ACTION_FREQ / 1852.0
W_PROGRESS    = 1.4 / _STEP_DIST_NM    # ≈ 3.46 — straight flight → +1.4/step
W_SEP         = 3.0                    # nearest nbr: 0 at SEP_NM, −3 at 0 NM
INTRUSION_PEN = -10.0
EXIT_CORRECT  =  3.0
EXIT_WRONG    = -5.0
W_ACTION      =  0.1                   # penalty per heading-change action (0 or 1)

_DELTA = [-10, 10, 0, None]            # None = snap to original heading

_bs_initialized = False


def _wrap180(a):
    return (a + 180) % 360 - 180


from bluesky.simulation import ScreenIO

class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0):
        pass


# ── Environment ───────────────────────────────────────────────────────────────

class AirspaceEnvV3(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized

        obs_dim = 6 + N_NEIGHBOURS * 5             # 26 — fixed
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True

        bs.scr = _ScreenDummy()
        bs.stack.stack(f'DT {SIM_DT};FF')

        # Episode state — set in reset()
        self.n_aircraft  = 0
        self._slots      = []          # list of length n_aircraft: cs or None
        self._active     = set()
        self._dest_ll    = {}
        self._orig_hdg   = {}
        self._queue      = None
        self._next_id    = 0
        self._step_count = 0
        self.polygon     = None

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bs.traf.reset()

        # Sample stochastic scenario
        area_km2   = random.uniform(AREA_KM2_MIN, AREA_KM2_MAX)
        density    = random.uniform(DENSITY_MIN,  DENSITY_MAX)
        n_ac       = int(np.clip(round(area_km2 / density), 2, MAX_AGENTS))

        shapely_poly = make_random_polygon(area_km2 * KM_TO_NM**2, _CFG)
        while True:
            aircraft_list = _place_aircraft(shapely_poly, n_ac, _CFG)
            if aircraft_list is not None:
                break

        self.n_aircraft  = n_ac
        self.polygon     = np.array(shapely_poly.exterior.coords[:-1])
        self._slots      = [None] * n_ac
        self._active     = set()
        self._dest_ll    = {}
        self._orig_hdg   = {}
        self._next_id    = n_ac
        self._step_count = 0

        # Register airspace with BlueSky
        flat = []
        for v in self.polygon:
            ll = nm_to_latlon(CENTER_LL, v[0], v[1])
            flat += [float(ll[0]), float(ll[1])]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat)

        # Spawn initial batch
        for i, ac in enumerate(aircraft_list):
            cs = f'AC{i:02d}'
            bs.traf.cre(cs, actype=AC_TYPE,
                        aclat=float(ac['spawn_ll'][0]), aclon=float(ac['spawn_ll'][1]),
                        achdg=float(ac['heading']), acspd=AC_SPEED, acalt=ALTITUDE)
            self._dest_ll[cs]  = ac['dest_ll']
            self._orig_hdg[cs] = float(ac['heading'])
            self._slots[i]     = cs
            self._active.add(cs)

        self._queue = SpawnQueue(shapely_poly, n_ac, _CFG)
        bs.stack.stack('ASAS ON')

        return self._get_all_obs(), {}

    def step(self, actions):
        # actions: array of length n_aircraft
        for slot_i, cs in enumerate(self._slots):
            if cs is not None and cs in self._active:
                self._apply_action(cs, int(actions[slot_i]))

        # Snapshot distances before simulation
        prev_dists = {}
        for cs in self._active:
            idx = bs.traf.id2idx(cs)
            if idx >= 0:
                _, d = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[idx], bs.traf.lon[idx],
                    float(self._dest_ll[cs][0]), float(self._dest_ll[cs][1]),
                )
                prev_dists[cs] = d

        for _ in range(ACTION_FREQ):
            bs.sim.step()

        self._step_count += 1

        # Per-slot rewards
        rewards = np.zeros(self.n_aircraft, dtype=np.float32)
        for slot_i, cs in enumerate(self._slots):
            if cs is not None and cs in self._active:
                rewards[slot_i] = self._agent_reward(cs, prev_dists.get(cs, 0.0))
                if int(actions[slot_i]) in (0, 1):
                    rewards[slot_i] -= W_ACTION

        # Exits — reward before deleting
        exited = self._get_exited()
        for cs in exited:
            slot_i = self._slots.index(cs)
            rewards[slot_i] += self._exit_reward(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active.discard(cs)
            self._slots[slot_i] = None
            self._queue.notify_exit(self._step_count)

        # Spawn replacements into freed slots
        active_lls = [
            (bs.traf.lat[bs.traf.id2idx(cs)], bs.traf.lon[bs.traf.id2idx(cs)])
            for cs in self._active if bs.traf.id2idx(cs) >= 0
        ]
        free_slots = [i for i, cs in enumerate(self._slots) if cs is None]
        for ac, slot_i in zip(self._queue.try_flush(active_lls, self._step_count), free_slots):
            cs = f'AC{self._next_id:02d}'
            self._next_id += 1
            bs.traf.cre(cs, actype=AC_TYPE,
                        aclat=float(ac['spawn_ll'][0]), aclon=float(ac['spawn_ll'][1]),
                        achdg=float(ac['heading']), acspd=AC_SPEED, acalt=ALTITUDE)
            self._dest_ll[cs]  = ac['dest_ll']
            self._orig_hdg[cs] = float(ac['heading'])
            self._slots[slot_i] = cs
            self._active.add(cs)

        terminated = self._queue.exhausted and len(self._active) == 0
        truncated  = self._step_count >= MAX_STEPS

        info = {
            'los_pairs':  list(bs.traf.cd.lospairs),
            'active':     list(self._active),
            'n_aircraft': self.n_aircraft,
        }
        return self._get_all_obs(), rewards, terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        delta   = _DELTA[action_idx]
        new_hdg = self._orig_hdg[cs] if delta is None else bs.traf.hdg[idx] + delta
        bs.stack.stack(f'HDG {cs} {new_hdg:.1f}')

    def _agent_obs(self, cs):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        own_lat = bs.traf.lat[own_idx]
        own_lon = bs.traf.lon[own_idx]
        own_hdg = bs.traf.hdg[own_idx]
        own_gs  = bs.traf.gs[own_idx]
        own_nm  = latlon_to_nm(CENTER_LL, own_lat, own_lon)
        hdg_rad = math.radians(own_hdg)

        own_x  = own_nm[0] / SPATIAL_SCALE
        own_y  = own_nm[1] / SPATIAL_SCALE
        own_vx = own_gs * math.sin(hdg_rad) / AC_SPEED
        own_vy = own_gs * math.cos(hdg_rad) / AC_SPEED

        dest    = self._dest_ll[cs]
        wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
            own_lat, own_lon, float(dest[0]), float(dest[1])
        )
        drift = _wrap180(own_hdg - wpt_qdr)

        obs = [own_x, own_y, own_vx, own_vy,
               math.cos(math.radians(drift)), math.sin(math.radians(drift))]

        # 4 nearest neighbours — sorted by distance, zero-padded if fewer exist
        neighbours = []
        for other_cs in self._active:
            if other_cs == cs:
                continue
            int_idx = bs.traf.id2idx(other_cs)
            if int_idx < 0:
                continue
            int_lat = bs.traf.lat[int_idx]
            int_lon = bs.traf.lon[int_idx]
            int_hdg = math.radians(bs.traf.hdg[int_idx])
            int_gs  = bs.traf.gs[int_idx]
            int_nm  = latlon_to_nm(CENTER_LL, int_lat, int_lon)
            _, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)
            neighbours.append((
                dist,
                (int_nm[0] - own_nm[0]) / SPATIAL_SCALE,
                (int_nm[1] - own_nm[1]) / SPATIAL_SCALE,
                (int_gs * math.sin(int_hdg) - own_gs * math.sin(hdg_rad)) / AC_SPEED,
                (int_gs * math.cos(int_hdg) - own_gs * math.cos(hdg_rad)) / AC_SPEED,
                dist / SPATIAL_SCALE,
            ))

        neighbours.sort(key=lambda x: x[0])

        for k in range(N_NEIGHBOURS):
            if k < len(neighbours):
                _, rel_x, rel_y, rel_vx, rel_vy, dist_n = neighbours[k]
            else:
                rel_x = rel_y = rel_vx = rel_vy = 0.0
                dist_n = 1.0             # normalised max distance
            obs += [rel_x, rel_y, rel_vx, rel_vy, dist_n]

        return np.array(obs, dtype=np.float32)

    def _get_all_obs(self):
        obs = []
        for cs in self._slots:
            if cs is not None:
                obs.append(self._agent_obs(cs))
            else:
                obs.append(np.zeros(self.observation_space.shape, dtype=np.float32))
        return np.stack(obs)   # shape (n_aircraft, 26)

    def _agent_reward(self, cs, dist_prev):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return 0.0

        own_lat = bs.traf.lat[own_idx]
        own_lon = bs.traf.lon[own_idx]

        _, dist_now = bs.tools.geo.kwikqdrdist(
            own_lat, own_lon,
            float(self._dest_ll[cs][0]), float(self._dest_ll[cs][1]),
        )
        r_progress = W_PROGRESS * (dist_prev - dist_now)

        min_dist = float('inf')
        for other_cs in self._active:
            if other_cs == cs:
                continue
            other_idx = bs.traf.id2idx(other_cs)
            if other_idx >= 0:
                _, d = bs.tools.geo.kwikqdrdist(
                    own_lat, own_lon,
                    bs.traf.lat[other_idx], bs.traf.lon[other_idx],
                )
                if d < min_dist:
                    min_dist = d
        r_sep = W_SEP * min(0.0, min_dist / SEP_NM - 1.0) if min_dist < float('inf') else 0.0

        intrusion = any(cs in pair for pair in bs.traf.cd.lospairs)

        return float(r_progress + r_sep + (INTRUSION_PEN if intrusion else 0.0))

    def _get_exited(self):
        exited = []
        for cs in list(self._active):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs); continue
            inside = bs.tools.areafilter.checkInside(
                'AIRSPACE',
                np.array([bs.traf.lat[idx]]),
                np.array([bs.traf.lon[idx]]),
                np.array([ALTITUDE * 30.48]),
            )
            if not inside[0]:
                exited.append(cs)
        return exited

    def _exit_reward(self, cs):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        dest    = self._dest_ll[cs]
        wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            float(dest[0]), float(dest[1]),
        )
        c = math.cos(math.radians(_wrap180(bs.traf.hdg[idx] - wpt_qdr)))
        return float(EXIT_CORRECT * c if c >= 0 else EXIT_WRONG * c)


# ── Single-agent wrapper for SB3 ─────────────────────────────────────────────

class MARLSingleAgentWrapper(gym.Env):
    """
    Presents the N-agent env as a single-agent stream for SB3.
    n_aircraft varies per episode — the wrapper reads it from the env after reset.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        self._env = AirspaceEnvV3()
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

        self._n_agents        = 2          # updated at reset
        self._ep_rewards      = np.zeros(MAX_AGENTS)
        self._ep_los          = 0
        self._ep_steps        = 0
        self._ep_actions      = []
        self._obs_stack       = None
        self._agent_idx       = 0
        self._pending_actions = []

    def reset(self, seed=None, options=None):
        obs_stack, info        = self._env.reset(seed=seed, options=options)
        self._n_agents         = self._env.n_aircraft
        self._obs_stack        = obs_stack
        self._agent_idx        = 0
        self._pending_actions  = []
        self._ep_rewards       = np.zeros(self._n_agents)
        self._ep_los           = 0
        self._ep_steps         = 0
        self._ep_actions       = []
        return obs_stack[0], info

    def step(self, action):
        self._pending_actions.append(int(action))
        self._ep_actions.append(int(action))

        if len(self._pending_actions) < self._n_agents:
            self._agent_idx += 1
            return self._obs_stack[self._agent_idx], 0.0, False, False, {}

        actions = np.array(self._pending_actions)
        self._pending_actions = []
        self._agent_idx       = 0

        obs_stack, rewards, terminated, truncated, info = self._env.step(actions)
        self._obs_stack = obs_stack

        self._ep_rewards += rewards
        self._ep_los     += int(len(info['los_pairs']) > 0)
        self._ep_steps   += 1

        extra_info = {}
        if terminated or truncated:
            extra_info = {
                'episode_rewards':     self._ep_rewards.tolist(),
                'mean_episode_reward': float(self._ep_rewards.mean()),
                'ep_los_steps':        self._ep_los,
                'ep_length':           self._ep_steps,
                'n_aircraft':          self._n_agents,
                'action_distribution': np.bincount(self._ep_actions, minlength=4).tolist(),
            }

        return obs_stack[0], float(rewards.mean()), terminated, truncated, extra_info
