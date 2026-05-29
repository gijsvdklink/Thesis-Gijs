"""
AirspaceEnv v2 — MARL, discrete heading changes.

All NUM_AIRCRAFT are controlled by the same shared policy (CTDE).

Action space  (Discrete 4):
  0 → -10°   1 → +10°   2 → 0° (hold)   3 → follow original heading

Observation per agent  (16 floats):
  Ownship   : x, y, vx, vy, cos(drift), sin(drift)
  Intruder i: rel_x, rel_y, rel_vx, rel_vy, distance      × N_NEIGHBOURS

Reward per agent per step:
  R = W_PROGRESS × (dist_prev − dist_now)          ← progress toward destination
    + W_SEP      × min(0, dist_nearest/SEP_NM − 1) ← smooth separation penalty
    + INTRUSION_PEN  if LoS                         ← hard cliff at violation
  Exit toward destination  → +EXIT_CORRECT (+3)
  Exit away from destination → +EXIT_WRONG  (−5)

  Straight flight → ~+1.4/step
  dist_nearest = SEP_NM (5 NM) → sep term = 0
  dist_nearest = 2.5 NM        → sep term = −1.5/step
  dist_nearest = 0 NM          → sep term = −3.0/step  + LoS −10

Episode ends when all agents have exited or MAX_STEPS is reached.
"""

import math
import os
import sys
import importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import bluesky as bs

sys.path.insert(0, os.path.dirname(__file__))
sg = importlib.import_module('scenario_generation')

# ── Environment constants ─────────────────────────────────────────────────────
N_AGENTS       = sg.NUM_AIRCRAFT
N_NEIGHBOURS   = N_AGENTS - 1

SPATIAL_SCALE  = 50.0          # NM — normalises x/y and distances
ACTION_FREQ    = 10             # BlueSky steps between RL actions
MAX_STEPS      = 600

INTRUSION_PEN  = -10.0         # hard penalty per step while in LoS
EXIT_CORRECT   =  3.0          # reward for exiting toward destination
EXIT_WRONG     = -5.0          # penalty for exiting away from destination

# Progress reward: straight flight → +1.4/step, moving away → −1.4/step
_STEP_DIST_NM  = sg.AC_SPEED * sg.SIM_DT * ACTION_FREQ / 1852.0
W_PROGRESS     = 1.4 / _STEP_DIST_NM                # ≈ 3.46

# Smooth separation penalty: 0 when safe, negative inside SEP_NM bubble
W_SEP          = 3.0                                 # at 0 NM → −3.0/step

# Heading deltas; None = snap back to original heading
_DELTA = [-10, 10, 0, None]   # None = snap back to original heading

_bs_initialized = False


def _wrap180(a):
    return (a + 180) % 360 - 180


class AirspaceEnvV2(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        global _bs_initialized

        obs_dim = 6 + N_NEIGHBOURS * 5
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True

        bs.scr = sg.ScreenDummy()
        bs.stack.stack(f'DT {sg.SIM_DT};FF')

        self._dest_ll    = {}
        self._orig_hdg   = {}
        self._active     = []
        self._step_count = 0
        self.polygon       = None
        self.aircraft_list = None

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bs.traf.reset()

        polygon, aircraft_list, _, _ = sg.build_scenario()
        self.polygon       = polygon
        self.aircraft_list = aircraft_list
        self._step_count   = 0
        self._dest_ll      = {}
        self._orig_hdg     = {}

        flat = []
        for v in polygon:
            ll = sg.nm_to_latlon(sg.CENTER_LL, v[0], v[1])
            flat += [float(ll[0]), float(ll[1])]
        bs.tools.areafilter.defineArea('AIRSPACE', 'POLY', flat)

        for i, ac in enumerate(aircraft_list):
            cs = f'AC{i:02d}'
            bs.traf.cre(
                cs, actype=sg.AC_TYPE,
                aclat=float(ac['spawn_ll'][0]), aclon=float(ac['spawn_ll'][1]),
                achdg=float(ac['heading']),     acspd=sg.AC_SPEED, acalt=sg.ALTITUDE,
            )
            self._dest_ll[cs]  = ac['dest_ll']
            self._orig_hdg[cs] = float(ac['heading'])

        self._active = [f'AC{i:02d}' for i in range(N_AGENTS)]
        bs.stack.stack('ASAS ON')

        return self._get_all_obs(), {}

    def step(self, actions):
        for i in range(N_AGENTS):
            cs = f'AC{i:02d}'
            if cs in self._active:
                self._apply_action(cs, int(actions[i]))

        # Record distances BEFORE sim steps to compute progress
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

        rewards = np.array(
            [self._agent_reward(f'AC{i:02d}', prev_dists.get(f'AC{i:02d}', 0.0))
             if f'AC{i:02d}' in self._active else 0.0
             for i in range(N_AGENTS)],
            dtype=np.float32,
        )

        # Compute exit rewards BEFORE deleting so BlueSky still has position/heading
        exited = self._get_exited()
        for cs in exited:
            i = int(cs[2:])
            rewards[i] += self._exit_reward(cs)
            bs.traf.delete(bs.traf.id2idx(cs))
            self._active.remove(cs)

        terminated = len(self._active) == 0
        truncated  = self._step_count >= MAX_STEPS

        info = {
            'los_pairs': list(bs.traf.cd.lospairs),
            'active':    list(self._active),
        }
        return self._get_all_obs(), rewards, terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_action(self, cs, action_idx):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return
        delta = _DELTA[action_idx]
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

        # Ownship position in NM relative to reference point
        own_nm  = sg.latlon_to_nm(sg.CENTER_LL, own_lat, own_lon)
        own_x   = own_nm[0] / SPATIAL_SCALE
        own_y   = own_nm[1] / SPATIAL_SCALE

        # Ownship velocity components (normalised)
        hdg_rad = math.radians(own_hdg)
        own_vx  = own_gs * math.sin(hdg_rad) / sg.AC_SPEED
        own_vy  = own_gs * math.cos(hdg_rad) / sg.AC_SPEED

        # Track deviation to destination
        dest = self._dest_ll[cs]
        wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
            own_lat, own_lon, float(dest[0]), float(dest[1])
        )
        drift = _wrap180(own_hdg - wpt_qdr)

        obs = [
            own_x,
            own_y,
            own_vx,
            own_vy,
            math.cos(math.radians(drift)),
            math.sin(math.radians(drift)),
        ]

        # Intruder features — sorted nearest first
        neighbours = []
        for other_cs in (f'AC{j:02d}' for j in range(N_AGENTS)):
            if other_cs == cs:
                continue
            int_idx = bs.traf.id2idx(other_cs)
            if int_idx >= 0:
                int_lat = bs.traf.lat[int_idx]
                int_lon = bs.traf.lon[int_idx]
                int_hdg = math.radians(bs.traf.hdg[int_idx])
                int_gs  = bs.traf.gs[int_idx]

                int_nm  = sg.latlon_to_nm(sg.CENTER_LL, int_lat, int_lon)
                _, dist = bs.tools.geo.kwikqdrdist(own_lat, own_lon, int_lat, int_lon)

                rel_x  = (int_nm[0] - own_nm[0]) / SPATIAL_SCALE
                rel_y  = (int_nm[1] - own_nm[1]) / SPATIAL_SCALE
                rel_vx = (int_gs * math.sin(int_hdg) - own_gs * math.sin(hdg_rad)) / sg.AC_SPEED
                rel_vy = (int_gs * math.cos(int_hdg) - own_gs * math.cos(hdg_rad)) / sg.AC_SPEED
                neighbours.append((dist, rel_x, rel_y, rel_vx, rel_vy, dist / SPATIAL_SCALE))
            else:
                neighbours.append((SPATIAL_SCALE, 0.0, 0.0, 0.0, 0.0, 1.0))

        neighbours.sort(key=lambda x: x[0])
        for _, rel_x, rel_y, rel_vx, rel_vy, dist_n in neighbours[:N_NEIGHBOURS]:
            obs += [rel_x, rel_y, rel_vx, rel_vy, dist_n]

        return np.array(obs, dtype=np.float32)

    def _get_all_obs(self):
        return np.stack([self._agent_obs(f'AC{i:02d}') for i in range(N_AGENTS)])

    def _agent_reward(self, cs, dist_prev):
        own_idx = bs.traf.id2idx(cs)
        if own_idx < 0:
            return 0.0

        own_lat = bs.traf.lat[own_idx]
        own_lon = bs.traf.lon[own_idx]
        dest    = self._dest_ll[cs]

        # Progress toward destination
        _, dist_now = bs.tools.geo.kwikqdrdist(
            own_lat, own_lon, float(dest[0]), float(dest[1])
        )
        r_progress = W_PROGRESS * (dist_prev - dist_now)

        # Smooth separation penalty — zero when safe, negative inside SEP_NM bubble
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
        r_sep = W_SEP * min(0.0, min_dist / sg.SEP_NM - 1.0) if min_dist < float('inf') else 0.0

        intrusion = any(cs in pair for pair in bs.traf.cd.lospairs)

        return float(r_progress + r_sep + (INTRUSION_PEN if intrusion else 0.0))

    def _get_exited(self):
        """Return callsigns that have left the airspace (without deleting them yet)."""
        exited = []
        for cs in list(self._active):
            idx = bs.traf.id2idx(cs)
            if idx < 0:
                exited.append(cs)
                continue
            inside = bs.tools.areafilter.checkInside(
                'AIRSPACE',
                np.array([bs.traf.lat[idx]]),
                np.array([bs.traf.lon[idx]]),
                np.array([sg.ALTITUDE * 30.48]),
            )
            if not inside:
                exited.append(cs)
        return exited

    def _exit_reward(self, cs):
        """
        Asymmetric exit reward based on heading alignment with destination:
          drift =   0° → +EXIT_CORRECT (+3)  correct exit
          drift =  90° →  0
          drift = 180° → +EXIT_WRONG   (-5)  wrong boundary
        """
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            return 0.0
        own_lat = bs.traf.lat[idx]
        own_lon = bs.traf.lon[idx]
        own_hdg = bs.traf.hdg[idx]
        dest    = self._dest_ll[cs]
        wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
            own_lat, own_lon, float(dest[0]), float(dest[1])
        )
        c = math.cos(math.radians(_wrap180(own_hdg - wpt_qdr)))
        return float(EXIT_CORRECT * c if c >= 0 else EXIT_WRONG * c)


# ── Single-agent wrapper for SB3 / SubprocVecEnv ─────────────────────────────

class MARLSingleAgentWrapper(gym.Env):
    """
    Presents the N-agent env as a single-agent stream for SB3.
    Collects one action per agent sequentially, steps the real env once,
    then serves mean reward back to SB3.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        self._env = AirspaceEnvV2()
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

        self._ep_rewards      = np.zeros(N_AGENTS)
        self._ep_los          = 0
        self._ep_steps        = 0
        self._ep_actions      = []
        self._obs_stack       = None
        self._agent_idx       = 0
        self._pending_actions = []

    def reset(self, seed=None, options=None):
        obs_stack, info       = self._env.reset(seed=seed, options=options)
        self._obs_stack       = obs_stack
        self._agent_idx       = 0
        self._pending_actions = []
        self._ep_rewards      = np.zeros(N_AGENTS)
        self._ep_los          = 0
        self._ep_steps        = 0
        self._ep_actions      = []
        return obs_stack[0], info

    def step(self, action):
        self._pending_actions.append(int(action))
        self._ep_actions.append(int(action))

        if len(self._pending_actions) < N_AGENTS:
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
                'action_distribution': np.bincount(self._ep_actions, minlength=4).tolist(),
            }

        return obs_stack[0], float(rewards.mean()), terminated, truncated, extra_info
