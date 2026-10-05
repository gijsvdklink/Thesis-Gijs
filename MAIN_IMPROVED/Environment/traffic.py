# A lightweight kinematic traffic model, in place of BlueSky.
#
# Every constant here was MEASURED from BlueSky's A320 at FL350 rather than assumed
# (Validation/validate_kinematics.py reproduces the measurement and checks the match):
#
#   turns     BlueSky holds a fixed 25.0 deg bank, so the rate is g*tan(25)/V. It is
#             1.1328 deg/s at M 0.78 and varies with TAS, which is why the rate is
#             recomputed from the current speed rather than fixed.
#   speed     a constant 0.5 m/s^2 toward the commanded Mach, both accelerating and
#             decelerating; one Mach step of 0.02 takes 12 s.
#   position  straight-line integration in the flat east/north NM frame the controller
#             already reasons in. Over 75 NM this differs from BlueSky's great-circle
#             track by 0.046 NM, under 1% of the 5 NM separation minimum.
#
# Nothing here is global state. BlueSky was a process-wide singleton, which forced
# n_envs = 1; every AirspaceEnv now owns its own Traffic, so environments vectorise.

import math

import numpy as np

from .config import CONFIG, KT_PER_MACH

M_PER_NM   = 1852.0
NMS_PER_MS = 1.0 / M_PER_NM          # m/s -> NM/s
KT_TO_NMS  = 1.0 / 3600.0            # kt   -> NM/s

# -- Measured from BlueSky; see the module docstring ---------------------------
BANK_DEG      = 25.0                 # implied bank, constant across the speed envelope
ACCEL_MS2     = 0.5                  # m/s^2, both signs
G_MS2         = 9.81

_TURN_NUM     = G_MS2 * math.tan(math.radians(BANK_DEG))   # rad/s when divided by V [m/s]
ACCEL_NMS2    = ACCEL_MS2 * NMS_PER_MS                     # NM/s^2


def mach_to_nms(mach):
    return mach * KT_PER_MACH * KT_TO_NMS


class Traffic:
    """The aircraft of one environment: position, heading and speed, integrated per second.

    Positions are east/north NM about CONFIG['center_ll']; speeds are NM/s; headings are
    degrees clockwise from north. The attribute names mirror the BlueSky fields the
    environment used to read (`id`, `hdg`, `tas`, `pos`), so the call sites read the same.
    """

    __slots__ = ('id', '_idx', 'pos', 'hdg', 'tas', 'cmd_hdg', 'cmd_tas')

    def __init__(self):
        self.reset()

    def reset(self):
        self.id   = []
        self._idx = {}
        self.pos     = np.zeros((0, 2))
        self.hdg     = np.zeros(0)
        self.tas     = np.zeros(0)
        self.cmd_hdg = np.zeros(0)
        self.cmd_tas = np.zeros(0)

    # -- Lifecycle -------------------------------------------------------------

    def create(self, cs, east_nm, north_nm, hdg_deg, mach):
        # Created established: already at the commanded heading and speed, which is what
        # BlueSky settled to within a few seconds of a cre() at cruise.
        speed = mach_to_nms(mach)
        self.id.append(cs)
        self._idx[cs] = len(self.id) - 1
        self.pos     = np.vstack([self.pos, (float(east_nm), float(north_nm))])
        self.hdg     = np.append(self.hdg,     float(hdg_deg) % 360.0)
        self.tas     = np.append(self.tas,     speed)
        self.cmd_hdg = np.append(self.cmd_hdg, float(hdg_deg) % 360.0)
        self.cmd_tas = np.append(self.cmd_tas, speed)

    def delete_idx(self, i):
        if not 0 <= i < len(self.id):
            return
        del self.id[i]
        self.pos     = np.delete(self.pos, i, axis=0)
        self.hdg     = np.delete(self.hdg, i)
        self.tas     = np.delete(self.tas, i)
        self.cmd_hdg = np.delete(self.cmd_hdg, i)
        self.cmd_tas = np.delete(self.cmd_tas, i)
        self._idx = {cs: k for k, cs in enumerate(self.id)}

    def id2idx(self, cs):
        return self._idx.get(cs, -1)

    # -- Instructions ----------------------------------------------------------

    def set_heading(self, cs, hdg_deg):
        i = self._idx.get(cs, -1)
        if i >= 0:
            self.cmd_hdg[i] = float(hdg_deg) % 360.0

    def set_mach(self, cs, mach):
        i = self._idx.get(cs, -1)
        if i >= 0:
            self.cmd_tas[i] = mach_to_nms(mach)

    # -- Integration -----------------------------------------------------------

    def step(self, dt=1.0):
        if not self.id:
            return

        # Speed first, so the turn rate below uses the speed actually flown this second.
        dv = np.clip(self.cmd_tas - self.tas, -ACCEL_NMS2 * dt, ACCEL_NMS2 * dt)
        self.tas += dv

        # Bank-limited turn, shortest way round. The rate falls as the aircraft speeds up,
        # exactly as a fixed bank angle requires.
        rate_deg = np.degrees(_TURN_NUM / (self.tas * M_PER_NM)) * dt
        dh = (self.cmd_hdg - self.hdg + 180.0) % 360.0 - 180.0
        self.hdg = (self.hdg + np.clip(dh, -rate_deg, rate_deg)) % 360.0

        h = np.radians(self.hdg)
        self.pos[:, 0] += self.tas * np.sin(h) * dt
        self.pos[:, 1] += self.tas * np.cos(h) * dt

    # -- Read-out --------------------------------------------------------------

    def states(self, indices):
        """Positions (NM) and velocities (NM/s) of the given rows, as the controller sees them."""
        idx   = np.asarray(indices, dtype=int)
        hdg_r = np.radians(self.hdg[idx])
        speed = self.tas[idx]
        vel   = np.stack([speed * np.sin(hdg_r), speed * np.cos(hdg_r)], axis=1)
        return self.pos[idx].copy(), vel
