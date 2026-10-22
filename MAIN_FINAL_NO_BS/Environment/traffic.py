"""A kinematic traffic model in place of BlueSky, for faster training.

Every constant was measured from BlueSky's A320 at FL350, not assumed; Validation/
validate_kinematics.py repeats the measurement and reports the agreement:

    turns     BlueSky holds a fixed 25 deg bank, so the rate is g*tan(25)/V: 1.1328 deg/s at
              M 0.78. It is recomputed from the current speed rather than fixed.
    speed     a constant 0.5 m/s^2 toward the commanded Mach, both signs; one Mach step of
              0.02 takes 12 s.
    position  straight-line integration in the flat east/north NM frame the sector lives in.
              Over 200 s this differs from BlueSky's great-circle track by 0.016 NM, 0.3 % of
              the 5 NM separation minimum.

There is no global state: each AirspaceEnv owns its Traffic, where BlueSky was a process-wide
singleton. The attribute names mirror the BlueSky fields the environment reads.
"""

import math

import numpy as np

from .config import CONFIG, KT_PER_MACH

M_PER_NM   = 1852.0
NMS_PER_MS = 1.0 / M_PER_NM
KT_TO_NMS  = 1.0 / 3600.0

BANK_DEG  = 25.0
ACCEL_MS2 = 0.5
G_MS2     = 9.81

_TURN_NUM  = G_MS2 * math.tan(math.radians(BANK_DEG))
ACCEL_NMS2 = ACCEL_MS2 * NMS_PER_MS


def mach_to_nms(mach):
    return mach * KT_PER_MACH * KT_TO_NMS


class Traffic:
    """The aircraft of one environment, integrated every simulated second.

    Positions are east/north NM about CONFIG['center_ll'], speeds NM/s, headings degrees
    clockwise from north.
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

    def create(self, cs, east_nm, north_nm, hdg_deg, mach):
        """Created established, at the commanded heading and speed, as BlueSky settles within
        seconds of a cre() at cruise."""
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

    def set_heading(self, cs, hdg_deg):
        i = self._idx.get(cs, -1)
        if i >= 0:
            self.cmd_hdg[i] = float(hdg_deg) % 360.0

    def set_mach(self, cs, mach):
        i = self._idx.get(cs, -1)
        if i >= 0:
            self.cmd_tas[i] = mach_to_nms(mach)

    def step(self, dt=1.0):
        """Speed first, so the bank-limited turn rate uses the speed actually flown."""
        if not self.id:
            return

        dv = np.clip(self.cmd_tas - self.tas, -ACCEL_NMS2 * dt, ACCEL_NMS2 * dt)
        self.tas += dv

        rate_deg = np.degrees(_TURN_NUM / (self.tas * M_PER_NM)) * dt
        dh = (self.cmd_hdg - self.hdg + 180.0) % 360.0 - 180.0
        self.hdg = (self.hdg + np.clip(dh, -rate_deg, rate_deg)) % 360.0

        h = np.radians(self.hdg)
        self.pos[:, 0] += self.tas * np.sin(h) * dt
        self.pos[:, 1] += self.tas * np.cos(h) * dt

    def states(self, indices):
        """Positions (NM) and velocities (NM/s) of the given rows."""
        idx   = np.asarray(indices, dtype=int)
        hdg_r = np.radians(self.hdg[idx])
        speed = self.tas[idx]
        vel   = np.stack([speed * np.sin(hdg_r), speed * np.cos(hdg_r)], axis=1)
        return self.pos[idx].copy(), vel
