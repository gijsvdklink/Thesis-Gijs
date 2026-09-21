# The BlueSky bridge: starting it once per process, and reading traffic into our frame.

import numpy as np

import bluesky as bs
from bluesky.simulation import ScreenIO

from .config import CONFIG
from .geometry import NMS_PER_MS, latlon_to_nm

_started = False


class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0):
        pass


def start_bluesky():
    global _started
    if not _started:
        bs.init(mode='sim', detached=True)
        _started = True
    bs.scr = _ScreenDummy()
    bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")


def traffic_states(indices):
    idx = np.asarray(indices, dtype=int)

    pos   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
    speed = bs.traf.tas[idx] * NMS_PER_MS
    hdg   = np.radians(bs.traf.hdg[idx])

    vel = np.stack([speed * np.sin(hdg), speed * np.cos(hdg)], axis=1)
    return pos, vel
