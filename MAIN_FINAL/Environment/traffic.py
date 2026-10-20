
import numpy as np

import bluesky as bs
from bluesky.simulation import ScreenIO

from .config import CONFIG
from .geometry import latlon_to_nm

M_PER_NM   = 1852.0
NMS_PER_MS = 1.0 / M_PER_NM

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
    """Positions (NM) and velocities (NM/s) of the given BlueSky rows, in our frame."""
    idx = np.asarray(indices, dtype=int)

    pos   = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
    speed = bs.traf.tas[idx] * NMS_PER_MS
    hdg   = np.radians(bs.traf.hdg[idx])

    vel = np.stack([speed * np.sin(hdg), speed * np.cos(hdg)], axis=1)
    return pos, vel
