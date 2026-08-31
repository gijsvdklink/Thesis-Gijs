# The BlueSky bridge: starting it once per process, and reading traffic into our frame.

import math

import numpy as np

import bluesky as bs
from bluesky.simulation import ScreenIO

from .config import CONFIG
from .geometry import NMS_PER_MS

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
