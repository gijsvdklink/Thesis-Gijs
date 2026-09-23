"""Check the lightweight traffic model against BlueSky, which it replaces.

The environment no longer imports BlueSky; this script does, so that the kinematics the
model was calibrated on can be re-measured and the two flown side by side. It is the
evidence behind the constants in Environment/traffic.py.

    python Validation/validate_kinematics.py

Reports, for a set of manoeuvres, the largest difference in track position, heading and
speed between BlueSky's A320 at FL350 and the model. Anything well inside the 5 NM
separation minimum means the replacement does not change the conflicts the agent sees.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from larger_netw_exp.Environment.config import CONFIG, CRUISE_ALT_M, KT_PER_MACH
from larger_netw_exp.Environment.traffic import Traffic

import bluesky as bs
from bluesky.simulation import ScreenIO
from bluesky.stack.stackbase import Stack as _BsStack


class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0):
        pass


def _start():
    bs.init(mode='sim', detached=True)
    bs.scr = _ScreenDummy()


def _bs_reset(hdg, mach):
    _BsStack.cmdstack.clear()
    bs.traf.reset()
    bs.stack.stack("DT 1.0;FF")
    bs.traf.cre('AC00', actype=CONFIG['ac_type'], aclat=0.0, aclon=0.0,
                achdg=hdg, acspd=mach, acalt=CRUISE_ALT_M)
    bs.stack.stack('ALT AC00 FL350')
    bs.stack.stack(f'SPD AC00 {mach}')
    bs.stack.stack('ASAS OFF')
    for _ in range(80):          # let BlueSky settle onto the commanded cruise state
        bs.sim.step()
    return np.array([bs.traf.lon[0] * 60.0, bs.traf.lat[0] * 60.0])


def _bs_state(origin):
    return (np.array([bs.traf.lon[0] * 60.0, bs.traf.lat[0] * 60.0]) - origin,
            float(bs.traf.hdg[0]),
            float(bs.traf.tas[0]) / 1852.0)


def run_case(name, hdg0, mach0, commands, seconds):
    """commands: {second: ('hdg', deg) | ('mach', m)} applied to both models alike."""
    origin = _bs_reset(hdg0, mach0)

    traf = Traffic()
    traf.create('AC00', 0.0, 0.0, hdg0, mach0)

    worst_pos = worst_hdg = worst_spd = 0.0
    for t in range(1, seconds + 1):
        if t in commands:
            kind, value = commands[t]
            if kind == 'hdg':
                bs.stack.stack(f'HDG AC00 {value}')
                traf.set_heading('AC00', value)
            else:
                bs.stack.stack(f'SPD AC00 {value}')
                traf.set_mach('AC00', value)

        bs.sim.step()
        traf.step(1.0)

        bs_pos, bs_hdg, bs_spd = _bs_state(origin)
        lw_pos, lw_hdg, lw_spd = traf.pos[0], float(traf.hdg[0]), float(traf.tas[0])

        worst_pos = max(worst_pos, float(np.hypot(*(bs_pos - lw_pos))))
        worst_hdg = max(worst_hdg, abs((bs_hdg - lw_hdg + 180.0) % 360.0 - 180.0))
        worst_spd = max(worst_spd, abs(bs_spd - lw_spd) * 3600.0)

    flown = float(np.hypot(*_bs_state(origin)[0]))
    print(f'  {name:<34} {seconds:4d} s  {flown:6.1f} NM flown   '
          f'max dpos {worst_pos:6.3f} NM   dhdg {worst_hdg:5.2f} deg   dspd {worst_spd:5.2f} kt')
    return worst_pos, worst_hdg, worst_spd


def main():
    _start()
    m = CONFIG['ac_mach']
    step = CONFIG['mach_step']

    print('BlueSky A320 FL350 vs the lightweight model, identical commands:')
    cases = [
        ('straight and level',            0.0,  m, {},                        900),
        ('turn +30',                     90.0,  m, {10: ('hdg', 120.0)},      900),
        ('turn -60',                     90.0,  m, {10: ('hdg', 30.0)},       900),
        ('turn +90 then back',           45.0,  m, {10: ('hdg', 135.0),
                                                    300: ('hdg', 45.0)},      900),
        ('speed up one step',             0.0,  m, {10: ('mach', m + step)},  900),
        ('slow down one step',            0.0,  m, {10: ('mach', m - step)},  900),
        ('turn and speed up together',   20.0,  m, {10: ('hdg', 80.0),
                                                    10.5: ('mach', m + step)}, 900),
        ('repeated turns',                0.0,  m, {10: ('hdg', 60.0),
                                                    120: ('hdg', 0.0),
                                                    240: ('hdg', 300.0),
                                                    400: ('hdg', 45.0)},      900),
    ]
    worst = [0.0, 0.0, 0.0]
    for name, hdg0, mach0, cmds, secs in cases:
        cmds = {int(k): v for k, v in cmds.items()}
        r = run_case(name, hdg0, mach0, cmds, secs)
        worst = [max(a, b) for a, b in zip(worst, r)]

    print(f'\n  worst over all cases: {worst[0]:.3f} NM, {worst[1]:.2f} deg, {worst[2]:.2f} kt')
    print(f'  separation minimum is {CONFIG["sep_nm"]:.0f} NM, so the position error is '
          f'{100 * worst[0] / CONFIG["sep_nm"]:.2f}% of it.')


if __name__ == '__main__':
    main()
