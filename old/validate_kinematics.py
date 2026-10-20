"""Does the kinematic traffic model fly like BlueSky? (Environment/traffic.py replaces BlueSky.)

python Validation/validate_kinematics.py        needs BlueSky installed; the environment itself does not

One A320 at FL350 and M 0.78 is flown for 300 s in BlueSky and in Traffic, with the same
instructions at the same seconds: a 30 deg turn, a Mach step up, a 45 deg turn back and a Mach
step down. Printed per manoeuvre: the largest difference in heading, speed and position, and the
turn rate and acceleration each model actually flew.
"""

import contextlib
import io
import os
import sys

import numpy as np

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MAIN_DIR)

from Environment.config import CONFIG, CRUISE_ALT_M
from Environment.traffic import Traffic

SECONDS = 300
MACH, STEP = CONFIG['ac_mach'], CONFIG['mach_step']
SCRIPT = {10: ('hdg', 120.0), 70: ('spd', MACH + STEP), 130: ('hdg', 75.0), 190: ('spd', MACH - STEP)}
WINDOWS = [('turn +30', 10, 70), ('Mach +0.02', 70, 130), ('turn -45', 130, 190),
           ('Mach -0.04', 190, SECONDS)]
KTS = 0.514444


def bluesky_track():
    import bluesky as bs
    from bluesky.simulation import ScreenIO

    class _Quiet(ScreenIO):
        def echo(self, text='', flags=0):
            pass

    with contextlib.redirect_stdout(io.StringIO()):
        bs.init(mode='sim', detached=True)
    bs.scr = _Quiet()
    bs.stack.stack(f"DT {CONFIG['sim_dt']};FF")
    bs.traf.reset()
    bs.traf.cre('AC0', actype=CONFIG['ac_type'], aclat=0.0, aclon=0.0, achdg=90.0, acspd=MACH,
                acalt=CRUISE_ALT_M)
    bs.stack.stack(f'SPD AC0 {MACH}')
    bs.stack.stack(f"ALT AC0 FL{CONFIG['altitude']}")
    rows = []
    for t in range(SECONDS):
        if t in SCRIPT:
            kind, value = SCRIPT[t]
            bs.stack.stack(f'HDG AC0 {value:.1f}' if kind == 'hdg' else f'SPD AC0 {value:.3f}')
        bs.sim.step()
        rows.append((bs.traf.lon[0] * 60.0, bs.traf.lat[0] * 60.0, bs.traf.hdg[0], bs.traf.tas[0] / KTS))
    return np.array(rows)


def kinematic_track():
    traf = Traffic()
    traf.create('AC0', 0.0, 0.0, 90.0, MACH)
    rows = []
    for t in range(SECONDS):
        if t in SCRIPT:
            kind, value = SCRIPT[t]
            traf.set_heading('AC0', value) if kind == 'hdg' else traf.set_mach('AC0', value)
        traf.step(CONFIG['sim_dt'])
        rows.append((traf.pos[0, 0], traf.pos[0, 1], traf.hdg[0], traf.tas[0] * 3600.0))
    return np.array(rows)


def main():
    blue, kin = bluesky_track(), kinematic_track()
    dhdg = np.abs((blue[:, 2] - kin[:, 2] + 180.0) % 360.0 - 180.0)
    dspd = np.abs(blue[:, 3] - kin[:, 3])
    dpos = np.hypot(blue[:, 0] - kin[:, 0], blue[:, 1] - kin[:, 1])
    rate = lambda track, a, b: np.abs(np.diff(track[a:b, 2])).max()
    accel = lambda track, a, b: np.abs(np.diff(track[a:b, 3])).max()
    print(f'{"manoeuvre":<12} {"max dhdg":>9} {"max dspd":>9} {"max dpos":>9}   '
          f'{"turn rate BS / model":>21}   {"accel BS / model":>18}')
    for name, a, b in WINDOWS:
        print(f'{name:<12} {dhdg[a:b].max():7.2f} deg {dspd[a:b].max():6.2f} kt {dpos[a:b].max():6.3f} NM   '
              f'{rate(blue, a, b):7.3f} / {rate(kin, a, b):5.3f} deg/s   '
              f'{accel(blue, a, b):6.3f} / {accel(kin, a, b):5.3f} kt/s')
    print(f'after {SECONDS} s ({blue[-1, 3] / 3600 * SECONDS:.0f} NM flown): position differs by '
          f'{dpos[-1]:.3f} NM, against a separation minimum of {CONFIG["sep_nm"]:g} NM')


if __name__ == '__main__':
    main()
