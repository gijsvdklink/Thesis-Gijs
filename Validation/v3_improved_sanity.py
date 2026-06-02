"""
Sanity checks for v3_improved environment.

Run:
    python -m Validation.v3_improved_sanity
"""

import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bluesky as bs
import numpy as np
from Environments.v3_improved import AirspaceEnv, CONFIG

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
WARN = '\033[93mWARN\033[0m'

def check(label, cond, detail=''):
    status = PASS if cond else FAIL
    print(f'  [{status}] {label}' + (f'  →  {detail}' if detail else ''))
    return cond

def main():
    print('\n=== v3_improved sanity checks ===\n')

    env = AirspaceEnv()
    obs, _ = env.reset(seed=42)

    # ── 1. Altitude units ────────────────────────────────────────────────────
    print('1. Altitude units')
    cs_list = list(env._active_callsigns)
    if not cs_list:
        print(f'  [{WARN}] no aircraft spawned — skipping altitude check')
    else:
        cs  = cs_list[0]
        idx = bs.traf.id2idx(cs)
        alt_m = float(bs.traf.alt[idx])          # BlueSky stores altitude in metres
        fl350_m = 35000 * 0.3048                 # 10668.0 m
        check('alt ≈ 10 668 m (FL350)',
              abs(alt_m - fl350_m) < 500,
              f'bs.traf.alt = {alt_m:.1f} m  (expected {fl350_m:.1f} m)')
        if alt_m < 1000:
            print(f'  [{WARN}] alt = {alt_m:.1f} — looks like feet were passed as metres; '
                  f'change acalt to CONFIG["altitude"] * 30.48')
        elif alt_m > 20_000 and abs(alt_m - fl350_m) > 500:
            print(f'  [{WARN}] alt = {alt_m:.1f} — may be in feet (35000); '
                  f'change acalt to CONFIG["altitude"] * 30.48')

    # ── 2. Speed units ───────────────────────────────────────────────────────
    print('\n2. Speed units (TAS)')
    if cs_list:
        tas_ms = float(bs.traf.tas[idx])
        tas_kt = tas_ms / 0.514444
        expected_kt = CONFIG['ac_speed']          # 450 kt nominal TAS (approx)
        check('TAS ≈ 230–280 m/s  (450 kt at FL350)',
              200 < tas_ms < 320,
              f'bs.traf.tas = {tas_ms:.1f} m/s  ({tas_kt:.0f} kt)')
        if tas_ms > 400:
            print(f'  [{WARN}] TAS = {tas_ms:.1f} m/s — acspd may be interpreted as m/s; '
                  f'should be kts. Change acspd to ~231.5 (m/s equivalent of 450 kt).')

    # ── 3. SPD command semantics ─────────────────────────────────────────────
    print('\n3. Speed actions — commanded Mach bookkeeping')
    if cs_list:
        from Environments.v3_improved import SPEED_DELTAS
        env._commanded_speed[cs] = CONFIG['ac_mach']
        env._apply_action(cs, 6)  # M−0.04
        m6 = env._commanded_speed[cs]
        check('Action 6 (M−0.04) → commanded_speed = 0.74',
              abs(m6 - (CONFIG['ac_mach'] + SPEED_DELTAS[0])) < 1e-6,
              f'commanded Mach = {m6:.4f}')
        env._commanded_speed[cs] = CONFIG['ac_mach']
        env._apply_action(cs, 8)  # M+0.02
        m8 = env._commanded_speed[cs]
        check('Action 8 (M+0.02) → commanded_speed = 0.80',
              abs(m8 - (CONFIG['ac_mach'] + SPEED_DELTAS[2])) < 1e-6,
              f'commanded Mach = {m8:.4f}')
        env._commanded_speed[cs] = CONFIG['ac_mach']  # restore
        print(f'  [NOTE] MACH cmd takes ~60–120 sim-steps to reflect in bs.traf.M '
              f'(verified manually: M0.78→0.73 after 60 steps)')

    # ── 4. Sim stepping — distance per RL step ──────────────────────────────
    print('\n4. Sim stepping — distance per RL step')
    if cs_list:
        from Environments.v3_improved import latlon_to_nm
        nm_before = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        for _ in range(CONFIG['action_freq']):
            bs.sim.step()
        nm_after = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        dist_nm = float(np.linalg.norm(nm_after - nm_before))
        step_s  = CONFIG['action_freq'] * CONFIG['sim_dt']   # 10 s
        expected_nm = CONFIG['ac_speed'] / 3600.0 * step_s  # 1.25 nm at 450 kt
        check(f'aircraft moves ≈ {expected_nm:.2f} nm per RL step',
              abs(dist_nm - expected_nm) < 0.3,
              f'moved {dist_nm:.3f} nm  (expected ~{expected_nm:.2f} nm)')

    # ── 5. Reset cleanliness ─────────────────────────────────────────────────
    print('\n5. Reset cleanliness')
    n_before = bs.traf.ntraf
    obs2, _  = env.reset(seed=99)
    n_after  = bs.traf.ntraf
    # After reset, traf should only contain freshly spawned aircraft (not accumulated)
    check('aircraft count after 2nd reset ≤ max_agents',
          bs.traf.ntraf <= CONFIG['max_agents'],
          f'bs.traf.ntraf = {bs.traf.ntraf}')
    check('step counter reset to 0',
          env._step_count == 0,
          f'_step_count = {env._step_count}')
    check('pending spawns cleared',
          len(env._pending_spawns) == 0,
          f'_pending_spawns = {env._pending_spawns}')

    # ── 6. Observation range ─────────────────────────────────────────────────
    print('\n6. Observation range — all values in [-1, 1] or [0, 1]')
    obs_arr = np.array(obs2)
    check('obs has correct dimension (21)',
          len(obs_arr) == 21,
          f'dim = {len(obs_arr)}')
    check('no NaN/Inf in obs',
          np.all(np.isfinite(obs_arr)),
          f'non-finite at indices {np.where(~np.isfinite(obs_arr))[0].tolist()}')
    out_of_range = np.where(np.abs(obs_arr) > 1.01)[0]
    check('all obs values in [-1, 1]',
          len(out_of_range) == 0,
          f'out-of-range indices {out_of_range.tolist()}: {obs_arr[out_of_range]}')

    # ── 7. Quick episode roll-out ─────────────────────────────────────────────
    print('\n7. Quick 20-step roll-out (no crash)')
    obs3, _ = env.reset(seed=7)
    ok = True
    for t in range(20):
        action = env.action_space.sample()
        try:
            obs3, rew, term, trunc, info = env.step(action)
            if not np.all(np.isfinite(obs3)):
                print(f'  [{FAIL}] non-finite obs at step {t}')
                ok = False
                break
        except Exception as e:
            print(f'  [{FAIL}] exception at step {t}: {e}')
            ok = False
            break
    if ok:
        print(f'  [{PASS}] 20 steps completed, last reward = {rew:.4f}')

    env.close()
    print('\n=== done ===\n')

if __name__ == '__main__':
    main()
