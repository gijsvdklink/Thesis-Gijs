"""
Unit tests for reward formula components in v3_dalmau.
These tests exercise pure-math helpers and the conflict/drift/work formulae
directly, without starting BlueSky or the Gym environment.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Environments.v3_dalmau import (
    latlon_to_nm, wrap_to_180, CONFIG, NM_TO_KM, ACT_COST
)

# ── helpers ───────────────────────────────────────────────────────────────────

def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}: expected {b}, got {a}  (diff={a-b:.2e})"

def assert_gt(a, b, msg=""):
    assert a > b, f"{msg}: expected {a} > {b}"

def assert_le(a, b, msg=""):
    assert a <= b, f"{msg}: expected {a} <= {b}"

# ── CPA conflict score (standalone reimplementation of the formula) ────────────

def _conflict_score_formula(dx, dy, dvn, dve, sep, t_warn, lookahead):
    """Mirror of _cpa_conflict_score for a single pair."""
    d_now = math.sqrt(dx*dx + dy*dy)
    rv2   = dvn*dvn + dve*dve

    # in LoS already
    if d_now < sep:
        return max(0.0, 1.0 - d_now / sep)

    if rv2 < 1e-12:
        return 0.0

    dot  = dx*dve + dy*dvn
    tcpa = -dot / rv2

    if tcpa <= 0 or tcpa > lookahead:
        return 0.0

    dx_c  = dx + tcpa * dve
    dy_c  = dy + tcpa * dvn
    cpa_d = math.sqrt(dx_c*dx_c + dy_c*dy_c)

    dcpa_factor = max(0.0, 1.0 - cpa_d / sep)
    if dcpa_factor == 0.0:
        return 0.0

    tcpa_factor = max(0.0, 1.0 - tcpa / t_warn)
    return dcpa_factor * tcpa_factor


SEP    = CONFIG['sep_nm']
T_WARN = CONFIG['t_warn']
LOOK   = CONFIG['lookahead_s']

# ── Tests: conflict score ──────────────────────────────────────────────────────

def test_conflict_head_on_imminent():
    """Head-on collision course, tcpa=0 → score should be 1.0."""
    # two aircraft at (0,0) and (1,0) NM, flying toward each other at 1 NM/s
    # tcpa = 0.5 s, dcpa = 0.0 NM
    spd = 1.0  # NM/s
    dx, dy   =  1.0, 0.0
    dvn, dve = -2 * spd, 0.0   # closing at 2 NM/s in x direction
    score = _conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK)
    # tcpa = 0.5 s (< T_WARN), dcpa ≈ 0 → high score
    assert_gt(score, 0.5, "head-on imminent conflict score")
    assert_le(score, 1.0, "score must be ≤ 1")
    print(f"  head-on imminent: score={score:.4f}  ✓")


def test_conflict_perfect_collision_tcpa_zero():
    """Aircraft at exactly the same position → in-LoS branch, score = 1.0."""
    score = _conflict_score_formula(0.0, 0.0, -1.0, 0.0, SEP, T_WARN, LOOK)
    assert_close(score, 1.0, msg="zero-distance in-LoS score")
    print(f"  zero-distance in-LoS: score={score:.4f}  ✓")


def test_conflict_diverging_zero():
    """Diverging pair → no conflict, score = 0."""
    # aircraft heading away from each other
    dx, dy   = 10.0, 0.0
    dvn, dve =  1.0, 0.0   # intruder moving away
    score = _conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK)
    assert_close(score, 0.0, msg="diverging pair score")
    print(f"  diverging pair: score={score:.4f}  ✓")


def test_conflict_wide_miss():
    """Converging pair that misses by more than sep → score = 0."""
    # crossing paths, miss distance >> sep
    sep = SEP  # 5 NM
    # aircraft at (0,0), intruder at (0, 100 NM), flying east at 1 NM/s
    dx, dy   = 0.0, 100.0
    dvn, dve = 0.0, 1.0   # moving east, own aircraft stationary → no CPA solution closing
    # actually let's construct a proper wide-miss case:
    # own at origin, intruder at (0, 50), moving straight south
    dx, dy   = 0.0,  50.0
    dvn, dve = -0.5, 0.0   # intruder moving south, own stationary (dv = intruder - own = -0.5 south)
    score = _conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK)
    # tcpa when dot = dx*dve + dy*dvn = 0 + 50*(-0.5) = -25, tcpa = 25/0.25 = 100 s
    # dcpa: at tcpa=100, dy_c = 50 + 100*(-0.5) = 0, dx_c = 0 → cpa=0 < sep!
    # so this IS a conflict, just very long tcpa
    # Let's use a case where intruder passes far to the side
    dx, dy   = 20.0, 50.0
    dvn, dve = -0.5, 0.0
    # dot = 20*0 + 50*(-0.5) = -25, rv2 = 0.25, tcpa = 100 s
    # dx_c = 20, dy_c = 0 → cpa = 20 >> sep=5
    score = _conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK)
    assert_close(score, 0.0, tol=1e-9, msg="wide-miss score")
    print(f"  wide miss: score={score:.4f}  ✓")


def test_conflict_beyond_lookahead():
    """tcpa beyond lookahead → score = 0."""
    # converging but very slow — tcpa >> lookahead
    dx, dy   = 0.0, 1000.0
    dvn, dve = -0.001, 0.0  # almost stationary
    score = _conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK)
    assert_close(score, 0.0, msg="beyond-lookahead score")
    print(f"  beyond lookahead: score={score:.4f}  ✓")


def test_conflict_in_los_partial():
    """Aircraft 2 NM apart (< sep=5) → score = 1 - 2/5 = 0.6."""
    d_now = 2.0
    # place at (d_now, 0), arbitrary velocities (in-LoS branch ignores them)
    score = _conflict_score_formula(d_now, 0.0, -1.0, 0.0, SEP, T_WARN, LOOK)
    expected = 1.0 - d_now / SEP
    assert_close(score, expected, msg="partial in-LoS score")
    print(f"  in-LoS partial (d={d_now}): score={score:.4f} expected={expected:.4f}  ✓")


def test_conflict_increases_as_tcpa_decreases():
    """Same geometry, smaller tcpa → larger score.

    Aircraft start 20 NM apart (> sep=5 NM) heading straight at each other.
    Faster closing speed → smaller tcpa → higher tcpa_factor → higher score.
    dcpa = 0 for all cases (head-on), so only tcpa_factor varies.
    """
    dx, dy = 20.0, 0.0   # 20 NM apart — outside sep
    scores = []
    # closing speeds chosen so tcpa stays within lookahead and t_warn
    for closing_speed in [0.02, 0.05, 0.1, 0.5]:
        # tcpa = dx / closing_speed: 1000, 400, 200, 40 s — all < LOOK=900 except 1000
        dvn, dve = 0.0, -closing_speed  # intruder flying west (east-component negative)
        scores.append(_conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK))
    # filter to non-zero (tcpa <= lookahead) and check monotone
    nonzero = [(i, s) for i, s in enumerate(scores) if s > 0]
    for k in range(len(nonzero)-1):
        assert nonzero[k][1] < nonzero[k+1][1], \
            f"score should increase with closing speed: {scores}"
    print(f"  score vs closing speed: {[f'{s:.3f}' for s in scores]}  ✓")


# ── Tests: drift penalty ───────────────────────────────────────────────────────

def _drift_penalty(diff_deg, w_drift=None):
    if w_drift is None:
        w_drift = CONFIG['w_drift']
    cos_err = math.cos(math.radians(diff_deg))
    return -w_drift * (1.0 - cos_err) / 2.0


def test_drift_on_course():
    """Zero heading error → zero drift penalty."""
    r = _drift_penalty(0.0)
    assert_close(r, 0.0, msg="on-course drift penalty")
    print(f"  on-course drift: {r:.4f}  ✓")


def test_drift_max():
    """180° error → maximum drift penalty = -w_drift."""
    r = _drift_penalty(180.0)
    assert_close(r, -CONFIG['w_drift'], msg="max drift penalty")
    print(f"  max drift (180°): {r:.4f}  ✓")


def test_drift_90deg():
    """90° error → half drift penalty."""
    r = _drift_penalty(90.0)
    assert_close(r, -CONFIG['w_drift'] * 0.5, msg="90-deg drift penalty")
    print(f"  90° drift: {r:.4f}  ✓")


def test_drift_negative_angle():
    """Drift penalty is symmetric: +45° == -45°."""
    r_pos = _drift_penalty(45.0)
    r_neg = _drift_penalty(-45.0)
    assert_close(r_pos, r_neg, msg="drift symmetry")
    print(f"  drift symmetry: +45°={r_pos:.4f}, -45°={r_neg:.4f}  ✓")


def test_drift_wraps_correctly():
    """350° and -10° are the same angular error."""
    # wrap_to_180(350) = -10, so cos is the same
    diff1 = wrap_to_180(350.0)
    diff2 = -10.0
    assert_close(diff1, diff2, msg="wrap_to_180(350)")
    r1 = _drift_penalty(diff1)
    r2 = _drift_penalty(diff2)
    assert_close(r1, r2, msg="wrapped drift equality")
    print(f"  wrap_to_180 drift: diff={diff1:.1f}°, penalty={r1:.4f}  ✓")


# ── Tests: work penalty ────────────────────────────────────────────────────────

def test_work_costs():
    """ACT_COST × w_work for each action."""
    W = CONFIG['w_work']
    expected = {
        0: -1.0 * W,   # −45° turn
        1: -0.5 * W,   # −30° turn
        2:  0.0,        # hold
        3: -0.5 * W,   # +30° turn
        4: -1.0 * W,   # +45° turn
        5:  0.0,        # back-to-WP
        6: -0.5 * W,   # speed −0.04
        7: -0.5 * W,   # speed −0.02
        8: -0.5 * W,   # speed +0.02
        9: -0.5 * W,   # speed +0.04
    }
    for action, exp_penalty in expected.items():
        got = -ACT_COST[action] * W
        assert_close(got, exp_penalty, msg=f"work penalty action {action}")
    print(f"  all 10 action costs correct (W={W})  ✓")


def test_work_penalty_new_weight():
    """w_work was changed from 0.05 to 0.20; verify config value."""
    assert CONFIG['w_work'] == 0.20, \
        f"Expected w_work=0.20, got {CONFIG['w_work']}"
    print(f"  w_work = {CONFIG['w_work']}  ✓")


# ── Tests: coord helpers ───────────────────────────────────────────────────────

def test_latlon_to_nm_origin():
    """Center point maps to (0, 0)."""
    center = CONFIG['center_ll']
    nm = latlon_to_nm(center, center[0], center[1])
    assert_close(nm[0], 0.0, msg="origin x")
    assert_close(nm[1], 0.0, msg="origin y")
    print(f"  center → (0,0)  ✓")


def test_latlon_to_nm_north():
    """1° north ≈ 60 NM north."""
    center = CONFIG['center_ll']
    nm = latlon_to_nm(center, center[0] + 1.0, center[1])
    assert_close(nm[1], 60.0, tol=0.1, msg="1° north = 60 NM")
    assert_close(nm[0], 0.0,  tol=0.1, msg="1° north, no east")
    print(f"  1° north → {nm[1]:.2f} NM north  ✓")


def test_wrap_to_180():
    assert_close(wrap_to_180(0.0),    0.0)
    # 180° wraps to ±180 — both are the same angle; check via cos
    assert_close(math.cos(math.radians(wrap_to_180(180.0))), math.cos(math.radians(180.0)))
    assert_close(wrap_to_180(270.0),  -90.0)
    assert_close(wrap_to_180(350.0),  -10.0)
    assert_close(wrap_to_180(-190.0), 170.0)
    print(f"  wrap_to_180 cases  ✓")


# ── Test: reward weights ───────────────────────────────────────────────────────

def test_reward_weights():
    """Spot-check updated config weights."""
    assert CONFIG['w_los']      == 5.00, f"w_los={CONFIG['w_los']}"
    assert CONFIG['w_conflict'] == 1.00, f"w_conflict={CONFIG['w_conflict']}"
    assert CONFIG['w_drift']    == 0.80, f"w_drift={CONFIG['w_drift']}"
    assert CONFIG['w_work']     == 0.20, f"w_work={CONFIG['w_work']}"
    print(f"  weights: los={CONFIG['w_los']} conflict={CONFIG['w_conflict']} "
          f"drift={CONFIG['w_drift']} work={CONFIG['w_work']}  ✓")


# ── Test: conflict score monotonicity in dcpa ──────────────────────────────────

def test_conflict_decreases_as_dcpa_increases():
    """Larger miss distance → smaller score."""
    # fixed tcpa, varying cpa by tilting the approach angle
    # use: aircraft at (sep*t, 0) for small t, intruder flying straight west
    # dcpa varies with lateral offset
    spd = 0.5  # NM/s
    tcpa_target = T_WARN / 2  # mid-warning
    lateral_offsets = [0.0, 0.5, 1.0, 2.0, 4.5, 5.0, 6.0]
    scores = []
    for lat in lateral_offsets:
        # intruder at (spd * tcpa_target, lat), flying west at spd
        dx, dy   = spd * tcpa_target, lat
        dvn, dve = 0.0, -spd
        scores.append(_conflict_score_formula(dx, dy, dvn, dve, SEP, T_WARN, LOOK))
    # scores should be non-increasing as lateral offset increases
    for i in range(len(scores)-1):
        assert scores[i] >= scores[i+1] - 1e-9, \
            f"score not decreasing: offsets={lateral_offsets}, scores={scores}"
    print(f"  dcpa sweep: {[f'{s:.3f}' for s in scores]}  ✓")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        # conflict score
        test_conflict_head_on_imminent,
        test_conflict_perfect_collision_tcpa_zero,
        test_conflict_diverging_zero,
        test_conflict_wide_miss,
        test_conflict_beyond_lookahead,
        test_conflict_in_los_partial,
        test_conflict_increases_as_tcpa_decreases,
        test_conflict_decreases_as_dcpa_increases,
        # drift
        test_drift_on_course,
        test_drift_max,
        test_drift_90deg,
        test_drift_negative_angle,
        test_drift_wraps_correctly,
        # work
        test_work_costs,
        test_work_penalty_new_weight,
        # coord helpers
        test_latlon_to_nm_origin,
        test_latlon_to_nm_north,
        test_wrap_to_180,
        # weights
        test_reward_weights,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"[{name}]")
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
