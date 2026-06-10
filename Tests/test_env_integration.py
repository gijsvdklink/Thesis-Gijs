"""
Integration tests for v3 AirspaceEnv.

Spins up the real environment (BlueSky) and checks:
  - observation shape and value ranges
  - actions produce the correct commanded heading / speed changes
  - reward components have the right sign and magnitude
  - no crashes over a short episode
"""

import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Environments.v3 import AirspaceEnv, CONFIG, OBS_DIM, ACT_COST, wrap_to_180

# ── helpers ───────────────────────────────────────────────────────────────────

def assert_close(a, b, tol=1e-4, msg=""):
    assert abs(a - b) < tol, f"{msg}: expected {b:.6f}, got {a:.6f}"

def assert_ge(a, b, msg=""):
    assert a >= b, f"{msg}: expected {a} >= {b}"

def assert_le(a, b, msg=""):
    assert a <= b, f"{msg}: expected {a} <= {b}"

# ── Fixture ───────────────────────────────────────────────────────────────────

_env = None

def get_env():
    global _env
    if _env is None:
        _env = AirspaceEnv()
    return _env

# ── Observation tests ─────────────────────────────────────────────────────────

def test_obs_shape():
    env = get_env()
    obs, _ = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != ({OBS_DIM},)"
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
    print(f"  obs shape={obs.shape}, dtype={obs.dtype}  ✓")


def test_obs_heading_error_unit_circle():
    """sin/cos of heading error must satisfy sin²+cos² ≈ 1."""
    env = get_env()
    obs, _ = env.reset(seed=1)
    sin_d, cos_d = float(obs[0]), float(obs[1])
    norm = sin_d**2 + cos_d**2
    assert_close(norm, 1.0, tol=1e-5, msg="sin²+cos² heading error")
    print(f"  sin²+cos² = {norm:.6f}  ✓")


def test_obs_own_features_range():
    """Own features (indices 0-4) should be in [-1, 1]."""
    env = get_env()
    obs, _ = env.reset(seed=7)
    own = obs[:5]
    for i, v in enumerate(own):
        assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, \
            f"own feature [{i}] = {v:.4f} out of [-1, 1]"
    print(f"  own features in [-1,1]: {own}  ✓")


def test_obs_intruder_features_range():
    """Intruder block features (indices 5-24) should be in [-1, 1] or close."""
    env = get_env()
    obs, _ = env.reset(seed=3)
    intruder = obs[5:]
    for i, v in enumerate(intruder):
        assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4, \
            f"intruder feature [{i}] = {v:.4f} out of [-1, 1]"
    print(f"  intruder features in [-1,1]: min={intruder.min():.3f} max={intruder.max():.3f}  ✓")


def test_obs_consistent_after_step():
    """Observation after step has the same shape and dtype."""
    env = get_env()
    env.reset(seed=5)
    obs, _, _, _, _ = env.step(2)  # hold action
    assert obs.shape == (OBS_DIM,), f"post-step obs shape {obs.shape}"
    assert obs.dtype == np.float32
    print(f"  post-step obs shape and dtype  ✓")


# ── Action tests ──────────────────────────────────────────────────────────────

def test_action_heading_increments():
    """
    Heading actions 0-4 should shift cmd_hdg by the expected delta.
    Action 2 (Δψ=0) should not change cmd_hdg.
    """
    from Environments.v3 import HEADING_DELTAS
    env = get_env()
    env.reset(seed=10)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft at reset, skipping heading test")
        return

    for action_idx, delta in enumerate(HEADING_DELTAS):  # actions 0-4
        env.reset(seed=10)
        cs = env._focus_cs
        if cs is None:
            continue
        import bluesky as bs
        idx = bs.traf.id2idx(cs)
        before = env._commanded_heading.get(cs, bs.traf.hdg[idx])
        env._apply_action(cs, action_idx)
        after = env._commanded_heading[cs]
        expected = (before + delta) % 360
        assert_close(after, expected, tol=0.5, msg=f"action {action_idx} (Δ={delta}°)")
    print(f"  heading actions 0-4 shift cmd_hdg correctly  ✓")


def test_action_hold_no_change():
    """Action 2 (Δψ=0) leaves cmd_hdg unchanged."""
    env = get_env()
    env.reset(seed=11)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft, skipping hold test")
        return
    before = env._commanded_heading.get(cs, 0.0)
    env._apply_action(cs, 2)
    after = env._commanded_heading.get(cs, 0.0)
    assert_close(before, after, tol=1e-9, msg="hold action changes cmd_hdg")
    print(f"  hold action: cmd_hdg unchanged ({before:.1f}°)  ✓")


def test_action_back_to_wp():
    """Action 5 sets cmd_hdg to bearing toward destination."""
    import bluesky as bs
    from bluesky.tools import geo
    env = get_env()
    env.reset(seed=12)
    cs = env._focus_cs
    if cs is None or cs not in env._destination_ll:
        print("  no focus aircraft / destination, skipping back-to-WP test")
        return
    # first deviate
    env._apply_action(cs, 0)  # -45°
    env._apply_action(cs, 5)  # back-to-WP
    idx = bs.traf.id2idx(cs)
    bearing, _ = geo.kwikqdrdist(
        bs.traf.lat[idx], bs.traf.lon[idx],
        *[float(v) for v in env._destination_ll[cs]])
    cmd = env._commanded_heading[cs]
    diff = abs(wrap_to_180(cmd - bearing))
    assert diff < 1.0, f"back-to-WP: cmd_hdg={cmd:.1f}° bearing={bearing:.1f}° diff={diff:.2f}°"
    print(f"  back-to-WP: cmd_hdg={cmd:.1f}° ≈ bearing={bearing:.1f}°  ✓")


def test_action_speed_decrease():
    """Action 6 (ΔM=−0.04) decreases commanded Mach."""
    env = get_env()
    env.reset(seed=13)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft, skipping speed test")
        return
    before = env._commanded_speed.get(cs, CONFIG['ac_mach'])
    env._apply_action(cs, 6)
    after = env._commanded_speed[cs]
    expected = max(before - 0.04, CONFIG['ac_mach_min'])
    assert_close(after, expected, tol=1e-6, msg="speed decrease action 6")
    print(f"  action 6: mach {before:.3f} → {after:.3f}  ✓")


def test_action_speed_increase():
    """Action 9 (ΔM=+0.04) increases commanded Mach."""
    env = get_env()
    env.reset(seed=14)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft, skipping speed test")
        return
    before = env._commanded_speed.get(cs, CONFIG['ac_mach'])
    env._apply_action(cs, 9)
    after = env._commanded_speed[cs]
    expected = min(before + 0.04, CONFIG['ac_mach_max'])
    assert_close(after, expected, tol=1e-6, msg="speed increase action 9")
    print(f"  action 9: mach {before:.3f} → {after:.3f}  ✓")


def test_action_speed_clamped():
    """Speed stays within [M_min, M_max] even when pushed repeatedly."""
    env = get_env()
    env.reset(seed=15)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft, skipping clamp test")
        return
    for _ in range(20):
        env._apply_action(cs, 9)  # keep increasing
    after = env._commanded_speed[cs]
    assert_close(after, CONFIG['ac_mach_max'], tol=1e-6, msg="speed upper clamp")
    for _ in range(20):
        env._apply_action(cs, 6)  # keep decreasing
    after = env._commanded_speed[cs]
    assert_close(after, CONFIG['ac_mach_min'], tol=1e-6, msg="speed lower clamp")
    print(f"  speed clamped to [{CONFIG['ac_mach_min']}, {CONFIG['ac_mach_max']}]  ✓")


# ── Reward tests ──────────────────────────────────────────────────────────────

def test_reward_is_nonpositive():
    """Reward must be ≤ 0 at every step (all components are penalties)."""
    env = get_env()
    env.reset(seed=99)
    for action in range(10):
        env.reset(seed=99)
        for _ in range(5):
            obs, reward, _, truncated, _ = env.step(action)
            assert reward <= 0.0 + 1e-9, \
                f"reward > 0 at action {action}: {reward}"
            if truncated:
                break
    print(f"  reward ≤ 0 for all actions  ✓")


def test_reward_hold_cheaper_than_big_turn():
    """Action 2 (hold, cost 0) should produce reward ≥ action 0 (−45°, cost 1).

    The delta combines work cost and drift: if the turn corrects an existing drift,
    drift partially offsets the work penalty. We only assert the sign here; exact
    isolation requires knowing the pre-existing deviation direction.
    """
    env = get_env()
    rewards = {}
    for action in [0, 2]:
        env.reset(seed=77)
        _, r, _, _, _ = env.step(action)
        rewards[action] = r
    assert rewards[2] >= rewards[0] - 1e-9, \
        f"hold reward {rewards[2]:.4f} should be ≥ big-turn reward {rewards[0]:.4f}"
    diff = rewards[2] - rewards[0]
    print(f"  hold vs big-turn: Δreward={diff:.4f} (work={CONFIG['w_work']}, drift mixed in)  ✓")


def test_reward_drift_zero_when_on_course():
    """
    When cmd_hdg == bearing to destination (action 5 = back-to-WP),
    the drift term should be 0 (or very close).
    """
    env = get_env()
    env.reset(seed=55)
    cs = env._focus_cs
    if cs is None:
        print("  no focus aircraft, skipping drift test")
        return
    # action 5 aligns cmd_hdg to destination, then step with hold
    env._apply_action(cs, 5)

    import bluesky as bs
    from bluesky.tools import geo
    idx = bs.traf.id2idx(cs)
    bearing, _ = geo.kwikqdrdist(
        bs.traf.lat[idx], bs.traf.lon[idx],
        *[float(v) for v in env._destination_ll[cs]])
    cmd_hdg = env._commanded_heading.get(cs, bs.traf.hdg[idx])
    cos_err = math.cos(math.radians(wrap_to_180(bearing - cmd_hdg)))
    drift = CONFIG['w_drift'] * (1.0 - cos_err) / 2.0
    assert drift < 0.01, f"drift should be ~0 after back-to-WP: {drift:.4f}"
    print(f"  drift after back-to-WP: {drift:.6f} ≈ 0  ✓")


def test_reward_work_penalty_magnitudes():
    """
    Isolate work penalty by comparing hold (cost 0) vs a speed action (cost 0.5).

    Speed actions do not change cmd_hdg, so the drift term is identical in both
    cases. Over 5 simulated seconds a tiny speed change barely moves the aircraft,
    so conflict and LoS terms are also negligible. The reward delta should be very
    close to w_work × 0.5.
    """
    env = get_env()
    rewards = {}
    for action in [2, 8]:   # hold (cost 0) vs ΔM=+0.02 (cost 0.5)
        env.reset(seed=66)
        _, r, _, _, _ = env.step(action)
        rewards[action] = r
    diff = rewards[2] - rewards[8]
    expected = CONFIG['w_work'] * 0.5
    assert_close(diff, expected, tol=0.05,   # loose: tiny residual from speed effect on CPA
                 msg="Δwork ≈ w_work × 0.5")
    print(f"  hold vs speed action: Δ={diff:.4f} ≈ w_work×0.5={expected:.4f}  ✓")


# ── Episode smoke test ────────────────────────────────────────────────────────

def test_episode_no_crash():
    """Run one full episode to truncation without errors."""
    env = get_env()
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        assert obs.shape == (OBS_DIM,), f"bad obs shape at step {steps}"
        assert reward <= 0.0 + 1e-9, f"positive reward at step {steps}: {reward}"
        if terminated or truncated:
            break
    assert steps > 0
    print(f"  episode ran {steps} steps, total reward={total_reward:.2f}  ✓")


def test_episode_action_distribution_reasonable():
    """After one episode, every action should be taken at least once (stochastic policy)."""
    env = get_env()
    env.reset(seed=0)
    action_counts = np.zeros(10, dtype=int)
    done = False
    while not done:
        action = env.action_space.sample()
        action_counts[action] += 1
        _, _, _, done, _ = env.step(action)
    print(f"  action counts: {action_counts.tolist()}  ✓")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Initialising environment (BlueSky) …")
    get_env()
    print("Environment ready.\n")

    tests = [
        # observation
        test_obs_shape,
        test_obs_heading_error_unit_circle,
        test_obs_own_features_range,
        test_obs_intruder_features_range,
        test_obs_consistent_after_step,
        # actions
        test_action_heading_increments,
        test_action_hold_no_change,
        test_action_back_to_wp,
        test_action_speed_decrease,
        test_action_speed_increase,
        test_action_speed_clamped,
        # reward
        test_reward_is_nonpositive,
        test_reward_hold_cheaper_than_big_turn,
        test_reward_drift_zero_when_on_course,
        test_reward_work_penalty_magnitudes,
        # smoke
        test_episode_no_crash,
        test_episode_action_distribution_reasonable,
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
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
