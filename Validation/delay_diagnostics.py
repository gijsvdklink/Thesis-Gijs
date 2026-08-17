# Diagnostics for the delay experiment. Run this BEFORE committing to a training budget.
#
# Answers two questions:
#   1. Do the four arms actually see the same scenarios at the same seed?
#   2. What delays do pilots actually get, and is the faster "engaged" branch ever used?
#      If the arms behave alike, this is where it shows up.
#
#   python -m Validation.delay_diagnostics --episodes 2
#   python -m Validation.delay_diagnostics --episodes 3 --model path/to/best_model.zip

import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environments.v4 import AirspaceEnv, DELAY_MODES
from Environments.v4.delays import FIRST_S, NEXT_S

HIST_MAX_S = 90.0   # histogram range only; the probabilistic arm has no upper bound

SEEDS = (10_001, 10_002, 10_003, 10_004, 10_005)


def run_episode(mode, seed, steps, policy):
    """One episode, recording every delay the pilots were given."""
    env = AirspaceEnv(delay_mode=mode)
    obs, _ = env.reset(seed=seed)

    # Record the realised delay of every instruction the pilots actually flew; the
    # episode summary only keeps a running mean. Measured at execution rather than at
    # issue, because the probabilistic arm has no deadline to read off.
    recorded, real_is_due = [], env.response_delay.is_due

    def spy(cmd, now):
        due = real_is_due(cmd, now)
        if due:
            recorded.append(now - cmd['issued_at_s'])
        return due
    env.response_delay.is_due = spy

    scenario = {'n_aircraft': env.n_aircraft, 'ep_length': env._max_steps,
                'sector_nm2': float(abs(env._polygon_shape.area))}

    for _ in range(steps or env._max_steps):
        obs, _, _, truncated, info = env.step(policy(obs))
        if truncated:
            break

    return scenario, env._episode_summary(), recorded


def describe(values):
    """min / mean / median / p90 / max of a sample, as a one-line string."""
    if not values:
        return 'no draws'
    ordered = sorted(values)
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    return (f'min {ordered[0]:5.1f}   mean {statistics.fmean(values):5.1f}   '
            f'median {statistics.median(values):5.1f}   p90 {p90:5.1f}   max {ordered[-1]:5.1f}')


def histogram(values, width=48, bins=14):
    """Coarse text histogram, so the shapes can be compared by eye. The top bin is a
    catch-all, since the probabilistic arm's tail is unbounded."""
    if not values:
        return []
    edges  = np.linspace(0, HIST_MAX_S, bins + 1)
    counts, _ = np.histogram(np.clip(values, 0, HIST_MAX_S), bins=edges)
    peak = max(counts.max(), 1)
    return [f'    {edges[i]:5.1f}-{edges[i+1]:5.1f} s  '
            f'{"#" * int(width * counts[i] / peak):<{width}} {counts[i]:>6}'
            for i in range(bins)]


def make_policy(model_path, seed):
    """A trained policy if one is given, otherwise uniform random actions."""
    if not model_path:
        rng = np.random.default_rng(seed)
        return lambda obs: int(rng.integers(0, 10))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    model   = PPO.load(model_path, device='cpu')
    vecnorm = VecNormalize.load(model_path.replace('.zip', '_vecnorm.pkl'), None)
    vecnorm.training = False

    def policy(obs):
        action, _ = model.predict(vecnorm.normalize_obs(obs.reshape(1, -1)), deterministic=True)
        return int(np.asarray(action).flat[0])
    return policy


def main():
    ap = argparse.ArgumentParser(description='Delay-experiment diagnostics.')
    ap.add_argument('--episodes', type=int, default=2, help='episodes per mode')
    ap.add_argument('--steps', type=int, default=1500,
                    help='steps per episode (0 = run the full episode, ~2100-4800 steps)')
    ap.add_argument('--model', default=None,
                    help='trained checkpoint to drive the traffic; random actions if omitted')
    args = ap.parse_args()

    seeds  = SEEDS[:max(1, args.episodes)]
    policy = make_policy(args.model, seeds[0])
    print(f'{args.episodes} episode(s) x {args.steps or "full"} steps per mode, '
          f'{"trained policy" if args.model else "random actions"}\n')

    scenarios, results, samples = {}, {}, {}
    for mode in DELAY_MODES:
        summaries, delays, scen = [], [], []
        for seed in seeds:
            s, summary, recorded = run_episode(mode, seed, args.steps, policy)
            scen.append(s)
            summaries.append(summary)
            delays += recorded
        scenarios[mode] = scen
        results[mode]   = summaries
        samples[mode]   = delays
        print(f'  {mode:<15} done')

    # 1. Did every arm get the same traffic?
    print('\n=== SCENARIO MATCHING (same seed must give the same airspace) ===')
    reference = scenarios[DELAY_MODES[0]]
    for mode in DELAY_MODES:
        match = 'identical' if scenarios[mode] == reference else '*** DIFFERS ***'
        print(f'  {mode:<15} {match}')
    for seed, s in zip(seeds, reference):
        print(f'    seed {seed}: {s["n_aircraft"]} aircraft, '
              f'{s["ep_length"]} steps, {s["sector_nm2"]:,.0f} NM^2')

    # 2. What delays did the pilots actually get?
    print('\n=== REALISED RESPONSE DELAYS ===')
    for mode in DELAY_MODES:
        if mode == 'none':
            continue
        print(f'\n  {mode}   ({len(samples[mode]):,} advisories)')
        print(f'    {describe(samples[mode])}')
        for line in histogram(samples[mode]):
            print(line)

    # 3. Is the faster branch ever reached?
    print('\n=== ENGAGEMENT AND WORKLOAD ===')
    print(f'  {"mode":<15} {"engaged %":>10} {"mean delay":>11} {"focus hold":>11} {"discarded":>12}')
    for mode in DELAY_MODES:
        r = results[mode]
        print(f'  {mode:<15} '
              f'{100 * statistics.fmean(x["ep_delay_next_frac"] for x in r):>9.1f}% '
              f'{statistics.fmean(x["ep_delay_mean_s"] for x in r):>10.1f}s '
              f'{statistics.fmean(x["ep_focus_hold_steps"] for x in r):>10.1f} '
              f'{statistics.fmean(x["ep_discarded"] for x in r):>12.0f}')
    print(f'\n  "engaged %" is how often an advisory took the {NEXT_S:g} s branch rather than')
    print(f'  the {FIRST_S:g} s one. Near 0 means the 30/15 split never fires and the arms are')
    print(f'  effectively single-delay; near 100 means the {FIRST_S:g} s branch is the rare one.')

    # 4. The KPIs the arms are compared on.
    print('\n=== KPIs ===')
    print(f'  {"mode":<15} {"LoS/fh":>9} {"deviation":>11} {"arrivals":>9} '
          f'{"manoeuvred":>11} {"reward":>10}')
    for mode in DELAY_MODES:
        r = results[mode]
        handled = statistics.fmean(x['ep_manoeuvred_exits'] for x in r)
        exits   = statistics.fmean(x['ep_exits'] for x in r)
        print(f'  {mode:<15} '
              f'{statistics.fmean(x["ep_los_events_per_fh"] for x in r):>9.3f} '
              f'{statistics.fmean(x["ep_exit_deviation_nm"] for x in r):>10.1f}N '
              f'{statistics.fmean(x["ep_arrival_rate"] for x in r):>9.3f} '
              f'{handled:>6.0f}/{exits:<4.0f} '
              f'{statistics.fmean(x["ep_reward_total"] for x in r):>10.1f}')
    print('\n  Deviation and arrivals are scored over MANOEUVRED aircraft only (the')
    print('  "manoeuvred" column is how many of the exits those were). Untouched traffic')
    print('  exits perfectly on route and would otherwise swamp both numbers.')
    print('  With random actions these KPIs mean little -- pass --model for trained arms.')


if __name__ == '__main__':
    main()
