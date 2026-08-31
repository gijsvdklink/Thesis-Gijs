"""Regression harness: fly fixed scenarios under a fixed action sequence and record what came out.

The refactor is allowed to change behaviour, but only where it means to. This records what the
environment does now, so that every later change is either provably invisible or a delta that can be
named and justified.

    python -m Environments.main.replay_check --write     # capture the baseline, before touching anything
    python -m Environments.main.replay_check --compare   # diff the current code against it

Actions come from a seeded PRNG rather than a trained policy, so no model files are needed and the
sequence is identical on both sides of a change.
"""

import argparse
import json
import os
import random

from .config import N_ACTIONS, VALIDATION_SEEDS
from .delays import DELAY_MODES
from .env import AirspaceEnv, episode_summary

# The matrix: every delay mode over the same scenarios, so the delay paths are all exercised.
SCENARIOS   = tuple(VALIDATION_SEEDS[:3])
STEPS       = 400            # ~20 exits and a handful of LoS events per run; see the docstring
DELAY_MEAN  = 30.0
ACTION_SEED = 12345

BASELINE = os.path.join(os.path.dirname(__file__), 'replay_baseline.json')

# Excluded from the record: 'actions' is a 400-element list, summarised by action_distribution.
_SKIP_STATS = ('actions',)


def fly(delay_mode, scenario_seed, steps):
    """One run: fixed scenario, fixed action sequence. Returns the raw counters and the summary."""
    env = AirspaceEnv(delay_mode=delay_mode, delay_mean_s=DELAY_MEAN, seed=0)
    env.reset(options={'scenario_seed': scenario_seed})

    actions = random.Random(ACTION_SEED)
    for _ in range(steps):
        env.step(actions.randrange(N_ACTIONS))

    stats = {k: v for k, v in env._ep_stats.items() if k not in _SKIP_STATS}
    return {
        'delay_mode':    delay_mode,
        'scenario_seed': scenario_seed,
        'n_aircraft':    env.n_aircraft,
        'rho':           env.rho,
        'max_steps':     env._max_steps,
        'stats':         stats,
        'summary':       episode_summary(env._ep_stats),
    }


def record(steps=STEPS):
    """The whole matrix, in a fixed order."""
    runs = []
    for delay_mode in DELAY_MODES:
        for scenario_seed in SCENARIOS:
            runs.append(fly(delay_mode, scenario_seed, steps))
            print(f'  {delay_mode:<14} scenario {scenario_seed}  '
                  f'exits={runs[-1]["stats"]["exits"]:>3}  '
                  f'los={runs[-1]["stats"]["los_events"]:>3}', flush=True)
    return {'steps': steps, 'action_seed': ACTION_SEED, 'runs': runs}


# -- Comparison ----------------------------------------------------------------

def _flatten(run):
    """One run's numbers as a flat {path: value} map, so a diff can name exactly what moved."""
    flat = {}
    for section in ('stats', 'summary'):
        for key, value in run[section].items():
            if isinstance(value, list):          # action_distribution
                for i, count in enumerate(value):
                    flat[f'{section}.{key}[{i}]'] = count
            else:
                flat[f'{section}.{key}'] = value
    return flat


def compare(baseline, current):
    """Print every key that moved. Returns the number of runs that differ."""
    if len(baseline['runs']) != len(current['runs']):
        print(f'run count changed: {len(baseline["runs"])} -> {len(current["runs"])}')
        return len(current['runs'])

    changed_runs = 0
    for old, new in zip(baseline['runs'], current['runs']):
        label = f'{new["delay_mode"]} / {new["scenario_seed"]}'
        if old['delay_mode'] != new['delay_mode'] or old['scenario_seed'] != new['scenario_seed']:
            print(f'{label}: matrix order changed')
            changed_runs += 1
            continue

        old_flat, new_flat = _flatten(old), _flatten(new)
        deltas = [(key, old_flat.get(key), new_flat.get(key))
                  for key in sorted(set(old_flat) | set(new_flat))
                  if old_flat.get(key) != new_flat.get(key)]
        if not deltas:
            continue

        changed_runs += 1
        print(f'\n{label}:')
        for key, was, now in deltas:
            print(f'  {key:<44} {was!r:>22}  ->  {now!r}')

    return changed_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='capture the baseline')
    parser.add_argument('--compare', action='store_true', help='diff against the baseline')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--baseline', default=BASELINE)
    args = parser.parse_args()

    if not (args.write or args.compare):
        parser.error('pass --write or --compare')

    print(f'{len(DELAY_MODES) * len(SCENARIOS)} runs of {args.steps} steps', flush=True)
    current = record(args.steps)

    if args.write:
        with open(args.baseline, 'w') as handle:
            json.dump(current, handle, indent=1, sort_keys=True)
        print('\nwrote', args.baseline)
        return

    with open(args.baseline) as handle:
        baseline = json.load(handle)

    if baseline['steps'] != current['steps']:
        raise SystemExit(f'baseline is {baseline["steps"]} steps, this run is {current["steps"]}')

    changed = compare(baseline, current)
    print('\nIDENTICAL' if not changed else f'\n{changed} of {len(current["runs"])} runs differ')
    raise SystemExit(1 if changed else 0)


if __name__ == '__main__':
    main()
