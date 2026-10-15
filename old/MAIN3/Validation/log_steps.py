"""Every RL step of a set of models, flown in one delay condition.

The test harness records one row per EPISODE. This records one row per STEP: the action the
policy chose, the reward it earned and the three terms that reward is made of, and the
aircraft the reward was charged to. It is for looking at what a policy actually does inside
an episode -- which advisory, when, and what it cost -- rather than at the episode totals.

The three reward terms are accumulated inside the environment and never returned, so each
step's contribution is recovered as the change in the running total across that step. Nothing
in Environment/ is modified.

One file per seed, so several terminals can share the work without colliding:

    python Validation/log_steps.py --seeds 1 2
    python Validation/log_steps.py --seeds 3 4

Defaults to the no-delay models flown in deterministic 60 s, 20 episodes each.
"""

import argparse
import csv
import importlib.util
import os
import sys

HERE     = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, MAIN_DIR)

spec = importlib.util.spec_from_file_location(
    'rts', os.path.join(HERE, 'run_test_scenarios.py'))
rts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rts)

from Environment.config import (CONFIG, VALIDATION_SEEDS, TURN_DELTAS, SPEED_ACTIONS,
                                HOLD_ACTION, RETURN_TO_INITIAL_HDG_ACTION, N_ACTIONS)

MODELS_DIR  = os.path.join(MAIN_DIR, 'Models')
RESULTS_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'Step_logs')

# What each action index means, read off the action layout rather than restated here, so this
# stays correct whether or not the folder has speed actions.
LABEL = {HOLD_ACTION: 'hold', RETURN_TO_INITIAL_HDG_ACTION: 'return'}
for _index, _delta in TURN_DELTAS.items():
    LABEL[_index] = f'turn_{_delta:+d}'
for _index, _sign in SPEED_ACTIONS.items():
    LABEL[_index] = 'speed_up' if _sign > 0 else 'speed_down'
ACTION_LABELS = [LABEL.get(i, f'action_{i}') for i in range(N_ACTIONS)]

STEP_FIELDS = ['model_type', 'seed', 'episode', 'scenario_seed', 'step', 'acting_cs',
               'action', 'action_label', 'reward', 'r_los', 'r_drift', 'r_work',
               'los_seconds', 'n_aircraft']
TERMS = ('reward_los', 'reward_drift', 'reward_work')


def log_seed(model_type, seed, law, mean, episodes, models, out):
    from Environment import AirspaceEnv

    step_path    = os.path.join(out, f'steps_{model_type}_seed{seed}.csv')
    episode_path = os.path.join(out, f'episodes_{model_type}_seed{seed}.csv')
    if os.path.exists(step_path):
        print(f'{os.path.basename(step_path)} already done', flush=True)
        return

    steps, checkpoint = rts.find_model(models, model_type, seed)
    policy = rts.load_policy(checkpoint)
    env = (AirspaceEnv(delay_mode=law, delay_mean_s=float(mean)) if mean
           else AirspaceEnv(delay_mode='none'))
    print(f'{model_type} seed {seed} ({steps:,} steps) in {law} {mean} s: '
          f'{episodes} episodes', flush=True)

    episode_rows = []
    with open(step_path + '.part', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
        writer.writeheader()

        for episode in range(episodes):
            scenario_seed = VALIDATION_SEEDS[episode]
            observation, _ = env.reset(options={'scenario_seed': scenario_seed})
            executed = rts.count_executed(env.atco)
            previous = {key: env._ep_stats[key] for key in TERMS}
            index = 0

            while True:
                # The focus BEFORE the step is the aircraft the reward is charged to;
                # info['focus_cs'] is already the NEXT step's focus.
                acting = env.focus_cs
                action = rts.choose(policy, observation)
                observation, reward, _, truncated, info = env.step(action)

                current = {key: env._ep_stats[key] for key in TERMS}
                writer.writerow({
                    'model_type': model_type, 'seed': seed, 'episode': episode,
                    'scenario_seed': scenario_seed, 'step': index,
                    'acting_cs': acting or '', 'action': int(action),
                    'action_label': ACTION_LABELS[int(action)], 'reward': reward,
                    'r_los':   current['reward_los']   - previous['reward_los'],
                    'r_drift': current['reward_drift'] - previous['reward_drift'],
                    'r_work':  current['reward_work']  - previous['reward_work'],
                    'los_seconds': info['los_seconds'], 'n_aircraft': info['n_aircraft'],
                })
                previous = current
                index += 1

                if truncated:
                    kpis = {k.removeprefix('ep_'): v for k, v in info.items()
                            if k.startswith('ep_')}
                    executed['advisories'] = executed['turns'] + executed['speed_changes']
                    for kind, count in executed.items():
                        kpis[f'{kind}_executed'] = count
                        kpis[f'{kind}_executed_per_fh'] = count / kpis['flight_hours']
                    episode_rows.append({'model_type': model_type, 'seed': seed,
                                         'episode': episode, 'scenario_seed': scenario_seed,
                                         'test_law': law, 'test_delay': mean,
                                         'model_steps': steps, **kpis})
                    break

            if (episode + 1) % 5 == 0 or episode + 1 == episodes:
                print(f'  {episode + 1}/{episodes} episodes', flush=True)

    with open(episode_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    os.replace(step_path + '.part', step_path)
    print(f'  -> {step_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='which training seeds this terminal handles, e.g. --seeds 1 2')
    parser.add_argument('--model-type', default='none', choices=list(rts.MODEL_TYPES),
                        help='which models to fly (default none)')
    parser.add_argument('--delay', default='deterministic',
                        help='delay law to fly them in (default deterministic)')
    parser.add_argument('--delay-mean', type=int, default=60,
                        help='delay magnitude in seconds (default 60); 0 for no delay')
    parser.add_argument('--episodes', type=int, default=20,
                        help='scenarios per model, from the start of the validation set')
    parser.add_argument('--n-ac', type=int, default=None, metavar='N',
                        help='hold the aircraft count at N instead of drawing it per episode')
    parser.add_argument('--density', type=float, default=None, metavar='RHO',
                        help='hold the traffic density at RHO ac/km^2 instead of drawing it')
    parser.add_argument('--models', default=MODELS_DIR, help=f'default {MODELS_DIR}')
    parser.add_argument('--out', default=RESULTS_DIR, help=f'default {RESULTS_DIR}')
    args = parser.parse_args()

    # A fixed airspace, if asked for. Set before any environment is built.
    if args.n_ac is not None:
        CONFIG['n_aircraft'] = lambda rng: args.n_ac
    if args.density is not None:
        CONFIG['rho'] = lambda rng: args.density

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for seed in args.seeds:
        log_seed(args.model_type, seed, args.delay, args.delay_mean,
                 args.episodes, os.path.abspath(args.models), out)


if __name__ == '__main__':
    main()
