"""Test the trained models on the held-out test scenarios.

Every model type (7 seeds each) is flown in the undelayed environment and in the deterministically
and lognormally delayed environments of 15, 30, 45 and 60 s; no CR is flown once, as it never
transmits an advisory. Every run flies the same VALIDATION_SEEDS scenarios, which training never
draws. Each model is tested at its furthest-trained checkpoint.

python Validation/run_test_scenarios.py --terminal 1      one terminal's share of the runs (1 to 24)
python Validation/run_test_scenarios.py --merge           every run in one CSV, once all terminals are done

A terminal skips the runs it has already finished, so a stopped terminal is resumed by giving
the same command again.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MAIN_DIR)

from Environment.config import HOLD_ACTION, TRAINING_SEEDS, VALIDATION_SEEDS

MODELS_DIR  = os.path.join(MAIN_DIR, 'Models')
RESULTS_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'Test_results')

MODEL_TYPES = {'none': 'none', 'deterministic': 'deterministic_30s', 'lognormal': 'lognormal_30s'}
PREFIX      = {'none': 'N', 'deterministic': 'D', 'lognormal': 'L'}
NO_CR       = 'no_cr'

NO_DELAY     = ('none', 0)
TEST_DELAYS  = [NO_DELAY] + [(law, mean) for law in ('deterministic', 'lognormal')
                             for mean in (15, 30, 45, 60)]
TERMINALS    = 24
PRINT_EVERY  = 25


def all_runs():
    """Every test run as (model type, seed, delay law, mean delay): 3 x 7 x 9 + no CR = 190."""
    runs = [(model_type, seed, law, mean) for model_type in MODEL_TYPES
            for seed in TRAINING_SEEDS for law, mean in TEST_DELAYS]
    return runs + [(NO_CR, 0) + NO_DELAY]


def terminal_share(terminal, terminals):
    return all_runs()[terminal - 1::terminals]


def model_name(model_type, seed):
    return 'NoCR' if model_type == NO_CR else f'{PREFIX[model_type]}{seed}'


def run_path(out, model_type, seed, law, mean):
    test = 'no_delay' if law == 'none' else f'{law}_{mean}s'
    return os.path.join(out, 'runs', f'{model_name(model_type, seed)}_{test}.csv')


def checkpoint_steps(path):
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read('data').decode())['num_timesteps'])


def find_model(models, model_type, seed):
    """The furthest-trained checkpoint of one model, and its step count."""
    folder = MODEL_TYPES[model_type]
    run_dir = os.path.join(models, folder, f'{folder}_seed{seed}')
    found = [(checkpoint_steps(path), path)
             for path in (os.path.join(run_dir, name) for name in ('final_model.zip', 'last_model.zip'))
             if os.path.exists(path)]
    if not found:
        sys.exit(f'no model in {run_dir}')
    return max(found)


class _Ignored:
    def __init__(self, *args):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    """Reads the observation statistics under any numpy version. The random state stored beside
    them is not needed to test, and differs between numpy versions, so it is skipped."""

    def find_class(self, module, name):
        if module.startswith('numpy.random'):
            return _Ignored
        if np.__version__.startswith('1.'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(path):
    from stable_baselines3 import PPO

    with open(path.replace('.zip', '_vecnorm.pkl'), 'rb') as handle:
        normaliser = _Unpickler(handle).load()
    mean = np.asarray(normaliser.obs_rms.mean, dtype=np.float32)
    std = np.sqrt(np.asarray(normaliser.obs_rms.var, dtype=np.float32) + normaliser.epsilon)
    return PPO.load(path, device='cpu'), mean, std, float(normaliser.clip_obs)


def choose(policy, observation):
    if policy is None:
        return HOLD_ACTION
    model, mean, std, clip = policy
    action, _ = model.predict(np.clip((observation - mean) / std, -clip, clip), deterministic=True)
    return int(np.asarray(action).flat[0])


def count_executed(atco):
    """Counts the instructions the ATCO actually passes on. The environment counts the advisories
    sent to the ATCO, but a revision replaces the one being evaluated, so not all are flown."""
    executed = {'turns': 0, 'speed_changes': 0}
    act_if_ready = atco.act_if_ready

    def counted(*args):
        ready = act_if_ready(*args)
        if ready is not None:
            executed['speed_changes' if 'target_mach' in ready[1] else 'turns'] += 1
        return ready

    atco.act_if_ready = counted
    return executed


def fly(env, policy, scenario_seed):
    """One scenario, and every KPI of its episode, raw and per flight hour."""
    observation, _ = env.reset(options={'scenario_seed': scenario_seed})
    executed = count_executed(env.atco)
    while True:
        observation, _, _, truncated, info = env.step(choose(policy, observation))
        if truncated:
            kpis = {key.removeprefix('ep_'): value for key, value in info.items()
                    if key.startswith('ep_')}
            executed['advisories'] = executed['turns'] + executed['speed_changes']
            for kind, count in executed.items():
                kpis[f'{kind}_executed'] = count
                kpis[f'{kind}_executed_per_fh'] = count / kpis['flight_hours']
            return kpis


def test_run(model_type, seed, law, mean, models, out, episodes):
    from Environment import AirspaceEnv

    path = run_path(out, model_type, seed, law, mean)
    if os.path.exists(path):
        print(f'{os.path.basename(path)} already done', flush=True)
        return

    if model_type == NO_CR:
        policy, steps = None, 0
    else:
        steps, checkpoint = find_model(models, model_type, seed)
        policy = load_policy(checkpoint)
    env = AirspaceEnv(delay_mode=law, delay_mean_s=float(mean)) if mean else AirspaceEnv(delay_mode='none')

    scenarios = VALIDATION_SEEDS[:episodes]
    print(f'{model_name(model_type, seed)} ({steps:,} steps) in {law} {mean} s: '
          f'{len(scenarios)} scenarios', flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    started = time.time()
    with open(path + '.part', 'w', newline='') as handle:
        writer = None
        for scenario, scenario_seed in enumerate(scenarios):
            kpis = fly(env, policy, scenario_seed)
            row = {'model': model_name(model_type, seed), 'train_type': model_type, 'seed': seed,
                   'model_steps': steps, 'test_type': law, 'test_delay': mean,
                   'scenario': scenario, 'scenario_seed': scenario_seed,
                   'n_aircraft': env.n_aircraft, 'rho': env.rho, **kpis}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            done = scenario + 1
            if done % PRINT_EVERY == 0 or done == len(scenarios):
                rate = (time.time() - started) / done
                print(f'  {done:>3}/{len(scenarios)}  {rate:4.1f} s/scenario  '
                      f'{rate * (len(scenarios) - done) / 60:5.1f} min to go', flush=True)
    os.replace(path + '.part', path)


def run_terminal(terminal, terminals, models, out, episodes):
    """This terminal's runs one after another, each in its own process: BlueSky is a
    process-wide singleton, so every test environment gets a fresh one."""
    share = terminal_share(terminal, terminals)
    print(f'terminal {terminal} of {terminals}: {len(share)} runs -> {out}', flush=True)
    failed = []
    for model_type, seed, law, mean in share:
        command = [sys.executable, os.path.abspath(__file__), '--run', model_type, str(seed),
                   law, str(mean), '--models', models, '--out', out, '--episodes', str(episodes)]
        if subprocess.run(command).returncode != 0:
            failed.append(os.path.basename(run_path(out, model_type, seed, law, mean)))
    print(f'terminal {terminal} finished' + (f'; failed: {", ".join(failed)}' if failed else ''))


def merge(out):
    missing = [os.path.basename(path) for path in (run_path(out, *run) for run in all_runs())
               if not os.path.exists(path)]
    if missing:
        sys.exit(f'{len(missing)} runs are not finished yet:\n  ' + '\n  '.join(missing))

    frame = pd.concat([pd.read_csv(run_path(out, *run)) for run in all_runs()], ignore_index=True)
    path = os.path.join(out, 'test_results.csv')
    frame.to_csv(path, index=False)
    print(f'{len(frame):,} episodes of {frame["model"].nunique()} models in '
          f'{frame.groupby(["model", "test_type", "test_delay"]).ngroups} runs -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--terminal', type=int, help=f'which terminal, 1 to {TERMINALS}')
    parser.add_argument('--terminals', type=int, default=TERMINALS)
    parser.add_argument('--merge', action='store_true', help='combine all runs into one CSV')
    parser.add_argument('--run', nargs=4, metavar=('TYPE', 'SEED', 'LAW', 'MEAN'),
                        help='a single test run, e.g. --run lognormal 3 deterministic 45')
    parser.add_argument('--models', default=MODELS_DIR, help=f'default {MODELS_DIR}')
    parser.add_argument('--out', default=RESULTS_DIR, help=f'default {RESULTS_DIR}')
    parser.add_argument('--episodes', type=int, default=len(VALIDATION_SEEDS),
                        help='test scenarios per run, from the start of the set')
    args = parser.parse_args()
    out, models = os.path.abspath(args.out), os.path.abspath(args.models)

    if args.merge:
        merge(out)
    elif args.run:
        model_type, seed, law, mean = args.run
        test_run(model_type, int(seed), law, int(mean), models, out, args.episodes)
    elif args.terminal:
        run_terminal(args.terminal, args.terminals, models, out, args.episodes)
    else:
        parser.error('give --terminal, --run or --merge')


if __name__ == '__main__':
    main()
