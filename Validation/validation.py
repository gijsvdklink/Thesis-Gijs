"""Phase 1: fly one trained policy in one delay world and write one row per episode.

python Validation/validation.py --condition lognormal --run-seed 2000 --delay-mean 45
python Validation/validation.py --commands      # write "Terminal commands.txt"
"""

import argparse
import csv
import glob
import itertools
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environments.v4.config import VALIDATION_SEEDS

# -- The experiment ------------------------------------------------------------

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXPERIMENT  = os.path.join(ROOT, 'NEW DELAYS EXPERIMENT')
RUNS_ROOT   = os.path.join(EXPERIMENT, 'seed_study')
RESULTS_DIR = os.path.join(EXPERIMENT, 'results')
FIGURES_DIR = os.path.join(EXPERIMENT, 'figures')

# Training condition -> the run directory Training/v4_train.py wrote it to.
CONDITIONS = {
    'none':          'none',
    'deterministic': 'deterministic_30s',
    'geometric':     'probabilistic_30s',   # 'geometric' in the report, 'probabilistic' in delays.py
    'lognormal':     'lognormal_30s',
}

COLOURS = {'none':          'tab:blue',
           'deterministic': 'tab:red',
           'geometric':     'tab:orange',
           'lognormal':     'tab:green'}

LABELS = {'none':          'trained without delay',
          'deterministic': 'trained with 30 s deterministic delay',
          'geometric':     'trained with 30 s geometric delay',
          'lognormal':     'trained with 30 s lognormal delay'}

NO_CR       = 'no_cr'      # a condition name, so it must survive a command line unquoted
NO_CR_LABEL = 'no CR'

# The five training runs per condition, spaced so no two share part of their training scenarios.
TRAINING_SEEDS = [0, 1000, 2000, 3000, 4000]

# Mean pilot response time at test time, always lognormally distributed. 0 is the undelayed world.
DELAYS_S = [0, 15, 30, 45, 60, 90, 120]

# The held-out scenarios, named directly: every policy at every delay level flies exactly these.
BASE_SEEDS = list(VALIDATION_SEEDS)
EPISODES   = len(BASE_SEEDS)

TERMINALS     = 21
COMMANDS_FILE = os.path.join(os.path.dirname(__file__), 'Terminal commands.txt')

HOLD = 3


# -- One evaluation run --------------------------------------------------------

def find_model(condition, run_seed):
    """The checkpoint for one training run, located by condition and seed."""
    pattern = os.path.join(RUNS_ROOT, CONDITIONS[condition],
                           f'*seed{run_seed}_*', 'last_model.zip')
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f'no model for {condition} seed {run_seed}: {pattern}')
    if len(matches) > 1:
        sys.exit(f'{len(matches)} models match {pattern}; expected one')
    return matches[0]


class _Unpickler(pickle.Unpickler):
    """Reads a checkpoint written with a different numpy version."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(model_path):
    """(model, mean, std, clip) for normalising observations, or None to fly no CR."""
    if model_path is None:
        return None
    from stable_baselines3 import PPO

    stats_path = model_path.replace('.zip', '_vecnorm.pkl')
    if not os.path.exists(stats_path):
        sys.exit(f'no observation statistics beside the model: {stats_path}')
    with open(stats_path, 'rb') as stats_file:
        stats = _Unpickler(stats_file).load()

    mean = np.asarray(stats.obs_rms.mean, dtype=np.float32)
    std  = np.sqrt(np.asarray(stats.obs_rms.var, dtype=np.float32) + stats.epsilon)
    return PPO.load(model_path, device='cpu'), mean, std, float(stats.clip_obs)


def pick_action(policy, observation):
    """What to do this step. Without a policy that is always HOLD."""
    if policy is None:
        return HOLD
    model, mean, std, clip = policy
    normalised = np.clip((observation - mean) / std, -clip, clip)
    action, _ = model.predict(normalised, deterministic=True)
    return int(np.asarray(action).flat[0])


def make_env(delay_mean_s):
    """The test world: a mean of 0 is the undelayed one, every other level is lognormal."""
    from Environments.v4 import AirspaceEnv

    if delay_mean_s == 0:
        return AirspaceEnv(delay_mode='none')
    return AirspaceEnv(delay_mode='lognormal', delay_mean_s=float(delay_mean_s))


def run_episode(env, policy, scenario_seed):
    """Fly one named scenario and return its end-of-episode numbers."""
    observation, _ = env.reset(options={'scenario_seed': scenario_seed})
    while True:
        observation, _, _, truncated, info = env.step(pick_action(policy, observation))
        if truncated:
            return {key: value for key, value in info.items()
                    if isinstance(value, (int, float))}


def evaluate(condition, run_seed, delay_mean_s, base_seeds, path):
    """Fly the fixed scenario set in one delay world and write it to one CSV."""
    model_path = None if condition == NO_CR else find_model(condition, run_seed)
    policy     = load_policy(model_path)
    env        = make_env(delay_mean_s)

    started = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='') as handle:
        writer = None
        for episode, base_seed in enumerate(base_seeds):
            summary = run_episode(env, policy, base_seed)
            row = {'condition': condition, 'run_seed': run_seed,
                   'delay_mean_s': delay_mean_s, 'episode': episode,
                   'episode_seed': env.episode_seed,
                   'n_ac': env.n_aircraft, 'rho': env.rho, **summary}

            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()             # readable while the run continues

            done = episode + 1
            if done in (1, 5) or done % 10 == 0 or done == len(base_seeds):
                elapsed = time.time() - started
                left    = elapsed / done * (len(base_seeds) - done)
                print(f'  {done:>4}/{len(base_seeds)}  {elapsed / done:5.1f} s/episode  '
                      f'{left / 60:5.1f} min to go', flush=True)

    print('done ->', path, flush=True)


def output_name(condition, run_seed, delay_mean_s):
    """The CSV one evaluation run writes to."""
    if condition == NO_CR:
        return os.path.join(RESULTS_DIR, 'no_cr.csv')
    return os.path.join(RESULTS_DIR, f'{condition}_s{run_seed}_{delay_mean_s:g}s.csv')


# -- The sweep, dealt over the terminals ---------------------------------------

def grid():
    """Every evaluation run: the whole sweep, plus the delay-independent no-CR reference."""
    cells = list(itertools.product(CONDITIONS, TRAINING_SEEDS, DELAYS_S))
    cells.append((NO_CR, 0, 0))
    return cells


def command(condition, run_seed, delay_mean_s):
    return (f'python Validation/validation.py --condition {condition} '
            f'--run-seed {run_seed} --delay-mean {delay_mean_s:g}')


def share(cells):
    """The cells dealt round-robin, so every terminal gets the same amount of work."""
    return [cells[terminal::TERMINALS] for terminal in range(TERMINALS)]


def write_commands(path=COMMANDS_FILE):
    """The two phases, written out command by command."""
    cells = grid()
    lines = ['PHASE 1 -- VALIDATION',
             f'{len(cells)} evaluation runs of {EPISODES} episodes, over {TERMINALS} terminals.',
             'Run every terminal from the project root; they are independent and may be',
             'started in any order.',
             '']

    for terminal, work in enumerate(share(cells), start=1):
        lines.append(f'terminal {terminal} ({len(work)} runs):')
        lines += [command(*cell) for cell in work]
        lines.append('')

    lines += ['',
              'PHASE 2 -- PLOTTING',
              'One terminal, after every run of Phase 1 has finished. It reports what is on',
              'disk, then writes fig_degradation.png; it stops with a list of missing runs',
              'if Phase 1 is incomplete.',
              '',
              'python Validation/create_plots.py',
              '']

    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    print('wrote', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--condition', choices=list(CONDITIONS) + [NO_CR])
    parser.add_argument('--run-seed', type=int, default=0,
                        help='which training run of that condition to evaluate')
    parser.add_argument('--delay-mean', type=float, default=30.0,
                        help='mean pilot response time in seconds; 0 is the undelayed world')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help='how many of the fixed scenarios to fly, from the start of the set')
    parser.add_argument('--out', default=None, help='override the output CSV path')
    parser.add_argument('--commands', action='store_true',
                        help='write "Terminal commands.txt" instead of evaluating')
    args = parser.parse_args()

    if args.commands:
        write_commands()
        return
    if args.condition is None:
        parser.error('--condition is required (or pass --commands)')

    path = args.out or output_name(args.condition, args.run_seed, args.delay_mean)
    print(f'{args.condition} seed {args.run_seed}  {args.delay_mean:g} s  '
          f'{args.episodes} episodes -> {os.path.basename(path)}', flush=True)
    evaluate(args.condition, args.run_seed, args.delay_mean, BASE_SEEDS[:args.episodes], path)


if __name__ == '__main__':
    main()
