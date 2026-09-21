"""Phase 1: fly one trained policy in one delay world and write one row per episode.

Run from the MAIN folder:

python Validation/validation.py --condition lognormal --run-seed 2 --delay-law lognormal --delay-mean 30
python Validation/validation.py --commands      # write "Terminal commands.txt"
"""

import argparse
import csv
import glob
import itertools
import json
import os
import pickle
import zipfile
import sys
import time
from random import Random

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.config import N_ACTIONS, TRAINING_SEEDS, VALIDATION_SEEDS

# -- The experiment ------------------------------------------------------------

# Models live inside MAIN, at the path Training/train.py writes them to. Output lives OUTSIDE
# MAIN, beside it: MAIN is replaced wholesale on deployment, output is not.
MAIN_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT        = os.path.abspath(os.path.join(MAIN_DIR, '..'))
RUNS_ROOT   = os.path.join(MAIN_DIR, 'Models')      # where Training/train.py writes
RESULTS_DIR = os.path.join(ROOT, 'Results')
# Beside Results rather than inside it, so that globbing Results never picks the merge up.
COMBINED_CSV = os.path.join(ROOT, 'all_episodes.csv')
FIGURES_DIR = os.path.join(ROOT, 'Figures')
TABLES_DIR  = os.path.join(ROOT, 'Tables')

# Training condition -> the run directory Training/train.py wrote it to.
CONDITIONS = {
    'none':          'none',
    'deterministic': 'deterministic_30s',
    'lognormal':     'lognormal_30s',
}

NO_CR       = 'no_cr'      # a condition name, so it must survive a command line unquoted
NO_CR_LABEL = 'no CR'

RANDOM       = 'random'    # the untrained reference: one action drawn uniformly per step
RANDOM_LABEL = 'random actions'
RANDOM_SEED  = 0           # fixed, so a random-reference run repeats exactly

# Neither reference has training runs, so both are flown at run seed 0.
REFERENCES = (NO_CR, RANDOM)

COLOURS = {'none':          'tab:blue',
           'deterministic': 'tab:red',
           'lognormal':     'tab:green',
           RANDOM:          'tab:orange'}

LABELS = {'none':          'trained without delay',
          'deterministic': 'trained with 30 s deterministic delay',
          'lognormal':     'trained with 30 s lognormal delay',
          RANDOM:          RANDOM_LABEL}

# The test worlds: the response law and its mean. The undelayed world is shared by both laws --
# at a mean of 0 there is nothing to distribute -- so it is flown once and plotted on both curves.
DELAY_MEANS_S = [15, 30, 45, 60]
DELAY_LAWS    = ['lognormal', 'deterministic']

NO_DELAY = ('none', 0)

DELAY_LEVELS = [NO_DELAY] + [(law, mean) for law in DELAY_LAWS for mean in DELAY_MEANS_S]

# The means each law is drawn at, with the shared undelayed world at the front of both curves.
def levels_for(law):
    return [NO_DELAY] + [(law, mean) for mean in DELAY_MEANS_S]

# The held-out scenarios, named directly: every policy at every delay level flies exactly these.
BASE_SEEDS = list(VALIDATION_SEEDS)
EPISODES   = len(BASE_SEEDS)

TERMINALS     = 24
COMMANDS_FILE = os.path.join(os.path.dirname(__file__), 'Terminal commands.txt')

HOLD = 3


# -- One evaluation run --------------------------------------------------------

def checkpoint_steps(path):
    """The step a checkpoint was written at, read from the zip without loading the policy."""
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read('data').decode())['num_timesteps'])


def find_model(condition, run_seed):
    """The FURTHEST-TRAINED policy of one run, whichever file holds it.

    Training/train.py writes final_model from a finally block, so it is written on a crash or a
    Ctrl-C as well as on a clean finish. A run interrupted and then resumed therefore leaves
    final_model at the step it was interrupted on and last_model far beyond it: the name says
    nothing about which is later, so the step count in the checkpoint decides.
    """
    # An exact path, not a glob: Training/train.py writes exactly one directory per
    # (delay type, seed), so there is nothing to disambiguate.
    folder  = CONDITIONS[condition]
    run_dir = os.path.join(RUNS_ROOT, folder, f'{folder}_seed{run_seed}')

    found = [(checkpoint_steps(path), path)
             for path in (os.path.join(run_dir, name)
                          for name in ('final_model.zip', 'last_model.zip'))
             if os.path.exists(path)]
    if not found:
        sys.exit(f'no model for {condition} seed {run_seed}: nothing in {run_dir}')

    steps, path = max(found)
    if len(found) > 1:
        print(f'  {os.path.basename(path)} @ {steps:,} steps '
              f'(over {min(found)[1] and os.path.basename(min(found)[1])} '
              f'@ {min(found)[0]:,})', flush=True)
    return path


class _Unpickler(pickle.Unpickler):
    """Reads a checkpoint written with a different numpy version."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(model_path):
    """(model, mean, std, clip) for normalising observations."""
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
    """What to do this step: HOLD without a policy, a uniform draw for the random reference."""
    if policy is None:
        return HOLD
    if isinstance(policy, Random):
        return policy.randrange(N_ACTIONS)
    model, mean, std, clip = policy
    normalised = np.clip((observation - mean) / std, -clip, clip)
    action, _ = model.predict(normalised, deterministic=True)
    return int(np.asarray(action).flat[0])


def make_env(delay_law, delay_mean_s):
    """The test world: one response law at one mean. A mean of 0 is the undelayed world."""
    from Environment import AirspaceEnv

    if delay_law == 'none' or delay_mean_s == 0:
        return AirspaceEnv(delay_mode='none')
    return AirspaceEnv(delay_mode=delay_law, delay_mean_s=float(delay_mean_s))


def run_episode(env, policy, scenario_seed):
    """Fly one named scenario and return its end-of-episode numbers."""
    observation, _ = env.reset(options={'scenario_seed': scenario_seed})
    while True:
        observation, _, _, truncated, info = env.step(pick_action(policy, observation))
        if truncated:
            return {key: value for key, value in info.items()
                    if isinstance(value, (int, float))}


def evaluate(condition, run_seed, delay_law, delay_mean_s, base_seeds, path):
    """Fly the fixed scenario set in one delay world and write it to one CSV."""
    if condition == NO_CR:
        policy = None                       # HOLD every step: no advisory is ever transmitted
    elif condition == RANDOM:
        policy = Random(RANDOM_SEED)        # seeded once per run, so the run repeats exactly
    else:
        policy = load_policy(find_model(condition, run_seed))
    env = make_env(delay_law, delay_mean_s)

    started = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='') as handle:
        writer = None
        for episode, base_seed in enumerate(base_seeds):
            summary = run_episode(env, policy, base_seed)
            row = {'condition': condition, 'run_seed': run_seed,
                   'delay_law': delay_law, 'delay_mean_s': delay_mean_s, 'episode': episode,
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


def combine(path=COMBINED_CSV):
    """Every per-episode row Phase 1 wrote, in one file.

    Phase 1 keeps one CSV per evaluation run rather than appending to a shared file: two dozen
    terminals writing to one handle would interleave their rows, and a run that died could not
    then be repeated on its own. The merge happens here instead, once Phase 1 has finished.
    """
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.csv')))
    if not paths:
        sys.exit(f'no result CSVs in {RESULTS_DIR}; run Phase 1 first')

    episodes = 0
    with open(path, 'w', newline='') as out:
        writer = None
        for source in paths:
            with open(source, newline='') as handle:
                for row in csv.DictReader(handle):
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    episodes += 1

    print(f'{episodes:,} episodes from {len(paths)} runs -> {path}', flush=True)


def output_name(condition, run_seed, delay_law, delay_mean_s):
    """The CSV one evaluation run writes to."""
    if condition == NO_CR:
        return os.path.join(RESULTS_DIR, 'no_cr.csv')
    if condition == RANDOM:
        return os.path.join(RESULTS_DIR, f'random_{delay_law}_{delay_mean_s:g}s.csv')
    return os.path.join(RESULTS_DIR,
                        f'{condition}_s{run_seed}_{delay_law}_{delay_mean_s:g}s.csv')


# -- The sweep, dealt over the terminals ---------------------------------------

def grid():
    """Every evaluation run: the trained sweep, the random reference over it, and no CR once."""
    cells = [(condition, seed, law, mean)
             for condition, seed, (law, mean)
             in itertools.product(CONDITIONS, TRAINING_SEEDS, DELAY_LEVELS)]

    # The random reference transmits advisories, so the response delay changes what it flies and
    # it is swept like a trained model. No CR transmits nothing at all, so the delay never
    # applies and every world would repeat the same episode; it is flown once.
    cells += [(RANDOM, 0, law, mean) for law, mean in DELAY_LEVELS]
    cells.append((NO_CR, 0) + NO_DELAY)
    return cells


def command(condition, run_seed, delay_law, delay_mean_s):
    return (f'python Validation/validation.py --condition {condition} '
            f'--run-seed {run_seed} --delay-law {delay_law} --delay-mean {delay_mean_s:g}')


def share(cells):
    """The cells dealt round-robin, so every terminal gets the same amount of work."""
    return [cells[terminal::TERMINALS] for terminal in range(TERMINALS)]


def write_commands(path=COMMANDS_FILE):
    """The two phases, written out command by command."""
    cells = grid()
    lines = ['PHASE 1 -- VALIDATION',
             f'{len(cells)} evaluation runs of {EPISODES} episodes, over {TERMINALS} terminals.',
             'Run every terminal from the MAIN folder. They are independent and may be',
             'started in any order.',
             '']

    for terminal, work in enumerate(share(cells), start=1):
        lines.append(f'terminal {terminal} ({len(work)} runs):')
        lines += [command(*cell) for cell in work]
        lines.append('')

    lines += ['',
              'PHASE 1b -- MERGE',
              'One terminal, after every run of Phase 1 has finished. It gathers every',
              'per-episode CSV into one file beside Results.',
              '',
              'python Validation/validation.py --combine',
              '',
              '',
              'PHASE 2 -- PLOTTING',
              'One terminal, after every run of Phase 1 has finished. It reports what is on',
              'disk, then writes the figures; it stops with a list of missing runs if Phase 1',
              'is incomplete.',
              '',
              'python Validation/create_plots.py',
              '']

    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    print('wrote', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--condition', choices=list(CONDITIONS) + list(REFERENCES))
    parser.add_argument('--run-seed', type=int, default=0,
                        help='which training run of that condition to evaluate')
    parser.add_argument('--delay-law', choices=['none'] + DELAY_LAWS, default='lognormal',
                        help='the response law of the test world (default lognormal)')
    parser.add_argument('--delay-mean', type=float, default=30.0,
                        help='mean pilot response time in seconds; 0 is the undelayed world')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help='how many of the fixed scenarios to fly, from the start of the set')
    parser.add_argument('--out', default=None, help='override the output CSV path')
    parser.add_argument('--commands', action='store_true',
                        help='write "Terminal commands.txt" instead of evaluating')
    parser.add_argument('--combine', action='store_true',
                        help='merge every Phase 1 CSV into one file instead of evaluating')
    args = parser.parse_args()

    if args.commands:
        write_commands()
        return
    if args.combine:
        combine()
        return
    if args.condition is None:
        parser.error('--condition is required (or pass --commands)')

    law = 'none' if args.delay_mean == 0 else args.delay_law
    path = args.out or output_name(args.condition, args.run_seed, law, args.delay_mean)
    print(f'{args.condition} seed {args.run_seed}  {law} {args.delay_mean:g} s  '
          f'{args.episodes} episodes -> {os.path.basename(path)}', flush=True)
    evaluate(args.condition, args.run_seed, law, args.delay_mean,
             BASE_SEEDS[:args.episodes], path)


if __name__ == '__main__':
    main()
