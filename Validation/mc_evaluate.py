import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environments.v4 import AirspaceEnv
from Environments.v4.config import TEST_SEEDS
from Environments.v4.delays import DELAY_MODES, MEAN_DELAY_S

HOLD = 3
EPISODES = 200

# Everything this run produces lives under Validation/report_results: the per-episode CSVs
# in results/, the figures in figures/. Paths are anchored to the repository rather than to
# the working directory, so the scripts can be started from anywhere, and a bare file name
# on the command line lands in the right folder automatically.
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR  = os.path.join(ROOT, 'Validation', 'report_results')
RESULTS_DIR = os.path.join(REPORT_DIR, 'results')
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')


def in_folder(path, folder):
    """A bare name goes in `folder`; anything with a directory in it is left alone."""
    return path if os.path.dirname(path) else os.path.join(folder, path)



def parse_means(text):
    """'30' -> [30.0].  '15,45,60' -> [15.0, 45.0, 60.0]."""
    return [float(part) for part in text.split(',')]


class _Unpickler(pickle.Unpickler):
    """Reads a checkpoint written with a different numpy version."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(model_path):
    """(model, mean, std, clip) for normalising observations, or None for no CR.

    The saved VecNormalize is unpickled directly for its statistics rather than through
    VecNormalize.load, which insists on being attached to a live vectorised environment.
    All we need are the numbers the policy was trained with.
    """
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


def run_episode(env, policy, seed):
    """Fly one episode and return the environment's end-of-episode numbers.

    seed=None continues to the next episode in the pool; a seed restarts the sequence.
    """
    observation, _ = env.reset(seed=seed)
    while True:
        observation, _, _, truncated, info = env.step(pick_action(policy, observation))
        if truncated:
            return {key: value for key, value in info.items()
                    if isinstance(value, (int, float))}


def evaluate(env, policy, policy_name, delay, mean_s, episodes, first_seed, path):
    """Fly `episodes` episodes in one delay world and write them to one CSV."""
    env.delay_mean_s = mean_s          # reset() rebuilds the delay model from this
    started = time.time()

    with open(path, 'w', newline='') as handle:
        writer = None
        for episode in range(episodes):
            summary = run_episode(env, policy, first_seed if episode == 0 else None)
            row = {'policy': policy_name, 'delay': delay, 'delay_mean_s': mean_s,
                   'episode': episode, 'episode_seed': env.episode_seed,
                   'n_ac': env.n_aircraft, 'rho': env.rho, **summary}

            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()             # readable while the run continues

            done = episode + 1
            if done in (1, 5) or done % 10 == 0 or done == episodes:
                elapsed     = time.time() - started
                per_episode = elapsed / done
                left        = per_episode * (episodes - done)
                finish      = time.strftime('%H:%M', time.localtime(time.time() + left))
                print(f'  {done:>4}/{episodes}  {per_episode:5.1f} s/episode  '
                      f'{elapsed / 3600:4.1f} h gone  {left / 3600:4.1f} h to go  '
                      f'finish ~{finish}', flush=True)

    print('done ->', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=None,
                        help='PPO checkpoint; leave it out to evaluate no CR')
    parser.add_argument('--out', required=True,
                        help='output CSV; a bare name lands in Validation/report_results/results')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                        help=f'episodes to fly (default {EPISODES})')
    parser.add_argument('--delay', default='none', choices=list(DELAY_MODES))
    parser.add_argument('--delay-mean', default=str(MEAN_DELAY_S),
                        help='mean pilot response time in seconds. A comma list sweeps '
                             'several, writing <out>_<mean>s.csv for each')
    parser.add_argument('--seed', type=int, default=1,
                        help='where in the test pool the run starts; the same value gives '
                             'every policy the same episodes')
    args = parser.parse_args()

    if args.model and not os.path.exists(args.model):
        sys.exit(f'model not found: {args.model}')

    args.out = in_folder(args.out, RESULTS_DIR)

    means       = [0.0] if args.delay == 'none' else parse_means(args.delay_mean)
    policy_name = 'policy' if args.model else 'no CR'

    # One CSV per delay magnitude; a single magnitude keeps --out exactly as given.
    stem  = args.out[:-4] if args.out.endswith('.csv') else args.out
    files = {m: (args.out if len(means) == 1 else f'{stem}_{m:g}s.csv') for m in means}

    print(f'{policy_name}: {args.episodes} episodes per delay magnitude, seed {args.seed}')
    print(f'delay: {args.delay}'
          + ('' if args.delay == 'none' else
             f", mean {', '.join(f'{m:g}' for m in means)} s"))
    for path in files.values():
        print(f'out:   {path}')
    sys.stdout.flush()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    env    = AirspaceEnv(delay_mode=args.delay, seed_pool=TEST_SEEDS)
    policy = load_policy(args.model)

    for mean_s, path in files.items():
        evaluate(env, policy, policy_name, args.delay, mean_s,
                 args.episodes, args.seed, path)


if __name__ == '__main__':
    main()
