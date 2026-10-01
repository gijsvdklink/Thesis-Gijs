import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import sys
import zipfile
import time
from collections import deque
from random import Random

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecMonitor,
                                              VecNormalize)

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment import AirspaceEnv, CONFIG, DELAY_MODES
from Environment.config import TRAINING_SEEDS
from Environment.stats import ADVISORY_LABELS

TREND_WINDOW = 200

STATS_WINDOW = 100


TOTAL_TIMESTEPS = 300_000_000

# THE one difference from MAIN2: four environments instead of one. The rollout is still
# 4096 transitions per update (4 x 1024), so the update SIZE matches MAIN2 and MAIN, but
# each update now spans four independent scenarios rather than one long trajectory.
# Correlated minibatches are the usual reason a single environment learns worse, and the
# price is that GAE truncates over four 1024-step segments instead of one of 4096 -- at
# gamma 0.995 the discount horizon is 200 steps, well inside 1024, so that cost is small.
N_ENVS     = 4
N_STEPS    = 1024         # rollout = N_ENVS x N_STEPS, so 4 x 1024 = 4096 per update
BATCH_SIZE = 512

GAMMA    = 0.995
ENT_COEF = 0.01

SAVE_EVERY     = 500_000
PROGRESS_EVERY = 50_000

RUNS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models'))

METRICS = [
    ('ep_reward_total',      'episode/reward_total'),
    ('ep_reward_per_fh',     'episode/reward_per_flight_hour'),

    ('ep_reward_los_per_fh',   'reward/los_per_flight_hour'),
    ('ep_reward_drift_per_fh', 'reward/drift_per_flight_hour'),
    ('ep_reward_work_per_fh',  'reward/work_per_flight_hour'),

    ('ep_los_events_per_fh', 'safety/los_events_per_flight_hour'),
    ('ep_conflicts_per_fh',  'safety/conflicts_per_flight_hour'),

    ('ep_path_ratio',        'route/distance_ratio'),
    ('ep_on_route_rate',     'route/on_route_exits'),

    ('ep_turns_per_fh',         'actions/heading_changes_per_flight_hour'),
    ('ep_advisories_per_fh',    'actions/advisories_per_flight_hour'),

    ('ep_delay_mean_s',      'delay/mean_response_s'),
    ('ep_discarded',         'delay/advisories_discarded'),
    ('ep_repeats',           'delay/advice_re_selected'),
]

# One row per advisory the action set actually contains. Hand-writing these breaks the
# moment the action set changes: this folder has no speed actions, so episode_summary
# never emits ep_speed_up_per_fh and a hard-coded row raises KeyError on the first
# finished episode. ADVISORY_LABELS is built from TURN_DELTAS and SPEED_ACTIONS, so it
# always matches whatever the environment can issue.
METRICS += [(f'ep_{label}_per_fh', f'advisory/{label}') for label in ADVISORY_LABELS.values()]


def save(model, run_dir, name):
    """A checkpoint and its VecNormalize statistics, which must travel together as <name>_vecnorm.pkl."""
    model.save(os.path.join(run_dir, name))
    model.get_env().save(os.path.join(run_dir, f'{name}_vecnorm.pkl'))


class LogEpisodes(BaseCallback):
    """Log each finished training episode. These come from the EXPLORING policy."""

    def __init__(self):
        super().__init__()
        self.recent  = deque(maxlen=TREND_WINDOW)
        self.windows = {key: deque(maxlen=STATS_WINDOW) for key, _ in METRICS}

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'ep_reward_total' not in info:
                continue
            for key, tag in METRICS:
                window = self.windows[key]
                window.append(info[key])
                self.logger.record(tag, sum(window) / len(window))

            self.recent.append((self.num_timesteps, info['ep_reward_total']))
            if len(self.recent) == TREND_WINDOW:
                mean_step   = sum(t for t, _ in self.recent) / TREND_WINDOW
                mean_reward = sum(r for _, r in self.recent) / TREND_WINDOW
                spread      = sum((t - mean_step) ** 2 for t, _ in self.recent)
                if spread > 0:
                    covariance = sum((t - mean_step) * (r - mean_reward)
                                     for t, r in self.recent)
                    self.logger.record('episode/reward_slope_per_1M',
                                       covariance / spread * 1e6)
        return True


class Checkpoint(BaseCallback):
    """Write last_model every `every` steps, so a crash does not cost the whole run."""

    def __init__(self, save_dir, every):
        super().__init__()
        self.save_dir  = save_dir
        self.every     = every
        self.last_save = 0

    def _on_training_start(self):
        self.last_save = self.num_timesteps

    def _on_step(self):
        if not self.every or self.num_timesteps - self.last_save < self.every:
            return True
        self.last_save = self.num_timesteps
        save(self.model, self.save_dir, 'last_model')
        print(f'  saved @ {self.num_timesteps:>10,}', flush=True)
        return True


class Progress(BaseCallback):
    """One line every PROGRESS_EVERY steps: how far along, how fast, how much longer."""

    def _on_training_start(self):
        self.t0 = time.time()
        self.last = self.num_timesteps
        self.start_steps = self.num_timesteps
        self.total = self.start_steps + self.locals.get('total_timesteps', TOTAL_TIMESTEPS)

    def _on_step(self):
        if self.num_timesteps - self.last < PROGRESS_EVERY:
            return True
        self.last = self.num_timesteps
        elapsed = time.time() - self.t0
        rate    = (self.num_timesteps - self.start_steps) / max(elapsed, 1e-9)
        eta_h   = (self.total - self.num_timesteps) / max(rate, 1e-9) / 3600
        print(f'{self.num_timesteps:>10,} / {self.total:,} '
              f'({100 * self.num_timesteps / self.total:5.1f}%)   '
              f'{rate:6.1f} steps/s   elapsed {elapsed / 3600:5.2f} h   '
              f'eta {eta_h:5.1f} h', flush=True)
        return True


def delay_type_name(delay_mode, delay_mean_s):
    """Directory name for one delay type, e.g. 'lognormal_30s'; the baseline stays plain 'none'."""
    return 'none' if delay_mode == 'none' else f'{delay_mode}_{delay_mean_s:g}s'


def train(delay_mode, seed, total_timesteps, n_envs, save_every, delay_mean_s,
          runs_root=RUNS_ROOT, overwrite=False, resume=False):
    delay_type = delay_type_name(delay_mode, delay_mean_s)
    run_dir = os.path.join(runs_root, delay_type, f'{delay_type}_seed{seed}')
    if resume:
        # The checkpoint that is FURTHEST ALONG, not the first name that happens to exist.
        # final_model is written when a run ends or is interrupted; last_model every
        # save_every steps. A run killed without unwinding (no KeyboardInterrupt, so the
        # finally block never ran) leaves final_model behind at an OLDER step count than
        # last_model. Preferring it by name then silently discards the newer weights, and
        # the next checkpoint overwrites them. Read the step count and take the larger.
        checkpoint, best_steps = None, -1
        for name in ('last_model', 'final_model'):
            zip_path = os.path.join(run_dir, f'{name}.zip')
            # A checkpoint without its VecNormalize statistics cannot be resumed from.
            if not (os.path.exists(zip_path)
                    and os.path.exists(os.path.join(run_dir, f'{name}_vecnorm.pkl'))):
                continue
            with zipfile.ZipFile(zip_path) as archive:
                steps = json.loads(archive.read('data').decode()).get('num_timesteps', -1)
            if steps > best_steps:
                checkpoint, best_steps = name, steps
        if checkpoint is None:
            sys.exit(f'--resume: no checkpoint with its vecnorm in {run_dir}')
    elif os.path.exists(run_dir) and not overwrite:
        sys.exit(f'{run_dir} already exists. Delete it, move it aside, or pass --overwrite '
                 f'to train over it.')
    os.makedirs(run_dir, exist_ok=True)

    # Every worker needs its OWN scenario stream. Constructed from the run seed alone, all
    # n_envs workers would draw the same episodes and the rollout would be n_envs copies of
    # one trajectory. The streams are drawn in order from the run seed, so worker k always
    # gets the same stream whatever n_envs is, and --seed still fixes the whole run.
    stream       = Random(seed)
    worker_seeds = [stream.randrange(2 ** 63) for _ in range(n_envs)]

    def make_worker(worker_seed):
        def _init():
            return AirspaceEnv(delay_mode=delay_mode, delay_mean_s=delay_mean_s,
                               seed=worker_seed)
        return _init

    factories = [make_worker(ws) for ws in worker_seeds]

    # Subprocesses only pay for themselves once there is more than one environment; a single
    # worker stays in-process and skips the pickling and IPC altogether.
    venv = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories)
    if resume:
        env = VecNormalize.load(os.path.join(run_dir, f'{checkpoint}_vecnorm.pkl'),
                                VecMonitor(venv))
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)

    if resume:
        model = PPO.load(os.path.join(run_dir, checkpoint), env=env,
                         tensorboard_log=run_dir, device='cpu')
        print(f'resumed from {checkpoint} at {model.num_timesteps:,} steps', flush=True)
    else:
        model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                    n_steps=N_STEPS, batch_size=BATCH_SIZE, gamma=GAMMA, ent_coef=ENT_COEF)

    callbacks = CallbackList([Progress(), LogEpisodes(), Checkpoint(run_dir, save_every)])

    print(f'{delay_type}  seed {seed}  {total_timesteps:,} steps  {n_envs} envs  '
          f'save every {save_every:,}  -> {run_dir}', flush=True)
    try:
        remaining = total_timesteps - model.num_timesteps if resume else total_timesteps
        if remaining <= 0:
            print(f'already at {model.num_timesteps:,} of {total_timesteps:,} steps; '
                  f'nothing to do', flush=True)
        else:
            model.learn(remaining, callback=callbacks, reset_num_timesteps=not resume)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        save(model, run_dir, 'final_model')
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train one delay type.')
    parser.add_argument('--delay', required=True, choices=list(DELAY_MODES),
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=TRAINING_SEEDS[0], choices=TRAINING_SEEDS,
                        help=f'which training run: one of {list(TRAINING_SEEDS)}. All delay '
                             f'types at the same seed share weights and scenarios.')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help=f'training steps (default {TOTAL_TIMESTEPS:,})')
    parser.add_argument('--n-envs', type=int, default=N_ENVS,
                        help=f'parallel environments (default {N_ENVS}); above 1 each runs '
                             f'in its own subprocess. The rollout is n_envs x N_STEPS '
                             f'({N_ENVS} x {N_STEPS} = {N_ENVS * N_STEPS}), so raising this '
                             f'multiplies the update size unless N_STEPS is lowered to match.')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                        help=f'steps between last_model checkpoints '
                             f'(default {SAVE_EVERY:,}); 0 saves only at the end')
    parser.add_argument('--overwrite', action='store_true',
                        help='train into an existing run directory instead of refusing')
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run from its last checkpoint, keeping '
                             'the step count and the TensorBoard curves continuous')
    parser.add_argument('--runs-root', default=RUNS_ROOT,
                        help='where the run directory is created, so a new set of models '
                             'can sit beside an old one (default Runs_saved/experiments)')
    default_mean_s = CONFIG['delay_mean_s']
    parser.add_argument('--delay-mean', '--delay-first', dest='delay_mean',
                        type=float, default=default_mean_s,
                        help=f'delay magnitude: the MEAN pilot response time in seconds '
                             f'(default {default_mean_s:g}). Every advisory is drawn from '
                             f'this distribution. Ignored when --delay none.')
    args = parser.parse_args()

    train(args.delay, args.seed, args.timesteps, args.n_envs,
          args.save_every, args.delay_mean, args.runs_root, args.overwrite, args.resume)


if __name__ == '__main__':
    main()
