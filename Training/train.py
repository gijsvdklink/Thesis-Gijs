# PPO trainer for the response-delay experiment, one delay type per process: python -m Training.train --delay none|deterministic|lognormal|geometric. The same --seed sees identical scenarios in every type.

import os

# Must precede the torch import: a 64x64 MLP gains nothing from intra-op threading, and one OpenMP pool per worker would spin-wait more than it computes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
import time
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecMonitor,
                                              VecNormalize)

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.main import AirspaceEnv, DELAY_MODES
from Environments.main.atco import MEAN_DELAY_S as DEFAULT_MEAN_S

# -- Settings ------------------------------------------------------------------

# A large budget: the BlueSky-Gym benchmark (Groot et al., SID 2024) found 2M too few for PPO to converge. Pilot at 10M first.
TOTAL_TIMESTEPS = 50_000_000

# Seed stream spacing between workers: worker r of run S uses stream S * WORKER_SEEDS + r,
# so no two workers and no two runs share traffic. Any n_envs below this cannot collide.
WORKER_SEEDS = 10_000

# Parallel envs per delay type. Keep N_ENVS x (types running at once) at or below the physical core count.
N_ENVS     = 2
N_STEPS    = 4096                 # per env; rollout = 2 x 4096 = 8192 steps
BATCH_SIZE = 512                  # 8192 / 512 = 16 minibatches per epoch

GAMMA    = 0.995
ENT_COEF = 0.01

# No evaluation during training: the reported policy is the one training ended on, scored afterwards by Validation/validation.py on VALIDATION_SEEDS. What is left is a periodic save against a crash.
SAVE_EVERY     = 500_000
PROGRESS_EVERY = 50_000

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'experiments'))

ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'return', 'spd+', 'spd-']

# Episode-summary key -> TensorBoard tag, deliberately short; the evaluation CSVs keep the rest.
METRICS = [
    ('ep_reward_total',      'episode/reward_total'),

    # Safety: the per-flight-hour rate is the comparable one, the raw count the easier one to sanity check.
    ('ep_los_events_per_fh', 'safety/los_events_per_flight_hour'),
    ('ep_los_events',        'safety/los_events'),

    # Route keeping, over every aircraft that leaves; drift is over all airborne traffic, every step.
    ('ep_arrival_rate',      'route/arrival_rate'),
    ('ep_exit_deviation_nm', 'route/exit_deviation_nm'),
    ('ep_mean_drift_deg',    'route/mean_drift_deg'),

    # Instruction load, split by kind and normalised by traffic.
    ('ep_turns_per_fh',         'actions/heading_changes_per_flight_hour'),
    ('ep_speed_changes_per_fh', 'actions/speed_changes_per_flight_hour'),

    # What the pilots actually did, and what the controller wasted.
    ('ep_delay_mean_s',      'delay/mean_response_s'),
    ('ep_discarded',         'delay/advisories_discarded'),
    ('ep_repeats',           'delay/advice_re_selected'),
]


# -- Saving --------------------------------------------------------------------

def save(model, run_dir, name):
    """A checkpoint and its VecNormalize statistics, which must travel together as <name>_vecnorm.pkl."""
    model.save(os.path.join(run_dir, name))
    model.get_env().save(os.path.join(run_dir, f'{name}_vecnorm.pkl'))


# -- Callbacks -----------------------------------------------------------------

class LogEpisodes(BaseCallback):
    """Log each finished training episode. These come from the EXPLORING policy."""

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'ep_reward_total' not in info:
                continue
            for key, tag in METRICS:
                self.logger.record_mean(tag, info[key])

            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(ACTION_LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
        return True


class Checkpoint(BaseCallback):
    """Write last_model every `every` steps, so a crash does not cost the whole run."""

    def __init__(self, save_dir, every):
        super().__init__()
        self.save_dir  = save_dir
        self.every     = every
        self.last_save = 0

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
        self.last = 0
        self.total = self.locals.get('total_timesteps', TOTAL_TIMESTEPS)

    def _on_step(self):
        if self.num_timesteps - self.last < PROGRESS_EVERY:
            return True
        self.last = self.num_timesteps
        elapsed = time.time() - self.t0
        rate    = self.num_timesteps / max(elapsed, 1e-9)
        eta_h   = (self.total - self.num_timesteps) / max(rate, 1e-9) / 3600
        print(f'{self.num_timesteps:>10,} / {self.total:,} '
              f'({100 * self.num_timesteps / self.total:5.1f}%)   '
              f'{rate:6.1f} steps/s   elapsed {elapsed / 3600:5.2f} h   '
              f'eta {eta_h:5.1f} h', flush=True)
        return True


# -- Training ------------------------------------------------------------------

def delay_type_name(delay_mode, delay_mean_s):
    """Directory name for one delay type, e.g. 'lognormal_30s'; the baseline stays plain 'none'."""
    return 'none' if delay_mode == 'none' else f'{delay_mode}_{delay_mean_s:g}s'


def train(delay_mode, seed, total_timesteps, n_envs, save_every, delay_mean_s,
          runs_root=RUNS_ROOT):
    delay_type = delay_type_name(delay_mode, delay_mean_s)
    run_name = f'{delay_type}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}'
    run_dir  = os.path.join(runs_root, delay_type, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Everything travels through the constructor so it reaches the worker PROCESSES; a CONFIG
    # edit here would not survive the spawn. A given --seed draws the same scenarios in every
    # delay type, which is what makes the delay the only variable between conditions.
    def make_worker(rank):
        def _init():
            return AirspaceEnv(delay_mode=delay_mode, delay_mean_s=delay_mean_s,
                               seed=seed * WORKER_SEEDS + rank)
        return _init

    # One environment does not need a worker process: SubprocVecEnv would pipe every step for no parallelism.
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = vec_env_cls([make_worker(rank) for rank in range(n_envs)])
    env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)

    # verbose=0: the Progress callback prints a compact line instead of SB3's full table.
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, gamma=GAMMA, ent_coef=ENT_COEF)

    callbacks = CallbackList([Progress(), LogEpisodes(), Checkpoint(run_dir, save_every)])

    print(f'{delay_type}  seed {seed}  {total_timesteps:,} steps  {n_envs} envs  '
          f'save every {save_every:,}  -> {run_dir}', flush=True)
    try:
        model.learn(total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        # Also written on Ctrl-C, so a hand-stopped run leaves the policy at the exact step it stopped on.
        save(model, run_dir, 'final_model')
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train one delay type.')
    parser.add_argument('--delay', required=True, choices=list(DELAY_MODES),
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed; all delay types at the same seed share their scenarios')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help=f'training steps (default {TOTAL_TIMESTEPS:,})')
    parser.add_argument('--n-envs', type=int, default=N_ENVS,
                        help=f'parallel environments, i.e. cores used (default {N_ENVS})')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                        help=f'steps between last_model checkpoints '
                             f'(default {SAVE_EVERY:,}); 0 saves only at the end')
    parser.add_argument('--runs-root', default=RUNS_ROOT,
                        help='where the run directory is created, so a new set of models '
                             'can sit beside an old one (default Runs_saved/experiments)')
    # --delay-first is the old spelling, kept so existing run scripts still work.
    parser.add_argument('--delay-mean', '--delay-first', dest='delay_mean',
                        type=float, default=DEFAULT_MEAN_S,
                        help=f'delay magnitude: the MEAN pilot response time in seconds '
                             f'(default {DEFAULT_MEAN_S:g}). Every advisory is drawn from '
                             f'this distribution. Ignored when --delay none.')
    args = parser.parse_args()

    train(args.delay, args.seed, args.timesteps, args.n_envs,
          args.save_every, args.delay_mean, args.runs_root)


if __name__ == '__main__':
    main()
