# PPO trainer for the action-response delay experiment. One arm per process:
#
#   python -m Training.v4_train --delay none
#   python -m Training.v4_train --delay deterministic
#   python -m Training.v4_train --delay lognormal
#   python -m Training.v4_train --delay probabilistic
#
# All arms launched with the same --seed see identical scenarios (see env.reset).

import os

# Must precede the torch import. The policy is a 64x64 MLP, far too small to benefit from
# intra-op threading, and one OpenMP pool per worker process would spend more time in
# spin-waits than in the network.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4 import AirspaceEnv, DELAY_MODES

# -- Settings ------------------------------------------------------------------

# The BlueSky-Gym benchmark (Groot et al., SID 2024) found 2M too few for PPO to converge
# on simpler conflict-resolution environments, hence a large budget. Pilot at 10M first.
TOTAL_TIMESTEPS = 50_000_000

# Parallel envs per arm. Keep N_ENVS x (arms running at once) at or below the machine's
# physical core count.
N_ENVS     = 2
N_STEPS    = 4096                 # per env; rollout = 2 x 4096 = 8192 steps
BATCH_SIZE = 512                  # 8192 / 512 = 16 minibatches per epoch

GAMMA    = 0.995
ENT_COEF = 0.01

# Deterministic evaluation, and the cadence at which a new best can be saved. The eval runs
# serially in this process while the workers idle, so it is not free: 5 episodes of
# 2100-4800 steps is ~17,000 single-worker steps (~90 s), against 100,000 training steps
# across 2 workers (~12 min). That is roughly 12% overhead for an eval point every 100k
# instead of every 250k. Lower --eval-every further and the eval starts to cost more than
# the training it interrupts; drop --eval-episodes first if you need it tighter than this.
# Each eval writes best_model if the score improved and last_model unconditionally -- a 50M
# arm runs for days, and a crash with only an early best_model on disk would be painful.
EVAL_EVERY     = 100_000
EVAL_SEEDS     = (10_001, 10_002, 10_003, 10_004, 10_005)
PROGRESS_EVERY = 50_000

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'experiments'))

ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'return', 'spd+', 'spd-']

# Episode-summary key -> TensorBoard tag, shared by the training and eval logs.
# Deliberately short. _episode_summary carries more (flight hours, engagement branch,
# focus hold); Validation/delay_diagnostics.py reports those rather than cluttering
# TensorBoard with numbers that barely move.
METRICS = [
    ('ep_reward_total',      'episode/reward_total'),

    # Safety, normalised for traffic: episodes differ in aircraft count, sector size and
    # length, so a raw LoS count is not comparable between them.
    ('ep_los_events_per_fh', 'safety/los_events_per_flight_hour'),

    # Route keeping, scored over MANOEUVRED aircraft only -- see env._score_arrival.
    ('ep_arrival_rate',      'route/arrival_rate'),
    ('ep_exit_deviation_nm', 'route/exit_deviation_nm'),

    # What the pilots actually did, and what the controller wasted.
    ('ep_delay_mean_s',      'delay/mean_response_s'),
    ('ep_discarded',         'delay/instructions_discarded'),
]


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


class EvaluateAndCheckpoint(BaseCallback):
    """Measure what the agent can ACTUALLY do, and keep the best policy seen.

    PPO's policy is stochastic, so the episode/ curves include exploratory actions. These
    rollouts use predict(deterministic=True) on the fixed EVAL_SEEDS scenarios -- the same
    episodes every time, in every arm -- and log under eval/.

    Note for the write-up: best_model maximises a small-sample score, so its eval number is
    optimistically biased. Report final_model on a larger held-out set instead.
    """

    def __init__(self, eval_env, save_dir, seeds, every):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.seeds    = seeds
        self.every    = every
        self.last_eval = 0
        self.best = -float('inf')

    def _save(self, name):
        """Model + its VecNormalize stats. The two must travel together."""
        stem = os.path.join(self.save_dir, name)
        self.model.save(stem)
        self.model.get_env().save(stem + '_vecnorm.pkl')

    def _one_episode(self, seed):
        """A full deterministic episode, observations normalised with the training stats."""
        vecnorm = self.model.get_env()
        obs, _ = self.eval_env.reset(seed=seed)
        while True:
            action, _ = self.model.predict(vecnorm.normalize_obs(obs.reshape(1, -1)),
                                           deterministic=True)
            obs, _, _, truncated, _ = self.eval_env.step(int(np.asarray(action).flat[0]))
            if truncated:
                return self.eval_env._episode_summary()

    def _on_step(self):
        if not self.every or self.num_timesteps - self.last_eval < self.every:
            return True
        self.last_eval = self.num_timesteps

        runs = [self._one_episode(seed) for seed in self.seeds]
        for key, tag in METRICS:
            self.logger.record(f'eval/{tag}', sum(r[key] for r in runs) / len(runs))

        score = sum(r['ep_reward_total'] for r in runs) / len(runs)
        if score > self.best:
            self.best = score
            self._save('best_model')
            note = 'NEW BEST, saved'
        else:
            note = f'best {self.best:>9.1f}'

        self._save('last_model')          # crash insurance, independent of the score
        print(f'  eval @ {self.num_timesteps:>10,}   reward {score:>9.1f}   {note}', flush=True)
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

def train(delay_mode, seed, total_timesteps, n_envs, eval_seeds, eval_every):
    run_name = f'v4_{delay_mode}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}'
    run_dir  = os.path.join(RUNS_ROOT, delay_mode, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # delay_mode travels via env_kwargs so it reaches the worker processes; editing CONFIG
    # here would not survive the spawn.
    venv = make_vec_env(AirspaceEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs={'delay_mode': delay_mode})
    env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)

    # verbose=0: the Progress callback prints a compact line instead of SB3's full table.
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, gamma=GAMMA, ent_coef=ENT_COEF)

    eval_env  = AirspaceEnv(delay_mode=delay_mode)   # lives here, alone with BlueSky
    callbacks = CallbackList([
        Progress(), LogEpisodes(),
        EvaluateAndCheckpoint(eval_env, run_dir, eval_seeds, eval_every)])

    print(f'{delay_mode}  seed {seed}  {total_timesteps:,} steps  {n_envs} envs  '
          f'eval every {eval_every:,} x{len(eval_seeds)} eps  -> {run_dir}', flush=True)
    try:
        model.learn(total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        env.save(os.path.join(run_dir, 'final_vecnorm.pkl'))
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train one arm of the delay experiment.')
    parser.add_argument('--delay', required=True, choices=list(DELAY_MODES),
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed; all arms at the same seed share their scenarios')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help=f'training steps (default {TOTAL_TIMESTEPS:,})')
    parser.add_argument('--n-envs', type=int, default=N_ENVS,
                        help=f'parallel environments, i.e. cores used (default {N_ENVS})')
    parser.add_argument('--eval-episodes', type=int, default=len(EVAL_SEEDS),
                        help=f'held-out episodes per evaluation (max {len(EVAL_SEEDS)})')
    parser.add_argument('--eval-every', type=int, default=EVAL_EVERY,
                        help=f'steps between evaluations (default {EVAL_EVERY:,}); '
                             f'0 disables evaluation and checkpointing entirely')
    args = parser.parse_args()

    train(args.delay, args.seed, args.timesteps, args.n_envs,
          EVAL_SEEDS[:max(1, args.eval_episodes)], args.eval_every)


if __name__ == '__main__':
    main()
