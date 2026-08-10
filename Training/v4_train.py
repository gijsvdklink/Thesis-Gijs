"""
PPO training for one arm of the action-response delay experiment.

Three arms, one environment each, stock PPO. Run them in three terminals with the
same seed, so the only difference between the curves is the delay condition:

    python -m Training.v4_train --delay none          --seed 42
    python -m Training.v4_train --delay deterministic --seed 42
    python -m Training.v4_train --delay probabilistic --seed 42
    tensorboard --logdir Runs_saved/experiments

PPO runs on stable-baselines3 DEFAULTS -- nothing is tuned here. Two things are not
negotiable, though, and both are about correctness rather than performance:

1. VecNormalize(norm_obs=True). The environment emits raw NM / kt / s / rad, so this
   is the only thing standardising the inputs. The *_vecnorm.pkl saved beside each
   model MUST be loaded again at evaluation or visualisation time.

2. The training environment runs in a SUBPROCESS even though there is only one of
   them. BlueSky's traffic is a process-global singleton, so an evaluation episode
   in this process would call bs.traf.reset() and wipe the training environment's
   aircraft mid-run. The subprocess keeps the two apart.

Throughput is roughly 44 steps/s: stock PPO does 10 epochs x 32 minibatches = 320
gradient updates per 2048 environment steps, which costs more than the simulation.
Budget about 6 hours per million steps and set TOTAL_TIMESTEPS accordingly.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4 import AirspaceEnv

# -- Settings ------------------------------------------------------------------

TOTAL_TIMESTEPS = 5_000_000       # ~32 h at 44 steps/s
# Deterministic evaluation. Two full episodes cost roughly a minute, so at 44 steps/s
# this cadence spends about 9% of the run on measurement and gives ~200 eval points.
# Lower it for a finer curve, at proportionally more overhead.
EVAL_EVERY      = 25_000
EVAL_SEEDS      = (10_001, 10_002)   # held-out scenarios, identical for all three arms
# best_model is written the moment a new best is measured, and re-written at least this
# often regardless, so the file on disk is never more than SAVE_EVERY steps stale.
SAVE_EVERY      = 100_000
PROGRESS_EVERY  = 10_000          # throughput line in the terminal

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'experiments'))

ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'return', 'spd+', 'spd-']

# Episode-summary key -> TensorBoard tag, shared by the training and eval logs.
METRICS = [('ep_reward_total', 'reward_total'), ('ep_length', 'length'),
           ('ep_los_events', 'los_events'), ('ep_los_steps', 'los_steps'),
           ('ep_arrival_rate', 'arrival_rate')]


# -- Callbacks -----------------------------------------------------------------

class LogEpisodes(BaseCallback):
    """Log each finished training episode. These come from the EXPLORING policy."""

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'ep_reward_total' not in info:
                continue
            for key, tag in METRICS:
                self.logger.record_mean(f'episode/{tag}', info[key])

            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(ACTION_LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
        return True


class EvaluateAndCheckpoint(BaseCallback):
    """Every EVAL_EVERY steps, measure what the agent can ACTUALLY do, and keep the
    best policy seen.

    PPO's policy is stochastic, so the episode/ curves include exploratory actions the
    agent would not choose if asked for its best guess. These rollouts use
    predict(deterministic=True) on held-out scenarios and log under eval/ -- expect them
    to sit above the training curves. best_model is checkpointed on this measured score
    rather than on the noisy training reward.
    """

    def __init__(self, eval_env, save_dir):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.last_eval = 0
        self.last_save = 0
        self.best = -float('inf')

    def _save_best(self):
        """Write best_model + its VecNormalize stats, overwriting the previous pair.
        The two must travel together: the policy is meaningless without the obs stats
        it was trained under."""
        stem = os.path.join(self.save_dir, 'best_model')
        self.model.save(stem)
        self.model.get_env().save(stem + '_vecnorm.pkl')
        self.last_save = self.num_timesteps

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
        if not EVAL_EVERY or self.num_timesteps - self.last_eval < EVAL_EVERY:
            return True
        self.last_eval = self.num_timesteps

        runs = [self._one_episode(seed) for seed in EVAL_SEEDS]
        for key, tag in METRICS:
            self.logger.record(f'eval/{tag}', sum(r[key] for r in runs) / len(runs))

        score = sum(r['ep_reward_total'] for r in runs) / len(runs)
        if score > self.best:
            self.best = score
            self._save_best()
            note = 'NEW BEST, saved'
        elif self.num_timesteps - self.last_save >= SAVE_EVERY:
            self._save_best()                     # keep the on-disk copy fresh
            note = f'best {self.best:>9.1f}, re-saved'
        else:
            note = f'best {self.best:>9.1f}'
        print(f'  eval @ {self.num_timesteps:>10,}   reward {score:>9.1f}   {note}',
              flush=True)
        return True


class Progress(BaseCallback):
    """One line every PROGRESS_EVERY steps: how far along, how fast, how much longer."""

    def _on_training_start(self):
        self.t0 = time.time()
        self.last = 0

    def _on_step(self):
        if self.num_timesteps - self.last < PROGRESS_EVERY:
            return True
        self.last = self.num_timesteps
        elapsed = time.time() - self.t0
        rate    = self.num_timesteps / max(elapsed, 1e-9)
        eta_h   = (TOTAL_TIMESTEPS - self.num_timesteps) / max(rate, 1e-9) / 3600
        print(f'{self.num_timesteps:>10,} / {TOTAL_TIMESTEPS:,} '
              f'({100 * self.num_timesteps / TOTAL_TIMESTEPS:5.1f}%)   '
              f'{rate:6.1f} steps/s   elapsed {elapsed / 3600:5.2f} h   '
              f'eta {eta_h:5.1f} h', flush=True)
        return True


# -- Training ------------------------------------------------------------------

def train(delay_mode, seed):
    run_name = f'v4_{delay_mode}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}'
    run_dir  = os.path.join(RUNS_ROOT, delay_mode, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # One environment, but in a subprocess -- see the module docstring. delay_mode
    # travels via env_kwargs so it reaches the worker; editing CONFIG here would not.
    venv = make_vec_env(AirspaceEnv, n_envs=1, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs={'delay_mode': delay_mode})
    env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0)

    # verbose=0: the Progress callback below prints a compact throughput line instead
    # of SB3's full table after every 2048-step rollout.
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir)

    # The evaluation environment lives here in the main process, alone with BlueSky.
    eval_env = AirspaceEnv(delay_mode=delay_mode)
    callbacks = CallbackList([Progress(), LogEpisodes(),
                              EvaluateAndCheckpoint(eval_env, run_dir)])

    print(f'{delay_mode}  seed {seed}  {TOTAL_TIMESTEPS:,} steps  -> {run_dir}', flush=True)
    try:
        model.learn(TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        env.save(os.path.join(run_dir, 'final_vecnorm.pkl'))
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--delay', required=True,
                        choices=['none', 'deterministic', 'probabilistic'],
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: random)')
    args = parser.parse_args()
    train(args.delay, args.seed if args.seed is not None else random.randint(0, 99_999))


if __name__ == '__main__':
    main()
