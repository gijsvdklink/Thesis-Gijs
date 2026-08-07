"""
PPO training for the v4 ATC environment (Environments/v4).

That environment emits RAW PHYSICAL UNITS (NM, kt, s, rad) with no hand-picked
normalisers and no clipping, so VecNormalize is not a convenience here -- it is the
only thing standardising the inputs. norm_obs=True is mandatory, and the *_vecnorm.pkl
saved next to each model MUST be loaded at eval/visualisation time. Feeding a policy
raw observations without it produces nonsense: ranges are in NM and speeds in kt, so
the inputs are two orders of magnitude apart before the network ever sees them.

Edit the SETTINGS / PPO_KWARGS below to configure a run.

Command-line flags:
  --delay MODE        none | deterministic | probabilistic  (required; THE experiment variable)
  --seed N            fix the random seed (default: random)
  --n-envs N          parallel environments (default N_ENVS below)
  --tag NAME          label added to the run directory name

Experiment 1 trains the same environment under three action-response delay conditions,
so keep the seed fixed across arms and vary only --delay:

  python -m Training.v4_train --delay none          --seed 42 --n-envs 8
  python -m Training.v4_train --delay deterministic --seed 42 --n-envs 8
  python -m Training.v4_train --delay probabilistic --seed 42 --n-envs 8

Monitor:   tensorboard --logdir Runs_saved/experiments
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from Environments.v4 import AirspaceEnv

# -- Settings ------------------------------------------------------------------

N_ENVS           = 48            # parallel environments (tune to the machine's cores)
TOTAL_TIMESTEPS  = 20_000_000

# Periodic DETERMINISTIC evaluation -- the agent's actual performance. PPO's policy is
# stochastic, so the episode/ curves come from an agent still sampling exploratory
# actions and understate what it can really do. These run predict(deterministic=True)
# on held-out scenarios and log under eval/. Set EVAL_EVERY = 0 to switch off.
EVAL_EVERY       = 1_000_000     # ~20 evaluation points over a 20M-step run
EVAL_SEEDS       = (10_001, 10_002)   # held-out scenarios, identical for all three arms
CROSS_EVAL_STEPS = 600           # short partial episode for the transfer check

RUNS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'experiments'))

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 256,         # rollout = n_steps * N_ENVS; sets the update/log frequency
    batch_size    = 2048,        # 6 minibatches x 10 epochs per rollout (large N_ENVS -> big rollout)
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    vf_coef       = 0.5,
    verbose       = 0,
    policy_kwargs = dict(net_arch=[128, 128]),
)

# -- Callbacks -----------------------------------------------------------------

class EpisodeStatsCallback(BaseCallback):
    """Log per-episode reward, separation losses, arrival rate and the action mix."""
    ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'return', 'spd+', 'spd-']

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record_mean('episode/mean_reward', info['mean_episode_reward'])
            self.logger.record_mean('episode/length',      info['ep_length'])

            self.logger.record_mean('safety/los_steps',    info['ep_los_steps'])
            self.logger.record_mean('safety/los_events',   info['ep_los_events'])

            # fraction of aircraft leaving the sector without drift (on route heading)
            self.logger.record_mean('arrival/rate',        info['ep_arrival_rate'])

            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self.ACTION_LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
            if len(dist) >= 10:
                # fractions, so they are comparable to the per-action fractions above
                turns = sum(dist) - dist[3] - dist[7] - dist[8] - dist[9]   # 6 stack-turns (excl. hold/return/speed)
                self.logger.record_mean('actions/turns_total', turns / total)
                self.logger.record_mean('actions/speed_total', (dist[8] + dist[9]) / total)
        return True


class EvalCallback(BaseCallback):
    """Every EVAL_EVERY steps, measure what the agent can ACTUALLY do.

    PPO's policy is stochastic: during training it samples from the action distribution,
    so the episode/ curves include exploratory actions the agent would not choose if
    asked for its best guess. These rollouts call predict(deterministic=True) instead --
    argmax, no exploration -- on scenarios never trained on, and log under eval/. Expect
    them to sit ABOVE the training curves.

    Three jobs, in one place because they share the evaluation environment:
      eval/...           full episodes in this run's own delay condition (the headline)
      cross/<mode>/...   short rollouts under the other conditions (the transfer question)
      best_model         checkpointed on the deterministic score, not the noisy training
                         reward, so "best" means best measured performance

    Uses ONE evaluation environment, re-used sequentially with its delay_mode swapped.
    BlueSky is a process-global singleton, so these must not run concurrently -- they are
    separated by a full reset(), and the SubprocVecEnv training workers are in other
    processes entirely.
    """
    TAGS = [('ep_reward_total', 'reward_total'), ('ep_los_events', 'los_events'),
            ('ep_los_steps', 'los_steps'), ('ep_arrival_rate', 'arrival_rate'),
            ('ep_length', 'length')]

    def __init__(self, save_path, seed, delay_mode):
        super().__init__()
        self.save_path  = save_path
        self.seed       = seed
        self.delay_mode = delay_mode
        self.last_step  = 0
        self.best       = -float('inf')
        self.env        = None

    def _rollout(self, mode, seed, max_steps=None):
        """One deterministic rollout; returns the episode summary. Runs to the episode's
        own truncation unless max_steps caps it short."""
        vecnorm = self.model.get_env()        # training VecNormalize; supplies obs stats
        self.env.delay_mode = mode
        obs, _ = self.env.reset(seed=seed)
        steps = 0
        while True:
            norm = vecnorm.normalize_obs(obs.reshape(1, -1))
            action, _ = self.model.predict(norm, deterministic=True)
            obs, _, _, truncated, _ = self.env.step(int(np.asarray(action).flat[0]))
            steps += 1
            if truncated or (max_steps and steps >= max_steps):
                return self.env._episode_summary()

    def _on_step(self):
        if not EVAL_EVERY or self.num_timesteps - self.last_step < EVAL_EVERY:
            return True
        self.last_step = self.num_timesteps
        if self.env is None:
            self.env = AirspaceEnv()          # first use: pays the BlueSky init once

        # 1. actual performance: full episodes under this run's own delay condition
        runs = [self._rollout(self.delay_mode, s) for s in EVAL_SEEDS]
        mean = lambda key: sum(r[key] for r in runs) / len(runs)
        for key, tag in self.TAGS:
            self.logger.record(f'eval/{tag}', mean(key))

        # 2. transfer: the same policy under every condition, short and comparable
        for mode in ('none', 'deterministic', 'probabilistic'):
            summary = self._rollout(mode, EVAL_SEEDS[0], max_steps=CROSS_EVAL_STEPS)
            for key, tag in self.TAGS:
                self.logger.record(f'cross/{mode}/{tag}', summary[key])

        # 3. checkpoint on the measured score rather than the exploring one
        score = mean('ep_reward_total')
        if score <= self.best:
            print(f'[{self.seed}] {self.num_timesteps:,}  eval_reward={score:.1f} '
                  f'(best {self.best:.1f}, not saved)', flush=True)
            return True

        self.best = score
        stem = os.path.join(self.save_path, 'best_model')
        self.model.save(stem)
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(stem + '_vecnorm.pkl')
        print(f'[{self.seed}] {self.num_timesteps:,}  NEW BEST eval_reward={score:.1f} '
              f'-> saved best_model', flush=True)
        return True


class ProgressCallback(BaseCallback):
    """Print a throughput line every 10k steps."""
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_print = 0
        print(f'[{self.seed}] target {TOTAL_TIMESTEPS:,} steps', flush=True)

    def _on_step(self):
        if self.num_timesteps - self.last_print < 10_000:
            return True
        elapsed = (time.time() - self.start_time) / 60
        rate    = self.num_timesteps / max(time.time() - self.start_time, 1)
        pct     = 100 * self.num_timesteps / TOTAL_TIMESTEPS
        self.last_print = self.num_timesteps
        print(f'[{self.seed}] {self.num_timesteps:>10,}  ({pct:.1f}%)  '
              f'{elapsed:.1f} min  {rate:.0f} steps/s', flush=True)
        return True

# -- Training run --------------------------------------------------------------

def train(seed, delay_mode, n_envs, tag=''):
    tag_part = f'_{tag}' if tag else ''
    run_name = f"v4exp_{delay_mode}_seed{seed}{tag_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(RUNS_ROOT, delay_mode, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    # delay_mode travels via env_kwargs so it reaches every SubprocVecEnv worker. Mutating
    # the module-level CONFIG in this process would NOT propagate to spawned workers.
    venv = make_vec_env(AirspaceEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs={'delay_mode': delay_mode})
    # norm_obs=True is REQUIRED here: this environment emits raw NM/kt/s/rad.
    env  = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    model = PPO('MlpPolicy', env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)
    callbacks = CallbackList([
        ProgressCallback(seed),
        EpisodeStatsCallback(),
        EvalCallback(ckpt_dir, seed, delay_mode),
    ])

    print(f'[{seed}] delay={delay_mode}  n_envs={n_envs}  -> {run_dir}', flush=True)
    try:
        model.learn(TOTAL_TIMESTEPS, callback=callbacks, tb_log_name='ppo')
    except KeyboardInterrupt:
        pass
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        env.save(os.path.join(run_dir, 'final_vecnorm.pkl'))
        env.close()
        print(f'[{seed}] saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='PPO training for the v4 ATC environment (raw-unit observations).')
    parser.add_argument('--delay', required=True,
                        choices=['none', 'deterministic', 'probabilistic'],
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: random)')
    parser.add_argument('--n-envs', type=int, default=N_ENVS,
                        help=f'parallel environments (default {N_ENVS}); lower it when '
                             f'running the three delay arms concurrently')
    parser.add_argument('--tag', type=str, default='',
                        help='label added to the run directory name')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 99_999)
    train(seed, delay_mode=args.delay, n_envs=args.n_envs, tag=args.tag)


if __name__ == '__main__':
    main()
