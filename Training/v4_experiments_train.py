"""
PPO training for the v4-experiments ATC environment (Environments/v4_experiments).

That environment emits RAW PHYSICAL UNITS (NM, kt, s, rad) with no hand-picked
normalisers and no clipping, so VecNormalize is not a convenience here -- it is the
only thing standardising the inputs. norm_obs=True is mandatory, and the *_vecnorm.pkl
saved next to each model MUST be loaded at eval/visualisation time. Feeding raw
observations to the policy without it will produce nonsense, far more dramatically
than in v4 where the features were already O(1).

Edit the SETTINGS / PPO_KWARGS below to configure a run.

Command-line flags:
  --seed N            fix the random seed (default: random)
  --tag NAME          label added to the run directory name

Run:       python -m Training.v4_experiments_train --seed 42
Monitor:   tensorboard --logdir Runs_saved/experiments
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from Environments.v4_experiments import AirspaceEnv

# -- Settings ------------------------------------------------------------------

N_ENVS           = 48            # parallel environments (tune to the machine's cores)
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_EVERY = 300_000       # best-model check interval (steps)

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
    """Log per-episode reward, LoS steps, arrival rate, and the action distribution."""
    ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'direct', 'spd+', 'spd-']

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record_mean('episode/mean_reward',  info['mean_episode_reward'])
            self.logger.record_mean('episode/los_steps',    info['ep_los_steps'])
            self.logger.record_mean('episode/arrival_rate', info['ep_arrival_rate'])
            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self.ACTION_LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
            if len(dist) >= 10:
                # fractions, so they are comparable to the per-action fractions above
                turns = sum(dist) - dist[3] - dist[7] - dist[8] - dist[9]   # 6 stack-turns (excl. hold/fly-direct/speed)
                self.logger.record_mean('actions/turns_total', turns / total)
                self.logger.record_mean('actions/speed_total', (dist[8] + dist[9]) / total)
        return True


class BestModelCallback(BaseCallback):
    """Every CHECKPOINT_EVERY steps, save best_model (+ vecnorm) when the recent mean
    episode reward improves on the best seen so far. Rewards come from VecMonitor (inside
    VecNormalize), so they are un-normalised."""
    def __init__(self, save_path, seed):
        super().__init__()
        self.save_path = save_path
        self.seed      = seed
        self.last_step = 0
        self.best      = -float('inf')

    def _on_step(self):
        if self.num_timesteps - self.last_step < CHECKPOINT_EVERY:
            return True
        self.last_step = self.num_timesteps

        episodes = self.model.ep_info_buffer
        if not episodes:                       # no completed episodes yet
            return True
        mean_reward = sum(ep['r'] for ep in episodes) / len(episodes)
        if mean_reward <= self.best:
            print(f'[{self.seed}] {self.num_timesteps:,}  mean_ep_reward={mean_reward:.3f} '
                  f'(best {self.best:.3f}, not saved)', flush=True)
            return True

        self.best = mean_reward
        stem = os.path.join(self.save_path, 'best_model')
        self.model.save(stem)
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(stem + '_vecnorm.pkl')
        print(f'[{self.seed}] {self.num_timesteps:,}  NEW BEST mean_ep_reward='
              f'{mean_reward:.3f} -> saved best_model', flush=True)
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

def train(seed, tag=''):
    tag_part = f'_{tag}' if tag else ''
    run_name = f"v4exp_seed{seed}{tag_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed)
    # norm_obs=True is REQUIRED here: this environment emits raw NM/kt/s/rad.
    env  = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    model = PPO('MlpPolicy', env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)
    callbacks = CallbackList([
        ProgressCallback(seed),
        EpisodeStatsCallback(),
        BestModelCallback(ckpt_dir, seed),
    ])

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
        description='PPO training for the v4-experiments ATC environment (raw-unit observations).')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: random)')
    parser.add_argument('--tag', type=str, default='',
                        help='label added to the run directory name')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 99_999)
    train(seed, tag=args.tag)


if __name__ == '__main__':
    main()
