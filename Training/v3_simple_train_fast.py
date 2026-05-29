"""
Fast PPO training for v3_simple_env — single run, SubprocVecEnv.

Speedups vs v3_simple_train.py:
  1. SubprocVecEnv: N_ENVS parallel BlueSky instances in separate processes
  2. ASAS disabled in the env (pure-geometry urgency, no BlueSky CD needed)

Run:
    python -m Training.v3_simple_train_fast

Monitor:
    tensorboard --logdir Runs_saved/bs/non_delay/v3_simple
"""

import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from Environments.v3_simple_env import AirspaceEnv

# ── Settings ──────────────────────────────────────────────────────────────────

N_ENVS           = 4              # parallel BlueSky processes — tune to CPU cores
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_EVERY = 100_000
RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'bs', 'non_delay', 'v3_simple')
)

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,         # per env; must exceed episode length (~1200 steps)
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 0,
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    _ACTION_LABELS = ['-30deg', '-15deg', '0deg', '+15deg', '+30deg']

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record('episode/mean_reward', info['mean_episode_reward'])
            self.logger.record('episode/los_steps',   info['ep_los_steps'])
            self.logger.record('episode/length',       info['ep_length'])
            self.logger.record('episode/n_aircraft',   info.get('n_aircraft', 0))
            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self._ACTION_LABELS, dist):
                self.logger.record(f'actions/{label}', count / total)
        return True


class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, name_prefix):
        super().__init__()
        self._save_path   = save_path
        self._name_prefix = name_prefix
        self._last_saved  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= CHECKPOINT_EVERY:
            path = os.path.join(self._save_path,
                                f'{self._name_prefix}_{self.num_timesteps}_steps')
            self.model.save(path)
            self._last_saved = self.num_timesteps
            print(f'[ckpt] {path}.zip', flush=True)
        return True


class ProgressCallback(BaseCallback):
    def _on_training_start(self):
        self._start      = time.time()
        self._last_print = 0
        print(f'N_ENVS={N_ENVS}  target={TOTAL_TIMESTEPS:,} steps', flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= 10_000:
            elapsed          = (time.time() - self._start) / 60
            rate             = self.num_timesteps / max(time.time() - self._start, 1)
            pct              = 100 * self.num_timesteps / TOTAL_TIMESTEPS
            self._last_print = self.num_timesteps
            print(
                f'{self.num_timesteps:>10,} / {TOTAL_TIMESTEPS:,}'
                f'  ({pct:5.1f}%)  |  {elapsed:6.1f} min  |  {rate:.0f} steps/s',
                flush=True,
            )
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    seed     = random.randint(0, 99_999)
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}_x{N_ENVS}envs"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Run dir : {run_dir}', flush=True)

    env = VecMonitor(
        make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    )
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        seed            = seed,
        tensorboard_log = tb_dir,
        **PPO_KWARGS,
    )

    callbacks = CallbackList([
        ProgressCallback(),
        EpisodeStatsCallback(),
        CheckpointCallback(save_path=ckpt_dir, name_prefix='bs_v3_simple_fast'),
    ])

    model_path = os.path.join(run_dir, 'final_model')
    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = callbacks,
            tb_log_name         = 'ppo',
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save(model_path)
        print(f'Model saved: {model_path}.zip', flush=True)
        env.close()


if __name__ == '__main__':
    main()
