"""
PPO training for v2_env — MARL conflict resolution.

Running without arguments launches three independent training runs,
each as a separate subprocess with a different random seed.
All runs are visible together in TensorBoard.

Run:
    python -m Training.v2_train

Monitor:
    python -m tensorboard.main --logdir Runs_saved/bs/non_delay/complex
"""

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor

from Environments.v2_env import MARLVecEnv

# ── Settings ──────────────────────────────────────────────────────────────────

N_RUNS           = 3
CHECKPOINT_EVERY = 100_000
RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'bs', 'non_delay', 'complex')
)

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 0,
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode stats to TensorBoard."""

    _ACTION_LABELS = ['-20deg', '-10deg', 'hold', '+10deg', '+20deg', 'backWP']

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record('episode/mean_reward', info['mean_episode_reward'])
            self.logger.record('episode/los_steps',   info['ep_los_steps'])
            self.logger.record('episode/length',       info['ep_length'])
            self.logger.record('episode/n_aircraft',   info.get('n_aircraft', 0))
            total = max(sum(info['action_distribution']), 1)
            for label, count in zip(self._ACTION_LABELS, info['action_distribution']):
                self.logger.record(f'actions/{label}', count / total)
        return True


class CheckpointCallback(BaseCallback):
    """Saves the model every CHECKPOINT_EVERY timesteps."""

    def __init__(self, save_path, name_prefix, seed):
        super().__init__()
        self._save_path   = save_path
        self._name_prefix = name_prefix
        self._seed        = seed
        self._last_saved  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= CHECKPOINT_EVERY:
            path = os.path.join(self._save_path,
                                f'{self._name_prefix}_{self.num_timesteps}_steps')
            self.model.save(path)
            self._last_saved = self.num_timesteps
            print(f'[seed={self._seed}]  [ckpt] {path}.zip', flush=True)
        return True


class ProgressCallback(BaseCallback):
    """Prints a heartbeat every 5 000 steps."""

    def __init__(self, seed, run_dir):
        super().__init__()
        self._seed    = seed
        self._run_dir = run_dir

    def _on_training_start(self):
        self._start      = time.time()
        self._last_print = 0
        print(f'[seed={self._seed}]  Run dir : {self._run_dir}', flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= 5_000:
            elapsed          = (time.time() - self._start) / 60
            rate             = self.num_timesteps / max(time.time() - self._start, 1)
            self._last_print = self.num_timesteps
            print(f'[seed={self._seed}]  {self.num_timesteps:>10,} steps  |'
                  f'  {elapsed:6.1f} min  |  {rate:.0f} steps/s', flush=True)
        return True


# ── Single training run ───────────────────────────────────────────────────────

def train(seed):
    """Run one full training session with the given seed."""
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    env   = VecMonitor(MARLVecEnv())
    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        seed            = seed,
        tensorboard_log = tb_dir,
        **PPO_KWARGS,
    )

    callbacks = CallbackList([
        ProgressCallback(seed=seed, run_dir=run_dir),
        EpisodeStatsCallback(),
        CheckpointCallback(save_path=ckpt_dir, name_prefix='bs_complex_no_delay_v2', seed=seed),
    ])

    model_path = os.path.join(run_dir, 'model')
    try:
        model.learn(
            total_timesteps     = 100_000_000_000,
            callback            = callbacks,
            tb_log_name         = 'ppo',
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save(model_path)
        print(f'[seed={seed}]  Model saved: {model_path}.zip', flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='If given, run a single training with this seed')
    args = parser.parse_args()

    if args.seed is not None:
        # Worker mode: called by the launcher below
        train(args.seed)
    else:
        # Launcher mode: spawn N_RUNS independent subprocesses
        seeds = [random.randint(0, 99_999) for _ in range(N_RUNS)]
        print(f'\nLaunching {N_RUNS} training runs — seeds: {seeds}')
        print(f'TensorBoard: python -m tensorboard.main --logdir "{RUNS_ROOT}"\n')

        procs = [
            subprocess.Popen([sys.executable, __file__, '--seed', str(seed)])
            for seed in seeds
        ]
        try:
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            print('\nStopping all runs...')
            for p in procs:
                p.terminate()


if __name__ == '__main__':
    main()
