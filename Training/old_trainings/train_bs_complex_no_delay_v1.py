"""
PPO training on AirspaceEnvV3.

  - Stochastic airspace area (1 000–3 000 km²) per episode
  - 4-wave dynamic spawning (12 aircraft per episode)
  - Action penalty for unnecessary heading changes

Each run creates a timestamped folder:
    Runs_saved/bs/non_delay/complex/YYYYMMDD_HHMMSS_bs_complex_no_delay_v1/
        tensorboard/
        checkpoints/
        model.zip         (written on clean exit)

Stop at any time with Ctrl+C.

Monitor:
    python -m tensorboard.main --logdir Runs_saved/bs/non_delay/complex
"""

import os
import sys
import time
from datetime import datetime

_ENV_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Environments'))
sys.path.insert(0, _ENV_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from bs_complex_no_delay_v1 import MARLSingleAgentWrapper

# ── Settings ──────────────────────────────────────────────────────────────────
N_ENVS    = 4
RUNS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'bs', 'non_delay', 'complex'))


def make_run_dir():
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bs_complex_no_delay_v1"
    path = os.path.join(RUNS_ROOT, name)
    os.makedirs(os.path.join(path, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(path, 'checkpoints'), exist_ok=True)
    return path


def make_env():
    def _init():
        return MARLSingleAgentWrapper()
    return _init


# ── Callbacks ─────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):

    def __init__(self, run_dir: str):
        super().__init__()
        self.run_dir    = run_dir
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        tb_cmd = f'python -m tensorboard.main --logdir "{os.path.join(self.run_dir, "tensorboard")}"'
        print(f"Run folder  : {self.run_dir}")
        print(f"TensorBoard : {tb_cmd}")
        print(f"Training    : {N_ENVS} envs  |  unlimited  (Ctrl+C to stop)\n")

    def _on_step(self) -> bool:
        if self.n_calls % 2048 == 0:
            elapsed = time.time() - self.start_time
            print(f"  step {self.num_timesteps:>8,}  |  {elapsed/60:5.1f} min elapsed")
        return True


class RichTBCallback(BaseCallback):

    _ACTION_LABELS = ['-10', '+10', '0', 'back']

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record('marl/mean_ep_reward', info['mean_episode_reward'])
            self.logger.record('marl/ep_los_steps',   info['ep_los_steps'])
            self.logger.record('marl/ep_length',       info['ep_length'])
            for k, v in enumerate(info['episode_rewards']):
                self.logger.record(f'marl/reward_AC{k:02d}', v)
            for label, count in zip(self._ACTION_LABELS, info['action_distribution']):
                self.logger.record(f'actions/{label}', count)
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_dir    = make_run_dir()
    tb_dir     = os.path.join(run_dir, 'tensorboard')
    ckpt_dir   = os.path.join(run_dir, 'checkpoints')
    model_path = os.path.join(run_dir, 'model')

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)], start_method='spawn')
    env = VecMonitor(env)

    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 512,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        verbose         = 1,
        tensorboard_log = tb_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000 // N_ENVS,
        save_path   = ckpt_dir,
        name_prefix = 'bs_complex_no_delay_v1',
    )

    model.learn(
        total_timesteps     = 100_000_000,
        callback            = CallbackList([
            ProgressCallback(run_dir),
            RichTBCallback(),
            checkpoint_cb,
        ]),
        tb_log_name         = 'bs_complex_no_delay_v1',
        reset_num_timesteps = True,
    )

    model.save(model_path)
    print(f"\nModel      →  {model_path}.zip")
    print(f"Checkpoints →  {ckpt_dir}/")
    print(f"TensorBoard →  python -m tensorboard.main --logdir \"{tb_dir}\"")


if __name__ == '__main__':
    main()
