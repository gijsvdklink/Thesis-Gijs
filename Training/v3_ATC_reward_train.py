"""
PPO training for v3_improved.

Run:
    python -m Training.v3_improved_train
    python -m Training.v3_improved_train --multi   # 3 seeds in parallel

Monitor:
    tensorboard --logdir Runs_saved/v3_improved
"""

import os, random, subprocess, sys, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from Environments.v3_improved import AirspaceEnv

# ── Settings ──────────────────────────────────────────────────────────────────

N_ENVS           = 4
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_EVERY = 100_000
N_RUNS           = 3

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'v3_improved')
)

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.02,   # slightly higher than v3 to explore 9-action space
    vf_coef       = 0.5,
    verbose       = 0,
    policy_kwargs = dict(net_arch=[128, 128]),  # larger than v3's [64, 64]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    _LABELS = ['-30°', '-15°', 'direct', '+15°', '+30°', 'hold',
               'M-0.04', 'M-0.02', 'M+0.02', 'M+0.04']

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record('episode/mean_reward', info['mean_episode_reward'])
            self.logger.record('episode/los_steps',   info['ep_los_steps'])
            self.logger.record('episode/length',       info['ep_length'])
            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self._LABELS, dist):
                self.logger.record(f'actions/{label}', count / total)
        return True


class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, seed):
        super().__init__()
        self._save_path = save_path
        self._seed      = seed
        self._last      = 0

    def _on_step(self):
        if self.num_timesteps - self._last < CHECKPOINT_EVERY:
            return True
        stem = os.path.join(self._save_path, f'ckpt_{self.num_timesteps}')
        self.model.save(stem)
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(stem + '_vecnorm.pkl')
        self._last = self.num_timesteps
        print(f'[{self._seed}] checkpoint {self.num_timesteps:,}', flush=True)
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, seed):
        super().__init__()
        self._seed = seed

    def _on_training_start(self):
        self._t0         = time.time()
        self._last_print = 0
        print(f'[{self._seed}] target {TOTAL_TIMESTEPS:,} steps', flush=True)

    def _on_step(self):
        if self.num_timesteps - self._last_print < 10_000:
            return True
        elapsed          = (time.time() - self._t0) / 60
        rate             = self.num_timesteps / max(time.time() - self._t0, 1)
        pct              = 100 * self.num_timesteps / TOTAL_TIMESTEPS
        self._last_print = self.num_timesteps
        print(f'[{self._seed}] {self.num_timesteps:>10,}  ({pct:.1f}%)  '
              f'{elapsed:.1f} min  {rate:.0f} steps/s', flush=True)
        return True

# ── Training run ──────────────────────────────────────────────────────────────

def train(seed):
    run_name = f"improved_9act_obs25_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed)
    env  = VecNormalize(VecMonitor(venv),
                        norm_obs=False, norm_reward=False,
                        gamma=0.99)

    model = PPO('MlpPolicy', env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)

    callbacks = CallbackList([
        ProgressCallback(seed),
        EpisodeStatsCallback(),
        CheckpointCallback(ckpt_dir, seed),
    ])

    try:
        model.learn(TOTAL_TIMESTEPS, callback=callbacks,
                    tb_log_name='ppo', reset_num_timesteps=True)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        env.save(os.path.join(run_dir, 'final_vecnorm.pkl'))
        env.close()
        print(f'[{seed}] saved to {run_dir}', flush=True)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int, default=None)
    parser.add_argument('--multi', action='store_true',
                        help=f'Launch {N_RUNS} seeds in parallel')
    args = parser.parse_args()

    if args.multi:
        seeds = [random.randint(1, 99_999) for _ in range(N_RUNS)]
        print(f'Launching {N_RUNS} runs — seeds: {seeds}')
        procs = [subprocess.Popen([sys.executable, __file__, '--seed', str(s)])
                 for s in seeds]
        try:
            for p in procs: p.wait()
        except KeyboardInterrupt:
            for p in procs: p.terminate()
    else:
        train(args.seed if args.seed is not None else random.randint(0, 99_999))

if __name__ == '__main__':
    main()
