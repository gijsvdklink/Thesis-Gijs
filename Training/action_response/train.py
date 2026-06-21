"""
PPO training for the action-response-delay env (Environments/action_response/experimental.py).

Trains one policy under a chosen delay model so deterministic and probabilistic runs
can be compared (same MEAN delay -> isolates response variability). Observations are
range-scaled in-env; VecNormalize standardises obs + reward.

Run:
    python -m Training.action_response.train --delay probabilistic
    python -m Training.action_response.train --delay deterministic --mean 20.5
    python -m Training.action_response.train --delay probabilistic --multi   # 3 seeds

Monitor:
    tensorboard --logdir Runs_saved/action_response
"""

import os, random, subprocess, sys, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from Environments.action_response.experimental import AirspaceEnv

# ── Settings ──────────────────────────────────────────────────────────────────

N_ENVS           = 48
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_EVERY = 300_000
N_RUNS           = 3

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Runs_saved', 'action_response')
)

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 256,
    batch_size    = 2048,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.02,
    vf_coef       = 0.5,
    verbose       = 0,
    policy_kwargs = dict(net_arch=[128, 128]),
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    _LABELS = ['-60°', '-45°', '-30°', 'hold', '+30°', '+45°', '+60°', 'direct',
               'spd+', 'spd-']

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record_mean('episode/mean_reward',  info['mean_episode_reward'])
            self.logger.record_mean('episode/los_steps',    info['ep_los_steps'])
            self.logger.record_mean('episode/arrival_rate', info['ep_arrival_rate'])
            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self._LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
            if len(dist) >= 10:
                turn_count = sum(dist) - dist[3] - dist[7] - dist[8] - dist[9]
                self.logger.record_mean('actions/turns_total', turn_count)
                self.logger.record_mean('actions/speed_total', dist[8] + dist[9])
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

def train(seed, delay, mean_s, resume=None):
    resume_tag = "_resumed" if resume else ""
    run_name = (f"ardelay_{delay}_mean{mean_s:g}s_seed{seed}{resume_tag}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    env_kwargs = {'delay': delay, 'mean_s': mean_s}
    venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs=env_kwargs)
    monitored = VecMonitor(venv)

    vecnorm_path = resume.replace('.zip', '_vecnorm.pkl') if resume else None
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, monitored)
        env.training, env.norm_obs, env.norm_reward = True, True, True
    else:
        env = VecNormalize(monitored, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    if resume:
        model = PPO.load(resume, env=env, tensorboard_log=tb_dir)
        model.seed = seed
        print(f'[{seed}] resuming from {resume} @ {model.num_timesteps:,} steps', flush=True)
    else:
        model = PPO('MlpPolicy', env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)

    callbacks = CallbackList([
        ProgressCallback(seed),
        EpisodeStatsCallback(),
        CheckpointCallback(ckpt_dir, seed),
    ])

    try:
        model.learn(TOTAL_TIMESTEPS, callback=callbacks,
                    tb_log_name='ppo', reset_num_timesteps=(resume is None))
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
    parser.add_argument('--delay', choices=['deterministic', 'probabilistic'],
                        default='probabilistic')
    parser.add_argument('--mean',  type=float, default=20.5,
                        help='Mean initiation delay in seconds (shared by both modes)')
    parser.add_argument('--seed',  type=int,   default=None)
    parser.add_argument('--multi', action='store_true',
                        help=f'Launch {N_RUNS} seeds in parallel')
    parser.add_argument('--resume', default=None,
                        help='Path to a checkpoint .zip to continue training from')
    args = parser.parse_args()

    if args.multi:
        seeds = [random.randint(1, 99_999) for _ in range(N_RUNS)]
        print(f'Launching {N_RUNS} runs ({args.delay}, mean {args.mean}s) — seeds: {seeds}')
        procs = [subprocess.Popen([sys.executable, __file__, '--seed', str(s),
                                   '--delay', args.delay, '--mean', str(args.mean)])
                 for s in seeds]
        try:
            for p in procs: p.wait()
        except KeyboardInterrupt:
            for p in procs: p.terminate()
    else:
        train(args.seed if args.seed is not None else random.randint(0, 99_999),
              args.delay, args.mean, resume=args.resume)

if __name__ == '__main__':
    main()
