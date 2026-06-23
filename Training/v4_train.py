"""
PPO training for the v4 ACAS Xu-state env (Environments/normal/v4.py).

VecNormalize standardises both observations and reward (norm_obs=True,
norm_reward=True). The *_vecnorm.pkl saved alongside each model holds these stats
and MUST be loaded at eval/visualisation time (feeding raw obs to a norm_obs=True
policy collapses it).

Run:
    python -m Training.v4_train
    python -m Training.v4_train --multi   # 3 seeds in parallel

Monitor:
    tensorboard --logdir Runs_saved/normal
"""

import os, random, subprocess, sys, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from Environments.normal.v4 import AirspaceEnv

# ── Settings ──────────────────────────────────────────────────────────────────

N_ENVS           = 48   # tuned for this server's core count
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_EVERY = 300_000
N_RUNS           = 3

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'normal')
)

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 256,    # rollout = n_steps*N_ENVS = 12,288 (with N_ENVS=48)
                             # -- updates and reward/tensorboard dumps happen
                             # once per rollout, so this sets their frequency
    batch_size    = 2048,   # was 64: with a large N_ENVS the rollout is large,
                             # so batch_size=64 meant thousands of tiny
                             # minibatches x n_epochs, mostly Python/Adam
                             # overhead. 2048 -> 6 minibatches x 10 epochs = 60
                             # steps: far less wall-clock spent per rollout,
                             # and lower-variance gradients from bigger
                             # batches tends to help PPO stability too.
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
    # 10 actions: turns +-60/45/30, hold (3), fly-direct (7), speed up/down (8/9).
    _LABELS = ['-60°', '-45°', '-30°', 'hold', '+30°', '+45°', '+60°', 'direct',
               'spd+', 'spd-']

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            # record_mean averages across all episodes in the logging window
            # (plain record would keep only the last episode per dump)
            self.logger.record_mean('episode/mean_reward',  info['mean_episode_reward'])
            self.logger.record_mean('episode/los_steps',    info['ep_los_steps'])
            self.logger.record_mean('episode/arrival_rate', info['ep_arrival_rate'])
            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(self._LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
            # turn instructions issued per episode -- excludes hold (3), fly-direct (7)
            # and speed actions (8, 9): only real heading-change instructions
            if len(dist) >= 10:
                turn_count = sum(dist) - dist[3] - dist[7] - dist[8] - dist[9]
                self.logger.record_mean('actions/turns_total', turn_count)
                self.logger.record_mean('actions/speed_total', dist[8] + dist[9])
        return True


class BestModelCallback(BaseCallback):
    """Every CHECKPOINT_EVERY steps, keep the best model seen so far.

    Performance is the mean *raw* episode reward over the model's recent-episode
    buffer (filled by VecMonitor, which sits inside VecNormalize so its rewards are
    un-normalised). When that mean improves on the previous best, the model is saved
    as best_model.zip (+ best_model_vecnorm.pkl, needed at eval time). best_model is
    overwritten in place, so it always holds the best checkpoint to date."""
    def __init__(self, save_path, seed):
        super().__init__()
        self._save_path = save_path
        self._seed      = seed
        self._last      = 0
        self._best      = -float('inf')

    def _on_step(self):
        if self.num_timesteps - self._last < CHECKPOINT_EVERY:
            return True
        self._last = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf:                       # no completed episodes yet this early
            return True
        mean_reward = sum(ep['r'] for ep in buf) / len(buf)
        if mean_reward <= self._best:
            print(f'[{self._seed}] {self.num_timesteps:,}  mean_ep_reward={mean_reward:.3f} '
                  f'(best {self._best:.3f}, not saved)', flush=True)
            return True

        self._best = mean_reward
        stem = os.path.join(self._save_path, 'best_model')
        self.model.save(stem)
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(stem + '_vecnorm.pkl')
        print(f'[{self._seed}] {self.num_timesteps:,}  NEW BEST mean_ep_reward='
              f'{mean_reward:.3f} -> saved best_model', flush=True)
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

def train(seed, t_warn=None, resume=None, dummy_retn=False):
    t_warn_tag = f"_twarn{int(t_warn)}s" if t_warn is not None else ""
    dummy_tag  = "_dummyretn" if dummy_retn else ""
    resume_tag = "_resumed" if resume else ""
    run_name = f"acasxu_7state_obs22_seed{seed}{t_warn_tag}{dummy_tag}{resume_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    env_kwargs = {}
    if t_warn is not None:
        env_kwargs['t_warn'] = t_warn
    if dummy_retn:
        env_kwargs['dummy_retn'] = True
    venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs=env_kwargs)
    monitored = VecMonitor(venv)

    # Standardise BOTH observations and reward with VecNormalize (norm_obs=True,
    # norm_reward=True). Observations are pre-scaled in-env to roughly consistent
    # ranges; VecNormalize additionally standardises them with running mean/std, which
    # the policy relies on. The saved *_vecnorm.pkl holds the obs and reward stats and
    # MUST be loaded at eval/visualisation time -- feeding raw obs to a norm_obs=True
    # policy collapses it (e.g. onto HOLD).
    vecnorm_path = resume.replace('.zip', '_vecnorm.pkl') if resume else None
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, monitored)
        env.training, env.norm_obs, env.norm_reward = True, True, True
    else:
        env = VecNormalize(monitored, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    if resume:
        # warm-start from an existing policy and keep training it
        model = PPO.load(resume, env=env, tensorboard_log=tb_dir)
        model.seed = seed
        print(f'[{seed}] resuming from {resume} @ {model.num_timesteps:,} steps', flush=True)
    else:
        model = PPO('MlpPolicy', env, seed=seed, tensorboard_log=tb_dir, **PPO_KWARGS)

    callbacks = CallbackList([
        ProgressCallback(seed),
        EpisodeStatsCallback(),
        BestModelCallback(ckpt_dir, seed),
    ])

    try:
        # reset_num_timesteps=False -> counter & TensorBoard continue; the budget
        # below is then added on top of the steps already trained.
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
    parser.add_argument('--seed',   type=int,   default=None)
    parser.add_argument('--t_warn', type=float, default=None,
                        help='Warning horizon in seconds (default: 360 = 6 min)')
    parser.add_argument('--multi', action='store_true',
                        help=f'Launch {N_RUNS} seeds in parallel')
    parser.add_argument('--dummy-retn', dest='dummy_retn', action='store_true',
                        help='Ablation: hold the safe-to-return (retn_conf) obs at a '
                             'constant 0 so the policy cannot use it')
    parser.add_argument('--resume', default=None,
                        help='Path to a checkpoint .zip to continue training from')
    args = parser.parse_args()

    if args.multi:
        seeds = [random.randint(1, 99_999) for _ in range(N_RUNS)]
        print(f'Launching {N_RUNS} runs — seeds: {seeds}')
        extra = ['--t_warn', str(args.t_warn)] if args.t_warn is not None else []
        if args.dummy_retn:
            extra.append('--dummy-retn')
        procs = [subprocess.Popen([sys.executable, __file__, '--seed', str(s)] + extra)
                 for s in seeds]
        try:
            for p in procs: p.wait()
        except KeyboardInterrupt:
            for p in procs: p.terminate()
    else:
        train(args.seed if args.seed is not None else random.randint(0, 99_999),
              t_warn=args.t_warn, dummy_retn=args.dummy_retn)

if __name__ == '__main__':
    main()
