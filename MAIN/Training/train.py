# PPO trainer for the response-delay experiment, one delay type per process: python -m Training.train --delay none|deterministic|lognormal. The same --seed sees identical scenarios in every type.

import os

# Must precede the torch import: a 64x64 MLP gains nothing from intra-op threading, and one OpenMP pool per worker would spin-wait more than it computes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
import time
from collections import deque

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment import AirspaceEnv, CONFIG, DELAY_MODES
from Environment.config import TRAINING_SEEDS

# Episodes kept for the reward trend. At roughly 1,500 steps an episode, 200 of them span about
# 300k steps, so the slope reacts within a few rollouts without chasing single-episode noise.
TREND_WINDOW = 200

# Episodes averaged behind every logged KPI. A rollout is only ~3 episodes, and each episode is
# a different airspace -- aircraft count, density and geometry are all redrawn -- so a per-rollout
# mean is dominated by which scenarios happened to come up. R_los alone carries about 78% of the
# episode-to-episode spread. 100 matches SB3's own window for rollout/ep_rew_mean.
STATS_WINDOW = 100

# -- Settings ------------------------------------------------------------------

# The fixed training budget of every model, so that differences between models do not come from
# a different training length: runs are not stopped early. Ctrl-C still writes final_model, but
# only for aborted runs. The BlueSky-Gym benchmark (Groot et al., SID 2024) found 2M far too few
# for PPO to converge here.
TOTAL_TIMESTEPS = 300_000_000

# One environment per delay type: BlueSky is process-global, so a second env in the same process
# would share one simulation. Parallelism comes from running the delay types side by side, each
# in its own process, not from vectorising within a run.
N_ENVS     = 1
N_STEPS    = 4096                 # rollout = 4096 steps
BATCH_SIZE = 512                  # 4096 / 512 = 8 minibatches per epoch

GAMMA    = 0.995
ENT_COEF = 0.01

# No evaluation during training: the reported policy is the one training ended on, scored afterwards by Validation/validation.py on VALIDATION_SEEDS. What is left is a periodic save against a crash.
SAVE_EVERY     = 500_000
PROGRESS_EVERY = 50_000

# Beside Environment, Training and Validation; this is also where Validation/validation.py
# looks, so a finished run needs no moving.
RUNS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models'))

# Episode-summary key -> TensorBoard tag. The reported set is Table 2.4 of the report; the
# specific advisories and the delay diagnostics follow it.
METRICS = [
    ('ep_reward_total',      'episode/reward_total'),
    ('ep_reward_per_fh',     'episode/reward_per_flight_hour'),

    # Where the reward actually goes. The three add up to reward_per_flight_hour.
    ('ep_reward_los_per_fh',   'reward/los_per_flight_hour'),
    ('ep_reward_drift_per_fh', 'reward/drift_per_flight_hour'),
    ('ep_reward_work_per_fh',  'reward/work_per_flight_hour'),

    # LoS and conflicts, per flight hour so episodes of different size stay comparable.
    ('ep_los_events_per_fh', 'safety/los_events_per_flight_hour'),
    ('ep_conflicts_per_fh',  'safety/conflicts_per_flight_hour'),

    # Route efficiency.
    ('ep_path_ratio',        'route/distance_ratio'),
    ('ep_on_route_rate',     'route/on_route_exits'),

    # Instruction load, split by kind and in total.
    ('ep_turns_per_fh',         'actions/heading_changes_per_flight_hour'),
    ('ep_speed_changes_per_fh', 'actions/speed_changes_per_flight_hour'),
    ('ep_advisories_per_fh',    'actions/advisories_per_flight_hour'),

    # Which advisory, not just how many: the resolution strategy itself.
    ('ep_turn_m60_per_fh',   'advisory/turn_-60'),
    ('ep_turn_m45_per_fh',   'advisory/turn_-45'),
    ('ep_turn_m30_per_fh',   'advisory/turn_-30'),
    ('ep_turn_p30_per_fh',   'advisory/turn_+30'),
    ('ep_turn_p45_per_fh',   'advisory/turn_+45'),
    ('ep_turn_p60_per_fh',   'advisory/turn_+60'),
    ('ep_return_per_fh',     'advisory/return_to_initial_hdg'),
    ('ep_speed_up_per_fh',   'advisory/speed_up'),
    ('ep_speed_down_per_fh', 'advisory/speed_down'),

    # The delay pipeline: a mean response far above the nominal delay, or a discard count near
    # the advisory count, means instructions are being revised faster than the ATCO can act.
    ('ep_delay_mean_s',      'delay/mean_response_s'),
    ('ep_discarded',         'delay/advisories_discarded'),
    ('ep_repeats',           'delay/advice_re_selected'),
]


# -- Saving --------------------------------------------------------------------

def save(model, run_dir, name):
    """A checkpoint and its VecNormalize statistics, which must travel together as <name>_vecnorm.pkl."""
    model.save(os.path.join(run_dir, name))
    model.get_env().save(os.path.join(run_dir, f'{name}_vecnorm.pkl'))


# -- Callbacks -----------------------------------------------------------------

class LogEpisodes(BaseCallback):
    """Log each finished training episode. These come from the EXPLORING policy."""

    def __init__(self):
        super().__init__()
        self.recent  = deque(maxlen=TREND_WINDOW)     # (timestep, episode reward)
        self.windows = {key: deque(maxlen=STATS_WINDOW) for key, _ in METRICS}

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'ep_reward_total' not in info:
                continue
            # Every KPI is the mean over the last STATS_WINDOW episodes, not over the handful
            # in this rollout, so the curves show the policy rather than the luck of the draw.
            for key, tag in METRICS:
                window = self.windows[key]
                window.append(info[key])
                self.logger.record(tag, sum(window) / len(window))

            # Is the reward still climbing? Least-squares slope over the recent episodes, per
            # million steps, so it reads as "reward gained per 1M steps". Once it sits at zero
            # the run has stopped improving, which is the signal to stop it.
            self.recent.append((self.num_timesteps, info['ep_reward_total']))
            if len(self.recent) == TREND_WINDOW:
                mean_step   = sum(t for t, _ in self.recent) / TREND_WINDOW
                mean_reward = sum(r for _, r in self.recent) / TREND_WINDOW
                spread      = sum((t - mean_step) ** 2 for t, _ in self.recent)
                if spread > 0:
                    covariance = sum((t - mean_step) * (r - mean_reward)
                                     for t, r in self.recent)
                    self.logger.record('episode/reward_slope_per_1M',
                                       covariance / spread * 1e6)
        return True


class Checkpoint(BaseCallback):
    """Write last_model every `every` steps, so a crash does not cost the whole run."""

    def __init__(self, save_dir, every):
        super().__init__()
        self.save_dir  = save_dir
        self.every     = every
        self.last_save = 0

    def _on_step(self):
        if not self.every or self.num_timesteps - self.last_save < self.every:
            return True
        self.last_save = self.num_timesteps
        save(self.model, self.save_dir, 'last_model')
        print(f'  saved @ {self.num_timesteps:>10,}', flush=True)
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

def delay_type_name(delay_mode, delay_mean_s):
    """Directory name for one delay type, e.g. 'lognormal_30s'; the baseline stays plain 'none'."""
    return 'none' if delay_mode == 'none' else f'{delay_mode}_{delay_mean_s:g}s'


def train(delay_mode, seed, total_timesteps, n_envs, save_every, delay_mean_s,
          runs_root=RUNS_ROOT, overwrite=False):
    delay_type = delay_type_name(delay_mode, delay_mean_s)
    # One directory per (delay type, seed), with no timestamp. A timestamp meant every restart
    # left another copy behind, TensorBoard drew each seed twice, and validation.find_model
    # could no longer tell which run it was meant to score.
    run_dir = os.path.join(runs_root, delay_type, f'{delay_type}_seed{seed}')
    if os.path.exists(run_dir) and not overwrite:
        sys.exit(f'{run_dir} already exists. Delete it, move it aside, or pass --overwrite '
                 f'to train over it.')
    os.makedirs(run_dir, exist_ok=True)

    # Everything travels through the constructor so it reaches the worker PROCESSES; a CONFIG
    # edit here would not survive the spawn. A given --seed draws the same scenarios in every
    # delay type, which is what makes the delay the only variable between conditions.
    def make_worker():
        return AirspaceEnv(delay_mode=delay_mode, delay_mean_s=delay_mean_s, seed=seed)

    # DummyVecEnv throughout: one environment needs no worker process, and BlueSky being a
    # process-global singleton means more than one env per process is not safe anyway.
    venv = DummyVecEnv([make_worker for _ in range(n_envs)])
    env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)

    # verbose=0: the Progress callback prints a compact line instead of SB3's full table.
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, gamma=GAMMA, ent_coef=ENT_COEF)

    callbacks = CallbackList([Progress(), LogEpisodes(), Checkpoint(run_dir, save_every)])

    print(f'{delay_type}  seed {seed}  {total_timesteps:,} steps  {n_envs} envs  '
          f'save every {save_every:,}  -> {run_dir}', flush=True)
    try:
        model.learn(total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        # Also written on Ctrl-C, so a hand-stopped run leaves the policy at the exact step it stopped on.
        save(model, run_dir, 'final_model')
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train one delay type.')
    parser.add_argument('--delay', required=True, choices=list(DELAY_MODES),
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=TRAINING_SEEDS[0], choices=TRAINING_SEEDS,
                        help=f'which training run: one of {list(TRAINING_SEEDS)}. All delay '
                             f'types at the same seed share weights and scenarios.')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help=f'training steps (default {TOTAL_TIMESTEPS:,})')
    parser.add_argument('--n-envs', type=int, default=N_ENVS,
                        help=f'environments in this process (default {N_ENVS}); BlueSky is a '
                             f'singleton, so leave this at 1')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                        help=f'steps between last_model checkpoints '
                             f'(default {SAVE_EVERY:,}); 0 saves only at the end')
    parser.add_argument('--overwrite', action='store_true',
                        help='train into an existing run directory instead of refusing')
    parser.add_argument('--runs-root', default=RUNS_ROOT,
                        help='where the run directory is created, so a new set of models '
                             'can sit beside an old one (default Runs_saved/experiments)')
    # --delay-first is the old spelling, kept so existing run scripts still work.
    default_mean_s = CONFIG['delay_mean_s']
    parser.add_argument('--delay-mean', '--delay-first', dest='delay_mean',
                        type=float, default=default_mean_s,
                        help=f'delay magnitude: the MEAN pilot response time in seconds '
                             f'(default {default_mean_s:g}). Every advisory is drawn from '
                             f'this distribution. Ignored when --delay none.')
    args = parser.parse_args()

    train(args.delay, args.seed, args.timesteps, args.n_envs,
          args.save_every, args.delay_mean, args.runs_root, args.overwrite)


if __name__ == '__main__':
    main()
