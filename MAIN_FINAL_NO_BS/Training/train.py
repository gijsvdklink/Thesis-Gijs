"""Train one model: a single environment, PPO with the reported settings, 60,000 episodes.

    python Training/train.py --delay none --seed 1
    python Training/train.py --delay lognormal --delay-mean 30 --seed 3
    python Training/train.py --delay none --seed 1 --resume --max-episodes 19000
    python Training/train.py --delay none --seed 1 --sb3-default --runs-root SB3_default_models

--sb3-default builds PPO with SB3's own defaults (n_steps 2048, batch_size 64, n_epochs 10,
learning rate 3e-4, ...) instead of the reported settings. The environment, VecNormalize and
the stopping rule are the same.

A run writes to Models/<delay type>/<delay type>_seed<seed>/: the TensorBoard log, last_model
every 500,000 steps, final_model at the end, and beside each a <name>_vecnorm.pkl holding the
VecNormalize statistics. Those statistics are part of the model -- a checkpoint loaded without
them sees observations on the wrong scale.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import sys
import time
import zipfile
from collections import deque
from random import Random

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment import AirspaceEnv, CONFIG, DELAY_MODES
from Environment.config import TRAINING_SEEDS

MAX_EPISODES = 60_000
STEP_CEILING = 300_000_000

N_STEPS  = 2048
N_EPOCHS = 30

SAVE_EVERY     = 500_000
PROGRESS_EVERY = 50_000
STATS_WINDOW   = 100

RUNS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models'))

METRICS = [
    ('ep_reward_total',      'episode/reward_total'),
    ('ep_reward_per_fh',     'episode/reward_per_flight_hour'),

    ('ep_reward_los_per_fh',   'reward/los_per_flight_hour'),
    ('ep_reward_drift_per_fh', 'reward/drift_per_flight_hour'),
    ('ep_reward_work_per_fh',  'reward/work_per_flight_hour'),

    ('ep_los_events_per_fh', 'safety/los_events_per_flight_hour'),
    ('ep_conflicts_per_fh',  'safety/conflicts_per_flight_hour'),

    ('ep_path_ratio',        'route/distance_ratio'),
    ('ep_on_route_rate',     'route/on_route_exits'),

    ('ep_turns_per_fh',         'actions/heading_changes_per_flight_hour'),
    ('ep_speed_changes_per_fh', 'actions/speed_changes_per_flight_hour'),
    ('ep_advisories_per_fh',    'actions/advisories_per_flight_hour'),

    ('ep_turn_m60_per_fh',   'advisory/turn_-60'),
    ('ep_turn_m45_per_fh',   'advisory/turn_-45'),
    ('ep_turn_m30_per_fh',   'advisory/turn_-30'),
    ('ep_turn_p30_per_fh',   'advisory/turn_+30'),
    ('ep_turn_p45_per_fh',   'advisory/turn_+45'),
    ('ep_turn_p60_per_fh',   'advisory/turn_+60'),
    ('ep_return_per_fh',     'advisory/return_to_initial_hdg'),
    ('ep_speed_up_per_fh',   'advisory/speed_up'),
    ('ep_speed_down_per_fh', 'advisory/speed_down'),

    ('ep_delay_mean_s',      'delay/mean_response_s'),
    ('ep_discarded',         'delay/advisories_discarded'),
    ('ep_repeats',           'delay/advice_re_selected'),
]


def finished_episodes(infos):
    """The summary dict of every episode that ended on this step."""
    return [info for info in infos if 'ep_reward_total' in info]


def save(model, run_dir, name):
    """A checkpoint and its VecNormalize statistics, which must travel together."""
    model.save(os.path.join(run_dir, name))
    model.get_env().save(os.path.join(run_dir, f'{name}_vecnorm.pkl'))


class LogEpisodes(BaseCallback):
    """Rolling means of the episode metrics, to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.windows = {key: deque(maxlen=STATS_WINDOW) for key, _ in METRICS}

    def _on_step(self):
        for info in finished_episodes(self.locals.get('infos', [])):
            for key, tag in METRICS:
                window = self.windows[key]
                window.append(info[key])
                self.logger.record(tag, sum(window) / len(window))
        return True


class RunUntil(BaseCallback):
    """Count finished episodes, print progress, and stop the run at `limit` episodes.

    Episodes vary in length, so this is a different stopping rule from a step count. The count
    is not stored in the checkpoint: a resumed run starts from zero and needs the remaining
    episodes passed to it.
    """

    def __init__(self, limit):
        super().__init__()
        self.limit    = int(limit)
        self.episodes = 0

    def _on_training_start(self):
        self.t0          = time.time()
        self.start_steps = self.num_timesteps
        self.next_print  = self.num_timesteps + PROGRESS_EVERY

    def _on_step(self):
        self.episodes += len(finished_episodes(self.locals.get('infos', [])))
        self.logger.record('episode/count', self.episodes)

        if self.num_timesteps >= self.next_print:
            self.next_print = self.num_timesteps + PROGRESS_EVERY
            elapsed = time.time() - self.t0
            rate    = (self.num_timesteps - self.start_steps) / max(elapsed, 1e-9)
            eta_h   = (self.limit - self.episodes) / max(self.episodes / elapsed, 1e-9) / 3600
            print(f'{self.episodes:>7,} / {self.limit:,} episodes '
                  f'({100 * self.episodes / self.limit:5.1f}%)   '
                  f'{self.num_timesteps:>11,} steps   {rate:6.1f} steps/s   '
                  f'elapsed {elapsed / 3600:5.2f} h   eta {eta_h:5.1f} h', flush=True)

        if self.episodes >= self.limit:
            print(f'reached {self.episodes:,} episodes at {self.num_timesteps:,} steps; '
                  f'stopping', flush=True)
            return False
        return True


class Checkpoint(BaseCallback):
    """Write last_model every `every` steps, so a crash does not cost the whole run."""

    def __init__(self, run_dir, every):
        super().__init__()
        self.run_dir = run_dir
        self.every   = every

    def _on_training_start(self):
        self.next_save = self.num_timesteps + self.every

    def _on_step(self):
        if self.every and self.num_timesteps >= self.next_save:
            self.next_save = self.num_timesteps + self.every
            save(self.model, self.run_dir, 'last_model')
            print(f'  saved @ {self.num_timesteps:>10,}', flush=True)
        return True


def delay_type_name(delay_mode, delay_mean_s):
    """Directory name for one delay type, e.g. 'lognormal_30s'; the baseline stays 'none'."""
    return 'none' if delay_mode == 'none' else f'{delay_mode}_{delay_mean_s:g}s'


def latest_checkpoint(run_dir):
    """The furthest-trained checkpoint that still has its statistics beside it, or None."""
    best, best_steps = None, -1
    for name in ('last_model', 'final_model'):
        path = os.path.join(run_dir, f'{name}.zip')
        if not (os.path.exists(path)
                and os.path.exists(os.path.join(run_dir, f'{name}_vecnorm.pkl'))):
            continue
        with zipfile.ZipFile(path) as archive:
            steps = json.loads(archive.read('data').decode()).get('num_timesteps', -1)
        if steps > best_steps:
            best, best_steps = name, steps
    return best


def train(delay_mode, seed, delay_mean_s, save_every=SAVE_EVERY, runs_root=RUNS_ROOT,
          overwrite=False, resume=False, max_episodes=MAX_EPISODES, sb3_default=False):
    delay_type = delay_type_name(delay_mode, delay_mean_s)
    run_dir    = os.path.join(runs_root, delay_type, f'{delay_type}_seed{seed}')

    checkpoint = latest_checkpoint(run_dir) if resume else None
    if resume and checkpoint is None:
        sys.exit(f'--resume: no checkpoint with its vecnorm in {run_dir}')
    if not resume and os.path.exists(run_dir) and not overwrite:
        sys.exit(f'{run_dir} already exists. Delete it, move it aside, or pass --overwrite '
                 f'to train over it.')
    os.makedirs(run_dir, exist_ok=True)

    env_seed = Random(seed).randrange(2 ** 63)
    venv = DummyVecEnv([lambda: AirspaceEnv(delay_mode=delay_mode, delay_mean_s=delay_mean_s,
                                            seed=env_seed)])

    if resume:
        env = VecNormalize.load(os.path.join(run_dir, f'{checkpoint}_vecnorm.pkl'),
                                VecMonitor(venv))
        env.training, env.norm_reward = True, True
        model = PPO.load(os.path.join(run_dir, checkpoint), env=env,
                         tensorboard_log=run_dir, device='cpu')
        print(f'resumed from {checkpoint} at {model.num_timesteps:,} steps', flush=True)
    else:
        env = VecNormalize(VecMonitor(venv))
        if sb3_default:
            # Nothing set but what every run needs: the seed, the log and the CPU.
            model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                        device='cpu')
        else:
            model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                        device='cpu', n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=N_EPOCHS)

    callbacks = CallbackList([RunUntil(max_episodes), LogEpisodes(),
                              Checkpoint(run_dir, save_every)])

    print(f'{delay_type}  seed {seed}  '
          f'n_steps {model.n_steps}  batch {model.batch_size}  epochs {model.n_epochs}  '
          f'until {max_episodes:,} episodes  '
          f'save every {save_every:,}  -> {run_dir}', flush=True)
    try:
        model.learn(STEP_CEILING - model.num_timesteps, callback=callbacks,
                    reset_num_timesteps=not resume)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
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
    default_mean_s = CONFIG['delay_mean_s']
    parser.add_argument('--delay-mean', dest='delay_mean', type=float, default=default_mean_s,
                        help=f'delay magnitude: the MEAN response time in seconds '
                             f'(default {default_mean_s:g}). Ignored when --delay none.')
    parser.add_argument('--max-episodes', type=int, default=MAX_EPISODES,
                        help=f'stop once this many episodes have finished '
                             f'(default {MAX_EPISODES:,})')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                        help=f'steps between last_model checkpoints (default {SAVE_EVERY:,}); '
                             f'0 saves only at the end')
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run from its last checkpoint, keeping '
                             'the step count and the TensorBoard curves continuous')
    parser.add_argument('--overwrite', action='store_true',
                        help='train into an existing run directory instead of refusing')
    parser.add_argument('--sb3-default', action='store_true',
                        help='PPO with SB3 defaults instead of the reported settings; '
                             'ignored with --resume, which keeps the saved settings')
    parser.add_argument('--runs-root', default=RUNS_ROOT,
                        help=f'where the run directory is created (default {RUNS_ROOT})')
    args = parser.parse_args()
    if args.max_episodes < 1:
        parser.error('--max-episodes must be at least 1')

    train(args.delay, args.seed, args.delay_mean, save_every=args.save_every,
          runs_root=args.runs_root, overwrite=args.overwrite, resume=args.resume,
          max_episodes=args.max_episodes, sb3_default=args.sb3_default)


if __name__ == '__main__':
    main()
