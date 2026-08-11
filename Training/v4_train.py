"""
PPO training for one arm of the action-response delay experiment.

Three arms, one delay condition each. Run the three arms CONCURRENTLY with the same
seed, so the only difference between the curves is the delay condition, and repeat for
several seeds:

    python -m Training.v4_train --delay none          --seed 0
    python -m Training.v4_train --delay deterministic --seed 0
    python -m Training.v4_train --delay probabilistic --seed 0
    tensorboard --logdir Runs_saved/experiments

Run the three in three terminals at the same time. They are sized to share the machine:
2 workers each, 6 workers on 6 physical cores. Repeat with --seed 1000 and --seed 2000
for independent repeats, one seed at a time.

TUNED FOR A 2019 MacBook Pro (Core i7-9750H, 6 physical cores / 12 threads), and measured
in exactly this configuration rather than extrapolated: three arms at N_ENVS=2 sustain
113 / 119 / 133 steps/s, i.e. 365 env-steps/s aggregate. That beats a single run with 4
workers (317 steps/s) because the three main processes overlap their gradient updates with
each other's rollouts. Six workers land on six physical cores; hyperthreads add nothing
here (8 workers measured no faster than 4).

Running all three arms together also keeps them in lockstep, which matters more on a
laptop than on a server: if you stop early you have three comparable arms at 15M rather
than one finished arm at 45M.

Budget honestly. At ~122 steps/s per arm, 50M steps is about FIVE DAYS of fully pegged
CPU, and a laptop thermally throttles over that span, so call it five to six days for one
seed of all three arms. Pilot at 10M first -- about a day -- look at the curves, and only
then commit to the full budget:

    python -m Training.v4_train --delay none --seed 0 --timesteps 10000000

Seeds are spaced 1000 apart on purpose. PPO(seed=s) re-seeds the vectorised env, handing
worker i the seed s+i, so consecutive run seeds (0, 1, 2, ...) would replay each other's
scenarios with N_ENVS > 1 and the "independent" repeats would not be independent.

Everything below is stock stable-baselines3 PPO apart from four settings, each of which
is about this environment rather than about tuning for its own sake:

1. GAMMA = 0.995. One RL step is 5 s and the conflict horizon t_warn is 360 s, i.e. 72
   steps. The default 0.99 discounts a separation loss at the far edge of that horizon to
   0.99^72 = 0.48; 0.995 gives 0.70 and an effective horizon of 200 steps (1000 s), about
   2.8x t_warn. The delayed arms need this most: the delay inserts 6 dead steps between
   the instruction and the turn, while the cost of the radio call is charged immediately.

2. ENT_COEF = 0.01. The reward is purely negative and `hold` is the only free action, so
   the immediate gradient always points at "say nothing"; the payoff for turning arrives
   up to 72 steps later, discounted and noisy. With the SB3 default of 0.0 the policy can
   collapse onto hold before it ever experiences that turning prevents a loss of
   separation -- and it collapses SOONER under delay, which would make "delay cannot be
   learned" an artefact of exploration rather than a result.

3. N_ENVS = 2 with a matched rollout buffer, so that three concurrent arms put exactly one
   worker on each of the six physical cores. It also decorrelates the rollout: two sectors
   of different size and traffic count feed every update instead of one.

4. VecNormalize(gamma=GAMMA). VecNormalize keeps its OWN discount for the running return
   statistic that sets the reward scale, and it defaults to 0.99 regardless of PPO's
   gamma. Left unset it would normalise rewards for the wrong horizon.

Two things are not negotiable, and both are about correctness rather than performance:

* VecNormalize(norm_obs=True). The environment emits raw NM / kt / s / rad, so this is
  the only thing standardising the inputs. The *_vecnorm.pkl saved beside each model MUST
  be loaded again at evaluation or visualisation time.

* The training environments run in SUBPROCESSES. BlueSky's traffic is a process-global
  singleton, so an evaluation episode in this process would call bs.traf.reset() and wipe
  a training environment's aircraft mid-run. The subprocesses keep them apart.

Throughput scales with PHYSICAL cores, not threads: this machine reports 12 logical CPUs
but has 6 real ones, and hyperthreads add nothing here (measured). Six workers plus three
mostly-idle main processes fit it exactly, and need about 1.9 GB of RAM, since each worker
loads its own copy of BlueSky's navigation database (~215 MB measured).

CPU is the right target regardless -- SB3 recommends it for an MlpPolicy, and this laptop
has no usable torch GPU backend for it anyway.

Moving to a bigger machine later means changing one number: set N_ENVS so that
N_ENVS x 3 arms equals the physical core count (4 on a 12-core node), and scale N_STEPS
the other way so the rollout buffer stays at 8192.
"""

import os

# Must precede the torch import. The policy is a 64x64 MLP, far too small to benefit from
# intra-op threading, and three concurrent arms x 4 workers each would otherwise spawn one
# OpenMP pool per process and spend more time in spin-waits than in the network.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4 import AirspaceEnv

# -- Settings ------------------------------------------------------------------

# Overridable with --timesteps. The BlueSky-Gym benchmark (Groot et al., SESAR Innovation
# Days 2024) found 2M too few for PPO to converge on its conflict-resolution environments,
# and those are simpler than this one -- hence a large budget. See the header for what
# 50M actually costs on this laptop, and pilot at 10M first.
TOTAL_TIMESTEPS = 50_000_000

# Parallel environments. Keep N_ENVS x (number of arms run at once) at or below the
# machine's PHYSICAL core count: 2 x 3 arms = 6 on a 6-core MacBook Pro. Past that,
# throughput stops improving -- measured on this machine, 1/2/4/8 workers give
# 106/169/317/320 env-steps/s, so 8 workers on 6 cores buys nothing over 4.
N_ENVS     = 2
# Rollout buffer = 2 x 4096 = 8192 steps, which is 11 h of simulated traffic from two
# sectors. Sized for the 50M budget: it still gives ~6,100 policy updates, far more than
# enough, and a large rollout is the main defence against the variance in the advantage
# estimate -- the LoS penalty fires for ANY pair in the sector, most of which the focus
# aircraft cannot influence.
N_STEPS    = 4096                 # per env
BATCH_SIZE = 512                  # 8192 / 512 = 16 minibatches per epoch

# See the module docstring for why these two deviate from the SB3 defaults.
GAMMA    = 0.995
ENT_COEF = 0.01

# Deterministic evaluation, and the cadence at which a new best can be spotted and saved:
# best_model exists only as a function of an eval score, so the two are the same number.
#
# An episode is 2400-4700 steps and the eval runs SERIALLY in this process while both
# workers idle, so the cost is worth knowing. It is pure step-counting: 3 episodes ~=
# 10,650 single-worker steps, against 250,000 / 1.6 ~= 156,000 single-worker steps of
# training, so about 7% -- and less than that in practice, because the other two arms
# borrow the cores this one stops using. That buys 200 eval points and a best_model never
# more than 250k steps stale. (At the original 25k cadence the ratio was over 100%: the
# eval cost more than the training it interrupted.)
# Each eval writes best_model if the score improved, and last_model unconditionally -- a
# 50M arm runs for days, and a crash at hour 80 with only an early best_model on disk
# would be painful.
EVAL_EVERY      = 250_000
EVAL_SEEDS      = (10_001, 10_002, 10_003)   # held-out scenarios, identical for all arms
PROGRESS_EVERY  = 50_000          # throughput line in the terminal

RUNS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'experiments'))

ACTION_LABELS = ['-60', '-45', '-30', 'hold', '+30', '+45', '+60', 'return', 'spd+', 'spd-']

# Episode-summary key -> TensorBoard tag, shared by the training and eval logs.
METRICS = [('ep_reward_total', 'reward_total'), ('ep_length', 'length'),
           ('ep_los_events', 'los_events'), ('ep_los_steps', 'los_steps'),
           ('ep_arrival_rate', 'arrival_rate')]


# -- Callbacks -----------------------------------------------------------------

class LogEpisodes(BaseCallback):
    """Log each finished training episode. These come from the EXPLORING policy."""

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'ep_reward_total' not in info:
                continue
            for key, tag in METRICS:
                self.logger.record_mean(f'episode/{tag}', info[key])

            dist  = info.get('action_distribution', [])
            total = max(sum(dist), 1)
            for label, count in zip(ACTION_LABELS, dist):
                self.logger.record_mean(f'actions/{label}', count / total)
        return True


class EvaluateAndCheckpoint(BaseCallback):
    """Every EVAL_EVERY steps, measure what the agent can ACTUALLY do, and keep the
    best policy seen.

    PPO's policy is stochastic, so the episode/ curves include exploratory actions the
    agent would not choose if asked for its best guess. These rollouts use
    predict(deterministic=True) on held-out scenarios and log under eval/ -- expect them
    to sit above the training curves. best_model is checkpointed on this measured score
    rather than on the noisy training reward.

    Each eval writes up to two model/VecNormalize pairs into the run directory:
        best_model   whenever the measured score beats every previous eval
        last_model   every time, so a crashed multi-day run resumes from at most
                     EVAL_EVERY steps back
    plus final_model when learn() returns or is interrupted.

    Note for the write-up: best_model is selected by maximising a 3-episode score, so its
    eval/ number is optimistically biased. Report final_model on a larger held-out set
    (Validation/) rather than quoting the best eval point.
    """

    def __init__(self, eval_env, save_dir):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.last_eval = 0
        self.best = -float('inf')

    def _save(self, name):
        """Write a model + its VecNormalize stats, overwriting the previous pair. The two
        must travel together: the policy is meaningless without the obs stats it was
        trained under."""
        stem = os.path.join(self.save_dir, name)
        self.model.save(stem)
        self.model.get_env().save(stem + '_vecnorm.pkl')

    def _one_episode(self, seed):
        """A full deterministic episode, observations normalised with the training stats."""
        vecnorm = self.model.get_env()
        obs, _ = self.eval_env.reset(seed=seed)
        while True:
            action, _ = self.model.predict(vecnorm.normalize_obs(obs.reshape(1, -1)),
                                           deterministic=True)
            obs, _, _, truncated, _ = self.eval_env.step(int(np.asarray(action).flat[0]))
            if truncated:
                return self.eval_env._episode_summary()

    def _on_step(self):
        if not EVAL_EVERY or self.num_timesteps - self.last_eval < EVAL_EVERY:
            return True
        self.last_eval = self.num_timesteps

        runs = [self._one_episode(seed) for seed in EVAL_SEEDS]
        for key, tag in METRICS:
            self.logger.record(f'eval/{tag}', sum(r[key] for r in runs) / len(runs))

        score = sum(r['ep_reward_total'] for r in runs) / len(runs)
        if score > self.best:
            self.best = score
            self._save('best_model')
            note = 'NEW BEST, saved'
        else:
            note = f'best {self.best:>9.1f}'

        self._save('last_model')          # crash insurance, independent of the score
        print(f'  eval @ {self.num_timesteps:>10,}   reward {score:>9.1f}   {note}',
              flush=True)
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

def train(delay_mode, seed, total_timesteps):
    run_name = f'v4_{delay_mode}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}'
    run_dir  = os.path.join(RUNS_ROOT, delay_mode, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Subprocesses, one per env -- see the module docstring. delay_mode travels via
    # env_kwargs so it reaches the workers; editing CONFIG here would not.
    venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs={'delay_mode': delay_mode})
    env = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)

    # verbose=0: the Progress callback prints a compact throughput line instead of SB3's
    # full table after every rollout. Everything not named here is an SB3 default.
    model = PPO('MlpPolicy', env, seed=seed, verbose=0, tensorboard_log=run_dir,
                n_steps=N_STEPS, batch_size=BATCH_SIZE,
                gamma=GAMMA, ent_coef=ENT_COEF)

    # The evaluation environment lives here in the main process, alone with BlueSky.
    eval_env = AirspaceEnv(delay_mode=delay_mode)
    callbacks = CallbackList([Progress(), LogEpisodes(),
                              EvaluateAndCheckpoint(eval_env, run_dir)])

    print(f'{delay_mode}  seed {seed}  {total_timesteps:,} steps  '
          f'{N_ENVS} envs  -> {run_dir}', flush=True)
    try:
        model.learn(total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print('interrupted', flush=True)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        env.save(os.path.join(run_dir, 'final_vecnorm.pkl'))
        env.close()
        print(f'saved to {run_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--delay', required=True,
                        choices=['none', 'deterministic', 'probabilistic'],
                        help='action-response delay condition (the experiment variable)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed; space repeats 1000 apart (0, 1000, 2000, ...)')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help=f'training steps (default {TOTAL_TIMESTEPS:,})')
    args = parser.parse_args()
    train(args.delay, args.seed, args.timesteps)


if __name__ == '__main__':
    main()
