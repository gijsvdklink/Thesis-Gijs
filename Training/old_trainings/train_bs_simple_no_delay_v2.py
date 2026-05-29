"""
PPO training on bs_simple_no_delay_v2 with 4 parallel BlueSky simulations.

Each BlueSky simulation runs in its own subprocess (spawn) because BlueSky
uses global state and cannot be duplicated inside a single process.
SB3 sees  N_SIM × N_AIRCRAFT = 4 × 3 = 12  virtual environments.

Each run creates a timestamped folder:
    Runs_saved/bs/non_delay/simple/YYYYMMDD_HHMMSS_bs_simple_no_delay_v2/
        tensorboard/
        checkpoints/
        model.zip         (written on clean exit or Ctrl+C)

Monitor:
    python -m tensorboard.main --logdir Runs_saved/bs/non_delay/simple
"""

import multiprocessing as mp
import os
import sys
import time
from datetime import datetime

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from Environments.bs_simple_no_delay_v2 import AirspaceEnv, N_AIRCRAFT, OBS_DIM

# ── Settings ──────────────────────────────────────────────────────────────────

N_SIM              = 4
CHECKPOINT_EVERY   = 500_000      # save a checkpoint every this many timesteps
RUNS_ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Runs_saved', 'bs', 'non_delay', 'simple'))

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 1,
)

# ── Subprocess worker ─────────────────────────────────────────────────────────
# Must be at module level so multiprocessing can pickle it for spawn.

def _sim_worker(conn):
    """
    Runs one AirspaceEnv inside a dedicated subprocess.

    Tracks episode statistics and packages results in the same format that
    ParallelMARLVecEnv expects: (obs, rewards, dones, infos).
    Auto-resets on episode end and includes the summary in infos[0].
    """
    def blank_stats():
        return {'rewards': np.zeros(N_AIRCRAFT), 'los_steps': 0, 'steps': 0, 'actions': []}

    env   = AirspaceEnv()
    stats = blank_stats()
    conn.send('ready')

    while True:
        cmd, data = conn.recv()

        if cmd == 'reset':
            obs, _ = env.reset()
            stats  = blank_stats()
            conn.send(obs)

        elif cmd == 'step':
            obs, rewards, terminated, truncated, info = env.step(data)
            done  = terminated or truncated
            dones = np.full(N_AIRCRAFT, done)

            stats['rewards']   += rewards
            stats['los_steps'] += int(len(info['los_pairs']) > 0)
            stats['steps']     += 1
            stats['actions'].extend(data.tolist())

            infos = [{} for _ in range(N_AIRCRAFT)]
            if done:
                for i in range(N_AIRCRAFT):
                    infos[i]['terminal_observation'] = obs[i].copy()
                infos[0].update({
                    'mean_episode_reward': float(stats['rewards'].mean()),
                    'ep_los_steps':        stats['los_steps'],
                    'ep_length':           stats['steps'],
                    'action_distribution': np.bincount(stats['actions'], minlength=4).tolist(),
                })
                obs, _ = env.reset()
                stats  = blank_stats()

            conn.send((obs, rewards, dones, infos))

        elif cmd == 'close':
            conn.close()
            return


# ── Parallel VecEnv ───────────────────────────────────────────────────────────

class ParallelMARLVecEnv(VecEnv):
    """
    Runs N_SIM independent BlueSky simulations in separate subprocesses.

    Presents  num_envs = N_SIM × N_AIRCRAFT  virtual environments to SB3 PPO.
    Each agent still receives its own individual observation and reward —
    the parameter-sharing cooperative MARL structure is preserved.
    """

    def __init__(self, n_sim: int = N_SIM):
        self._n_sim      = n_sim
        self._n_per_sim  = N_AIRCRAFT

        ctx = mp.get_context('spawn')
        self._parent_conns = []
        self._procs        = []

        for _ in range(n_sim):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_sim_worker, args=(child_conn,), daemon=True)
            p.start()
            child_conn.close()          # parent only needs its end
            self._parent_conns.append(parent_conn)
            self._procs.append(p)

        for conn in self._parent_conns:
            assert conn.recv() == 'ready', 'Worker failed to start'

        obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Discrete(4)
        super().__init__(n_sim * N_AIRCRAFT, obs_space, act_space)

    # ── VecEnv core ───────────────────────────────────────────────────────────

    def reset(self):
        for conn in self._parent_conns:
            conn.send(('reset', None))
        chunks = [conn.recv() for conn in self._parent_conns]
        return np.concatenate(chunks, axis=0)       # (N_SIM × N_AIRCRAFT, OBS_DIM)

    def step_async(self, actions):
        n = self._n_per_sim
        for i, conn in enumerate(self._parent_conns):
            conn.send(('step', actions[i * n:(i + 1) * n]))

    def step_wait(self):
        results = [conn.recv() for conn in self._parent_conns]
        obs     = np.concatenate([r[0] for r in results], axis=0)
        rewards = np.concatenate([r[1] for r in results], axis=0)
        dones   = np.concatenate([r[2] for r in results], axis=0)
        infos   = [info for r in results for info in r[3]]
        return obs, rewards, dones, infos

    def close(self):
        for conn in self._parent_conns:
            try:
                conn.send(('close', None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10)

    # ── VecEnv interface boilerplate ──────────────────────────────────────────

    def get_attr(self, attr_name, indices=None):           return [None]  * self.num_envs
    def set_attr(self, attr_name, value, indices=None):    pass
    def env_method(self, method_name, *args, **kwargs):    return [None]  * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def seed(self, seed=None):                             return [None]  * self.num_envs


# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """Logs per-episode MARL stats to TensorBoard."""

    _ACTION_LABELS = ['-10deg', '+10deg', 'hold', 'snapback']

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'mean_episode_reward' not in info:
                continue
            self.logger.record('marl/mean_ep_reward', info['mean_episode_reward'])
            self.logger.record('marl/ep_los_steps',   info['ep_los_steps'])
            self.logger.record('marl/ep_length',       info['ep_length'])
            total = max(sum(info['action_distribution']), 1)
            for label, count in zip(self._ACTION_LABELS, info['action_distribution']):
                self.logger.record(f'actions/{label}', count / total)
        return True


class CheckpointCallback(BaseCallback):
    """Saves the model every CHECKPOINT_EVERY timesteps, independent of num_envs."""

    def __init__(self, save_path: str, name_prefix: str):
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
            print(f'  [ckpt] {path}.zip')
        return True


class ProgressCallback(BaseCallback):
    """Prints a heartbeat every 10k steps."""

    def _on_training_start(self):
        self._start      = time.time()
        self._last_print = 0
        print(f"\nRun dir     : {self._run_dir}")
        print(f"TensorBoard : python -m tensorboard.main --logdir \"{self._tb_dir}\"")
        print(f"Parallel    : {N_SIM} simulations × {N_AIRCRAFT} agents = {N_SIM * N_AIRCRAFT} virtual envs")
        print(f"Training    : 100 M steps  (Ctrl+C to stop early)\n")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= 10_000:
            elapsed          = (time.time() - self._start) / 60
            self._last_print = self.num_timesteps
            print(f"  {self.num_timesteps:>10,} steps  |  {elapsed:6.1f} min elapsed")
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_bs_simple_no_delay_v2"
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    tb_dir   = os.path.join(run_dir, 'tensorboard')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')

    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    env = VecMonitor(ParallelMARLVecEnv(N_SIM))

    model = PPO(
        policy          = 'MlpPolicy',
        env             = env,
        tensorboard_log = tb_dir,
        **PPO_KWARGS,
    )

    progress_cb          = ProgressCallback()
    progress_cb._run_dir = run_dir
    progress_cb._tb_dir  = tb_dir

    callbacks = CallbackList([
        progress_cb,
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_path   = ckpt_dir,
            name_prefix = 'bs_simple_no_delay_v2',
        ),
    ])

    model_path = os.path.join(run_dir, 'model')
    try:
        model.learn(
            total_timesteps     = 100_000_000,
            callback            = callbacks,
            tb_log_name         = 'ppo',
            reset_num_timesteps = True,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    finally:
        model.save(model_path)
        print(f"\nModel saved : {model_path}.zip")
        print(f"Visualise   : python -m Visualisation.visualise_bs_simple_v2 --mode \"{model_path}.zip\"")


if __name__ == '__main__':
    main()
