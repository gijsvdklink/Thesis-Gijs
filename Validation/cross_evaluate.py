"""
Cross-evaluation for the delay experiment: train under one action-response condition,
test under another. Writes TensorBoard scalars so the arms can be compared as curves.

The three arms share an identical observation space (27 features) and action space
(Discrete 10) BY DESIGN -- the 'pending' and 'wait_s' features exist in the no-delay arm
too, where they are simply always 0. That makes a checkpoint from one arm loadable against another,
and it is the question this script answers: what happens to a policy that learned assuming
instant compliance when it meets pilots who take 25 s?

    python -m Validation.cross_evaluate --runs Runs_saved/experiments --episodes 10
    tensorboard --logdir Runs_saved/cross_eval

Each (trained, tested) pair becomes its own TensorBoard run named "<trained>_on_<tested>",
so the baseline row shows up as none_on_none / none_on_deterministic / none_on_probabilistic
and can be overlaid directly.

CAVEAT on the baseline row: a policy trained with delay_mode='none' never saw pending=1 or
wait_s>0, so its VecNormalize statistics give both features ~zero variance and the normaliser
maps them to the clip bound at test time. Its off-diagonal cells therefore mix a wrong
strategy with an out-of-distribution input. The script prints an OOD factor per model so
you can see how large that effect is; do not report those cells as a pure strategy result.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from torch.utils.tensorboard import SummaryWriter

from Environments.v4 import AirspaceEnv, OBS_OWNSHIP_LABELS

MODES = ('none', 'deterministic', 'probabilistic')

# (episode-summary key, TensorBoard tag)
METRICS = [
    ('ep_los_per_fh',     'safety/los_per_fh'),
    ('ep_los_events',     'safety/los_events'),
    ('ep_los_steps',      'safety/los_steps'),
    ('ep_arr_on_route',   'arrival/on_route'),
    ('ep_arr_xtrack',     'arrival/xtrack'),
    ('ep_arr_ref',        'arrival/ref'),
    ('ep_reward_total',   'episode/reward_total'),
    ('ep_length',         'episode/length'),
    ('ep_tlos_at_issue',  'strategy/tlos_at_issue'),
    ('ep_turn_magnitude', 'strategy/turn_magnitude'),
    ('ep_pending_frac',   'delay/pending_frac'),
    ('ep_mean_delay_s',   'delay/mean_delay_s'),
    ('ep_amendments',     'delay/amendments'),
]

# Shown in the terminal summary; the rest are TensorBoard-only.
HEADLINE = ['ep_los_per_fh', 'ep_arr_on_route', 'ep_arr_ref', 'ep_reward_total']


def find_checkpoint(runs_root, mode):
    """Newest best_model.zip (falling back to final_model.zip) for one training arm.

    best_model lives in <run>/checkpoints/ beside best_model_vecnorm.pkl; final_model lives
    in <run>/ beside final_vecnorm.pkl -- different depths, so resolve the normaliser
    relative to each model rather than assuming one layout.
    """
    for stem, vecname in (('checkpoints/best_model', None), ('final_model', 'final_vecnorm.pkl')):
        hits = sorted(glob.glob(os.path.join(runs_root, mode, '*', f'{stem}.zip')))
        if not hits:
            continue
        model = hits[-1]
        vec = (model.replace('.zip', '_vecnorm.pkl') if vecname is None
               else os.path.join(os.path.dirname(model), vecname))
        if os.path.exists(vec):
            return model, vec
    return None, None


def pending_ood_factor(vecnorm_path):
    """Normalised value that pending=1 takes under this model's training statistics.
    Near 1 means the model saw delays in training; a huge value means it never did."""
    i = OBS_OWNSHIP_LABELS.index('pending')
    with open(vecnorm_path, 'rb') as f:
        vn = pickle.load(f)
    mean, var = float(vn.obs_rms.mean[i]), float(vn.obs_rms.var[i])
    return (1.0 - mean) / np.sqrt(var + vn.epsilon)


def evaluate(model_path, vecnorm_path, test_delay, episodes, n_envs, seed, writer):
    """Run `episodes` episodes under `test_delay`, logging each one to TensorBoard."""
    venv = make_vec_env(AirspaceEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed,
                        env_kwargs={'delay_mode': test_delay})
    env = VecNormalize.load(vecnorm_path, VecMonitor(venv))
    env.training = False          # freeze training-time statistics
    env.norm_reward = False       # raw rewards, so cells are comparable

    model = PPO.load(model_path, device='cpu')
    done_eps, obs = [], env.reset()

    while len(done_eps) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        for info in infos:
            if 'ep_los_per_fh' not in info or len(done_eps) >= episodes:
                continue
            for key, tag in METRICS:
                writer.add_scalar(tag, info[key], len(done_eps))
            done_eps.append(info)
    env.close()
    return {key: float(np.mean([e[key] for e in done_eps])) for key, _ in METRICS}


def main():
    ap = argparse.ArgumentParser(description='Cross-evaluate delay-experiment policies.')
    ap.add_argument('--runs', default='Runs_saved/experiments',
                    help='root holding none/ deterministic/ probabilistic/ run directories')
    ap.add_argument('--out', default='Runs_saved/cross_eval',
                    help='TensorBoard output root')
    ap.add_argument('--trained', nargs='*', choices=MODES, default=None,
                    help='which trained arms to evaluate (default: all found). '
                         'Use "--trained none" for just the baseline row.')
    ap.add_argument('--episodes', type=int, default=10, help='episodes per cell')
    ap.add_argument('--n-envs', type=int, default=6, help='parallel envs')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ckpts = {}
    for mode in MODES:
        if args.trained and mode not in args.trained:
            continue
        m, v = find_checkpoint(args.runs, mode)
        if m:
            ckpts[mode] = (m, v)
            print(f'{mode:<15} {os.path.relpath(m, args.runs)}')
        else:
            print(f'{mode:<15} no checkpoint -- skipping')
    if not ckpts:
        sys.exit(f'no checkpoints under {args.runs}')

    print('\npending=1 as the model\'s normaliser sees it (huge => never trained with delays):')
    for mode, (_, v) in ckpts.items():
        f = pending_ood_factor(v)
        print(f'  {mode:<15} {f:>12.1f}' + ('   <-- out of distribution' if abs(f) > 50 else ''))

    summary = {}
    for trained in ckpts:
        for tested in MODES:
            name = f'{trained}_on_{tested}'
            print(f'\n>>> {name}', flush=True)
            writer = SummaryWriter(os.path.join(args.out, name))
            summary[(trained, tested)] = evaluate(
                *ckpts[trained], tested, args.episodes, args.n_envs, args.seed, writer)
            writer.close()

    print(f'\n{"":<28}' + ''.join(f'{k.replace("ep_", ""):>18}' for k in HEADLINE))
    for (trained, tested), res in summary.items():
        label = f'{trained} on {tested}'
        print(f'{label:<28}' + ''.join(f'{res[k]:>18.3f}' for k in HEADLINE))

    print(f'\nTensorBoard:  tensorboard --logdir {args.out}')
    print('Each cell is a separate run, so overlay e.g. none_on_* to see how the baseline')
    print('policy degrades as the pilots get slower.')


if __name__ == '__main__':
    main()
