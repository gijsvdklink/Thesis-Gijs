"""The episode reward during training, one line per seed and one colour per delay type.

Plots rollout/ep_rew_mean, the unnormalised episode reward that SB3 logs, smoothed like
TensorBoard. A resumed run has a second event file, which takes over from its first step.
Reading the event files is slow, so the curves are cached as a CSV; --reload reads them again.

python Visualisation/training_curves.py [--runs DIR] [--out DIR] [--reload]
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TAG  = 'rollout/ep_rew_mean'

COLOURS = {'none':              '#2a78d6',
           'deterministic_30s': '#eb6834',
           'lognormal_30s':     '#1baf7a'}
LABELS  = {'none':              'No delay',
           'deterministic_30s': 'Deterministic delay, 30 s',
           'lognormal_30s':     'Lognormal delay, 30 s'}

SMOOTHING = 0.999   # on all ~9000 points per run; TensorBoard's 0.99 on its ~1000 samples


def read_run(run_dir):
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = sorted(glob.glob(os.path.join(run_dir, 'PPO_1', 'events.out.tfevents.*')),
                   key=lambda path: int(os.path.basename(path).split('.')[3]))
    points = {}
    for path in files:
        first = None
        for event in EventFileLoader(path).Load():
            for value in event.summary.value:
                if value.tag != TAG:
                    continue
                if first is None:
                    first  = event.step
                    points = {step: v for step, v in points.items() if step < first}
                points[event.step] = (value.simple_value if value.HasField('simple_value')
                                      else float(tensor_util.make_ndarray(value.tensor)))
    return run_dir, points


def load(runs_dir, cache, reload):
    if os.path.exists(cache) and not reload:
        return pd.read_csv(cache)
    rows = []
    with ProcessPoolExecutor() as pool:
        for run_dir, points in pool.map(read_run, sorted(glob.glob(os.path.join(runs_dir, '*', '*_seed*')))):
            delay_type = os.path.basename(os.path.dirname(run_dir))
            seed       = int(run_dir.rsplit('seed', 1)[1])
            rows += [(delay_type, seed, step, value) for step, value in sorted(points.items())]
            print(f'{delay_type} seed {seed}: {len(points)} points', flush=True)
    frame = pd.DataFrame(rows, columns=['delay_type', 'seed', 'step', 'reward'])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default=os.path.join(ROOT, 'Models', '35MLN steps'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'Figures'))
    parser.add_argument('--reload', action='store_true', help='read the event files again')
    args = parser.parse_args()

    frame = load(args.runs, os.path.join(args.out, 'training_reward.csv'), args.reload)

    # Only the smoothed curves: the raw values scatter too much to read seven seeds per type.
    frame['smoothed'] = frame.groupby(['delay_type', 'seed'])['reward'].transform(
        lambda reward: reward.ewm(alpha=1 - SMOOTHING).mean())

    figure, axis = plt.subplots(figsize=(10, 6))
    for (delay_type, seed), run in frame.groupby(['delay_type', 'seed']):
        axis.plot(run['step'] / 1e6, run['smoothed'], color=COLOURS[delay_type],
                  linewidth=1.5, label=LABELS[delay_type] if seed == 1 else None)

    # The reward is never positive, and the first two million steps would squash the rest.
    axis.set_ylim(frame.loc[frame['step'] > 2e6, 'reward'].min(), 0)
    axis.set_xlim(0, frame['step'].max() / 1e6)
    axis.set_xlabel('Training steps [millions]')
    axis.set_ylabel('Episode reward')
    axis.grid(alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    order = [labels.index(LABELS[delay_type]) for delay_type in COLOURS]
    axis.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right')

    for name in ('training_reward.pdf', 'training_reward.png'):
        figure.savefig(os.path.join(args.out, name), dpi=200, bbox_inches='tight')
    print('wrote', os.path.join(args.out, 'training_reward.pdf'))


if __name__ == '__main__':
    main()
