"""The reward and its three terms during training, one line per seed and one colour per delay type.

Plots episode/reward_per_flight_hour and the LoS, action and drift terms that add up to it,
as four panels. Everything is per flight hour, so the three terms sum to the total. A resumed
run has a second event file, which takes over from its first step. Reading the event files is
slow, so the curves are cached as a CSV; --reload reads them again.

python Visualisation/reward_terms.py [--runs DIR] [--out DIR] [--reload]
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

# The panels, in reading order: total, LoS, action, drift.
PANELS = [('episode/reward_per_flight_hour', 'Total reward'),
          ('reward/los_per_flight_hour',     'LoS penalty'),
          ('reward/work_per_flight_hour',    'Action penalty'),
          ('reward/drift_per_flight_hour',   'Drift penalty')]
TAGS = [tag for tag, _ in PANELS]

COLOURS = {'none':              '#2a78d6',
           'deterministic_30s': '#eb6834',
           'lognormal_30s':     '#1baf7a'}
LABELS  = {'none':              'No delay',
           'deterministic_30s': 'Deterministic delay, 30 s',
           'lognormal_30s':     'Lognormal delay, 30 s'}

SMOOTHING = 0.999   # on all points of a run, as in training_curves.py


def read_run(run_dir):
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = sorted(glob.glob(os.path.join(run_dir, 'PPO_1', 'events.out.tfevents.*')),
                   key=lambda path: int(os.path.basename(path).split('.')[3]))
    points = {tag: {} for tag in TAGS}
    for path in files:
        first = None
        for event in EventFileLoader(path).Load():
            for value in event.summary.value:
                if value.tag not in points:
                    continue
                if first is None:   # a resumed run overwrites what the earlier file logged
                    first = event.step
                    for tag in TAGS:
                        points[tag] = {step: v for step, v in points[tag].items() if step < first}
                points[value.tag][event.step] = (
                    value.simple_value if value.HasField('simple_value')
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
            for tag in TAGS:
                rows += [(delay_type, seed, tag, step, value)
                         for step, value in sorted(points[tag].items())]
            print(f'{delay_type} seed {seed}: ' +
                  ', '.join(f'{len(points[tag])} {tag.split("/")[-1]}' for tag in TAGS), flush=True)
    frame = pd.DataFrame(rows, columns=['delay_type', 'seed', 'tag', 'step', 'value'])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default=os.path.join(ROOT, '18septv2'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'Figures', '18septv2'))
    parser.add_argument('--reload', action='store_true', help='read the event files again')
    args = parser.parse_args()

    frame = load(args.runs, os.path.join(args.out, 'reward_terms.csv'), args.reload)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (tag, title) in zip(axes.ravel(), PANELS):
        panel = frame[frame['tag'] == tag]
        for (delay_type, seed), run in panel.groupby(['delay_type', 'seed']):
            colour = COLOURS[delay_type]
            steps  = run['step'] / 1e6
            axis.plot(steps, run['value'], color=colour, alpha=0.15, linewidth=0.8)
            axis.plot(steps, run['value'].ewm(alpha=1 - SMOOTHING).mean(), color=colour,
                      linewidth=1.5, label=LABELS[delay_type] if seed == 1 else None)
        # The reward is never positive, and the first two million steps would squash the rest.
        axis.set_ylim(panel.loc[panel['step'] > 2e6, 'value'].min(), 0)
        axis.set_xlim(0, panel['step'].max() / 1e6)
        axis.set_title(title)
        axis.set_ylabel('Reward per flight hour')
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel('Training steps [millions]')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(LABELS[delay_type]) for delay_type in COLOURS]
    figure.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout()

    for name in ('reward_terms.pdf', 'reward_terms.png'):
        figure.savefig(os.path.join(args.out, name), dpi=200, bbox_inches='tight')
    print('wrote', os.path.join(args.out, 'reward_terms.pdf'))


if __name__ == '__main__':
    main()
