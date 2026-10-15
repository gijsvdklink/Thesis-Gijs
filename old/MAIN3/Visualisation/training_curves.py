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

# A resumed run can be missing whole stretches of logging: the event file of the session
# that did the training was never written or never kept. Joining across such a stretch draws
# a straight line through steps that were never measured, which reads as a long plateau.
# A jump larger than this many steps is treated as a break: the line stops and restarts, and
# the smoothing restarts with it so the first points after the break are not dragged by the
# value from before it.
GAP_STEPS = 2_000_000


def read_run(run_dir):
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = sorted(glob.glob(os.path.join(run_dir, 'PPO_1', 'events.out.tfevents.*')),
                   key=lambda path: int(os.path.basename(path).split('.')[3]))
    points = {}
    for path in files:
        session = {}
        for event in EventFileLoader(path).Load():
            for value in event.summary.value:
                if value.tag != TAG:
                    continue
                session[event.step] = (value.simple_value if value.HasField('simple_value')
                                       else float(tensor_util.make_ndarray(value.tensor)))
        if not session:
            continue
        # A restart rewinds the step counter, so the newer session replaces the older curve
        # -- but ONLY over the steps it actually reaches. A session resumed from a stale
        # checkpoint and abandoned after a few rollouts would otherwise erase every later
        # point of the branch the weights really followed, leaving a gap the width of the
        # rewind. Supersede inside this session's own span and leave the rest standing.
        low, high = min(session), max(session)
        points = {step: value for step, value in points.items()
                  if step < low or step > high}
        points.update(session)
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

    # One segment per continuously logged stretch; see GAP_STEPS.
    frame = frame.sort_values(['delay_type', 'seed', 'step'])
    step_jump = frame.groupby(['delay_type', 'seed'])['step'].diff()
    frame['segment'] = (step_jump > GAP_STEPS).groupby(
        [frame['delay_type'], frame['seed']]).cumsum()

    # Only the smoothed curves: the raw values scatter too much to read seven seeds per type.
    frame['smoothed'] = frame.groupby(['delay_type', 'seed', 'segment'])['reward'].transform(
        lambda reward: reward.ewm(alpha=1 - SMOOTHING).mean())

    figure, axis = plt.subplots(figsize=(10, 6))
    labelled = set()
    for (delay_type, seed), run in frame.groupby(['delay_type', 'seed']):
        segments = [part for _, part in run.groupby('segment')]

        # Across a gap nothing was measured, so the join is drawn dashed and faint: the
        # curve stays readable end to end without claiming the missing stretch was flat.
        for before, after in zip(segments, segments[1:]):
            axis.plot([before['step'].iloc[-1] / 1e6, after['step'].iloc[0] / 1e6],
                      [before['smoothed'].iloc[-1], after['smoothed'].iloc[0]],
                      color=COLOURS[delay_type], linewidth=1.0,
                      linestyle=(0, (4, 4)), alpha=0.35, zorder=1)

        for part in segments:
            label = LABELS[delay_type] if delay_type not in labelled else None
            labelled.add(delay_type)
            axis.plot(part['step'] / 1e6, part['smoothed'], color=COLOURS[delay_type],
                      linewidth=1.5, label=label, zorder=2)

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
