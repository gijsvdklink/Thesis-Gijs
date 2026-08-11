"""One PNG per training metric, comparing the three delay arms.

Reads the TensorBoard logs under Runs_saved/experiments/<arm>/.
Regenerate with:  python figures/plot_training.py
"""
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'Runs_saved', 'experiments')

ARMS = [('none', 'C0'), ('deterministic', 'C1'), ('probabilistic', 'C2')]
SMOOTH = 25          # rolling-mean window, in rollouts (~8192 steps each)

TURNS = ['actions/-60', 'actions/-45', 'actions/-30',
         'actions/+30', 'actions/+45', 'actions/+60']

# filename, title, y label, tag(s) to sum, scale
PLOTS = [
    ('reward',       'Episode reward',       'Total reward',           'episode/reward_total',  1),
    ('los',          'Separation losses',    'LoS events per episode', 'episode/los_events',    1),
    ('arrival_rate', 'Arrival rate',         'On-route exits [%]',     'episode/arrival_rate',  100),
    ('turns',        'Heading instructions', 'Share of actions [%]',   TURNS,                   100),
    ('speed',        'Speed instructions',   'Share of actions [%]',   ['actions/spd+',
                                                                       'actions/spd-'],        100),
]


def load(arm):
    """Scalars from the most recent run of one arm THAT HAS DATA, as {tag: (steps, values)}.

    A run killed during start-up leaves a near-empty event file behind. Taking the newest
    directory blindly picks one of those over a 16-hour run and silently drops the arm
    from the plot, so skip anything without a real curve in it.
    """
    for run in sorted(glob.glob(os.path.join(RUNS, arm, 'v4_*', 'PPO_1')), reverse=True):
        ea = EventAccumulator(run, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags()['scalars']
        if 'episode/reward_total' in tags and len(ea.Scalars('episode/reward_total')) > 5:
            print(f'  {arm:<14} {os.path.basename(os.path.dirname(run))}')
            return {t: np.array([[e.step, e.value] for e in ea.Scalars(t)]).T for t in tags}
    print(f'  {arm:<14} no run with data')
    return None


def smooth(y, n=SMOOTH):
    """Rolling mean, same length as the input (the head is averaged over fewer points)."""
    n = min(n, len(y))
    if n < 2:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return np.concatenate([np.cumsum(y[:n - 1]) / np.arange(1, n), (c[n:] - c[:-n]) / n])


print('using runs:')
data = {arm: load(arm) for arm, _ in ARMS}

for name, title, ylabel, tags, scale in PLOTS:
    tags = [tags] if isinstance(tags, str) else tags
    fig, ax = plt.subplots(figsize=(6, 4))

    for arm, colour in ARMS:
        d = data[arm]
        if d is None or not all(t in d for t in tags):
            continue
        steps = d[tags[0]][0] / 1e6                     # millions of env steps
        y = sum(d[t][1] for t in tags) * scale
        ax.plot(steps, smooth(y), color=colour, lw=1.6, label=arm)

    ax.set_title(title)
    ax.set_xlabel('Environment steps [millions]')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(title='Delay condition', frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f'{name}.png'), dpi=200)
    plt.close(fig)
    print('written', f'{name}.png')
