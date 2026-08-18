# Three figures: one per delay shape, each overlaying that shape's three magnitudes so
# they can be compared directly.
#
#   delay_deterministic.png    delay_lognormal.png    delay_probabilistic.png
#
#   python figures/plot_delay_grids.py
#
# Samples Environments/v4/delays.py directly, so the figures cannot drift out of sync
# with what the environment actually draws.

import os
import sys
from random import Random

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4.delays import ResponseDelay

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MAGNITUDES = (15.0, 30.0, 45.0)      # --delay-first, matching the experiment arms
XMAX       = 150.0                   # both stochastic shapes have an unbounded tail
# One bin past the axis limit: numpy puts the right edge INSIDE the final bin, so a
# range ending exactly at XMAX would make that bin two seconds wide and draw a spike
# there that the distribution does not have.
BINS       = np.arange(0, XMAX + 2)  # 1 s bins

# Fewer samples for the probabilistic arm: draw() rolls its Markov chain second by
# second in a Python loop, so it is far slower per sample than the closed-form shapes.
N_SAMPLES = {'deterministic': 400_000, 'lognormal': 400_000, 'probabilistic': 60_000}

# Shade darkens with magnitude, matching the reward-comparison figures.
COLORS = {
    'deterministic': ('#6ec6ff', '#1565c0', '#0d1f6e'),
    'lognormal':     ('#ffb703', '#fb5607', '#9a0000'),
    'probabilistic': ('#7ae582', '#1fb954', '#0b6e2f'),
}
SHAPE_TITLES = {
    'deterministic': 'Deterministic delay',
    'lognormal':     'Log-normal delay',
    'probabilistic': 'Probabilistic delay (Markov chain)',
}

def plot_one(shape):
    """One figure: this shape's three magnitudes overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = N_SAMPLES[shape]

    for mean_s, color in zip(MAGNITUDES, COLORS[shape]):
        model  = ResponseDelay(shape, Random(0), mean_s=mean_s)
        target = model.expected_delay_s()
        sample = [model.sample_delay_s() for _ in range(n)]
        label  = f'{mean_s:g} s arm  (mean {target:g} s)'

        if shape == 'deterministic':
            ax.axvline(target, color=color, lw=3, label=label)
        else:
            ax.hist(sample, bins=BINS, density=True, color=color,
                    histtype='step', linewidth=2.0, label=label)
            ax.axvline(np.mean(sample), color=color, lw=1, ls='--', alpha=0.6)

    ax.set_xlabel('Action-response delay [s]')
    ax.set_ylabel('Probability density')
    ax.set_title(SHAPE_TITLES[shape])
    ax.set_xlim(0, XMAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'delay_{shape}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('written', os.path.basename(out))


for shape in ('deterministic', 'lognormal', 'probabilistic'):
    plot_one(shape)
