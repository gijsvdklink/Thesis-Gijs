"""The two delay figures of the report, drawn from atco.py so they cannot drift.

The environment rounds the drawn delay to a whole second, so the distribution the
simulation actually uses is discrete. It is plotted that way.

python Visualisation/delay_figures.py [--out DIR]
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.atco import ATCO
from Environment.config import CONFIG

TRAIN_MEAN = CONFIG['delay_mean_s']
TEST_MEANS = [15, 30, 45, 60]


def pmf(mean_s):
    """P(round(tau) = k) for every whole second k, for the truncated law the ATCO draws."""
    atco = ATCO('lognormal', np.random.default_rng(0), mean_s=mean_s)
    mu, sigma, cap = atco._mu, CONFIG['delay_sigma'], atco._cap

    def cdf(x):
        if x <= 0:
            return 0.0
        return 0.5 * (1.0 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2.0))))

    # A draw above the bound is rejected and redrawn, so the mass is renormalised by F(cap)
    # rather than piled onto the bound.
    seconds = np.arange(0, round(cap) + 1)
    probability = np.array([cdf(min(k + 0.5, cap)) - cdf(k - 0.5) for k in seconds]) / cdf(cap)
    return seconds, probability, cap


def save(figure, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    print('wrote', path, flush=True)


def figure_laws(out_dir):
    """The two response laws at the mean used in training."""
    seconds, probability, cap = pmf(TRAIN_MEAN)

    # Sized to sit beside the measured pilot data of Wuestenbecker et al., which is square.
    figure, axis = plt.subplots(figsize=(4.4, 3.7))
    axis.bar(seconds, probability, width=1.0,
             label=f'Lognormal, $\\bar\\tau$ = {TRAIN_MEAN:g} s')
    # A point mass of probability 1, so it is marked by its position, not by a bar.
    axis.axvline(TRAIN_MEAN, color='C1', linewidth=2,
                 label=f'Deterministic, $\\tau$ = {TRAIN_MEAN:g} s')

    axis.set_xlabel('Action-response delay $\\tau$ [s]')
    axis.set_ylabel('Probability')
    axis.set_xlim(0, cap + 4)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    axis.set_axisbelow(True)
    save(figure, out_dir, 'delay_distributions.pdf')


def figure_levels(out_dir):
    """The lognormal delay at each mean the trained models are tested at."""
    figure, axis = plt.subplots(figsize=(6.5, 3.2))

    for mean_s in TEST_MEANS:
        seconds, probability, _ = pmf(mean_s)
        axis.bar(seconds, probability, width=1.0, alpha=0.6,
                 label=f'$\\bar\\tau$ = {mean_s:g} s')

    axis.set_xlabel('Action-response delay $\\tau$ [s]')
    axis.set_ylabel('Probability')
    axis.set_xlim(0, 145)
    axis.legend()
    axis.grid(alpha=0.3)
    axis.set_axisbelow(True)
    save(figure, out_dir, 'delay_levels.pdf')


def main():
    default = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'Progress_Report_with_abb_and_mathematics_RESTRUCTURED (3)', 'images', 'midterm'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=default, help='where the PDFs are written')
    args = parser.parse_args()

    figure_laws(args.out)
    figure_levels(args.out)


if __name__ == '__main__':
    main()
