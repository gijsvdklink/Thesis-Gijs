# One landscape PDF page: the three action-response delay distributions, side by side
#
#   [deterministic]     [log-normal]     [geometric]
#
# with each panel overlaying the three magnitudes the experiment uses. One delay per
# instruction -- the env no longer has a faster follow-up branch, so there is one row of
# panels rather than a first/engaged pair.
#
#   python figures/build_delay_pdf.py
#
# Drawn natively rather than by embedding the PNGs, so every panel stays VECTOR: lines
# and text scale losslessly at any zoom.
#
# The curves are the models' CLOSED FORMS, not sampled histograms: sampling put visible
# Monte-Carlo noise in the tails, worst in the geometric arm whose draws are slow enough
# to limit the sample size. SIGMA and the mean parameterisation are imported from
# Environments/v4/delays.py, so the shape constants cannot drift from the environment,
# but the distributions below are stated independently -- if that file's maths changes,
# these formulas must be revisited.
#
# Every panel plots probability per 1 s, labelled as the density it is for the two
# stochastic shapes -- with 1 s bins the two numbers coincide. The deterministic arm is
# strictly a point mass, drawn as a unit stem: 1.0 means all of the probability at that
# one second, not a density, since its density is a Dirac delta no axis can show.

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4.delays import SIGMA

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MAGNITUDES = (15.0, 30.0, 45.0)      # --delay-first, matching the experiment arms
# Display limit. Both stochastic shapes are unbounded, but drawing the full tail squeezes
# every feature into the left third, so the axis stops here. A meaningful amount of the
# geometric tail falls beyond -- state it in the caption.
XMAX  = 75.0
EDGES = np.arange(0.0, XMAX + 1.0)   # 1 s bins: bin i covers [i, i+1)

# Built at its FINAL PRINTED SIZE, so it is placed 1:1 and every font size below is
# exactly what lands on the page. Scaling a larger figure down would shrink the text by
# the same factor, which is what makes figure text unreadable.
#
# LANDSCAPE, to be rotated onto a portrait page (sidewaysfigure). Rotated, the figure's
# WIDTH runs down the page, so it should match \textheight, and its height matches
# \textwidth:
#     9.7 in = 247 mm = A4 \textheight,  6.3 in = 160 mm = A4 \textwidth
# both minus 25 mm margins. Put \the\textheight in your document to check your own.
FIG_W_IN, FIG_H_IN = 9.7, 6.3

# Shade darkens with magnitude, matching the reward-comparison figures.
COLORS = {
    'deterministic': ('#6ec6ff', '#1565c0', '#0d1f6e'),
    'lognormal':     ('#ffb703', '#fb5607', '#9a0000'),
    'probabilistic': ('#7ae582', '#1fb954', '#0b6e2f'),
}
PANEL_TITLES = {
    'deterministic': 'Deterministic',
    'lognormal':     'Log-normal',
    # A fixed per-second compliance probability makes the delay geometric -- the
    # discrete analogue of the exponential, and the only memoryless one of the three.
    'probabilistic': 'Geometric',
}
SHAPES = ('deterministic', 'lognormal', 'probabilistic')

plt.rcParams.update({
    'font.size':        10,   # body text on the page, at 1:1
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.titlesize': 11,
    'pdf.fonttype': 42,     # embed real (subsetted) fonts, not Type 3 bitmap fonts
    'ps.fonttype': 42,
})


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def bin_probabilities(shape, mean_s):
    """P(delay lands in each 1 s bin), from the model's closed form.

    deterministic  all the mass in the bin holding mean_s.
    lognormal      continuous, parameterised on its MEAN: mu = ln(mean) - sigma^2/2,
                   so the bin probability is a difference of two normal CDFs in ln t.
    probabilistic  a per-second hazard of p = 1/mean makes the delay geometric on
                   {1, 2, ...} seconds: P(D = k) = p (1-p)^(k-1). The first roll is the
                   second AFTER the instruction, which is why the support starts at 1.
    """
    lo, hi = EDGES[:-1], EDGES[1:]
    if shape == 'deterministic':
        p = np.zeros_like(lo)
        p[int(mean_s)] = 1.0
        return p
    if shape == 'lognormal':
        mu = math.log(mean_s) - SIGMA ** 2 / 2.0
        with np.errstate(divide='ignore'):
            z_lo = np.where(lo > 0, (np.log(np.where(lo > 0, lo, 1.0)) - mu) / SIGMA, -np.inf)
            z_hi = (np.log(hi) - mu) / SIGMA
        return _phi(z_hi) - _phi(z_lo)
    p = 1.0 / mean_s
    k = lo                                   # a draw of exactly k s falls in bin [k, k+1)
    return np.where(k >= 1, p * (1.0 - p) ** (k - 1), 0.0)


def draw_panel(ax, shape):
    """This shape's three magnitudes overlaid."""
    for mean_s, color in zip(MAGNITUDES, COLORS[shape]):
        if shape == 'deterministic':
            # A point mass, drawn as a stem at the exact second rather than as a 1 s
            # bin: a stairs outline of one bin reads as two vertical lines, and the
            # delay is not spread across that second -- it is that second.
            ax.vlines(mean_s, 0.0, 1.0, color=color, lw=1.4, label=f'{mean_s:g} s')
            ax.plot([mean_s], [1.0], 'o', ms=3.5, color=color)
            continue
        ax.stairs(bin_probabilities(shape, mean_s), EDGES,
                  color=color, linewidth=1.4, label=f'{mean_s:g} s')
        ax.axvline(mean_s, color=color, lw=0.8, ls='--', alpha=0.6)

    ax.set_xlim(0, XMAX)
    ax.grid(True, alpha=0.3)
    # Headroom so the legend clears the curves rather than covering their peaks. The
    # deterministic panel is pinned instead: its mass is exactly 1.0, the ticks stop
    # there, and the extra room above only keeps the marker off the frame.
    if shape == 'deterministic':
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.28)
    ax.legend(loc='upper right', handlelength=1.4, borderpad=0.4, labelspacing=0.3)


fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))

for col, shape in enumerate(SHAPES):
    ax = axes[col]
    draw_panel(ax, shape)
    ax.set_title(PANEL_TITLES[shape])
    ax.set_xlabel('Action-response delay [s]')

# Named once, on the left: all three panels share the same quantity.
axes[0].set_ylabel('Probability density function')

fig.suptitle('Action-response delay distributions')
fig.tight_layout()

OUT = os.path.join(OUT_DIR, 'delay_distributions.pdf')
fig.savefig(OUT)
plt.close(fig)
print('written', os.path.basename(OUT), '-- all three panels vector, closed-form')
