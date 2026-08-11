"""Plot the action-response delay distributions: deterministic vs probabilistic.

Keep the four numbers below in sync with Environments/v4/config.py.
Regenerate with:  python figures/plot_delays.py
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

FIRST = 30      # config delay_first_s: first advisory to a newly selected ownship
NEXT  = 15      # config delay_next_s:  once one has executed while it holds focus
SIGMA = 0.4     # config delay_sigma:   log-normal shape (probabilistic only)
XMAX  = 70      # shared by all four panels, so they are directly comparable
YMAX  = 0.10

x = np.linspace(0.5, XMAX, 1000)

# Row 0 deterministic, row 1 probabilistic. Shared axes, so the reader cannot mistake a
# scale change for a difference between the conditions.
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)

for col, mean in enumerate((FIRST, NEXT)):
    # scipy's `scale` is the MEDIAN, so shift it to put the MEAN on `mean` -- the same
    # parameterisation env._draw_response_delay uses. Passing `mean` as the scale
    # directly is the classic log-normal mistake and shifts the curve ~8% right.
    dist = lognorm(SIGMA, scale=mean * math.exp(-SIGMA ** 2 / 2))

    axes[0, col].axvline(mean, color='C0', lw=2)          # all mass on one value
    axes[1, col].plot(x, dist.pdf(x), color='C1', lw=2)
    axes[1, col].axvline(mean, color='grey', lw=1, ls='--')

    for row, label in ((0, f'{mean:g} s'), (1, f'mean {mean:g} s')):
        axes[row, col].annotate(label, xy=(mean, YMAX * 0.92), xytext=(6, 0),
                                textcoords='offset points', va='top')
    axes[1, col].set_xlabel('Action-response delay [s]')

axes[0, 0].set_title('First instruction')
axes[0, 1].set_title('Follow-up instruction')
axes[0, 0].set_ylabel('Deterministic\n\nProbability density')
axes[1, 0].set_ylabel('Probabilistic\n\nProbability density')
axes[0, 0].set_xlim(0, XMAX)
axes[0, 0].set_ylim(0, YMAX)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delays.png')
fig.savefig(out, dpi=200)
print('written', os.path.basename(out))
