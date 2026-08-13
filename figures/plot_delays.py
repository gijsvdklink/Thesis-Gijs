# The three action-response delay distributions, drawn straight from delays.py so the
# figure cannot drift out of sync with the environment.
#   python figures/plot_delays.py

import os
import sys
from random import Random

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environments.v4.delays import ResponseDelay, FIRST_S, NEXT_S

N_SAMPLES = 400_000
XMAX = 90.0                      # display range; the probabilistic tail runs past it
BINS = np.arange(0, XMAX + 1)    # 1 s bins

ROWS = [('deterministic', 'Deterministic', 'C0'),
        ('lognormal',     'Log-normal',    'C1'),
        ('probabilistic', 'Probabilistic', 'C2')]

fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True, sharey=True)

for row, (mode, label, colour) in enumerate(ROWS):
    for col, target in enumerate((FIRST_S, NEXT_S)):
        ax = axes[row, col]
        model  = ResponseDelay(mode, Random(row * 10 + col))
        sample = [model.draw(engaged=bool(col)) for _ in range(N_SAMPLES)]

        ax.hist(sample, bins=BINS, density=True, color=colour,
                histtype='stepfilled', alpha=0.85)
        ax.axvline(np.mean(sample), color='grey', lw=1, ls='--')
        ax.annotate(f'mean {np.mean(sample):.1f} s', xy=(0.97, 0.9),
                    xycoords='axes fraction', ha='right', va='top')

    axes[row, 0].set_ylabel(f'{label}\n\nProbability density')

axes[0, 0].set_title(f'First instruction (target {FIRST_S:g} s)')
axes[0, 1].set_title(f'Follow-up instruction (target {NEXT_S:g} s)')
for ax in axes[2]:
    ax.set_xlabel('Action-response delay [s]')

# Shared axes so the reader cannot mistake a scale change for a difference between arms.
axes[0, 0].set_xlim(0, XMAX)
axes[0, 0].set_ylim(0, 0.10)


fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delays.png')
fig.savefig(out, dpi=200)
print('written', os.path.basename(out))
