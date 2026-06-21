"""Generate 2x2 action-response delay figure — one curve per panel."""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import lognorm

MEAN_S   = 20.5
SIGMA    = 0.656
MU       = math.log(MEAN_S) - SIGMA ** 2 / 2.0
SUB_MEAN = MEAN_S * 0.6
SUB_MU   = math.log(SUB_MEAN) - SIGMA ** 2 / 2.0

t = np.linspace(0.0, 65, 2000)

C_DET  = '#1a6ed8'
C_PROB = '#c0392b'

cases = [
    ('Deterministic\nFirst instruction',       'det',  MEAN_S,   MU,     C_DET),
    ('Deterministic\nSubsequent instruction',  'det',  SUB_MEAN, SUB_MU, C_DET),
    ('Probabilistic\nFirst instruction',       'prob', MEAN_S,   MU,     C_PROB),
    ('Probabilistic\nSubsequent instruction',  'prob', SUB_MEAN, SUB_MU, C_PROB),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
fig.patch.set_facecolor('white')

for ax, (title, model, mean, mu, color) in zip(axes.flat, cases):
    if model == 'det':
        y = np.clip(1.0 - t / mean, 0.0, 1.0)
        ax.plot(t, y, color=color, lw=2.4)
        ax.axvline(mean, color=color, lw=1.0, ls='--', alpha=0.6)
        ax.text(mean + 1, 0.55, f'd = {mean:.1f} s', color=color, fontsize=9)
        label = r'$r_\mathrm{pend}(t) = \max(0,\;1 - t/d)$'
    else:
        y = 1.0 - lognorm.cdf(t[1:], s=SIGMA, scale=np.exp(mu))
        ax.plot(t[1:], y, color=color, lw=2.4)
        ax.fill_between(t[1:], y, alpha=0.12, color=color)
        mode_val = math.exp(mu - SIGMA ** 2)
        ax.axvline(mode_val, color=color, lw=1.0, ls=':',  alpha=0.6)
        ax.axvline(mean,     color=color, lw=1.0, ls='--', alpha=0.6)
        ax.text(mode_val + 0.5, 0.82, f'mode\n{mode_val:.1f} s', fontsize=8, color=color)
        ax.text(mean     + 0.5, 0.45, f'mean\n{mean:.1f} s',     fontsize=8, color=color)
        label = r'$r_\mathrm{pend}(t) = 1 - F(t)$'

    ax.set_xlim(0, 65)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlabel('Time since instruction issued  (s)', fontsize=10)
    ax.set_ylabel(r'$r_\mathrm{pend}(t)$', fontsize=10)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.text(0.97, 0.95, label, transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5, color=color)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', lw=0.4, alpha=0.5)
    ax.grid(True, which='minor', lw=0.2, alpha=0.3)

fig.suptitle('Action-response delay models  '
             r'($\sigma = 0.656$,  subsequent factor $= 0.6$)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('action_response_delay.png', dpi=200, bbox_inches='tight')
print('Saved action_response_delay.png')
