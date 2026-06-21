"""Generate 3x2 action-response delay figure: deterministic, geometric, lognormal."""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import lognorm, expon

MEAN_S   = 20.5
SIGMA    = 0.656
MU       = math.log(MEAN_S) - SIGMA ** 2 / 2.0
SUB_MEAN = MEAN_S * 0.6
SUB_MU   = math.log(SUB_MEAN) - SIGMA ** 2 / 2.0
DT       = 5.0

t  = np.linspace(0.0, 60.0, 3000)
t1 = t[1:]   # avoid log(0) for lognormal

C_DET  = '#1a6ed8'
C_GEO  = '#27ae60'
C_LOGN = '#c0392b'

cases = [
    ('Deterministic — first instruction',      'det',  MEAN_S,   MU,     C_DET),
    ('Deterministic — subsequent instruction', 'det',  SUB_MEAN, SUB_MU, C_DET),
    ('Geometric — first instruction',          'geo',  MEAN_S,   None,   C_GEO),
    ('Geometric — subsequent instruction',     'geo',  SUB_MEAN, None,   C_GEO),
    ('Lognormal — first instruction',          'logn', MEAN_S,   MU,     C_LOGN),
    ('Lognormal — subsequent instruction',     'logn', SUB_MEAN, SUB_MU, C_LOGN),
]

fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharey=True)
fig.patch.set_facecolor('white')

for ax, (title, model, mean, mu, color) in zip(axes.flat, cases):

    if model == 'det':
        ax.bar([mean], [1.0], width=1.2, color=color, alpha=0.25,
               edgecolor=color, linewidth=0.8)
        cdf = np.where(t >= mean, 1.0, 0.0)
        ax.step(t, cdf, where='post', color=color, lw=2.4)
        ax.axvline(mean, color=color, lw=1.0, ls='--', alpha=0.7)
        ax.text(mean + 1, 0.45, f'd = {mean:.1f} s', color=color, fontsize=9)

    elif model == 'geo':
        p = DT / mean
        n_steps = int(60 / DT) + 2
        pmf_k = np.arange(0, n_steps - 1)
        pmf_t = (pmf_k + 1) * DT
        pmf_v = p * (1 - p) ** pmf_k
        steps   = np.arange(n_steps)
        t_steps = steps * DT
        cdf     = 1.0 - (1 - p) ** steps
        ax.step(t_steps, cdf, where='post', color=color, lw=2.4)
        ax.axvline(mean, color=color, lw=1.0, ls='--', alpha=0.7)
        ax.text(mean + 1, 0.45, f'mean = {mean:.1f} s', color=color, fontsize=9)

    else:  # logn
        cdf = lognorm.cdf(t1, s=SIGMA, scale=np.exp(mu))
        ax.plot(t1, cdf, color=color, lw=2.4, zorder=3)
        mode_val = math.exp(mu - SIGMA ** 2)
        ax.axvline(mode_val, color=color, lw=1.0, ls=':',  alpha=0.7)
        ax.axvline(mean,     color=color, lw=1.0, ls='--', alpha=0.7)
        ax.text(mode_val + 0.5, 0.25, f'mode\n{mode_val:.1f} s', fontsize=8, color=color)
        ax.text(mean     + 0.5, 0.55, f'mean\n{mean:.1f} s',     fontsize=8, color=color)

    ax.set_xlim(0, 60)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlabel('Time since instruction issued  (s)', fontsize=10)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', lw=0.4, alpha=0.5)
    ax.grid(True, which='minor', lw=0.2, alpha=0.3)

for ax in axes[:, 0]:
    ax.set_ylabel(r'P(executed)', fontsize=10)

fig.suptitle(
    r'Action-response delay models  '
    r'($\Delta t=5\,\mathrm{s}$,  $\sigma_\mathrm{logn}=0.656$,  subsequent factor $=0.6$)',
    fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('action_response_delay.png', dpi=200, bbox_inches='tight')
print('Saved action_response_delay.png')
