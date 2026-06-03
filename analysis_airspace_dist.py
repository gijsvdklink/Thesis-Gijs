"""
Airspace distribution analysis — v3_improved (uniform sampling).

Samples 10,000 episodes from the updated CONFIG distributions:
    n_aircraft ~ discrete U(2, 15)
    density    ~ U(5 000, 15 000) km²/aircraft
    area       = n_aircraft × density  (derived)

Plots:
    Left   — 2D heatmap: n_aircraft × density bin  (should be uniform)
    Middle — marginal n_aircraft bar chart          (should be flat)
    Right  — derived area distribution              (product of the two uniforms)

No BlueSky required — pure statistical sampling.

Run:  python analysis_airspace_dist.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Sampling ──────────────────────────────────────────────────────────────────

N_SAMPLES = 10_000
rng       = np.random.default_rng(seed=None)   # different each run

n_acs     = rng.integers(2, 16, N_SAMPLES)            # uniform discrete [2, 15]
densities = rng.uniform(5_000, 15_000, N_SAMPLES)     # km² per aircraft
areas     = n_acs * densities                          # derived sector area (km²)

# ── 2-D histogram: n_aircraft × density ──────────────────────────────────────

N_AC_VALUES      = np.arange(2, 16)
DENSITY_EDGES    = np.linspace(5_000, 15_000, 11)     # 10 bins of 1 000 km²
DENSITY_CENTRES  = (DENSITY_EDGES[:-1] + DENSITY_EDGES[1:]) / 2

H = np.zeros((len(N_AC_VALUES), len(DENSITY_CENTRES)))
for i, n in enumerate(N_AC_VALUES):
    mask = n_acs == n
    if mask.any():
        H[i], _ = np.histogram(densities[mask], bins=DENSITY_EDGES)

H_pct = H / N_SAMPLES * 100

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'wspace': 0.38})

# ── Left: 2-D heatmap ─────────────────────────────────────────────────────────

ax   = axes[0]
vmax = max(H_pct.max(), 0.01)
im   = ax.imshow(
    H_pct.T,
    origin='lower', aspect='auto', cmap='YlOrRd',
    extent=[1.5, 15.5, DENSITY_EDGES[0], DENSITY_EDGES[-1]],
    vmin=0, vmax=vmax,
)

for i, n in enumerate(N_AC_VALUES):
    for j, d_mid in enumerate(DENSITY_CENTRES):
        val = H_pct[i, j]
        if val >= 0.05:
            txt_col = 'white' if val > 0.6 * vmax else 'black'
            ax.text(n, d_mid, f'{val:.1f}%',
                    ha='center', va='center', fontsize=6.5, color=txt_col)

ax.set_xticks(N_AC_VALUES)
ax.set_xlabel('Number of aircraft', fontsize=11)
ax.set_ylabel('Density (km² per aircraft)', fontsize=11)
ax.set_yticks(DENSITY_CENTRES)
ax.set_yticklabels([f'{int(d/1000)}k' for d in DENSITY_CENTRES])
ax.set_title(f'Joint distribution  ({N_SAMPLES:,} samples)', fontsize=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('% of episodes', fontsize=9)

# ── Middle: marginal n_aircraft bar chart ─────────────────────────────────────

ax2    = axes[1]
counts = np.bincount(n_acs, minlength=16)[2:]
pct    = counts / N_SAMPLES * 100
expected = 100.0 / 14   # ~7.14% if perfectly uniform

bars = ax2.bar(N_AC_VALUES, pct, color='steelblue', edgecolor='white', linewidth=0.5)
ax2.axhline(expected, color='tomato', linestyle='--', linewidth=1.2,
            label=f'Expected ({expected:.1f}%)')

for bar, p in zip(bars, pct):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{p:.1f}%', ha='center', va='bottom', fontsize=7.5)

ax2.set_xticks(N_AC_VALUES)
ax2.set_xlabel('Number of aircraft', fontsize=11)
ax2.set_ylabel('% of episodes', fontsize=11)
ax2.set_title('Marginal: n_aircraft  (target: uniform)', fontsize=12)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, max(pct) * 1.25)
ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# ── Right: derived area distribution ──────────────────────────────────────────

ax3 = axes[2]
area_min, area_max = areas.min(), areas.max()
area_edges = np.linspace(0, 230_000, 24)   # 23 bins
ax3.hist(areas / 1_000, bins=area_edges / 1_000,
         color='steelblue', edgecolor='white', linewidth=0.4)

ax3.set_xlabel('Sector area (×10³ km²)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Derived: sector area distribution', fontsize=12)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

# Annotate range
ax3.text(0.97, 0.95,
         f'min:  {area_min/1000:.0f}k km²\nmax: {area_max/1000:.0f}k km²\n'
         f'mean: {areas.mean()/1000:.0f}k km²',
         transform=ax3.transAxes, ha='right', va='top',
         fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ── Save ──────────────────────────────────────────────────────────────────────

out = 'airspace_distribution.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved {out}')

print(f'\nn_aircraft distribution ({N_SAMPLES:,} samples)  — expected ~{expected:.1f}% each:')
for n, c, p in zip(N_AC_VALUES, counts, pct):
    deviation = p - expected
    bar = '█' * int(p / expected * 14)
    print(f'  n={n:2d}:  {c:5d}  ({p:5.1f}%  {deviation:+.1f}%)  {bar}')

print(f'\nDerived area:  min={areas.min()/1e3:.0f}k  '
      f'max={areas.max()/1e3:.0f}k  mean={areas.mean()/1e3:.0f}k km²')
