"""
Heatmaps from the Monte-Carlo evaluation CSVs (Validation/mc_evaluate.py output).

Reads the per-episode results for the trained policy and the HOLD-only NO-policy
baseline, averages each (aircraft x density) grid cell over all episodes, and draws
clear annotated heatmaps for the key metrics:

    los_fraction     fraction of steps with a loss of separation   (safety)
    los_steps        LoS steps per episode                          (safety, raw)
    actions_nonhold  instructions issued (non-HOLD)                 (workload)
    arrival_rate     fraction of aircraft exiting on-target         (efficiency)

For each metric: one panel for the trained policy, one for the NO-policy baseline,
and a difference panel (policy - baseline) so the effect of control is obvious.

Run (defaults match mc_evaluate's --out base):
    python -m Validation.mc_heatmaps                                   # Validation/mc_results_*.csv
    python -m Validation.mc_heatmaps --csv Validation/mc_results.csv   # explicit base
    python -m Validation.mc_heatmaps --metrics los_fraction arrival_rate
"""

import os, argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))

# col -> (display label, colormap, format-as-percent, vmin, vmax, higher_is_better)
METRICS = {
    'los_fraction':    ('LoS fraction',          'RdYlGn_r', True,  0.0,  None, False),
    'los_steps':       ('LoS steps / episode',   'RdYlGn_r', False, 0.0,  None, False),
    'los_pair_steps':  ('LoS pair-steps / ep',   'RdYlGn_r', False, 0.0,  None, False),
    'actions_nonhold': ('Instructions (non-HOLD)','YlOrRd',  False, 0.0,  None, False),
    'arrival_rate':    ('Arrival rate',          'RdYlGn',   True,  0.0,  1.0,  True),
}
DEFAULT_METRICS = ['los_fraction', 'actions_nonhold', 'arrival_rate']


def cell_grid(df, col):
    """Mean of `col` per (rho, n_ac) cell -> DataFrame (rows=rho asc, cols=n_ac asc)."""
    g = df.pivot_table(index='rho', columns='n_ac_target', values=col, aggfunc='mean')
    return g.sort_index(ascending=True).sort_index(axis=1, ascending=True)


def fmt_cell(val, pct):
    if not np.isfinite(val):
        return ''
    if pct:
        return f'{100*val:.1f}'
    if abs(val) >= 100:
        return f'{val:.0f}'
    return f'{val:.1f}'


def draw_panel(ax, grid, title, cmap, pct, vmin, vmax, diverging=False, annotate=False):
    data = grid.values.astype(float)
    if diverging:
        m = np.nanmax(np.abs(data)) or 1.0
        vmin, vmax = -m, m
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='upper', interpolation='nearest')

    n_rows, n_cols = data.shape
    if annotate:
        lo, hi = np.nanmin(data), np.nanmax(data)
        span = (hi - lo) or 1.0
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                rel = abs(v) / (max(abs(lo), abs(hi)) or 1.0) if diverging else (v - lo) / span
                ax.text(j, i, fmt_cell(v, pct), ha='center', va='center', fontsize=7,
                        color='white' if rel > 0.6 else 'black')

    # sparse ticks only (less clutter): show ~5 of each axis
    xstep = max(1, n_cols // 5)
    ystep = max(1, n_rows // 5)
    xt = range(0, n_cols, xstep)
    yt = range(0, n_rows, ystep)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([str(int(grid.columns[j])) for j in xt], fontsize=9)
    ax.set_yticks(list(yt))
    ax.set_yticklabels([f'{1/grid.index[i]/1000:.0f}' for i in yt], fontsize=9)
    ax.set_xlabel('Aircraft', fontsize=10)
    ax.set_ylabel('km$^2$/ac', fontsize=10)
    ax.set_title(title, fontsize=11)
    cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if pct and not diverging:
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def plot_metric(col, grids, eps_label, out_path, annotate=False):
    """grids: dict policy_name -> cell grid DataFrame for this metric."""
    label, cmap, pct, vmin, vmax, _higher_better = METRICS[col]
    has_pol, has_hold = 'policy' in grids, 'hold' in grids

    panels = []
    if has_pol:
        panels.append(('Trained policy', grids['policy'], cmap, pct, vmin, vmax, False))
    if has_hold:
        panels.append(('No policy (HOLD)', grids['hold'], cmap, pct, vmin, vmax, False))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6),
                             gridspec_kw={'wspace': 0.35})
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, grid, cm, p, vmn, vmx, div) in zip(axes, panels):
        draw_panel(ax, grid, title, cm, p, vmn, vmx, diverging=div, annotate=annotate)

    suffix = ' (%)' if pct else ''
    fig.suptitle(f'{label}{suffix}   —   {eps_label} episodes/cell', fontsize=13, y=1.0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out_path}')


def load_policy_csvs(base):
    """Return dict policy_name -> DataFrame from <base>_policy.csv / <base>_hold.csv,
    falling back to a single combined <base>.csv with a 'policy' column."""
    root, ext = os.path.splitext(base)
    frames = {}
    for name in ('policy', 'hold'):
        p = f'{root}_{name}{ext}'
        if os.path.exists(p):
            frames[name] = pd.read_csv(p)
    if not frames and os.path.exists(base):
        df = pd.read_csv(base)
        for name, sub in df.groupby('policy'):
            frames[str(name)] = sub
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=os.path.join(HERE, 'mc_results.csv'),
                    help='base CSV path; reads <base>_policy.csv and <base>_hold.csv')
    ap.add_argument('--out-dir', default=HERE, help='directory for the PNGs')
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=list(METRICS), help='metrics to plot')
    ap.add_argument('--annotate', action='store_true',
                    help='print the per-cell average value (off by default)')
    args = ap.parse_args()

    frames = load_policy_csvs(args.csv)
    if not frames:
        ap.error(f'no CSVs found for base {args.csv} '
                 '(expected <base>_policy.csv / <base>_hold.csv)')

    # Episodes per cell. A balanced/complete run has the SAME count in every cell;
    # if min != max the run is partial/uneven -> sparse cells have noisy averages
    # (the usual cause of a lone outlier). Report it so the plot can be trusted.
    cmin = cmax = None
    for name, df in frames.items():
        sizes = df.groupby(['n_ac_target', 'rho']).size()
        lo, hi = int(sizes.min()), int(sizes.max())
        cmin = lo if cmin is None else min(cmin, lo)
        cmax = hi if cmax is None else max(cmax, hi)
        print(f'  {name:6s}: {len(df)} rows, {sizes.size} cells, {lo}-{hi} episodes/cell')
    if cmin != cmax:
        print(f'  WARNING: uneven episode counts ({cmin}-{cmax}/cell) -- run looks '
              f'incomplete; cells with few episodes will have noisy averages/outliers.')
    eps_label = f'{cmin}' if cmin == cmax else f'{cmin}-{cmax}'

    os.makedirs(args.out_dir, exist_ok=True)
    for col in args.metrics:
        grids = {name: cell_grid(df, col) for name, df in frames.items() if col in df.columns}
        if grids:
            plot_metric(col, grids, eps_label, os.path.join(args.out_dir, f'mc_heat_{col}.png'),
                        annotate=args.annotate)


if __name__ == '__main__':
    main()
