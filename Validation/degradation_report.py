"""
Paired degradation report: what a policy loses when its pilots start responding late.

Reads the per-episode CSVs written by Validation/mc_evaluate.py -- one run per
(policy, test world) condition, all on the SAME seeds -- and answers three questions:

  1. How much better than doing nothing is the policy in the world it trained for?
     Every condition is compared against the HOLD-only baseline on the same scenarios.
  2. How much of that survives when the same checkpoint meets delayed pilots?
     Paired per-seed differences, so scenario luck cancels out.
  3. How much of the loss is the DELAY rather than the policy?
     A checkpoint trained under the same delay is the reference: whatever it recovers was
     strategy, whatever it does not is the cost of the world itself.

Conditions are passed as label=path pairs, pointing at the *_policy.csv / *_hold.csv
files mc_evaluate produced:

    python -m Validation.degradation_report \
        nodelay_on_none=Validation/deg_nodelay_none_policy.csv \
        nodelay_on_log45=Validation/deg_nodelay_logn45_policy.csv \
        log45_on_log45=Validation/deg_log45_logn45_policy.csv \
        --hold Validation/deg_nodelay_none_hold.csv \
        --baseline nodelay_on_none --degraded nodelay_on_log45

With no arguments it picks up the default filenames used by the run commands above.

The HOLD baseline is world-independent by construction: it issues no instruction, so no
response delay is ever drawn. The script checks that claim if it is given hold files from
more than one world.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# metric -> (label, lower_is_better)
METRICS = {
    'los_events_per_fh': ('LoS events / flight hour', True),
    'los_fraction':      ('Fraction of steps in LoS', True),
    'arrival_rate':      ('Arrival rate (on-route exits)', False),
    'exit_deviation_nm': ('Exit deviation (NM, manoeuvred)', True),
    'actions_nonhold':   ('Instructions issued', True),
    'discarded':         ('Instructions discarded unflown', True),
    'reward_total':      ('Episode reward', False),
}

DEFAULT_CONDITIONS = [
    ('nodelay_on_none',  'deg_nodelay_none_policy.csv'),
    ('nodelay_on_log45', 'deg_nodelay_logn45_policy.csv'),
    ('log45_on_log45',   'deg_log45_logn45_policy.csv'),
    ('log45_on_none',    'deg_log45_none_policy.csv'),
]
DEFAULT_HOLD = ['deg_nodelay_none_hold.csv']


def ci95(x):
    """Half-width of the 95% confidence interval on the mean."""
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load(label, path):
    df = pd.read_csv(path)
    if 'seed' not in df.columns:
        sys.exit(f'{path}: no seed column -- is this an mc_evaluate CSV?')
    df['condition'] = label
    return df


def check_hold_invariance(holds):
    """HOLD flies no instructions, so its episodes must be identical in every world.
    A mismatch means scenario generation is picking up delay randomness, which would
    invalidate every paired comparison below."""
    if len(holds) < 2:
        return
    ref_label, ref = holds[0]
    for label, df in holds[1:]:
        merged = ref.merge(df, on='seed', suffixes=('_a', '_b'))
        # Columns an older CSV never filled come through as NaN; compare only what both
        # files actually measured.
        bad = [m for m in METRICS
               if f'{m}_a' in merged
               and not merged[[f'{m}_a', f'{m}_b']].isna().any(axis=None)
               and not np.allclose(merged[f'{m}_a'], merged[f'{m}_b'])]
        status = 'MISMATCH: ' + ', '.join(bad) if bad else 'identical (as expected)'
        print(f'  HOLD A/A check  {ref_label} vs {label}: {status}')


def describe(frames):
    """Mean +- 95% CI per condition, per metric."""
    rows = []
    for label, df in frames.items():
        row = {'condition': label, 'episodes': len(df)}
        for m in METRICS:
            if m in df:
                row[m] = df[m].mean()
                row[f'{m}_ci'] = ci95(df[m])
        rows.append(row)
    return pd.DataFrame(rows).set_index('condition')


def paired(a, b, metric):
    """Paired per-seed difference b - a, with a t-test over the shared seeds."""
    m = a[['seed', metric]].merge(b[['seed', metric]], on='seed', suffixes=('_a', '_b'))
    if m.empty:
        return None
    d = m[f'{metric}_b'] - m[f'{metric}_a']
    t = stats.ttest_rel(m[f'{metric}_b'], m[f'{metric}_a'])
    return {'n': len(m), 'delta': d.mean(), 'ci': ci95(d), 'p': float(t.pvalue)}


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def print_table(title, header, rows):
    print(f'\n{title}')
    print('  ' + header)
    print('  ' + '-' * len(header))
    for r in rows:
        print('  ' + r)


def retained_benefit(clean, degraded, hold, metric):
    """Share of the policy's advantage over doing nothing that survives the delay.

    Measured on condition means rather than per seed: the denominator is a difference
    that can sit near zero for individual episodes, and a per-seed ratio would then be
    dominated by those. Returns None when the clean policy had no advantage to lose.
    """
    base = clean[metric].mean() - hold[metric].mean()
    if abs(base) < 1e-9:
        return None
    return (degraded[metric].mean() - hold[metric].mean()) / base


def make_figure(desc, frames, out_path):
    metrics = [m for m in METRICS if m in desc.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap('tab10').colors
    labels = list(frames)
    for ax, m in zip(axes, metrics):
        vals = [desc.loc[c, m] for c in labels]
        errs = [desc.loc[c, f'{m}_ci'] for c in labels]
        ax.bar(range(len(labels)), vals, yerr=errs, capsize=4,
               color=[colors[i % len(colors)] for i in range(len(labels))])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
        ax.set_title(METRICS[m][0], fontsize=8)
        ax.axhline(0, color='k', lw=0.6)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Per-episode KPIs by policy and test world (mean, 95% CI)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'\nfigure -> {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('conditions', nargs='*', metavar='LABEL=CSV',
                    help='policy conditions to compare (default: the four deg_*.csv runs)')
    ap.add_argument('--hold', nargs='*', default=None,
                    help='HOLD-only baseline CSV(s); several are cross-checked for the '
                         'world-independence of doing nothing')
    ap.add_argument('--baseline', default='nodelay_on_none',
                    help='condition the degradation is measured FROM')
    ap.add_argument('--degraded', default='nodelay_on_log45',
                    help='condition the degradation is measured TO')
    ap.add_argument('--reference', default='log45_on_log45',
                    help='policy trained for the degraded world (strategy vs. world split)')
    ap.add_argument('--out-fig', default=os.path.join(HERE, 'degradation.png'))
    args = ap.parse_args()

    specs = [c.split('=', 1) for c in args.conditions] if args.conditions else \
            [(lbl, os.path.join(HERE, f)) for lbl, f in DEFAULT_CONDITIONS]
    frames = {lbl: load(lbl, p) for lbl, p in specs if os.path.exists(p)}
    missing = [p for _, p in specs if not os.path.exists(p)]
    for p in missing:
        print(f'skipping (not found): {p}')
    if not frames:
        sys.exit('no condition CSVs found -- run Validation/mc_evaluate.py first')

    hold_paths = args.hold if args.hold is not None else \
                 [os.path.join(HERE, f) for f in DEFAULT_HOLD]
    holds = [(os.path.basename(p), load(os.path.basename(p), p))
             for p in hold_paths if os.path.exists(p)]
    hold = holds[0][1] if holds else None

    print('conditions:')
    for lbl, df in frames.items():
        world = df['delay'].iloc[0] if 'delay' in df else '?'
        mean_s = df['delay_mean_s'].iloc[0] if 'delay_mean_s' in df else 0
        print(f'  {lbl:<20} {len(df):>4} episodes   test world: {world}'
              + ('' if world == 'none' else f' {mean_s:g}s'))
    if hold is not None:
        print(f'  {"hold":<20} {len(hold):>4} episodes   (do-nothing baseline)')
        check_hold_invariance(holds)

    desc = describe({**frames, **({'hold': hold} if hold is not None else {})})

    header = f'{"metric":<34}' + ''.join(f'{c:>22}' for c in desc.index)
    rows = []
    for m, (label, _) in METRICS.items():
        if m not in desc.columns:
            continue
        cells = ''.join(f'{desc.loc[c, m]:>14.3f} +-{desc.loc[c, f"{m}_ci"]:>6.3f}'
                        for c in desc.index)
        rows.append(f'{label:<34}' + cells)
    print_table('MEANS (95% CI)', header, rows)

    def paired_block(title, a_label, b_label):
        if a_label not in frames or b_label not in frames:
            print(f'\n{title}: skipped ({a_label} or {b_label} missing)')
            return
        hdr = f'{"metric":<34}{"delta":>12}{"95% CI":>14}{"p":>10}{"relative":>12}'
        out = []
        for m, (label, _) in METRICS.items():
            r = paired(frames[a_label], frames[b_label], m)
            if r is None:
                continue
            base = frames[a_label][m].mean()
            rel = f'{100 * r["delta"] / base:+.1f}%' if abs(base) > 1e-9 else '--'
            ci = '+-' + f'{r["ci"]:.3f}'
            out.append(f'{label:<34}{r["delta"]:>+12.3f}{ci:>14}'
                       f'{fmt_p(r["p"]):>10}{rel:>12}')
        print_table(f'{title}  ({b_label} - {a_label}, paired by seed)', hdr, out)

    paired_block('DEGRADATION', args.baseline, args.degraded)
    paired_block('RECOVERED BY A DELAY-TRAINED POLICY', args.degraded, args.reference)

    if hold is not None and args.baseline in frames and args.degraded in frames:
        hdr = f'{"metric":<34}{"clean vs hold":>16}{"delayed vs hold":>18}{"retained":>12}'
        out = []
        for m, (label, _) in METRICS.items():
            if m not in hold:
                continue
            clean_gain = frames[args.baseline][m].mean() - hold[m].mean()
            deg_gain   = frames[args.degraded][m].mean() - hold[m].mean()
            share = retained_benefit(frames[args.baseline], frames[args.degraded], hold, m)
            out.append(f'{label:<34}{clean_gain:>+16.3f}{deg_gain:>+18.3f}'
                       + (f'{100 * share:>11.1f}%' if share is not None else f'{"--":>12}'))
        print_table('BENEFIT OVER DOING NOTHING, AND HOW MUCH SURVIVES THE DELAY', hdr, out)
        print('  retained = (delayed - hold) / (clean - hold); 100% = delay costs nothing,')
        print('  0% = the policy is worth no more than doing nothing, <0% = worse than nothing.')

    make_figure(desc, {**frames, **({'hold': hold} if hold is not None else {})},
                args.out_fig)


if __name__ == '__main__':
    main()
