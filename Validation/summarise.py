"""
Monte-Carlo summary of the evaluation runs: one mean per condition, one figure per KPI.

Reads the per-episode CSVs written by Validation/mc_evaluate.py, one per condition, and
reports the average of every KPI with a 95% confidence interval. The interval comes from a
PERCENTILE BOOTSTRAP rather than a normal approximation: LoS events are mostly zero with a
long tail, so the sampling distribution of their mean is not symmetric and 1.96 x SEM would
misstate it.

    python -m Validation.summarise \
        base_on_none=Validation/results/tw360_base_none_policy.csv \
        base_on_log30=Validation/results/tw360_base_log30_policy.csv \
        "no CR=Validation/results/tw360_hold_hold.csv" \
        --out Validation/results/tw360

Writes the table to stdout. The figures are Validation/plot_report.py's job.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# CSV column -> axis label. The columns are the environment's own names.
METRICS = {
    'ep_los_events_per_fh':      'LoS events / flight hour',
    'ep_los_fraction':           'Fraction of time in LoS',
    'ep_arrival_rate':           'Arrival rate (on-route exits)',
    'ep_exit_deviation_nm':      'Exit deviation [NM]',
    'ep_turns_per_fh':           'Heading changes / flight hour',
    'ep_speed_changes_per_fh':   'Speed changes / flight hour',
    'ep_discarded':              'Advisories discarded unflown',
    'ep_delay_mean_s':           'Realised pilot response [s]',
    'ep_reward_total':           'Episode reward',
}

RESAMPLES = 10_000

# Everything this run produces lives under Validation/report_results: the per-episode CSVs
# in results/, the figures in figures/. Paths are anchored to the repository rather than to
# the working directory, so the scripts can be started from anywhere, and a bare file name
# on the command line lands in the right folder automatically.
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR  = os.path.join(ROOT, 'Validation', 'report_results')
RESULTS_DIR = os.path.join(REPORT_DIR, 'results')
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')


def in_folder(path, folder):
    """A bare name goes in `folder`; anything with a directory in it is left alone."""
    return path if os.path.dirname(path) else os.path.join(folder, path)



def mean_and_ci(values):
    """The mean of these episodes, and a 95% confidence interval around it.

    The episodes we ran are the only picture of the population we have, so we treat them
    as if they were the population: draw a new data set of the same size from them, with
    replacement, and write down its mean. Do that RESAMPLES times and the means spread
    out; the middle 95% of that spread is the interval. Nothing is assumed about the
    shape of the underlying distribution, which matters here because most episodes have
    zero LoS events and a few have several.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean()), float(values.mean())

    rng = np.random.default_rng(0)          # fixed seed, so the figures are reproducible
    resampled_means = []
    for _ in range(RESAMPLES):
        resample = rng.choice(values, size=len(values), replace=True)
        resampled_means.append(resample.mean())

    lower = np.percentile(resampled_means, 2.5)
    upper = np.percentile(resampled_means, 97.5)
    return float(values.mean()), float(lower), float(upper)


def load(label, path):
    df = pd.read_csv(path)
    print(f'  {label:<18} {len(df):>5} episodes   {os.path.basename(path)}')
    return df


def summarise(frames):
    """{label: dataframe} -> {metric: {label: (mean, lo, hi)}}."""
    table = {}
    for metric in METRICS:
        per_label = {}
        for label, df in frames.items():
            if metric in df and df[metric].notna().any():
                per_label[label] = mean_and_ci(df[metric].dropna())
        if per_label:
            table[metric] = per_label
        else:
            print(f'  skipping {metric}: not present in any run')
    return table


def print_table(table, labels):
    width = max(len(l) for l in labels) + 2
    print(f'\n{"metric":<34}' + ''.join(f'{l:>{width + 16}}' for l in labels))
    for metric, per_label in table.items():
        cells = ''
        for label in labels:
            if label in per_label:
                mean, lo, hi = per_label[label]
                cells += f'{mean:>{width + 4}.3f} [{lo:.3f}, {hi:.3f}]'
            else:
                cells += f'{"--":>{width + 16}}'
        print(f'{METRICS[metric]:<34}' + cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('conditions', nargs='+', metavar='LABEL=CSV',
                    help='one per condition, e.g. base_on_none=tw360_base_none.csv')
    args = ap.parse_args()

    print('conditions:')
    frames = {}
    for spec in args.conditions:
        label, _, path = spec.partition('=')
        path = in_folder(path, RESULTS_DIR)
        if not os.path.exists(path):
            sys.exit(f'not found: {path}')
        frames[label] = load(label, path)

    table = summarise(frames)
    print_table(table, list(frames))



if __name__ == '__main__':
    main()
