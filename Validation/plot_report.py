"""
The five figures for the results chapter, each combining what belongs together.

    python -m Validation.plot_report --results "report_results/results new test rounds" \
        --runs report_results/experiments --out report_results/figures

    fig_training_reward.png   Experiment 1: every delay type learning
    fig_degradation.png       the four headline KPIs against growing delay, one curve per model
    fig_degradation_all.png   the same, with every KPI of the results table
    fig_action_mix.png        which instructions the models transmit, and what is discarded

Colours are consistent throughout: a delay type keeps its colour family in every panel,
and darker means a longer mean delay.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Validation.summarise import METRICS, mean_and_ci

# Only the 360 s conflict horizon is reported.
# Everything this run produces lives under Validation/report_results: the per-episode CSVs
# in results/, the figures in figures/. Paths are anchored to the repository rather than to
# the working directory, so the scripts can be started from anywhere, and a bare file name
# on the command line lands in the right folder automatically.
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR  = os.path.join(ROOT, 'Validation', 'report_results')
RESULTS_DIR = os.path.join(REPORT_DIR, 'results')
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')


# Only the 360 s conflict horizon is reported.
HORIZON       = 'tw360'
HORIZON_LABEL = r'$t_\mathrm{warn} = 360$ s'

CONDITIONS = ['base_none', 'base_log30', 'log30_none', 'log30_log30', 'no_cr']
CONDITION_LABELS = {'base_none':   'no delay\ntested no delay',
                    'base_log30':  'no delay\ntested 30 s',
                    'log30_none':  '30 s\ntested no delay',
                    'log30_log30': '30 s\ntested 30 s',
                    'no_cr':       'no CR'}

DELAYS_S = [0, 15, 30, 45, 60, 90, 120]

# One entry per model: the CSV for each delay world it was evaluated in.
CURVE_FILES = {
    'trained without delay':                 'curve_base_{d:g}s.csv',
    'trained with 30 s lognormal delay':     'curve_log30_{d:g}s.csv',
    'trained with 30 s deterministic delay': 'curve_det30_{d:g}s.csv',
}
CURVE_COLOURS = {'trained without delay':                 'tab:blue',
                 'trained with 30 s lognormal delay':      'tab:green',
                 'trained with 30 s deterministic delay':  'tab:red'}

# One colour family per delay type, one shade per mean delay.
RUN_COLOURS = {'none': 'black'}
for family, cmap in [('deterministic', 'Blues'), ('lognormal', 'Greens'),
                     ('probabilistic', 'Oranges')]:
    for shade, mean_s in zip((0.45, 0.65, 0.9), (15, 30, 45)):
        RUN_COLOURS[f'{family}_{mean_s}s'] = plt.get_cmap(cmap)(shade)


# -- reading -------------------------------------------------------------------

def read_csv(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        sys.exit(f'not found: {path}')
    return pd.read_csv(path)


def read_reward_curve(runs_dir, run_name):
    """(steps, rewards) from the TensorBoard log of one training run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = glob.glob(os.path.join(runs_dir, run_name, '*', '*', 'events.out.tfevents.*'))
    if not files:
        return None
    accumulator = EventAccumulator(files[0], size_guidance={'scalars': 0})
    accumulator.Reload()
    if 'episode/reward_total' not in accumulator.Tags()['scalars']:
        return None
    points = accumulator.Scalars('episode/reward_total')
    return np.array([p.step for p in points]), np.array([p.value for p in points])


def rolling_mean(values, window=25):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


# -- figure 1: training --------------------------------------------------------

def figure_training_reward(runs_dir, out_dir):
    """Every delay type, raw curve faint and a rolling mean on top."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for run_name, colour in RUN_COLOURS.items():
        curve = read_reward_curve(runs_dir, run_name)
        if curve is None:
            print(f'  no log for {run_name}')
            continue
        plotted += 1
        steps, rewards = curve
        ax.plot(steps, rewards, color=colour, linewidth=0.4, alpha=0.2)
        smoothed = rolling_mean(rewards)
        ax.plot(steps[len(steps) - len(smoothed):], smoothed, color=colour,
                linewidth=1.6, label=run_name.replace('_', ' '))

    # Without logs the figure would be an empty frame, which is worse than no figure at
    # all: it would overwrite a good one built when the logs were still around.
    if not plotted:
        plt.close(fig)
        print(f'  no training logs under {runs_dir}, keeping any existing figure')
        return None

    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Episode reward')
    ax.set_title('Reward')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2, loc='lower right')
    fig.tight_layout()

    out = os.path.join(out_dir, 'fig_training_reward.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# -- figure 2: degradation -----------------------------------------------------

# The headline figure: the four KPIs the discussion turns on.
DEGRADATION_METRICS = ['ep_los_events_per_fh',  'ep_exit_deviation_nm',
                       'ep_advisories_per_fh',  'ep_reward_total']

# The companion figure: every KPI of the results table, in the order it is tabulated.
ALL_METRICS = ['ep_arrival_rate',        'ep_exit_deviation_nm',
               'ep_los_events_per_fh',   'ep_los_fraction',
               'ep_speed_changes_per_fh', 'ep_turns_per_fh',
               'ep_advisories_total',    'ep_advisories_per_fh',
               'ep_reward_total']

# Every non-hold action is one advisory transmitted, amendments included. The environment
# reports the two kinds separately, so the total is added here.
METRIC_LABELS = dict(METRICS, ep_advisories_total='Total advisories issued',
                     ep_advisories_per_fh='Advisories / flight hour')


def add_advisory_total(df):
    df['ep_advisories_total']  = df['ep_turns'] + df['ep_speed_changes']
    df['ep_advisories_per_fh'] = df['ep_turns_per_fh'] + df['ep_speed_changes_per_fh']
    return df


def read_degradation_curves(results_dir):
    """{model: {delay: episodes}} plus the no-CR reference."""
    # A model whose sweep has not been flown yet is skipped rather than failing here.
    curves = {}
    for model, pattern in CURVE_FILES.items():
        paths = {d: os.path.join(results_dir, pattern.format(d=d)) for d in DELAYS_S}
        missing = [os.path.basename(f) for f in paths.values() if not os.path.exists(f)]
        if missing:
            print(f'  skipping "{model}": {len(missing)} file(s) not there, '
                  f'e.g. {missing[0]}')
            continue
        curves[model] = {d: add_advisory_total(pd.read_csv(f)) for d, f in paths.items()}
    return curves, add_advisory_total(read_csv(results_dir, 'no_cr.csv'))


def degradation_figure(curves, no_cr, metrics, grid, figsize):
    """One panel per KPI on the given grid, spare cells dropped."""
    fig, axes = plt.subplots(*grid, figsize=figsize)
    for spare in axes.flat[len(metrics):]:
        spare.remove()
    for ax, metric in zip(axes.flat, metrics):
        span = []
        for model, per_delay in curves.items():
            stats = [mean_and_ci(per_delay[d][metric].dropna()) for d in DELAYS_S]
            ax.plot(DELAYS_S, [s[0] for s in stats], marker='o',
                    color=CURVE_COLOURS[model], label=model)
            ax.fill_between(DELAYS_S, [s[1] for s in stats], [s[2] for s in stats],
                            color=CURVE_COLOURS[model], alpha=0.15)
            span += [s[1] for s in stats] + [s[2] for s in stats]

        reference = no_cr[metric].mean()
        ax.axhline(reference, color='tab:blue', linestyle='--', linewidth=1.2,
                   label='no CR')

        # Room for the no-CR line: left to autoscale it lands on the frame and reads as
        # a border rather than as a reference value.
        low, high = min(span + [reference]), max(span + [reference])
        pad = 0.10 * (high - low)
        ax.set_ylim(0 if low >= 0 and low < pad else low - pad, high + pad)

        ax.set_xlabel('Mean lognormal delay time [s]')
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(DELAYS_S)
        ax.grid(alpha=0.3)

    # One legend for the whole figure, in a strip under the panels: inside a panel it
    # covers data, and in the right margin it wastes a column of space.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle('Models tested in lognormally delayed environments', fontsize=13)
    strip = 0.40 / figsize[1]
    fig.tight_layout(rect=(0, strip, 1, 1))
    fig.legend(handles, labels, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncols=4)
    return fig


def save(fig, out_dir, name):
    out = os.path.join(out_dir, name)
    # bbox_inches trims whatever margin the legend did not need.
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# -- figure 3: which instructions ----------------------------------------------

# The evaluation CSVs carry the two kinds of instruction, not the ten-way action
# histogram: mc_evaluate.py keeps the scalar entries of the info dict and the environment's
# 'action_distribution' is a list, so it is dropped. Splitting turns by their size would
# need the sweep re-flown.
ACTION_KINDS = [('ep_turns_per_fh',        'Heading changes', 'tab:purple'),
                ('ep_speed_changes_per_fh', 'Speed changes',  'tab:orange')]


def figure_action_mix(curves, out_dir):
    """What the models transmit: the mix per model, its balance, and what never got flown."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for spare in axes.flat[len(curves) + 2:]:
        spare.remove()

    # One stacked panel per model: the absolute rate, split by kind.
    ceiling = 0
    for ax, (model, per_delay) in zip(axes.flat, curves.items()):
        means = [[per_delay[d][key].mean() for d in DELAYS_S] for key, _, _ in ACTION_KINDS]
        ax.stackplot(DELAYS_S, *means, colors=[c for _, _, c in ACTION_KINDS],
                     labels=[label for _, label, _ in ACTION_KINDS], alpha=0.85)
        ceiling = max(ceiling, max(np.sum(means, axis=0)))
        ax.set_title(model, fontsize=10)
        ax.set_ylabel('Advisories / flight hour')

    # A shared ceiling: the panels are only comparable at a glance on one scale.
    for ax in axes.flat[:len(curves)]:
        ax.set_ylim(0, 1.05 * ceiling)

    balance, discarded = axes.flat[len(curves)], axes.flat[len(curves) + 1]
    for model, per_delay in curves.items():
        share = [per_delay[d]['ep_turns'].sum()
                 / max(per_delay[d]['ep_turns'].sum()
                       + per_delay[d]['ep_speed_changes'].sum(), 1)
                 for d in DELAYS_S]
        balance.plot(DELAYS_S, share, marker='o', color=CURVE_COLOURS[model], label=model)

        stats = [mean_and_ci(per_delay[d]['ep_discarded'].dropna()) for d in DELAYS_S]
        discarded.plot(DELAYS_S, [s[0] for s in stats], marker='o',
                       color=CURVE_COLOURS[model], label=model)
        discarded.fill_between(DELAYS_S, [s[1] for s in stats], [s[2] for s in stats],
                               color=CURVE_COLOURS[model], alpha=0.15)

    balance.set_ylabel('Heading share of all advisories')
    balance.set_ylim(0, 1)
    discarded.set_ylabel('Advisories discarded unflown / episode')

    for ax in fig.axes:
        ax.set_xlabel('Mean lognormal delay time [s]')
        ax.set_xticks(DELAYS_S)
        ax.grid(alpha=0.3)

    kinds = axes.flat[0].get_legend_handles_labels()
    models = balance.get_legend_handles_labels()
    fig.suptitle('Instructions issued in lognormally delayed environments', fontsize=13)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.legend(kinds[0] + models[0], kinds[1] + models[1], fontsize=10,
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncols=3)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=RESULTS_DIR, help='where the CSVs are')
    ap.add_argument('--runs', default=os.path.join(ROOT, 'Runs_saved', 'experiments'),
                    help='TensorBoard logs')
    ap.add_argument('--out', default=FIGURES_DIR, help='where the PNGs go')
    args = ap.parse_args()


    os.makedirs(args.out, exist_ok=True)
    reward_figure = figure_training_reward(args.runs, args.out)
    if reward_figure:
        print('figure ->', reward_figure)

    curves, no_cr = read_degradation_curves(args.results)
    headline = degradation_figure(curves, no_cr, DEGRADATION_METRICS, (2, 2), (12, 9))
    print('figure ->', save(headline, args.out, 'fig_degradation.png'))
    complete = degradation_figure(curves, no_cr, ALL_METRICS, (3, 3), (15, 13))
    print('figure ->', save(complete, args.out, 'fig_degradation_all.png'))
    print('figure ->', save(figure_action_mix(curves, args.out), args.out,
                            'fig_action_mix.png'))


if __name__ == '__main__':
    main()
