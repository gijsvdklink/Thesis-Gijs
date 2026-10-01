"""Analyse the test results of run_test_scenarios.py: the number of scenarios, the figures and the tests.

python Validation/Analysis.py

Reads Test_results/test_results.csv and writes to Test_results/analysis:

  kpis_per_model.csv     every KPI averaged over the test scenarios, a row per model and environment
  kpi_medians.csv        the median of the seven models of each type
  cv_stabilisation.csv   per KPI and model type: the c_v over all scenarios and n*, the number of
                         scenarios from which the running c_v stays within 2% of it
  statistics.csv         the hypothesis tests of experiments 1 to 4, Holm-corrected
  figures/               cv_stabilisation, test_performance and test_strategy, as PDF and PNG

A model's value is its mean over the test scenarios, and the seven models of a type are the
replicates. With seven values per type normality cannot be assumed, so all tests are
non-parametric and two-sided, at alpha = 0.05.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.multitest import multipletests

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(ROOT, 'Test_results')

KEYS        = ['model', 'train_type', 'seed', 'test_type', 'test_delay']
NOT_KPIS    = KEYS + ['model_steps', 'scenario', 'scenario_seed', 'n_aircraft', 'rho']
TRAIN_ORDER = ['none', 'deterministic', 'lognormal', 'no_cr']
TEST_ORDER  = ['none', 'deterministic', 'lognormal']

KPIS = {
    'los_events':           'LoS events',
    'los_events_per_fh':    'LoS events / flight hour',
    'conflicts':            'Conflicts',
    'conflicts_per_fh':     'Conflicts / flight hour',
    'flown_nm':             'Distance flown [NM]',
    'route_nm':             'Route length [NM]',
    'path_ratio':           'Distance ratio',
    'on_route':             'On-route exits',
    'exits':                'Exits',
    'on_route_rate':        'On-route exit rate',
    'turns':                'Heading changes',
    'turns_per_fh':         'Heading changes / flight hour',
    'speed_changes':        'Speed changes',
    'speed_changes_per_fh': 'Speed changes / flight hour',
    'advisories':           'Advisories',
    'advisories_per_fh':    'Advisories / flight hour',
    'turns_executed':                'Heading changes executed',
    'turns_executed_per_fh':         'Heading changes executed / flight hour',
    'speed_changes_executed':        'Speed changes executed',
    'speed_changes_executed_per_fh': 'Speed changes executed / flight hour',
    'reward_total':         'Episode reward',
    'reward_per_fh':        'Reward / flight hour',
}

ALPHA          = 0.05
TOLERANCE      = 0.02
RESAMPLES      = 1000
OWN_ENVIRONMENT = {'none': ('none', 0), 'deterministic': ('deterministic', 30),
                   'lognormal': ('lognormal', 30), 'no_cr': ('none', 0)}

COLOURS = {'none': '#2a78d6', 'deterministic': '#eb6834', 'lognormal': '#1baf7a',
           'no_cr': 'black'}
NAMES   = {'none': 'No-delay model', 'deterministic': 'Deterministic-delay model',
           'lognormal': 'Lognormal-delay model', 'no_cr': 'No CR'}
TEST_STYLES = {'lognormal': '-', 'deterministic': '--'}

plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 9.5, 'lines.linewidth': 1.8})
WIDTH_IN = 7.0


# -- Data ----------------------------------------------------------------------

def load(results):
    path = os.path.join(results, 'test_results.csv')
    if not os.path.exists(path):
        sys.exit(f'{path} does not exist; run "python Validation/run_test_scenarios.py --merge" first')
    episodes = pd.read_csv(path)
    episodes['train_type'] = pd.Categorical(episodes['train_type'], TRAIN_ORDER, ordered=True)
    episodes['test_type'] = pd.Categorical(episodes['test_type'], TEST_ORDER, ordered=True)

    runs = episodes.groupby(KEYS, observed=True).size()
    if runs.nunique() != 1:
        sys.exit(f'the runs differ in their number of scenarios:\n{runs.value_counts()}')
    checkpoints = episodes.groupby('model')['model_steps'].nunique()
    if (checkpoints > 1).any():
        print('WARNING: tested at more than one checkpoint: '
              + ', '.join(checkpoints[checkpoints > 1].index))
    missing = [kpi for kpi in KPIS if kpi not in episodes]
    if missing:
        print('WARNING: not in the results: ' + ', '.join(missing))
    print(f'{len(episodes):,} episodes: {len(runs)} runs of {runs.iloc[0]} scenarios')
    return episodes


def per_model(episodes):
    kpis = [column for column in episodes if column not in NOT_KPIS]
    models = (episodes.groupby(KEYS + ['model_steps'], observed=True)[kpis].mean()
              .reset_index().sort_values(['train_type', 'test_type', 'test_delay', 'seed']))
    medians = (models.groupby(['train_type', 'test_type', 'test_delay'], observed=True)[kpis]
               .median().reset_index())
    return models, medians


def environment(frame, train_type, test_type, test_delay):
    return frame[(frame['train_type'] == train_type) & (frame['test_type'] == test_type)
                 & (frame['test_delay'] == test_delay)]


# -- Number of scenarios: the running coefficient of variation -----------------

def running_cv(values, rng):
    """The mean c_v of RESAMPLES draws of n scenarios, with replacement, for n = 2 to N. A draw
    holds all seven models in each of its scenarios; values is (models, scenarios)."""
    models, total = values.shape
    first, second = values.sum(axis=0), (values ** 2).sum(axis=0)
    sizes = np.arange(2, total + 1)
    means = np.empty(len(sizes))
    with np.errstate(divide='ignore', invalid='ignore'):
        for index, n in enumerate(sizes):
            picks = rng.integers(0, total, size=(RESAMPLES, n))
            count = models * n
            mean = first[picks].sum(axis=1) / count
            variance = (second[picks].sum(axis=1) - count * mean ** 2) / (count - 1)
            means[index] = np.nanmean(np.sqrt(np.maximum(variance, 0)) / np.abs(mean))
        pooled = values.std(ddof=1) / abs(values.mean())
    return sizes, means, pooled


def settled_at(sizes, means, pooled):
    """The first n from which the running c_v stays within TOLERANCE of the pooled c_v."""
    outside = np.flatnonzero(np.abs(means - pooled) > TOLERANCE * pooled)
    if len(outside) == 0:
        return int(sizes[0])
    return int(sizes[outside[-1] + 1]) if outside[-1] + 1 < len(sizes) else None


def cv_stabilisation(episodes):
    rng = np.random.default_rng(0)
    rows, curves = [], {}
    for kpi in (kpi for kpi in KPIS if kpi in episodes):
        for train_type, (test_type, test_delay) in OWN_ENVIRONMENT.items():
            runs = environment(episodes, train_type, test_type, test_delay)
            values = runs.pivot(index='seed', columns='scenario', values=kpi).to_numpy(float)
            if values.mean() == 0 or values.std() == 0:
                continue
            sizes, means, pooled = running_cv(values, rng)
            n_star = settled_at(sizes, means, pooled)
            curves[kpi, train_type] = (sizes, means / pooled)
            rows.append({'kpi': kpi, 'train_type': train_type, 'test_type': test_type,
                         'test_delay': test_delay, 'pooled_cv': pooled, 'n_star': n_star,
                         'settled': n_star is not None})
    table = pd.DataFrame(rows)
    print(f'c_v settled within {TOLERANCE:.0%} for {table["settled"].sum()} of {len(table)} '
          f'KPI and model type pairs; largest n* = {table["n_star"].max():.0f}')
    return table, curves


def cv_figure(curves, total):
    kpis = [kpi for kpi in KPIS if any(key[0] == kpi for key in curves)]
    columns = 3
    rows = -(-len(kpis) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(WIDTH_IN, 1.55 * rows), sharex=True,
                                sharey=True, squeeze=False)
    for axis in axes.ravel()[len(kpis):]:
        axis.axis('off')
    for axis, kpi in zip(axes.ravel(), kpis):
        axis.axhspan(1 - TOLERANCE, 1 + TOLERANCE, color='0.6', alpha=0.35, linewidth=0)
        for train_type in OWN_ENVIRONMENT:
            if (kpi, train_type) in curves:
                sizes, relative = curves[kpi, train_type]
                axis.plot(sizes, relative, color=COLOURS[train_type], linewidth=1.2)
        axis.set_title(KPIS[kpi], fontsize=8.5)
        axis.set_xlim(0, total)
        axis.set_ylim(0.8, 1.1)
        axis.grid(alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel('$c_v$ / $c_v$ over all', fontsize=8.5)
    for axis in axes[-1]:
        axis.set_xlabel('Number of scenarios')
    figure.tight_layout(h_pad=0.6, w_pad=0.8)
    handles = [Line2D([], [], color=COLOURS[train_type], label=f'{NAMES[train_type]}')
               for train_type in OWN_ENVIRONMENT]
    handles.append(Patch(color='0.6', alpha=0.35, label=f'$\\pm${TOLERANCE:.0%}'))
    figure.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    return figure


# -- Figures: the median of the seven models against the mean test delay -------

def plot_kpi(axis, models, kpi, label):
    for train_type in ('none', 'deterministic', 'lognormal'):
        runs = models[models['train_type'] == train_type]
        for test_type, style in TEST_STYLES.items():
            medians = (runs[runs['test_type'].isin(['none', test_type])]
                       .groupby('test_delay')[kpi].median())
            axis.plot(medians.index, medians.values, style, marker='o', markersize=5,
                      color=COLOURS[train_type])
    axis.set_xticks(sorted(models['test_delay'].unique()))
    axis.set_ylabel(label)
    axis.grid(alpha=0.3)


def no_cr_value(models, kpi):
    return models.loc[models['train_type'] == 'no_cr', kpi].iloc[0]


def draw_no_cr(axis, value):
    axis.axhline(value, linestyle=':', color='black', linewidth=1.5)
    axis.annotate('no CR', xy=(0.01, value), xycoords=('axes fraction', 'data'), xytext=(0, 2),
                  textcoords='offset points', ha='left', va='bottom')


def off_scale(models, kpi):
    """+1 or -1 when the no-CR value lies so far above or below the curves that drawing it on
    their axis would flatten them, else 0."""
    medians = (models[models['train_type'] != 'no_cr']
               .groupby(['train_type', 'test_type', 'test_delay'], observed=True)[kpi].median())
    low, high = medians.min(), medians.max()
    reach, value = 0.5 * (high - low), no_cr_value(models, kpi)
    return 1 if value > high + reach else -1 if value < low - reach else 0


def break_axis(strip, panel, value, above):
    """The no-CR line in a thin strip beyond a break in the panel's vertical axis."""
    low, high = panel.get_ylim()
    strip.set_ylim(value - 0.08 * (high - low), value + 0.08 * (high - low))
    strip.set_yticks([value])
    text = f'{value:,.0f}' if abs(value) >= 1000 else f'{value:.2f}'
    strip.set_yticklabels([text.replace('-', '−')])
    strip.grid(alpha=0.3)
    draw_no_cr(strip, value)
    upper, lower = (strip, panel) if above else (panel, strip)
    upper.spines['bottom'].set_visible(False)
    lower.spines['top'].set_visible(False)
    upper.tick_params(axis='x', bottom=False)
    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle='none', color='black',
                 mec='black', mew=1, clip_on=False)
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **marks)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **marks)


def line_legend(figure, axes):
    handles = [Line2D([], [], color=COLOURS[train_type], linestyle=style, marker='o',
                      markersize=5, label=f'{NAMES[train_type]}, tested {law}')
               for law, style in TEST_STYLES.items()
               for train_type in ('none', 'deterministic', 'lognormal')]
    bottom = min(axis.get_position().y0 for axis in axes) - 0.45 / figure.get_figheight()
    figure.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, bottom), columnspacing=1.2, handlelength=2.6)


def performance_figure(models):
    """LoS events, conflicts and reward under each other. A no-CR value far off the curves gets
    a strip of its own beyond a break in the axis."""
    kpis = [('los_events_per_fh', 'LoS events\nper flight hour'),
            ('conflicts_per_fh', 'Conflicts\nper flight hour'),
            ('reward_total', 'Episode reward')]
    layout = []
    for kpi, _ in kpis:
        side = off_scale(models, kpi)
        layout += ([('above', kpi)] if side > 0 else []) + [('panel', kpi)] \
            + ([('below', kpi)] if side < 0 else [])
    ratios = [1 if kind == 'panel' else 0.28 for kind, _ in layout]
    figure = plt.figure(figsize=(7.0, 1.9 * sum(ratios)))
    grid = figure.add_gridspec(len(layout), 1, height_ratios=ratios, hspace=0.1)
    axes = []
    for row in range(len(layout)):
        axes.append(figure.add_subplot(grid[row], sharex=axes[0] if axes else None))

    panels = {kpi: axis for (kind, kpi), axis in zip(layout, axes) if kind == 'panel'}
    for kpi, label in kpis:
        plot_kpi(panels[kpi], models, kpi, label)
        if off_scale(models, kpi) == 0:
            draw_no_cr(panels[kpi], no_cr_value(models, kpi))
    for (kind, kpi), axis in zip(layout, axes):
        if kind != 'panel':
            break_axis(axis, panels[kpi], no_cr_value(models, kpi), above=kind == 'above')
    for axis in axes[:-1]:
        axis.tick_params(axis='x', labelbottom=False)
    axes[-1].set_xlabel('Mean test delay [s]')
    figure.align_ylabels(axes)
    line_legend(figure, axes)
    return figure


def strategy_figure(models):
    """Heading and speed changes: sent to the ATCO on top, executed by it below."""
    rows = [[('turns_per_fh', 'Heading changes sent\n/ flight hour'),
             ('speed_changes_per_fh', 'Speed changes sent\n/ flight hour')],
            [('turns_executed_per_fh', 'Heading changes executed\n/ flight hour'),
             ('speed_changes_executed_per_fh', 'Speed changes executed\n/ flight hour')]]
    rows = [row for row in rows if all(kpi in models for kpi, _ in row)]
    figure, axes = plt.subplots(len(rows), 2, squeeze=False,
                                figsize=(WIDTH_IN, len(rows) * (0.85 * WIDTH_IN / 2 + 0.3)))
    for row, kpis in zip(axes, rows):
        for axis, (kpi, label) in zip(row, kpis):
            axis.set_box_aspect(1)
            plot_kpi(axis, models, kpi, label)
            draw_no_cr(axis, no_cr_value(models, kpi))
    for axis in axes[-1]:
        axis.set_xlabel('Mean test delay [s]')
    figure.tight_layout(w_pad=2.5, h_pad=1.5)
    line_legend(figure, axes.ravel())
    return figure


def save(figure, folder, name):
    os.makedirs(folder, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(folder, f'{name}.{extension}'), dpi=200, bbox_inches='tight')
    plt.close(figure)


# -- Hypothesis tests ----------------------------------------------------------

DELAYED        = ['deterministic', 'lognormal']
NO_DELAY       = ('none', 0)
TRAINING_DELAY = {'none': NO_DELAY, 'deterministic': ('deterministic', 30),
                  'lognormal': ('lognormal', 30)}
TEST_MEANS     = [15, 30, 45, 60]
PERFORMANCE    = ['los_events_per_fh', 'conflicts_per_fh', 'reward_total']
STRATEGY       = ['turns_per_fh', 'speed_changes_per_fh']


def values(models, train_type, test, kpi):
    return environment(models, train_type, *test).set_index('seed')[kpi].sort_index()


def name(test):
    law, mean = test
    return 'no delay' if law == 'none' else f'{law} {mean} s'


def result(test, compared, tests, a, b, statistic, p):
    return {'test': test, 'compared': compared, 'test_environment': tests,
            'median_a': a.median(), 'median_b': b.median(), 'statistic': statistic, 'p': p}


def friedman(models, kpi, train_type, tests):
    groups = [values(models, train_type, test, kpi) for test in tests]
    statistic, p = friedmanchisquare(*groups)
    return result('Friedman', train_type, f'{name(tests[0])} to {name(tests[-1])}',
                  groups[0], groups[-1], statistic, p)


def mann_whitney(models, kpi, type_a, type_b, test):
    a, b = values(models, type_a, test, kpi), values(models, type_b, test, kpi)
    statistic, p = mannwhitneyu(a, b)
    return result('Mann-Whitney U', f'{type_a} vs {type_b}', name(test), a, b, statistic, p)


def signed_rank(models, kpi, train_type, test_a, test_b):
    a, b = values(models, train_type, test_a, kpi), values(models, train_type, test_b, kpi)
    statistic, p = wilcoxon(a, b)
    return result('Wilcoxon signed-rank', train_type, f'{name(test_a)} vs {name(test_b)}',
                  a, b, statistic, p)


def against_no_cr(models, kpi, train_type, test):
    a, reference = values(models, train_type, test, kpi), no_cr_value(models, kpi)
    statistic, p = sign_test(a.to_numpy(), reference)
    return result('Sign test', f'{train_type} vs no CR', name(test), a,
                  pd.Series([reference]), statistic, p)


def families(models):
    """Every test, grouped as (experiment, KPI, tests); Holm corrects within each group."""
    for kpi in PERFORMANCE:
        yield 1, kpi, [friedman(models, kpi, 'none', [NO_DELAY] + [(law, mean) for mean in TEST_MEANS])
                       for law in DELAYED]
        yield 2, kpi, [mann_whitney(models, kpi, train_type, 'none', test)
                       for train_type in DELAYED for test in (NO_DELAY, TRAINING_DELAY[train_type])]
        yield 2, kpi, [against_no_cr(models, kpi, train_type, test)
                       for train_type, test in TRAINING_DELAY.items()]
        yield 4, kpi, ([friedman(models, kpi, train_type, [(train_type, mean) for mean in TEST_MEANS])
                        for train_type in DELAYED]
                       + [signed_rank(models, kpi, train_type, ('deterministic', 30), ('lognormal', 30))
                          for train_type in DELAYED]
                       + [mann_whitney(models, kpi, 'deterministic', 'lognormal', test)
                          for test in (('deterministic', 30), ('lognormal', 30))])
    for kpi in STRATEGY:
        yield 3, kpi, [mann_whitney(models, kpi, train_type, 'none', test)
                       for train_type in DELAYED for test in (NO_DELAY, TRAINING_DELAY[train_type])]


def hypothesis_tests(models):
    rows = []
    for experiment, kpi, tests in families(models):
        significant, p_holm, _, _ = multipletests([test['p'] for test in tests], alpha=ALPHA,
                                                  method='holm')
        rows += [{'experiment': experiment, 'kpi': kpi, **test, 'p_holm': adjusted,
                  'significant': reject}
                 for test, adjusted, reject in zip(tests, p_holm, significant)]
    table = pd.DataFrame(rows).sort_values(['experiment', 'kpi'], kind='stable')
    print(f'{table["significant"].sum()} of {len(table)} tests significant after Holm')
    return table


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default=RESULTS_DIR, help=f'default {RESULTS_DIR}')
    args = parser.parse_args()
    out = os.path.join(args.results, 'analysis')
    figures = os.path.join(out, 'figures')
    os.makedirs(out, exist_ok=True)

    episodes = load(args.results)
    models, medians = per_model(episodes)
    models.to_csv(os.path.join(out, 'kpis_per_model.csv'), index=False)
    medians.to_csv(os.path.join(out, 'kpi_medians.csv'), index=False)

    table, curves = cv_stabilisation(episodes)
    table.to_csv(os.path.join(out, 'cv_stabilisation.csv'), index=False)
    save(cv_figure(curves, episodes['scenario'].nunique()), figures, 'cv_stabilisation')

    save(performance_figure(models), figures, 'test_performance')
    save(strategy_figure(models), figures, 'test_strategy')

    hypothesis_tests(models).to_csv(os.path.join(out, 'statistics.csv'), index=False)
    print(f'written to {out}')


if __name__ == '__main__':
    main()
