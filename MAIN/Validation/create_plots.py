"""Phase 2: the score matrices, the bootstrap and the degradation figure.

Run once every evaluation run of Phase 1 has finished.

python Validation/create_plots.py
"""

import glob
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validation as v

# -- KPIs ----------------------------------------------------------------------

# Every KPI but the last is normalised -- a rate per flight hour, a fraction, a ratio or a
# per-step average -- so that scenarios of different size and episode length stay comparable.
# The episode total is plotted beside the per-step reward, and is comparable between policies
# only because they all fly the same scenarios.
# Table 2.4 of the report, in its order.
KPIS = [
    ('ep_los_events_per_fh',    'LoS events / flight hour'),
    ('ep_conflicts_per_fh',     'Conflicts / flight hour'),
    ('ep_path_ratio',           'Distance ratio'),
    ('ep_on_route_rate',        'On-route exits'),
    ('ep_turns_per_fh',         'Heading changes / flight hour'),
    ('ep_speed_changes_per_fh', 'Speed changes / flight hour'),
    ('ep_advisories_per_fh',    'Advisories / flight hour'),
    ('ep_reward_total',         'Episode reward'),
]

# The specific advisory issued, for the resolution-strategy question.
ADVISORY_KPIS = [
    ('ep_turn_m60_per_fh',   'Turn -60 / flight hour'),
    ('ep_turn_m45_per_fh',   'Turn -45 / flight hour'),
    ('ep_turn_m30_per_fh',   'Turn -30 / flight hour'),
    ('ep_turn_p30_per_fh',   'Turn +30 / flight hour'),
    ('ep_turn_p45_per_fh',   'Turn +45 / flight hour'),
    ('ep_turn_p60_per_fh',   'Turn +60 / flight hour'),
    ('ep_return_per_fh',     'Return to route / flight hour'),
    ('ep_speed_up_per_fh',   'Speed up / flight hour'),
    ('ep_speed_down_per_fh', 'Speed down / flight hour'),
]

COLUMNS = 4       # panels per row; the grid grows downwards with the KPI list

# -- The score matrices --------------------------------------------------------

def load_results():
    """Every per-episode CSV in one frame."""
    paths = sorted(glob.glob(os.path.join(v.RESULTS_DIR, '*.csv')))
    if not paths:
        sys.exit(f'no result CSVs in {v.RESULTS_DIR}; run Phase 1 first')

    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def score_matrix(frame, kpi, condition, level):
    """One (runs x scenarios) matrix: a row per training run, a column per scenario."""
    law, mean = level
    rows = frame[(frame['condition'] == condition)
                 & (frame['delay_law'] == law)
                 & (frame['delay_mean_s'] == mean)]

    # pivot rather than pivot_table: a duplicated (run, scenario) pair is a mistake, not something to average.
    table = rows.pivot(index='run_seed', columns='episode_seed', values=kpi)
    if table.isna().to_numpy().any():
        sys.exit(f'{condition} at {law} {mean:g} s: not every run flew every scenario')
    return table.to_numpy(dtype=float)


def no_cr_matrix(frame, kpi):
    """The no-CR reference as a one-row matrix; it is delay-independent, so it is flown once."""
    rows = frame[frame['condition'] == v.NO_CR]
    return rows.pivot(index='run_seed', columns='episode_seed', values=kpi).to_numpy(dtype=float)


def describe(frame):
    """What Phase 1 produced, and what it still owes."""
    print(f'{len(frame):,} episodes, {frame["episode_seed"].nunique()} distinct scenarios\n')

    missing = []
    for condition in v.CONDITIONS:
        for seed in v.TRAINING_SEEDS:
            for law, mean in v.DELAY_LEVELS:
                flown = len(frame[(frame['condition'] == condition)
                                  & (frame['run_seed'] == seed)
                                  & (frame['delay_law'] == law)
                                  & (frame['delay_mean_s'] == mean)])
                if flown != v.EPISODES:
                    missing.append(f'{condition} seed {seed} at {law} {mean:g} s: '
                                   f'{flown} of {v.EPISODES} episodes')
    if v.NO_CR not in set(frame['condition']):
        missing.append('no_cr: not evaluated')

    for condition in list(v.CONDITIONS) + [v.NO_CR]:
        rows = frame[frame['condition'] == condition]
        print(f'  {condition:<14} {len(rows):>6,} episodes, '
              f'{rows["run_seed"].nunique()} runs, '
              f'{rows.groupby(["delay_law", "delay_mean_s"]).ngroups} delay worlds')

    if missing:
        sys.exit('\nPhase 1 is not finished; these evaluation runs are missing:\n  '
                 + '\n  '.join(missing))
    print('\ngrid complete')


# -- The interquartile mean and its stratified bootstrap -----------------------

# The bootstrap resamples the RUNS with replacement, independently within each scenario, so
# the scenarios act as strata and their difficulty is held fixed.
RESAMPLES  = 50_000
CHUNK      = 2_000    # resamples built at once; keeps peak memory small
SEED       = 0        # fixed, so every figure is reproducible


def interquartile_mean(stack):
    """Mean of the middle half of the values, pooled over runs and scenarios."""
    flat = np.sort(stack.reshape(len(stack), -1), axis=1)
    quarter = flat.shape[1] // 4
    return flat[:, quarter:flat.shape[1] - quarter].mean(axis=1)


def estimate(matrix):
    """The interquartile mean of one score matrix, with a 95% confidence interval."""
    matrix = np.asarray(matrix, dtype=float)
    point  = float(interquartile_mean(matrix[None])[0])
    if matrix.shape[0] < 2:
        return point, point, point          # one run: nothing to resample

    runs, scenarios = matrix.shape
    rng    = np.random.default_rng(SEED)
    draws  = []
    for start in range(0, RESAMPLES, CHUNK):
        count = min(CHUNK, RESAMPLES - start)
        picks = rng.integers(0, runs, size=(count, runs, scenarios))
        resampled = np.take_along_axis(np.broadcast_to(matrix, (count, runs, scenarios)),
                                       picks, axis=1)
        draws.append(interquartile_mean(resampled))

    low, high = np.percentile(np.concatenate(draws), [2.5, 97.5])
    return point, float(low), float(high)


# -- The figure ----------------------------------------------------------------

BAND_ALPHA = 0.18


def curve(frame, kpi, condition, levels):
    """Point estimate and interval for one condition across one law's delay sweep."""
    estimates = [estimate(score_matrix(frame, kpi, condition, level)) for level in levels]
    points, lows, highs = zip(*estimates)
    return np.array(points), np.array(lows), np.array(highs)


def figure_for_law(frame, law, kpis, name):
    """One panel per KPI: the interquartile mean against the mean test delay, for one law."""
    levels = v.levels_for(law)
    means  = [mean for _, mean in levels]

    rows = math.ceil(len(kpis) / COLUMNS)
    figure, axes = plt.subplots(rows, COLUMNS, figsize=(5.5 * COLUMNS, 4.5 * rows),
                               squeeze=False)
    for spare in axes.ravel()[len(kpis):]:
        spare.axis('off')

    for axis, (kpi, label) in zip(axes.ravel(), kpis):
        for condition in v.CONDITIONS:
            points, lows, highs = curve(frame, kpi, condition, levels)
            colour = v.COLOURS[condition]
            axis.plot(means, points, marker='o', color=colour, label=v.LABELS[condition])
            axis.fill_between(means, lows, highs, color=colour, alpha=BAND_ALPHA)

        reference, _, _ = estimate(no_cr_matrix(frame, kpi))
        axis.axhline(reference, linestyle='--', color='grey', linewidth=1, label=v.NO_CR_LABEL)

        axis.set_xlabel(f'Mean {law} delay time [s]')
        axis.set_ylabel(label)
        axis.set_xticks(means)
        axis.grid(alpha=0.3)

    axes.ravel()[0].legend(fontsize=8)
    figure.suptitle(f'Models tested in {law}ly delayed environments '
                    f'({len(v.TRAINING_SEEDS)} training runs, {v.EPISODES} scenarios; '
                    f'interquartile mean, 95% stratified bootstrap)')
    figure.tight_layout()

    os.makedirs(v.FIGURES_DIR, exist_ok=True)
    path = os.path.join(v.FIGURES_DIR, name)
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print('wrote', path, flush=True)


def main():
    frame = load_results()
    describe(frame)
    for law in v.DELAY_LAWS:
        figure_for_law(frame, law, KPIS, f'fig_degradation_{law}.png')
        figure_for_law(frame, law, ADVISORY_KPIS, f'fig_advisories_{law}.png')


if __name__ == '__main__':
    main()
