"""Phase 2: the score matrices, the bootstrap and the degradation figure.

Run once every evaluation run of Phase 1 has finished.

python Validation/create_plots.py
"""

import glob
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

# Every KPI is normalised -- a rate per flight hour, a fraction, a ratio or a per-step average --
# so that scenarios of different size and episode length stay comparable.
KPIS = [
    ('ep_los_events_per_fh',    'LoS events / flight hour'),
    ('ep_conflicts_per_fh',     'Conflicts / flight hour'),
    ('ep_los_fraction',         'Fraction of time in LoS'),
    ('ep_path_ratio',           'Distance flown / minimum'),
    ('ep_arrival_rate',         'Arrival rate (on-route exits)'),
    ('ep_exit_deviation_nm',    'Exit deviation [NM]'),
    ('ep_turns_per_fh',         'Heading changes / flight hour'),
    ('ep_speed_changes_per_fh', 'Speed changes / flight hour'),
    ('ep_advisories_per_fh',    'Advisories / flight hour'),
    ('mean_episode_reward',     'Reward / step'),
]

# -- The score matrices --------------------------------------------------------

def load_results():
    """Every per-episode CSV in one frame, with the one KPI the environment does not write."""
    paths = sorted(glob.glob(os.path.join(v.RESULTS_DIR, '*.csv')))
    if not paths:
        sys.exit(f'no result CSVs in {v.RESULTS_DIR}; run Phase 1 first')

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame['ep_advisories_per_fh'] = frame['ep_turns_per_fh'] + frame['ep_speed_changes_per_fh']
    return frame


def score_matrix(frame, kpi, condition, delay_mean_s):
    """One (runs x scenarios) matrix: a row per training run, a column per scenario."""
    rows = frame[(frame['condition'] == condition) & (frame['delay_mean_s'] == delay_mean_s)]

    # pivot rather than pivot_table: a duplicated (run, scenario) pair is a mistake, not something to average.
    table = rows.pivot(index='run_seed', columns='episode_seed', values=kpi)
    if table.isna().to_numpy().any():
        sys.exit(f'{condition} at {delay_mean_s:g} s: not every run flew every scenario')
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
            for delay in v.DELAYS_S:
                flown = len(frame[(frame['condition'] == condition)
                                  & (frame['run_seed'] == seed)
                                  & (frame['delay_mean_s'] == delay)])
                if flown != v.EPISODES:
                    missing.append(f'{condition} seed {seed} at {delay:g} s: '
                                   f'{flown} of {v.EPISODES} episodes')
    if v.NO_CR not in set(frame['condition']):
        missing.append('no_cr: not evaluated')

    for condition in list(v.CONDITIONS) + [v.NO_CR]:
        rows = frame[frame['condition'] == condition]
        print(f'  {condition:<14} {len(rows):>6,} episodes, '
              f'{rows["run_seed"].nunique()} runs, {rows["delay_mean_s"].nunique()} delays')

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


def curve(frame, kpi, condition):
    """Point estimate and interval for one condition across the whole delay sweep."""
    estimates = [estimate(score_matrix(frame, kpi, condition, delay)) for delay in v.DELAYS_S]
    points, lows, highs = zip(*estimates)
    return np.array(points), np.array(lows), np.array(highs)


def figure_degradation(frame):
    """One panel per KPI: the interquartile mean against the mean test delay."""
    figure, axes = plt.subplots(2, 5, figsize=(26, 9))

    for axis, (kpi, label) in zip(axes.ravel(), KPIS):
        for condition in v.CONDITIONS:
            points, lows, highs = curve(frame, kpi, condition)
            colour = v.COLOURS[condition]
            axis.plot(v.DELAYS_S, points, marker='o', color=colour, label=v.LABELS[condition])
            axis.fill_between(v.DELAYS_S, lows, highs, color=colour, alpha=BAND_ALPHA)

        reference, _, _ = estimate(no_cr_matrix(frame, kpi))
        axis.axhline(reference, linestyle='--', color='grey', linewidth=1, label=v.NO_CR_LABEL)

        axis.set_xlabel('Mean lognormal delay time [s]')
        axis.set_ylabel(label)
        axis.set_xticks(v.DELAYS_S)
        axis.grid(alpha=0.3)

    axes.ravel()[0].legend(fontsize=8)
    figure.suptitle(f'Models tested in lognormally delayed environments '
                    f'({len(v.TRAINING_SEEDS)} training runs, {v.EPISODES} scenarios; '
                    f'interquartile mean, 95% stratified bootstrap)')
    figure.tight_layout()

    os.makedirs(v.FIGURES_DIR, exist_ok=True)
    path = os.path.join(v.FIGURES_DIR, 'fig_degradation.png')
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print('wrote', path, flush=True)


def main():
    frame = load_results()
    describe(frame)
    figure_degradation(frame)


if __name__ == '__main__':
    main()
