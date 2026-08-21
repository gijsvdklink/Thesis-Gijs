# Checks on the response-delay models. No BlueSky, no environment -- pure maths.
#   python -m Environments.v4.test_delays

import statistics
import sys

import numpy as np

from .delays import ResponseDelay, DELAY_MODES, MEAN_DELAY_S

N = 100_000
# Every claim is checked at more than one magnitude: with a single number, a model that
# ignores mean_s entirely would still pass.
MAGNITUDES = (15.0, 30.0, 45.0)
failures = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        failures.append(name)


def close(a, b, tol):
    return abs(a - b) <= tol * abs(b)


def draws(mode, mean_s=MEAN_DELAY_S, n=N, seed=0):
    model = ResponseDelay(mode, np.random.default_rng(seed), mean_s=mean_s)
    return [model.sample_delay_s() for _ in range(n)]


def main():
    for mode in DELAY_MODES:
        check(f'{mode}: reproducible from a seed',
              draws(mode, n=500, seed=7) == draws(mode, n=500, seed=7))

    check('none: exactly 0 s', set(draws('none', n=100)) == {0.0})
    for mean_s in MAGNITUDES:
        check(f'deterministic: exactly {mean_s:g} s',
              set(draws('deterministic', mean_s, 100)) == {mean_s})

    # The realised mean must match what expected_delay_s() claims, in both stochastic modes.
    for mode in ('lognormal', 'probabilistic'):
        for mean_s in MAGNITUDES:
            sample  = statistics.fmean(draws(mode, mean_s))
            claimed = ResponseDelay(mode, np.random.default_rng(0), mean_s=mean_s).expected_delay_s()
            check(f'{mode}/{mean_s:g}: sample mean matches expected_delay_s()',
                  close(sample, claimed, 0.02),
                  f'sample {sample:.2f} s, claimed {claimed:.2f} s')

    # Guards the mu = log(target) - sigma^2/2 correction: without it the mean is ~8% high.
    for mean_s in MAGNITUDES:
        sample = statistics.fmean(draws('lognormal', mean_s))
        check(f'lognormal: mean is the target {mean_s:g} s, not the median',
              close(sample, mean_s, 0.02), f'{sample:.2f} s')

    for mean_s in MAGNITUDES:
        sample = draws('probabilistic', mean_s)
        p = 1.0 / mean_s
        # No delay type has a ceiling, so the stochastic ones must run well past their mean.
        check(f'probabilistic/{mean_s:g}: whole seconds from 1, unbounded above',
              all(float(d).is_integer() for d in sample)
              and min(sample) == 1.0 and max(sample) > 2 * mean_s)

        # Memorylessness: the chance of responding in the next second does not depend on
        # how long the pilot has already been silent. This is what separates it from
        # the lognormal delay type, where a long silence means a response is overdue.
        for waited in (5, 15, 25):
            still_waiting = [d for d in sample if d > waited]
            responds_next = sum(1 for d in still_waiting if d == waited + 1) / len(still_waiting)
            check(f'probabilistic/{mean_s:g}: memoryless after {waited} s',
                  close(responds_next, p, 0.15), f'{responds_next:.4f} vs {p:.4f}')

    # The geometric tail must match (1-p)^k, with no pile-up at any ceiling.
    sample = draws('probabilistic')
    p = 1.0 / MEAN_DELAY_S
    for k in (30, 60, 90):
        beyond = sum(1 for d in sample if d > k) / len(sample)
        check(f'probabilistic: P(delay > {k} s) matches the geometric tail',
              close(beyond, (1 - p) ** k, 0.10), f'{beyond:.4f} vs {(1 - p) ** k:.4f}')

    try:
        ResponseDelay('sometimes', np.random.default_rng(0))
        check('unknown mode is rejected', False)
    except ValueError:
        check('unknown mode is rejected', True)

    print()
    print(f'{len(failures)} failure(s)' if failures else 'all checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
