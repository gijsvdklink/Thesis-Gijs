# Checks on the response-delay models. No BlueSky, no environment -- pure maths.
#   python -m Environments.v4.test_delays

import statistics
import sys
from random import Random

from .delays import ResponseDelay, DELAY_MODES, FIRST_S, NEXT_S

N = 100_000
failures = []


def check(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        failures.append(name)


def close(a, b, tol):
    return abs(a - b) <= tol * abs(b)


def draws(mode, engaged, n=N, seed=0):
    model = ResponseDelay(mode, Random(seed))
    return [model.draw(engaged) for _ in range(n)]


def main():
    for mode in DELAY_MODES:
        check(f'{mode}: reproducible from a seed',
              draws(mode, False, 500, 7) == draws(mode, False, 500, 7))
        model = ResponseDelay(mode, Random(0))
        check(f'{mode}: engaged branch is never slower',
              model.mean_s(True) <= model.mean_s(False))

    check('none: exactly 0 s', set(draws('none', False, 100)) == {0.0})
    check(f'deterministic: exactly {FIRST_S:g} s / {NEXT_S:g} s',
          set(draws('deterministic', False, 100)) == {FIRST_S}
          and set(draws('deterministic', True, 100)) == {NEXT_S})

    # The realised mean must match what mean_s() claims, in both stochastic modes.
    for mode in ('lognormal', 'probabilistic'):
        for engaged, label in ((False, 'first'), (True, 'next')):
            sample  = statistics.fmean(draws(mode, engaged))
            claimed = ResponseDelay(mode, Random(0)).mean_s(engaged)
            check(f'{mode}/{label}: sample mean matches mean_s()', close(sample, claimed, 0.02),
                  f'sample {sample:.2f} s, claimed {claimed:.2f} s')

    # Guards the mu = log(target) - sigma^2/2 correction: without it the mean is ~8% high.
    for engaged, target in ((False, FIRST_S), (True, NEXT_S)):
        sample = statistics.fmean(draws('lognormal', engaged))
        check(f'lognormal: mean is the target {target:g} s, not the median',
              close(sample, target, 0.02), f'{sample:.2f} s')

    for engaged, target in ((False, FIRST_S), (True, NEXT_S)):
        sample = draws('probabilistic', engaged)
        p = 1.0 / target
        # No arm has a ceiling any more, so both stochastic ones must run past 70 s.
        check(f'probabilistic/{target:g}: whole seconds from 1, unbounded above',
              all(float(d).is_integer() for d in sample)
              and min(sample) == 1.0 and max(sample) > 70.0)

        # Memorylessness: the chance of responding in the next second does not depend on
        # how long the pilot has already been silent. This is what separates it from
        # the lognormal arm, where a long silence means a response is overdue.
        for waited in (5, 15, 25):
            still_waiting = [d for d in sample if d > waited]
            responds_next = sum(1 for d in still_waiting if d == waited + 1) / len(still_waiting)
            check(f'probabilistic/{target:g}: memoryless after {waited} s',
                  close(responds_next, p, 0.15), f'{responds_next:.4f} vs {p:.4f}')

    # The geometric tail must match (1-p)^k, with no pile-up at any ceiling.
    sample = draws('probabilistic', False)
    p = 1.0 / FIRST_S
    for k in (30, 60, 90):
        beyond = sum(1 for d in sample if d > k) / len(sample)
        check(f'probabilistic: P(delay > {k} s) matches the geometric tail',
              close(beyond, (1 - p) ** k, 0.10), f'{beyond:.4f} vs {(1 - p) ** k:.4f}')

    try:
        ResponseDelay('sometimes', Random(0))
        check('unknown mode is rejected', False)
    except ValueError:
        check('unknown mode is rejected', True)

    print()
    print(f'{len(failures)} failure(s)' if failures else 'all checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
