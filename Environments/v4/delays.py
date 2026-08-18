# Action-response delay: the seconds a pilot takes between an advisory being ISSUED and
# the action being EXECUTED.
#
#   none            0 s                            (baseline)
#   deterministic   exactly mean_s
#   lognormal       right-skewed, mean mean_s
#   probabilistic   geometric in whole seconds, mean mean_s
#
# All three delayed arms have the same mean and no upper bound, so at a given magnitude
# they differ only in SHAPE. The magnitude is the second experiment factor: pass it per
# run, so shape and magnitude can be crossed.
#
# ONE sample per advisory. The environment draws it the moment an advisory is issued to
# an aircraft that is not already waiting on one, and stores the execution time. The
# response time is hidden state: the agent never sees it, and once drawn it is fixed --
# amending the advisory changes WHAT the pilot will fly, never WHEN.

import math

DELAY_MODES = ('none', 'deterministic', 'lognormal', 'probabilistic')

MEAN_DELAY_S = 30.0   # default magnitude: the mean response time, in seconds
SIGMA        = 0.4    # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s


class ResponseDelay:
    """How long this arm's pilots take to act on an advisory.

    rng is an injected random.Random, never the global stream: that one generates
    sectors and traffic, and a delay draw taken from it would shift every later
    scenario decision.
    """

    def __init__(self, mode, rng, mean_s=MEAN_DELAY_S):
        if mode not in DELAY_MODES:
            raise ValueError(f'unknown delay mode {mode!r}; expected one of {DELAY_MODES}')
        self.mode   = mode
        self.rng    = rng
        self.mean_s = float(mean_s)

    def sample_delay_s(self):
        """One response time, in seconds. Called once per advisory."""
        if self.mode == 'none':
            return 0.0

        if self.mode == 'deterministic':
            return self.mean_s

        if self.mode == 'lognormal':
            # Parameterised on the MEAN, not the median: E[X] = exp(mu + sigma^2/2), so
            # without the correction the mean would land ~8% above the arm magnitude and
            # the arms would no longer be comparable.
            mu = math.log(self.mean_s) - SIGMA ** 2 / 2.0
            return self.rng.lognormvariate(mu, SIGMA)

        # Probabilistic: a constant 1/mean_s chance of responding in each second gives a
        # geometric delay on {1, 2, ...} seconds. Sampled in one step by inverse transform
        # rather than rolled second by second -- same distribution, and it gives the
        # execution time up front like the other arms.
        p = 1.0 / self.mean_s
        return math.ceil(math.log(1.0 - self.rng.random()) / math.log(1.0 - p))

    def expected_delay_s(self):
        """The mean this arm is aiming at. Diagnostics and checks only."""
        # Geometric with p = 1/mean_s has mean mean_s, and the lognormal is parameterised
        # on its mean, so all three delayed arms land on the same number.
        return 0.0 if self.mode == 'none' else self.mean_s
