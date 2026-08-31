# Response delay, in seconds between an advisory being ISSUED and EXECUTED: none, deterministic, lognormal or probabilistic, all with the same mean, so they differ only in SHAPE. One sample per piece of advice; changed advice draws again.

import numpy as np

DELAY_MODES = ('none', 'deterministic', 'lognormal', 'probabilistic')

MEAN_DELAY_S = 30.0   # default magnitude: the mean response time, in seconds
SIGMA        = 0.4    # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s


class ResponseDelay:

    def __init__(self, mode, rng, mean_s=MEAN_DELAY_S):
        if mode not in DELAY_MODES:
            raise ValueError(f'unknown delay mode {mode!r}; expected one of {DELAY_MODES}')
        self.mode   = mode
        self.rng    = rng
        self.mean_s = float(mean_s)

    def sample_delay_s(self):
        """One response time in WHOLE seconds, because the queue is tested once per simulated second."""
        if self.mode == 'none':
            return 0.0

        if self.mode == 'deterministic':
            return self.mean_s

        if self.mode == 'lognormal':
            # numpy takes the UNDERLYING NORMAL's parameters, so convert: E[D] = exp(mu + sigma^2/2).
            mu = np.log(self.mean_s) - SIGMA ** 2 / 2.0
            return float(np.round(self.rng.lognormal(mu, SIGMA)))

        # Geometric: a constant 1/mean_s chance of responding each second, drawn in one step at issue.
        return float(self.rng.geometric(1.0 / self.mean_s))

    def expected_delay_s(self):
        """The mean this delay type is aiming at. Diagnostics and checks only."""
        # Geometric with p = 1/mean_s and the mean-parameterised lognormal both land on mean_s.
        return 0.0 if self.mode == 'none' else self.mean_s
