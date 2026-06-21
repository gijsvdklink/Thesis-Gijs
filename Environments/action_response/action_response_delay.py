"""
action_response_delay -- pluggable pilot/ATCO action-response delay models.

The env (experimental.py) holds one of these models and uses it to
  * sample the INITIATION DELAY (seconds) before an issued instruction reaches the
    simulator  -- sample_delay(subsequent);
  * map response progress to the resp_pend observation -- pending_signal(elapsed, delay),
    where resp_pend in [0,1] is "probability the pilot has NOT responded yet".

Two models, after Kochenderfer et al. ("Robust Airborne Collision Avoidance through
Dynamic Programming", §6):

  DeterministicDelay  fixed delay = mean_s. The response time is known, so resp_pend is
                      an exact linear countdown.
  ProbabilisticDelay  delay ~ lognormal(mean_s, sigma) (mode ~10.75 s, mean 20.5 s, long
                      tail -- the empirical initiation-delay shape). The response time is
                      uncertain, so resp_pend is the lognormal survival 1 - CDF(elapsed).

Both share the same MEAN, so deterministic-vs-probabilistic isolates the effect of
response VARIABILITY. A subsequent (superseding) instruction draws a shorter delay
(subsequent_factor), mirroring the paper's 5 s -> 3 s.

Build one with make_delay_model('deterministic'|'probabilistic', **overrides).
"""

import math
import random

# Delay draws use their own RNG, separate from the global `random` stream that drives
# sector/spawn geometry -- so deterministic and probabilistic runs share the airspace
# and differ only in the response delay.
_RNG = random.Random()

def seed_delay(seed):
    _RNG.seed(seed)


class DeterministicDelay:
    """Fixed response delay; resp_pend is an exact linear countdown."""

    def __init__(self, mean_s=20.5, subsequent_factor=0.6, **_):
        self.mean_s            = mean_s
        self.subsequent_factor = subsequent_factor

    def sample_delay(self, subsequent=False):
        return self.mean_s * (self.subsequent_factor if subsequent else 1.0)

    def pending_signal(self, elapsed_s, delay_s):
        return max(0.0, 1.0 - elapsed_s / max(delay_s, 1e-9))


class ProbabilisticDelay:
    """Lognormal response delay; resp_pend = P(not yet responded) = 1 - CDF(elapsed)."""

    def __init__(self, mean_s=20.5, sigma=0.656, subsequent_factor=0.6, **_):
        self.mean_s            = mean_s
        self.sigma             = sigma
        self.subsequent_factor = subsequent_factor
        self.mu                = math.log(mean_s) - sigma ** 2 / 2.0   # E[delay] = mean_s

    def sample_delay(self, subsequent=False):
        mean = self.mean_s * (self.subsequent_factor if subsequent else 1.0)
        mu   = math.log(mean) - self.sigma ** 2 / 2.0
        return _RNG.lognormvariate(mu, self.sigma)

    def pending_signal(self, elapsed_s, delay_s):
        if elapsed_s <= 0.0:
            return 1.0
        cdf = 0.5 * (1.0 + math.erf((math.log(elapsed_s) - self.mu) / (self.sigma * math.sqrt(2.0))))
        return 1.0 - cdf


def make_delay_model(mode='probabilistic', **overrides):
    """mode in {'deterministic', 'probabilistic'}; overrides go to the constructor
    (e.g. mean_s=10.0, sigma=0.7)."""
    if mode == 'deterministic':
        return DeterministicDelay(**overrides)
    if mode == 'probabilistic':
        return ProbabilisticDelay(**overrides)
    raise ValueError(f"unknown delay mode: {mode!r}")


__all__ = ['DeterministicDelay', 'ProbabilisticDelay', 'make_delay_model', 'seed_delay']
