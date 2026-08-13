# Action-response delay: seconds between the controller issuing an instruction and the
# pilot acting on it.
#
#   none            0 s                                          (baseline)
#   deterministic   30 s, or 15 s once engaged
#   lognormal       mean 30 s / 15 s
#   probabilistic   1/30 chance per second, or 1/15 once engaged  (the Markov chain)
#
# All three delayed arms have exactly the same mean and no ceiling; they differ only in
# shape, which is what the experiment isolates.
#
# `engaged` = has this pilot already executed an advisory? The env decides it, and the
# rule is the same in every arm: False whenever the focus moves to a new aircraft, True
# from the moment an advisory executes, until the focus leaves.

import math

DELAY_MODES = ('none', 'deterministic', 'lognormal', 'probabilistic')

FIRST_S = 30.0    # first advisory to a new focus ship
NEXT_S  = 15.0    # once an advisory has executed and the aircraft still has the focus

SIGMA   = 0.4     # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s


class ResponseDelay:
    """Draws the response delay for one experiment arm.

    rng is an injected random.Random, never the global stream: that one generates
    sectors and traffic, and a delay draw taken from it would shift every later
    scenario decision.
    """

    def __init__(self, mode, rng):
        if mode not in DELAY_MODES:
            raise ValueError(f'unknown delay mode {mode!r}; expected one of {DELAY_MODES}')
        self.mode = mode
        self.rng  = rng

    def draw(self, engaged):
        """Seconds until the pilot acts on an instruction issued right now."""
        target = NEXT_S if engaged else FIRST_S

        if self.mode == 'none':
            return 0.0

        if self.mode == 'deterministic':
            return target

        if self.mode == 'lognormal':
            # Parameterised on the mean, not the median: E[X] = exp(mu + sigma^2/2).
            mu = math.log(target) - SIGMA ** 2 / 2.0
            return self.rng.lognormvariate(mu, SIGMA)

        # Markov chain: each second the pilot either stays in "not executed" (29/30, or
        # 14/15 once engaged) or executes. That is a geometric distribution, and the
        # per-second chance cannot change while the instruction is outstanding, so
        # drawing the whole delay now is equivalent and keeps the queue a deadline check.
        # Sampled by inverse transform, since P(delay > k) = (1 - p)^k.
        p = 1.0 / target
        return float(math.ceil(math.log(self.rng.random()) / math.log(1.0 - p)))

    def mean_s(self, engaged):
        """Expected delay on this branch. Diagnostics and tests only."""
        if self.mode == 'none':
            return 0.0
        # Geometric with p = 1/target has mean target, and the lognormal is parameterised
        # on its mean, so all three delayed arms land on exactly the same number.
        return NEXT_S if engaged else FIRST_S
