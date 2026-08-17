# Action-response delay: seconds between the controller issuing an instruction and the
# pilot acting on it.
#
#   none            0 s                                              (baseline)
#   deterministic   first_s, or next_s once engaged
#   lognormal       mean first_s / next_s
#   probabilistic   1/first_s chance per second, 1/next_s once engaged (the Markov chain)
#
# At a given magnitude all three delayed arms have exactly the same mean and no ceiling,
# so they differ only in SHAPE. The magnitude (first_s) is the second factor: pass it per
# run, so shape and magnitude can be crossed.
#
# `engaged` = has this pilot already executed an advisory? The env decides it, and the
# rule is the same in every arm: False whenever the focus moves to a new aircraft, True
# from the moment an advisory executes, until the focus leaves.
#
# Two hooks, because the arms decide at different moments:
#   on_issue  fixes whatever the model knows at issue time
#   is_due    asked once per SIMULATED SECOND while an instruction is outstanding
# The deterministic and lognormal arms settle the whole delay up front and is_due is
# then just a clock check. The probabilistic arm settles nothing: it rolls the chain,
# one transition per second, exactly as drawn.

import math

DELAY_MODES = ('none', 'deterministic', 'lognormal', 'probabilistic')

FIRST_S = 30.0    # default first advisory to a new focus ship
NEXT_S  = 15.0    # default once an advisory has executed and the aircraft keeps the focus
SIGMA   = 0.4     # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s

# Timing fields a pending instruction carries. An amendment inherits all of them, so
# re-issuing can never restart the clock or redraw the delay.
TIMING_FIELDS = ('issued_at_s', 'engaged', 'execute_at_s', 'hazard_p')


class ResponseDelay:
    """Decides when the pilot acts, for one experiment arm.

    rng is an injected random.Random, never the global stream: that one generates
    sectors and traffic, and a delay draw taken from it would shift every later
    scenario decision.
    """

    def __init__(self, mode, rng, first_s=FIRST_S, next_s=None):
        if mode not in DELAY_MODES:
            raise ValueError(f'unknown delay mode {mode!r}; expected one of {DELAY_MODES}')
        self.mode    = mode
        self.rng     = rng
        self.first_s = float(first_s)
        # The follow-up delay defaults to half the first: an engaged pilot responds twice
        # as fast. Crossing magnitude with shape then needs only one number per run.
        self.next_s  = float(next_s) if next_s is not None else self.first_s / 2.0

    def _target(self, engaged):
        """The mean this branch is aiming at, in seconds."""
        return self.next_s if engaged else self.first_s

    def on_issue(self, engaged, now):
        """Timing fields to store on the instruction the controller has just issued."""
        if self.mode == 'probabilistic':
            # Nothing is decided yet -- only the per-second transition probability, which
            # cannot change while this instruction is outstanding.
            return {'hazard_p': 1.0 / self._target(engaged)}
        return {'execute_at_s': now + self._deadline_delay(engaged)}

    def is_due(self, cmd, now):
        """Does the pilot act on this outstanding instruction at simulated second `now`?"""
        if 'hazard_p' in cmd:
            # One roll per second. Never on the issuing second itself, so the shortest
            # possible response is 1 s and the delay is geometric on {1, 2, ...}.
            return now > cmd['issued_at_s'] and self.rng.random() < cmd['hazard_p']
        return now >= cmd['execute_at_s']

    def _deadline_delay(self, engaged):
        """The whole delay, for the arms that settle it at issue time."""
        target = self._target(engaged)
        if self.mode == 'none':
            return 0.0
        if self.mode == 'deterministic':
            return target
        # Parameterised on the mean, not the median: E[X] = exp(mu + sigma^2/2).
        return self.rng.lognormvariate(math.log(target) - SIGMA ** 2 / 2.0, SIGMA)

    def draw(self, engaged):
        """One complete delay, start to finish.

        The environment never calls this -- it uses on_issue and is_due -- but the
        figure and the checks do. For the probabilistic arm it rolls the chain second
        by second, so it exercises the same code the simulation runs.
        """
        if self.mode != 'probabilistic':
            return self._deadline_delay(engaged)

        cmd = {'issued_at_s': 0.0, **self.on_issue(engaged, 0.0)}
        seconds = 0.0
        while not self.is_due(cmd, seconds):
            seconds += 1.0
        return seconds

    def mean_s(self, engaged):
        """Expected delay on this branch. Diagnostics and checks only."""
        if self.mode == 'none':
            return 0.0
        # Geometric with p = 1/target has mean target, and the lognormal is parameterised
        # on its mean, so all three delayed arms land on exactly the same number.
        return self._target(engaged)
