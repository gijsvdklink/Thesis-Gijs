# The ATCO: the human the CR tool advises, and the only route from an advisory to an aircraft.
#
# The CR tool (the urgency ranking plus the PPO model) hands the ATCO one advisory per step. The
# ATCO holds ONE instruction at a time -- one controller, one radio -- and takes a response delay
# to act on it. The whole controller-response model is this one class.
#
# One delay per aircraft TAKEN UP, not per advisory. Revising the advice for the aircraft already
# in hand changes what is flown, not when: the response time belongs to the controller getting to
# that aircraft, not to the words. Re-drawing on every revision would let the tool re-roll for a
# luckier draw, and -- because each draw starts from the current moment -- would keep the
# execution time ahead of the clock, so under a tool that advises every step nothing would ever
# be flown at all.
#
# Two shapes of delay, because the four modes are not the same kind of thing:
#
#   scheduled   none, deterministic, lognormal. A response time is drawn when the aircraft is
#               taken up, and the ATCO acts at that moment.
#
#   memoryless  geometric. A constant chance of acting each second, which is the ACAS Xa pilot
#               chain. Nothing is scheduled at all. Drawing a one-shot geometric and re-drawing
#               it on revision would destroy exactly the memorylessness the model represents.

import numpy as np

DELAY_MODES = ('none', 'deterministic', 'lognormal', 'geometric')

MEAN_DELAY_S = 30.0   # the mean time for the controller to act on an aircraft, in seconds
SIGMA        = 0.4    # lognormal shape; at mean 30 s, 80% of draws fall in 17-46 s


class ATCO:

    def __init__(self, mode, rng, mean_s=MEAN_DELAY_S):
        if mode not in DELAY_MODES:
            raise ValueError(f'unknown delay mode {mode!r}; expected one of {DELAY_MODES}')
        self.mode   = mode
        self.rng    = rng
        self.mean_s = float(mean_s)

        self.cs       = None      # the aircraft being worked on, None when free
        self.advisory = None      # what it is to be told

    def standing_for(self, cs):
        # The instruction in hand, but only if it is for this aircraft.
        return self.advisory if self.cs == cs else None

    def forget(self, cs):
        if self.cs == cs:
            self.cs = self.advisory = None

    def accept(self, cs, advisory, now_s):
        # Take one advisory. Anything already in hand is dropped -- including an instruction for
        # a DIFFERENT aircraft, which is then never flown.
        revision = self.cs == cs and self.advisory is not None
        advisory['issued_at_s'] = now_s               # when these words were chosen

        if revision:
            # Same aircraft: the clock already running keeps running.
            advisory['taken_up_at_s'] = self.advisory['taken_up_at_s']
            advisory['execute_at_s']  = self.advisory['execute_at_s']
        else:
            advisory['taken_up_at_s'] = now_s
            advisory['execute_at_s']  = (None if self.mode == 'geometric'
                                         else now_s + self.delay_s())

        self.cs, self.advisory = cs, advisory

    def due(self, now_s):
        # (cs, advisory) once the ATCO acts, and it is free again. None until then.
        if self.advisory is None:
            return None

        if self.mode == 'geometric':
            if self.rng.random() >= 1.0 / self.mean_s:
                return None
            self.advisory['execute_at_s'] = now_s     # known only now; the visualiser reads it
        elif now_s < self.advisory['execute_at_s']:
            return None

        ready = (self.cs, self.advisory)
        self.cs, self.advisory = None, None
        return ready

    def delay_s(self):
        # One scheduled response time in whole seconds; the ATCO is checked once per second.
        if self.mode == 'none':
            return 0.0
        if self.mode == 'deterministic':
            return self.mean_s
        # numpy takes the UNDERLYING NORMAL's parameters: E[D] = exp(mu + sigma^2/2).
        mu = np.log(self.mean_s) - SIGMA ** 2 / 2.0
        return float(np.round(self.rng.lognormal(mu, SIGMA)))
