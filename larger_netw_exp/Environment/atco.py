import math

import numpy as np

from .config import CONFIG

DELAY_MODES = ('none', 'deterministic', 'lognormal')


class ATCO:

    def __init__(self, delay_type, rng, mean_s=None):
        if delay_type not in DELAY_MODES:
            raise ValueError(f'unknown delay type {delay_type!r}; expected one of {DELAY_MODES}')
        self.delay_type = delay_type
        self.rng        = rng
        self.mean_s     = float(mean_s if mean_s is not None else CONFIG['delay_mean_s'])

        self._mu = self._mu_for_capped_mean() if delay_type == 'lognormal' else 0.0

        self.cs       = None
        self.advisory = None

    def pending_for(self, cs):
        return self.advisory if self.cs == cs else None

    def release(self, cs):
        if self.cs == cs:
            self.cs = self.advisory = None

    def receive(self, cs, advisory, now_s):
        held = self.pending_for(cs)
        if held is not None and self._same_instruction(held, advisory):
            return False

        # A revision of the instruction in hand is taken up faster than a fresh one; either way
        # the response time is drawn afresh, and anything already held is dropped.
        revision = held is not None
        tau      = self._draw_delay_s()
        if revision:
            tau = CONFIG['revision_kappa'] * tau

        execute_at = now_s + round(tau)
        if revision:
            # A revision may only POSTPONE the response, never advance it. The controller is
            # already committed to acting at the standing time, so revising cannot buy an
            # earlier one: with kappa = 0.7 a revision drawn 5 s in would otherwise land at
            # 5 + 21 = 26 s against a standing 30 s, and changing the advice would be rewarded.
            # Applies to every law, the drawn tau being the only thing that differs.
            execute_at = max(held['execute_at_s'], execute_at)

        advisory['issued_at_s']      = now_s
        advisory['response_start_s'] = self.advisory['response_start_s'] if revision else now_s
        advisory['execute_at_s']     = execute_at
        # Advisories discarded so far in this response: each one added a draw to the max above.
        advisory['revisions']        = held['revisions'] + 1 if revision else 0

        self.cs, self.advisory = cs, advisory
        return True

    def _same_instruction(self, held, advisory):
        for kind in ('target_hdg', 'target_mach'):
            if kind in held and kind in advisory:
                return abs(held[kind] - advisory[kind]) < 1e-9
        return False

    def act_if_ready(self, now_s, still_flying):
        # The once-a-second poll of Figure 2.5. The aircraft is checked BEFORE the clock: an
        # instruction whose aircraft has left the sector is deleted, never flown, and never
        # counted as a response the controller served.
        if self.advisory is None:
            return None

        if not still_flying(self.cs):
            self.cs = self.advisory = None
            return None

        if now_s < self.advisory['execute_at_s']:
            return None

        ready = (self.cs, self.advisory)
        self.cs, self.advisory = None, None
        return ready

    def _mu_for_capped_mean(self):
        # The cap removes mass from the upper tail, so the plain mu = ln(mean) - sigma^2/2
        # would leave the CAPPED mean below mean_s. Solve instead for the mu whose capped mean
        # is exactly mean_s, by bisection on the closed form of E[min(X, cap)].
        sigma, cap = CONFIG['delay_sigma'], CONFIG['delay_max_s']

        def phi(z):
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        def capped_mean(mu):
            lnc = math.log(cap)
            return (math.exp(mu + sigma ** 2 / 2.0) * phi((lnc - mu - sigma ** 2) / sigma)
                    + cap * (1.0 - phi((lnc - mu) / sigma)))

        lo, hi = math.log(self.mean_s) - 4 * sigma, math.log(self.mean_s) + 4 * sigma
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if capped_mean(mid) < self.mean_s:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _draw_delay_s(self):
        # Both laws have a mean of mean_s, so they differ in spread alone.
        if self.delay_type == 'none':
            return 0.0
        if self.delay_type == 'deterministic':
            return self.mean_s
        # Capped: a controller does not take minutes to act on a single conflict. mu is solved
        # so that the mean AFTER capping is exactly mean_s, keeping the two laws comparable.
        return float(min(self.rng.lognormal(self._mu, CONFIG['delay_sigma']),
                         CONFIG['delay_max_s']))
