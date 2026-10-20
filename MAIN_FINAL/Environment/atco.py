import math

from .config import CONFIG

DELAY_MODES = ('none', 'deterministic', 'lognormal')


class ATCO:

    def __init__(self, delay_type, rng, mean_s=None):
        if delay_type not in DELAY_MODES:
            raise ValueError(f'unknown delay type {delay_type!r}; expected one of {DELAY_MODES}')
        self.delay_type = delay_type
        self.rng        = rng
        self.mean_s     = float(mean_s if mean_s is not None else CONFIG['delay_mean_s'])

        self._mu = (math.log(self.mean_s) - CONFIG['delay_sigma'] ** 2 / 2.0
                    if delay_type == 'lognormal' else 0.0)

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

        revision = held is not None
        tau      = self._draw_delay_s()
        if revision:
            tau = CONFIG['revision_kappa'] * tau

        execute_at = now_s + round(tau)
        if revision:
            execute_at = max(held['execute_at_s'], execute_at)

        advisory['issued_at_s']      = now_s
        advisory['response_start_s'] = self.advisory['response_start_s'] if revision else now_s
        advisory['execute_at_s']     = execute_at

        self.cs, self.advisory = cs, advisory
        return True

    def _same_instruction(self, held, advisory):
        for kind in ('target_hdg', 'target_mach'):
            if kind in held and kind in advisory:
                return abs(held[kind] - advisory[kind]) < 1e-9
        return False

    def act_if_ready(self, now_s, still_flying):
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

    def _draw_delay_s(self):
        if self.delay_type == 'none':
            return 0.0
        if self.delay_type == 'deterministic':
            return self.mean_s
        return float(self.rng.lognormal(self._mu, CONFIG['delay_sigma']))
