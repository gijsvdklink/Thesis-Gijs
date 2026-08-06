"""
v4 ATC conflict-resolution environment.

Two properties define it:

1. The observation is reported in RAW PHYSICAL UNITS (NM, kt, s, rad). There are no
   hand-picked normalisers -- no D_WARN range scale, no division by cruise speed, no
   clipping to [0, 1] -- so all scaling is left to VecNormalize(norm_obs=True). Training
   and evaluation MUST load the matching *_vecnorm.pkl; raw observations fed to the
   policy directly produce nonsense.
2. Instructions are subject to an ACTION-RESPONSE DELAY: the pilot acts delay_s after
   the controller issues, selected per environment instance via

       AirspaceEnv(delay_mode='none' | 'deterministic' | 'probabilistic')

   The 'pending' observation feature (1 while an issued instruction has not yet been
   executed) exists in all three modes, so the three delay arms share one observation
   space and policies can be cross-evaluated across them. Checkpoints from any earlier
   25-feature version of this env are NOT loadable.

Public API (import from Environments.v4):
    AirspaceEnv          the gymnasium environment
    CONFIG               tunable settings dict
    OBS_DIM              observation length (26)
    OBS_OWNSHIP_LABELS   per-feature labels for the ownship part of the observation
    OBS_INTRUDER_LABELS  per-feature labels for each intruder slot
    latlon_to_nm, nm_to_latlon, wrap_to_180   coordinate helpers (used by the visualiser)

Internal layout:
    config.py    settings, derived constants, action layout, workload costs
    geometry.py  coordinate transforms and aircraft-state helpers
    conflict.py  separation geometry (time-to-LoS, urgency, conflict score, LoS checks)
    sector.py    sector polygon generation and aircraft placement
    env.py       the AirspaceEnv class
"""

from .config import (CONFIG, OBS_DIM, OBS_OWNSHIP_LABELS, OBS_INTRUDER_LABELS, NM_TO_KM)
from .geometry import latlon_to_nm, nm_to_latlon, wrap_to_180
from .env import AirspaceEnv

__all__ = ['AirspaceEnv', 'CONFIG', 'OBS_DIM', 'OBS_OWNSHIP_LABELS', 'OBS_INTRUDER_LABELS',
           'NM_TO_KM', 'latlon_to_nm', 'nm_to_latlon', 'wrap_to_180']
