"""
v4-experiments ATC conflict-resolution environment.

Identical to Environments/v4 except that the observation is reported in RAW PHYSICAL
UNITS (NM, kt, s, rad). The hand-picked normalisers are gone -- no D_WARN range scale,
no division by cruise speed, no clipping to [0, 1] -- so all scaling is left to
VecNormalize(norm_obs=True) at training time.

Public API (import from Environments.v4_experiments):
    AirspaceEnv          the gymnasium environment
    CONFIG               tunable settings dict
    OBS_DIM              observation length (25)
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
