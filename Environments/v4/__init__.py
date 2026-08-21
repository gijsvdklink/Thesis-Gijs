# v4 ATC conflict-resolution environment.
#
# Two properties define it:
#   1. The observation is in RAW PHYSICAL UNITS -- VecNormalize does all the scaling, so
#      training and evaluation MUST load the matching *_vecnorm.pkl.
#   2. Instructions are subject to an ACTION-RESPONSE DELAY, selected per instance:
#          AirspaceEnv(delay_mode='none' | 'deterministic' | 'lognormal' | 'probabilistic')
#      The 'pending' and 'wait_s' observation features exist in all four modes (constant 0
#      under 'none'), so the delay types share one observation space and can be cross-evaluated.
#
# Layout:  config.py  settings and constants      delays.py    response-delay models
#          geometry.py coordinate transforms      conflict.py  separation geometry
#          sector.py  sector and route generation env.py       the AirspaceEnv class

from .config import (CONFIG, OBS_DIM, OBS_OWNSHIP_LABELS, OBS_INTRUDER_LABELS, NM_TO_KM)
from .delays import DELAY_MODES
from .geometry import latlon_to_nm, nm_to_latlon, wrap_to_180
from .env import AirspaceEnv

__all__ = ['AirspaceEnv', 'CONFIG', 'DELAY_MODES', 'OBS_DIM', 'OBS_OWNSHIP_LABELS',
           'OBS_INTRUDER_LABELS', 'NM_TO_KM', 'latlon_to_nm', 'nm_to_latlon', 'wrap_to_180']
