"""Render the initial state of one episode of one training run, and fingerprint it.

    python render_episode.py <run_seed> <episode_index> <out_png>

The episode index counts draws from that run's scenario stream, so episode 12000 of run 5 is
the 12000th scenario AirspaceEnv(seed=5) would fly. The scenario seed is replayed directly
rather than by resetting 12000 times: both paths build the episode from the same number.
"""

import hashlib
import json
import os
import sys
from random import Random

# The MAIN folder, two levels up: Visualisation/seed_check/ -> MAIN.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Environment import AirspaceEnv
from Environment.config import TRAINING_SCENARIOS, HELD_OUT


def scenario_seed_for(run_seed, episode_index):
    """The episode_index-th scenario seed the run draws, held-out seeds skipped as in env.py."""
    stream = Random(int(run_seed))
    seed = None
    for _ in range(episode_index):
        seed = stream.randrange(TRAINING_SCENARIOS)
        while seed in HELD_OUT:
            seed = stream.randrange(TRAINING_SCENARIOS)
    return seed


def aircraft_arrow(axis, east, north, heading_deg, length):
    """An aircraft as a short arrow pointing along its heading (compass degrees)."""
    angle = np.radians(heading_deg)
    axis.arrow(east, north, length * np.sin(angle), length * np.cos(angle),
               head_width=length * 0.45, head_length=length * 0.45,
               fc='#0C2340', ec='#0C2340', linewidth=0.8, length_includes_head=True)


def main():
    run_seed      = int(sys.argv[1])
    episode_index = int(sys.argv[2])
    out_png       = sys.argv[3]

    seed = scenario_seed_for(run_seed, episode_index)

    env = AirspaceEnv(delay_mode='none', seed=run_seed)
    env.reset(options={'scenario_seed': seed})

    callsigns = sorted(env._aircraft)
    state = {
        'run_seed':      run_seed,
        'episode_index': episode_index,
        'scenario_seed': env.episode_seed,
        'n_aircraft':    int(env.n_aircraft),
        'rho':           float(env.rho),
        'polygon':       [[round(x, 9), round(y, 9)]
                          for x, y in env._polygon_shape.exterior.coords],
        'aircraft':      [{'cs': cs,
                           'initial_hdg':  round(float(env._aircraft[cs].initial_hdg), 9),
                           'spawn_pos_nm': [round(float(p), 9)
                                            for p in env._aircraft[cs].spawn_pos_nm]}
                          for cs in callsigns],
    }
    blob        = json.dumps(state, sort_keys=True).encode()
    fingerprint = hashlib.sha256(blob).hexdigest()[:16]

    # -- the picture -----------------------------------------------------------
    east, north = zip(*env._polygon_shape.exterior.coords)
    span   = max(max(east) - min(east), max(north) - min(north))
    length = span * 0.035

    figure, axis = plt.subplots(figsize=(7.5, 7.5))
    axis.fill(east, north, facecolor='#8CD7F4', alpha=0.16, zorder=0)
    axis.plot(east, north, color='#004D7D', linewidth=1.6, zorder=1)

    for cs in callsigns:
        record = env._aircraft[cs]
        e, n   = float(record.spawn_pos_nm[0]), float(record.spawn_pos_nm[1])
        # The 5 NM separation minimum around each aircraft, to scale.
        axis.add_patch(plt.Circle((e, n), 5.0, facecolor='#00A6D6', alpha=0.10,
                                  edgecolor='none', zorder=2))
        aircraft_arrow(axis, e, n, record.initial_hdg, length)
        axis.annotate(cs, (e, n), textcoords='offset points', xytext=(6, 6),
                      fontsize=6.5, color='#004D7D', zorder=4)

    axis.set_aspect('equal')
    axis.set_xlabel('east [NM]')
    axis.set_ylabel('north [NM]')
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.set_title(f'run seed {run_seed}, episode {episode_index:,}  '
                   f'(scenario seed {env.episode_seed})\n'
                   f'{state["n_aircraft"]} aircraft, '
                   f'rho = 1/{1 / state["rho"]:,.0f} per km$^2$   |   '
                   f'fingerprint {fingerprint}', fontsize=9)
    figure.tight_layout()
    figure.savefig(out_png, dpi=130)
    plt.close(figure)

    with open(out_png.replace('.png', '.json'), 'wb') as handle:
        handle.write(blob)

    print(f'{out_png}  scenario_seed={env.episode_seed}  n_ac={state["n_aircraft"]}  '
          f'fingerprint={fingerprint}')


if __name__ == '__main__':
    main()
