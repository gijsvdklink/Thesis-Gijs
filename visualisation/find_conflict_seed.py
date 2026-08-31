"""Find a SCENARIO SEED where two aircraft actually come into conflict, closest-approach first, for the visualiser."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environments.v4 import AirspaceEnv, CONFIG

HOLD = 3


def closest_approach_nm(env, scenario_seed):
    """Smallest distance between the two aircraft over one uninstructed episode."""
    env.reset(options={'scenario_seed': scenario_seed})
    closest = float('inf')
    while True:
        _, _, _, truncated, _ = env.step(HOLD)
        if len(env._pos) == 2:
            closest = min(closest, float(np.hypot(*(env._pos[0] - env._pos[1]))))
        if truncated:
            return closest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=40, help='how many scenarios to try')
    ap.add_argument('--first', type=int, default=0, help='first scenario seed')
    ap.add_argument('--n_ac', type=int, default=2, help='aircraft in the sector')
    ap.add_argument('--density', type=float, default=1 / 10000, help='aircraft per km^2')
    args = ap.parse_args()

    CONFIG['n_aircraft'] = lambda rng, n=args.n_ac: n
    CONFIG['rho']        = lambda rng, r=args.density: r

    env = AirspaceEnv()
    results = []
    for seed in range(args.first, args.first + args.seeds):
        closest = closest_approach_nm(env, seed)
        results.append((closest, seed))
        print(f'  seed {seed:>4}  closest approach {closest:7.1f} NM', flush=True)

    print('\nbest seeds (closest first):')
    for closest, seed in sorted(results)[:10]:
        note = 'loss of separation' if closest < CONFIG['sep_nm'] else ''
        print(f'  --seed {seed:<5} {closest:7.1f} NM  {note}')


if __name__ == '__main__':
    main()
