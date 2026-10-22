"""The routes flown in one scenario, as a static figure.

python Visualisation/route_plot.py --model <path to .zip> --n-ac 15 --rho 15000 --seed 0

Flies one scenario with a trained policy and draws what the aircraft actually did: the sector
boundary, the straight route each aircraft would have flown had it never been advised, and the
track it flew instead. Only the aircraft present at the start are followed, so the picture stays
readable; the run stops once they have all left the sector.
"""

import argparse, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import io

class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def _patched_json_to_data(json_string, custom_objects=None):
    import base64, json, warnings
    data = {}
    for key, item in json.loads(json_string).items():
        if custom_objects and key in custom_objects:
            data[key] = custom_objects[key]
        elif isinstance(item, dict) and ':serialized:' in item:
            try:
                data[key] = _NumpyShim(io.BytesIO(base64.b64decode(
                    item[':serialized:'].encode()))).load()
            except Exception as error:
                warnings.warn(f'could not deserialise {key}: {error}')
        else:
            data[key] = item
    return data


from stable_baselines3.common import save_util
save_util.json_to_data = _patched_json_to_data


class _Ignored:
    def __init__(self, *args): pass
    def __setstate__(self, state): pass


class _Unpickler(pickle.Unpickler):
    """Reads the observation statistics under any numpy version, as the evaluation does."""
    def find_class(self, module, name):
        if module.startswith('numpy.random'):
            return _Ignored
        if np.__version__.startswith('1.'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(path, env=None):
    from stable_baselines3 import PPO
    with open(path.replace('.zip', '_vecnorm.pkl'), 'rb') as handle:
        normaliser = _Unpickler(handle).load()
    mean = np.asarray(normaliser.obs_rms.mean, dtype=np.float32)
    std = np.sqrt(np.asarray(normaliser.obs_rms.var, dtype=np.float32) + normaliser.epsilon)
    custom = {'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0,
              'clip_range': lambda _: 0.0}
    if env is not None:
        custom['observation_space'] = env.observation_space
        custom['action_space'] = env.action_space
    return (PPO.load(path, device='cpu', custom_objects=custom),
            mean, std, float(normaliser.clip_obs))


def fly(model_path, n_ac, rho, scenario_seed, blind, max_steps):
    """One scenario. Returns the sector polygon and, per aircraft, its straight route and track."""
    from Environment.config import CONFIG
    CONFIG['n_aircraft'] = lambda rng: n_ac
    CONFIG['rho'] = lambda rng: rho
    from Environment import AirspaceEnv
    from Environment.config import OBS_OWNSHIP_LABELS
    response_state = [OBS_OWNSHIP_LABELS.index(name)
                      for name in ('pending', 'wait_s', 'pend_hdg', 'pend_spd')]

    env = AirspaceEnv(delay_mode='none')
    policy = load_policy(model_path, env)
    observation, _ = env.reset(options={'scenario_seed': scenario_seed})
    polygon = list(env.polygon) if env.polygon is not None else []

    followed = list(env._aircraft)
    straight = {cs: (np.array(env._aircraft[cs].spawn_pos_nm, dtype=float),
                     np.array(env._aircraft[cs].no_turn_exit_nm, dtype=float))
                for cs in followed}
    tracks = {cs: [] for cs in followed}

    model, mean, std, clip = policy
    for _ in range(max_steps):
        for cs in followed:
            index = env.traf.id2idx(cs)
            if index >= 0:
                tracks[cs].append(np.asarray(env.traf.pos[index], dtype=float))
        if blind:
            observation = observation.copy()
            observation[response_state] = 0.0
        action, _ = model.predict(np.clip((observation - mean) / std, -clip, clip),
                                  deterministic=True)
        observation, _, _, truncated, _ = env.step(int(np.asarray(action).flat[0]))
        if truncated or all(env.traf.id2idx(cs) < 0 for cs in followed):
            break
    return polygon, straight, {cs: np.array(t) for cs, t in tracks.items() if len(t) > 1}


def draw(polygon, straight, tracks, out):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    if polygon:
        ring = np.array(list(polygon) + [polygon[0]], dtype=float)
        axis.plot(ring[:, 0], ring[:, 1], color='0.15', linewidth=0.9, zorder=1)
    for cs, track in tracks.items():
        spawn, exit_point = straight[cs]
        axis.plot([spawn[0], exit_point[0]], [spawn[1], exit_point[1]], color='0.55',
                  linewidth=0.5, zorder=2)
        axis.plot(track[:, 0], track[:, 1], color='#1f4ea1', linewidth=1.3, zorder=3)
        axis.plot(track[-1, 0], track[-1, 1], 'o', markersize=4, markerfacecolor='white',
                  markeredgecolor='0.25', markeredgewidth=0.8, zorder=4)
    axis.set_aspect('equal')
    axis.axis('off')
    figure.tight_layout()
    for extension in ('pdf', 'png'):
        figure.savefig(f'{out}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {out}.pdf and {out}.png with {len(tracks)} tracks')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='path to the .zip checkpoint')
    ap.add_argument('--n-ac', type=int, default=15)
    ap.add_argument('--rho', type=float, default=15000, help='one aircraft per RHO km^2')
    ap.add_argument('--seed', type=int, default=0, help='scenario seed')
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--blind', action='store_true',
                    help='hide the response state, as for a no-delay model')
    ap.add_argument('--out', default='routes')
    args = ap.parse_args()

    polygon, straight, tracks = fly(args.model, args.n_ac, 1.0 / args.rho,
                                    args.seed, args.blind, args.max_steps)
    draw(polygon, straight, tracks, args.out)


if __name__ == '__main__':
    main()
