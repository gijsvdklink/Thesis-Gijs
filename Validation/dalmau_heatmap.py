"""
Heatmap validation for v3 -- policy vs. no-policy baseline.

Sweeps a 2-D grid of operating conditions:
    rows : aircraft density rho (aircraft/km^2)
    cols : number of aircraft

For each cell N_EPISODES episodes are run and the mean LoS fraction
(steps with at least one LoS / total steps) is recorded.

Run:
    python -m Validation.dalmau_heatmap                    # default checkpoint
    python -m Validation.dalmau_heatmap --model path.zip   # custom checkpoint
    python -m Validation.dalmau_heatmap --quick            # 2 episodes per cell
"""

import sys, os, argparse, pickle, io, random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── NumPy / SB3 compatibility shims ──────────────────────────────────────────

class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

from stable_baselines3.common import save_util

def _patched_json_to_data(json_string, custom_objects=None):
    import json, base64, warnings
    data = {}
    for key, item in json.loads(json_string).items():
        if custom_objects and key in custom_objects:
            data[key] = custom_objects[key]
        elif isinstance(item, dict) and ':serialized:' in item:
            try:
                raw = base64.b64decode(item[':serialized:'].encode())
                data[key] = _NumpyShim(io.BytesIO(raw)).load()
            except Exception as e:
                warnings.warn(f'Could not deserialise {key}: {e}')
        else:
            data[key] = item
    return data

save_util.json_to_data = _patched_json_to_data

# ─────────────────────────────────────────────────────────────────────────────

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Environments.v3 import AirspaceEnv, CONFIG

# ── Default checkpoint ────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = HERE
DEFAULT_MODEL = os.path.join(
    os.path.dirname(HERE),
    'Runs_saved', 'v3',
    'dalmau_8act_obs23_seed18592_20260606_132427',
    'checkpoints', 'ckpt_4600000.zip',
)

# ── Sweep grid ────────────────────────────────────────────────────────────────

N_AIRCRAFT = [8, 10, 12, 14]
# rho in aircraft/km^2; equivalent area/ac: 1/5000=5k, 1/8000=8k, 1/10000=10k, 1/15000=15k km^2/ac
RHO_VALUES = [1/5000, 1/8000, 1/10000, 1/15000]
N_EPISODES = 4          # reduce to 2 with --quick

# ── Config helpers ────────────────────────────────────────────────────────────

def _configure_for_cell(n_ac, rho):
    CONFIG['n_aircraft'] = lambda n=n_ac: n
    CONFIG['rho']        = lambda r=rho:  r

def _restore_config_defaults():
    CONFIG['n_aircraft'] = lambda: random.randint(2, 15)
    CONFIG['rho']        = lambda: random.uniform(1/15000, 1/5000)

# ── Model loading ─────────────────────────────────────────────────────────────

def _load_vecnorm(path, venv):
    try:
        with open(path, 'rb') as fh:
            vn = _NumpyShim(fh).load()
        vn.set_venv(venv)
        vn.training    = False
        vn.norm_reward = False
        return vn
    except Exception as e:
        print(f'  Warning: vecnorm skipped ({e})')
        return venv

# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(venv, model=None):
    obs  = venv.reset()
    done = False
    los_steps, total_steps, total_reward = 0, 0, 0.0

    while not done:
        if model is not None:
            action = model.predict(obs, deterministic=True)[0]
        else:
            action = np.array([venv.action_space.sample()])

        obs, reward, done_arr, infos = venv.step(action)
        total_reward += float(reward[0])
        total_steps  += 1
        done          = bool(done_arr[0])
        if infos[0].get('los_pairs', 0) > 0:
            los_steps += 1

    return los_steps / max(total_steps, 1), total_reward / max(total_steps, 1)

# ── Grid sweep ────────────────────────────────────────────────────────────────

def sweep(mode, model_path=None, vecnorm_path=None, n_episodes=N_EPISODES):
    n_rows, n_cols = len(RHO_VALUES), len(N_AIRCRAFT)
    los_grid = np.zeros((n_rows, n_cols, n_episodes))
    rew_grid = np.zeros_like(los_grid)

    venv  = DummyVecEnv([lambda: AirspaceEnv()])
    model = None

    if mode == 'policy' and model_path:
        if vecnorm_path and os.path.exists(vecnorm_path):
            venv = _load_vecnorm(vecnorm_path, venv)
        model = PPO.load(model_path, custom_objects={
            'observation_space': venv.observation_space,
            'action_space':      venv.action_space,
        })

    for di, rho in enumerate(RHO_VALUES):
        for ai, n_ac in enumerate(N_AIRCRAFT):
            _configure_for_cell(n_ac, rho)
            for ep in range(n_episodes):
                los_grid[di, ai, ep], rew_grid[di, ai, ep] = _run_episode(venv, model)
            print(f'  [{mode:8s}] rho={rho:.2e} ac/km^2  n={n_ac}'
                  f'  LoS={100*los_grid[di,ai].mean():.1f}%'
                  f'  rew={rew_grid[di,ai].mean():.3f}', flush=True)

    venv.close()
    _restore_config_defaults()
    return los_grid, rew_grid

# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, matrix, title, vmax):
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest',
                   cmap='RdYlGn_r', vmin=0, vmax=vmax)
    for di in range(len(RHO_VALUES)):
        for ai in range(len(N_AIRCRAFT)):
            val   = matrix[di, ai]
            color = 'white' if val > 0.55 * vmax else 'black'
            ax.text(ai, di, f'{100*val:.1f}%',
                    ha='center', va='center', fontsize=9, color=color)
    ax.set_xticks(range(len(N_AIRCRAFT)))
    ax.set_xticklabels([str(n) for n in N_AIRCRAFT])
    ax.set_yticks(range(len(RHO_VALUES)))
    ax.set_yticklabels([f'{int(1/r/1000)}k km2/ac' for r in RHO_VALUES])
    ax.set_xlabel('Aircraft in sector', fontsize=10)
    ax.set_ylabel('Density rho (ac/km^2)', fontsize=10)
    ax.set_title(title, fontsize=11)
    cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Mean LoS fraction', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def plot_heatmaps(pol_los, base_los, model_label, out_path):
    pol_mean  = pol_los.mean(axis=2)
    base_mean = base_los.mean(axis=2)
    vmax      = max(pol_mean.max(), base_mean.max(), 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={'wspace': 0.40})
    _draw_heatmap(axes[0], pol_mean,  f'PPO policy  ({model_label})', vmax)
    _draw_heatmap(axes[1], base_mean, 'No-policy baseline (random)',  vmax)
    fig.suptitle(
        f'v3 — Mean LoS fraction: policy vs. baseline\n'
        f'({pol_los.shape[2]} episodes per cell)',
        fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_summary(pol_los, base_los, out_path):
    n_cols = len(N_AIRCRAFT)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5*n_cols, 4),
                              sharey=True, gridspec_kw={'wspace': 0.12})
    # x-axis: area per aircraft in thousands of km^2 (= 1/rho / 1000), for readability
    x = np.array([1/r/1000 for r in RHO_VALUES])

    for ai, (ax, n_ac) in enumerate(zip(axes, N_AIRCRAFT)):
        for matrix, color, label in [(pol_los,  'steelblue', 'PPO'),
                                      (base_los, 'tomato',    'baseline')]:
            mu  = matrix[:, ai, :].mean(axis=1)
            sig = matrix[:, ai, :].std(axis=1)
            ax.fill_between(x, mu-sig, mu+sig, alpha=0.20, color=color)
            ax.plot(x, mu, 'o-', color=color, linewidth=1.8, label=label)
        ax.set_title(f'n = {n_ac} aircraft', fontsize=9)
        ax.set_xlabel('km^2/aircraft (x10^3)', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v)}' for v in x], fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ai == 0:
            ax.set_ylabel('LoS fraction')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        if ai == n_cols - 1:
            ax.legend(fontsize=8)

    fig.suptitle(f'v3 — LoS fraction vs. density  '
                 f'(mean ± 1 std, {pol_los.shape[2]} episodes each)', fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default=DEFAULT_MODEL,
                        help='Path to PPO checkpoint .zip')
    parser.add_argument('--quick',   action='store_true',
                        help='2 episodes per cell instead of 4 (fast smoke-test)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip baseline sweep, use hold action as proxy')
    args = parser.parse_args()

    n_ep         = 2 if args.quick else N_EPISODES
    model_path   = args.model if os.path.exists(args.model) else None
    vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl') if model_path else None
    model_label  = os.path.basename(model_path or 'no_model')

    if not model_path:
        print(f'WARNING: model not found at {args.model} — running baseline only')

    print(f'=== Policy sweep  ({n_ep} eps/cell) ===')
    pol_los, pol_rew = sweep('policy', model_path, vecnorm_path, n_ep)

    if not args.no_baseline:
        print(f'\n=== Baseline sweep ({n_ep} eps/cell) ===')
        base_los, base_rew = sweep('baseline', n_episodes=n_ep)
    else:
        base_los, base_rew = pol_los.copy(), pol_rew.copy()

    tag = 'quick_' if args.quick else ''
    npz_path = os.path.join(OUT_DIR, f'dalmau_heatmap_{tag}results.npz')
    np.savez(npz_path,
             pol_los=pol_los,  pol_rew=pol_rew,
             base_los=base_los, base_rew=base_rew,
             n_aircraft=np.array(N_AIRCRAFT),
             rho_values=np.array(RHO_VALUES))
    print(f'\nResults saved to {npz_path}')

    # Summary table
    print('\n── LoS summary (mean %) ─────────────────────────────────────────')
    print(f'{"rho (ac/km2)":>14}' + ''.join(f'  n={n}(P) n={n}(B)' for n in N_AIRCRAFT))
    for di, rho in enumerate(RHO_VALUES):
        row = f'{rho:>14.2e}'
        for ai in range(len(N_AIRCRAFT)):
            row += (f'  {100*pol_los[di,ai].mean():>5.1f}%'
                    f' {100*base_los[di,ai].mean():>5.1f}%')
        print(row)

    plot_heatmaps(pol_los, base_los, model_label,
                  os.path.join(OUT_DIR, f'dalmau_heatmap_{tag}.png'))
    plot_summary(pol_los, base_los,
                 os.path.join(OUT_DIR, f'dalmau_summary_{tag}.png'))


if __name__ == '__main__':
    main()
