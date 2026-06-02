"""
Heatmap validation for v3_improved — policy vs. no-policy baseline.

Sweeps a 2-D grid of:
    rows : sector density  (km² per aircraft)
    cols : number of aircraft

and records mean LoS fraction per cell.

Run:
    python -m Validation.v3_improved_heatmap
    python -m Validation.v3_improved_heatmap --model path/to/ckpt.zip
"""

import sys, os, argparse, pickle, io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── numpy 2.x → 1.x pickle shim ──────────────────────────────────────────────
class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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

from Environments.v3_improved import AirspaceEnv, CONFIG

# ── Sweep grid ────────────────────────────────────────────────────────────────

N_AIRCRAFT     = [2, 4, 6, 8]
DENSITY_PER_AC = [5_000, 8_000, 10_000, 15_000]   # km² per aircraft
N_EPISODES     = 8

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE

# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_cell(n_ac, d_per_ac):
    area = float(n_ac * d_per_ac)
    CONFIG['area_km2']    = lambda a=area: a
    CONFIG['density_km2'] = lambda n=n_ac, a=area: a / n

def _restore():
    CONFIG['area_km2']    = lambda: float(np.random.uniform(10_000, 80_000))
    CONFIG['density_km2'] = lambda: float(np.random.uniform(5_000, 15_000))

def _load_vecnorm(path, venv):
    try:
        with open(path, 'rb') as fh:
            vn = _NumpyShim(fh).load()
        vn.set_venv(venv)
        return vn
    except Exception as e:
        print(f'  Warning: vecnorm skipped ({e})')
        return venv

def _run_episode(venv, model=None):
    obs   = venv.reset()
    done  = False
    los, steps, rew = 0, 0, 0.0
    while not done:
        if model is not None:
            action = model.predict(obs, deterministic=True)[0]
        else:
            action = np.array([venv.action_space.sample()])
        obs, r, done_arr, infos = venv.step(action)
        rew   += float(r[0])
        steps += 1
        done   = bool(done_arr[0])
        if infos[0].get('los_pairs', 0) > 0:
            los += 1
    return los / max(steps, 1), rew / max(steps, 1)

def sweep(mode, model_path=None, vecnorm_path=None):
    los_grid = np.zeros((len(DENSITY_PER_AC), len(N_AIRCRAFT), N_EPISODES))
    rew_grid = np.zeros_like(los_grid)

    venv = DummyVecEnv([lambda: AirspaceEnv()])
    model = None
    if mode == 'policy' and model_path:
        if vecnorm_path and os.path.exists(vecnorm_path):
            venv = _load_vecnorm(vecnorm_path, venv)
            venv.training    = False
            venv.norm_reward = False
        model = PPO.load(model_path, custom_objects={
            'observation_space': venv.observation_space,
            'action_space':      venv.action_space,
        })

    for di, d in enumerate(DENSITY_PER_AC):
        for ai, n in enumerate(N_AIRCRAFT):
            _set_cell(n, d)
            tag = f'[{mode}] {d//1000}k km²/ac  n={n}'
            for ep in range(N_EPISODES):
                los_grid[di, ai, ep], rew_grid[di, ai, ep] = _run_episode(venv, model)
            print(f'  {tag}  LoS={100*los_grid[di,ai].mean():.1f}%', flush=True)

    venv.close()
    _restore()
    return los_grid, rew_grid

# ── Plotting ──────────────────────────────────────────────────────────────────

def _heatmap_ax(ax, mat, title, vmax, xlabels, ylabels):
    im = ax.imshow(mat, aspect='auto', interpolation='nearest',
                   cmap='RdYlGn_r', vmin=0, vmax=vmax)
    for di in range(len(DENSITY_PER_AC)):
        for ai in range(len(N_AIRCRAFT)):
            val = mat[di, ai]
            ax.text(ai, di, f'{100*val:.1f}%',
                    ha='center', va='center', fontsize=9,
                    color='white' if val > 0.55 * vmax else 'black')
    ax.set_xticks(range(len(N_AIRCRAFT)));      ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(DENSITY_PER_AC)));  ax.set_yticklabels(ylabels)
    ax.set_xlabel('Aircraft in sector', fontsize=10)
    ax.set_ylabel('Sector density (km² per aircraft)', fontsize=10)
    ax.set_title(title, fontsize=11)
    cbar = plt.gcf().colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Mean LoS fraction', fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    return im


def plot_heatmaps(pol_los, base_los, label, out_path):
    pol_mean  = pol_los.mean(axis=2)
    base_mean = base_los.mean(axis=2)
    vmax      = max(pol_mean.max(), base_mean.max(), 0.01)

    xlabels = [str(n) for n in N_AIRCRAFT]
    ylabels = [f'{d//1000}k' for d in DENSITY_PER_AC]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={'wspace': 0.40})
    _heatmap_ax(axes[0], pol_mean,  f'PPO policy  ({label})', vmax, xlabels, ylabels)
    _heatmap_ax(axes[1], base_mean, 'No-policy baseline (random)', vmax, xlabels, ylabels)

    fig.suptitle(
        f'Mean LoS fraction — policy vs. baseline\n'
        f'({N_EPISODES} episodes per cell · '
        f'{len(DENSITY_PER_AC)} density levels × {len(N_AIRCRAFT)} aircraft counts)',
        fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


def plot_summary(pol_los, base_los, out_path):
    n_cols = len(N_AIRCRAFT)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4),
                              sharey=True, gridspec_kw={'wspace': 0.12})
    x = np.array(DENSITY_PER_AC) / 1_000

    for ai, (ax, n) in enumerate(zip(axes, N_AIRCRAFT)):
        for mat, color, lbl in [(pol_los,  'steelblue', 'PPO'),
                                 (base_los, 'tomato',    'baseline')]:
            vals = mat[:, ai, :]
            mu   = vals.mean(axis=1)
            sig  = vals.std(axis=1)
            ax.fill_between(x, mu - sig, mu + sig, alpha=0.20, color=color)
            ax.plot(x, mu, 'o-', color=color, linewidth=1.8, label=lbl)
        ax.set_title(f'n = {n} aircraft', fontsize=9)
        ax.set_xlabel('km²/aircraft (×10³)', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v)}' for v in x], fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ai == 0:
            ax.set_ylabel('LoS fraction')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        if ai == n_cols - 1:
            ax.legend(fontsize=8)

    fig.suptitle(f'LoS fraction vs. density — mean ± 1 std  ({N_EPISODES} episodes each)',
                 fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        help='Path to PPO checkpoint .zip  (omit for baseline-only)')
    args = parser.parse_args()

    model_path   = args.model
    vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl') if model_path else None
    label        = os.path.basename(model_path or 'no_model')

    print('=== Baseline sweep ===')
    base_los, base_rew = sweep('baseline')

    if model_path:
        print('\n=== Policy sweep ===')
        pol_los, pol_rew = sweep('policy', model_path, vecnorm_path)
    else:
        print('\nNo model supplied — showing baseline only (policy = random)')
        pol_los, pol_rew = base_los.copy(), base_rew.copy()

    np.savez(os.path.join(OUT_DIR, 'v3_improved_heatmap_results.npz'),
             pol_los=pol_los, pol_rew=pol_rew,
             base_los=base_los, base_rew=base_rew,
             n_aircraft=np.array(N_AIRCRAFT),
             density_per_ac=np.array(DENSITY_PER_AC))

    print('\n── LoS summary (mean %) ──────────────────────────────────────────')
    header = f'{"density":>10}' + ''.join(f'  n={n}(P)  n={n}(B)' for n in N_AIRCRAFT)
    print(header)
    for di, d in enumerate(DENSITY_PER_AC):
        row = f'{d:>10,}'
        for ai in range(len(N_AIRCRAFT)):
            row += f'  {100*pol_los[di,ai].mean():>6.1f}%  {100*base_los[di,ai].mean():>6.1f}%'
        print(row)

    plot_heatmaps(pol_los, base_los, label,
                  os.path.join(OUT_DIR, 'v3_improved_heatmap.png'))
    plot_summary(pol_los, base_los,
                 os.path.join(OUT_DIR, 'v3_improved_summary.png'))


if __name__ == '__main__':
    main()
