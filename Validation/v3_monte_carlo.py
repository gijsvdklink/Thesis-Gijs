"""
Monte Carlo validation of the ramp180 policy (v3_simple_env).

2-D sweep:  n_aircraft  ×  sector density (km² per aircraft)
These two axes are independent: 4 aircraft in a tight 5 000 km²/ac sector
is harder than 4 aircraft in a wide 20 000 km²/ac sector.

Policy evaluated against a random-action baseline.

Per episode the script records:
  los_fraction : fraction of steps where ≥1 LoS pair was active
  mean_reward  : total reward / episode length

Outputs saved to Validation/:
  v3_los_heatmap.png   — 2-D mean LoS fraction grid, policy vs baseline
  v3_los_summary.png   — mean ± std LoS fraction vs density, per aircraft count
  v3_results.npz       — raw arrays for further analysis
"""

import sys, os, pickle, io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── numpy 2.x → 1.x pickle shim ──────────────────────────────────────────────
class Numpy2ShimUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module == 'numpy.random._pcg64' and name == 'PCG64':
            module = 'numpy.random'
        return super().find_class(module, name)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common import save_util

# Patch SB3's JSON deserialiser to route pickle through the numpy shim
def _patched_json_to_data(json_string, custom_objects=None):
    import json, base64, warnings
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects must be a dict or None")
    data = {}
    for key, item in json.loads(json_string).items():
        if custom_objects and key in custom_objects:
            data[key] = custom_objects[key]
        elif isinstance(item, dict) and ':serialized:' in item:
            try:
                raw = base64.b64decode(item[':serialized:'].encode())
                data[key] = Numpy2ShimUnpickler(io.BytesIO(raw)).load()
            except Exception as e:
                warnings.warn(f"Could not deserialise {key}: {e}")
        else:
            data[key] = item
    return data

save_util.json_to_data = _patched_json_to_data

from Environments.v3_simple_env import AirspaceEnv, CONFIG

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, '..'))
MODEL_DIR    = os.path.join(_ROOT, 'Runs_saved', 'v3',
                            'ramp180_obs21_sticky3_seed82563_20260601_123525')
MODEL_PATH   = os.path.join(MODEL_DIR, 'final_model.zip')
VECNORM_PATH = os.path.join(MODEL_DIR, 'final_vecnorm.pkl')

# ── Sweep grid ────────────────────────────────────────────────────────────────

N_AIRCRAFT     = [4, 8, 12]
DENSITY_PER_AC = [5_000, 10_000, 20_000]
N_EPISODES     = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_cell(n_ac, d_per_ac):
    area = float(n_ac * d_per_ac)
    CONFIG['area_km2']    = lambda a=area: a
    CONFIG['density_km2'] = lambda n=n_ac, a=area: a / n

def _restore():
    CONFIG['area_km2']    = lambda: 80_000.0
    CONFIG['density_km2'] = lambda: __import__('random').uniform(6_000.0, 10_000.0)

def _load_vecnorm(path, venv):
    try:
        with open(path, 'rb') as fh:
            vn = Numpy2ShimUnpickler(fh).load()
        vn.set_venv(venv)
        return vn
    except ValueError as e:
        if 'BitGenerator' in str(e):
            print(f"Warning: vecnorm skipped due to numpy mismatch: {e}")
            print("Proceeding without vecnorm (may affect results)")
            return venv
        raise

def _run_episode(venv, model=None):
    obs  = venv.reset()
    done = False
    ep_los, ep_len, ep_rew = 0, 0, 0.0
    while not done:
        action = model.predict(obs, deterministic=True)[0] if model is not None \
                 else np.array([venv.action_space.sample()])
        obs, rew, done_arr, infos = venv.step(action)
        ep_rew += float(rew[0])
        ep_len += 1
        done    = bool(done_arr[0])
        if infos[0].get('los_pairs', 0) > 0:
            ep_los += 1
    return ep_los / max(ep_len, 1), ep_rew / max(ep_len, 1)

# ── Grid sweep ────────────────────────────────────────────────────────────────

def sweep(mode):
    """
    Returns los_grid, rew_grid  shape (n_density_levels, n_aircraft_levels, n_episodes)
    mode: 'policy' | 'baseline'
    """
    los_grid = np.zeros((len(DENSITY_PER_AC), len(N_AIRCRAFT), N_EPISODES))
    rew_grid = np.zeros_like(los_grid)

    # Build env and model once — avoids reloading navdata and weights per cell
    venv = DummyVecEnv([lambda: AirspaceEnv()])
    if mode == 'policy':
        venv = _load_vecnorm(VECNORM_PATH, venv)
        venv.training    = False
        venv.norm_reward = False
        model = PPO.load(MODEL_PATH, custom_objects={
            'observation_space': venv.observation_space,
            'action_space':      venv.action_space,
        })
    else:
        model = None

    for di, d in enumerate(DENSITY_PER_AC):
        for ai, n in enumerate(N_AIRCRAFT):
            _set_cell(n, d)
            print(f'  [{mode}] d={d//1000}k km²/ac  n={n}', flush=True)
            for ep in range(N_EPISODES):
                los_grid[di, ai, ep], rew_grid[di, ai, ep] = _run_episode(venv, model)

    venv.close()
    _restore()
    return los_grid, rew_grid

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(pol_los, base_los, out_path):
    pol_mean  = pol_los.mean(axis=2)
    base_mean = base_los.mean(axis=2)
    vmax      = max(pol_mean.max(), base_mean.max(), 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'wspace': 0.40})
    ylabels   = [f'{d//1000}k' for d in DENSITY_PER_AC]
    xlabels   = [str(n) for n in N_AIRCRAFT]

    for ax, mat, title in zip(axes,
                               [pol_mean, base_mean],
                               ['PPO policy (ramp180)', 'No-policy baseline (random)']):
        im = ax.imshow(mat, aspect='auto', interpolation='nearest',
                       cmap='RdYlGn_r', vmin=0, vmax=vmax)
        for di in range(len(DENSITY_PER_AC)):
            for ai in range(len(N_AIRCRAFT)):
                val = mat[di, ai]
                ax.text(ai, di, f'{100*val:.1f}%',
                        ha='center', va='center', fontsize=8,
                        color='white' if val > 0.5 * vmax else 'black')
        ax.set_xticks(range(len(N_AIRCRAFT)));      ax.set_xticklabels(xlabels)
        ax.set_yticks(range(len(DENSITY_PER_AC)));  ax.set_yticklabels(ylabels)
        ax.set_xlabel('Aircraft in sector')
        ax.set_ylabel('Sector density (km² per aircraft)')
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Mean LoS fraction')
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.suptitle('Level of Separation (LoS) fraction — Monte Carlo sweep\n'
                 f'({N_EPISODES} episodes per cell  ·  '
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
        for mat, color, label in [(pol_los, 'steelblue', 'PPO'),
                                   (base_los, 'tomato', 'baseline')]:
            vals = mat[:, ai, :]
            mu   = vals.mean(axis=1)
            sig  = vals.std(axis=1)
            ax.fill_between(x, mu - sig, mu + sig, alpha=0.20, color=color)
            ax.plot(x, mu, 'o-', color=color, linewidth=1.8, label=label)
        ax.set_title(f'n = {n} aircraft', fontsize=9)
        ax.set_xlabel('km²/aircraft (×10³)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v)}' for v in x], fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ai == 0:
            ax.set_ylabel('LoS fraction')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        if ai == n_cols - 1:
            ax.legend(fontsize=8)

    fig.suptitle('LoS fraction vs. density — mean ± 1 std  '
                 f'({N_EPISODES} Monte Carlo episodes each)', fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=== Policy sweep ===')
    pol_los, pol_rew = sweep('policy')

    print('\n=== Baseline sweep ===')
    base_los, base_rew = sweep('baseline')

    np.savez(os.path.join(_HERE, 'v3_results.npz'),
             pol_los=pol_los, pol_rew=pol_rew,
             base_los=base_los, base_rew=base_rew,
             n_aircraft=np.array(N_AIRCRAFT),
             density_per_ac=np.array(DENSITY_PER_AC))
    print('Saved v3_results.npz')

    print('\n── Summary (mean LoS fraction %) ─────────────────────────────')
    header = f'{"density":>10}' + ''.join(f'  n={n:>2}(P)  n={n:>2}(B)' for n in N_AIRCRAFT)
    print(header)
    for di, d in enumerate(DENSITY_PER_AC):
        row = f'{d:>10,}'
        for ai in range(len(N_AIRCRAFT)):
            row += f'  {100*pol_los[di,ai].mean():>7.1f}%  {100*base_los[di,ai].mean():>7.1f}%'
        print(row)

    plot_heatmap(pol_los, base_los, os.path.join(_HERE, 'v3_los_heatmap.png'))
    plot_summary(pol_los, base_los, os.path.join(_HERE, 'v3_los_summary.png'))


if __name__ == '__main__':
    main()
