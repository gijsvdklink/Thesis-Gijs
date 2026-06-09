"""
policy_analysing.py — Monte Carlo policy evaluation for v3_dalmau.

Exhaustively samples every (n_ac, density_bin) combination with exactly
RUNS_PER_CELL deterministic episodes per cell.  Results are saved to JSON
and visualised as clean heatmaps (colour only, clear colorbar legend).

Total episodes = n_ac_values × density_bins × runs_per_cell
               = 14 × 5 × 3 = 210  (defaults)

Usage:
    python policy_analysing.py
    python policy_analysing.py --model path/to/ckpt.zip
    python policy_analysing.py --runs 5        # episodes per cell (default 3)
    python policy_analysing.py --no-policy     # hold baseline
    python policy_analysing.py --seed 42
"""

import argparse
import io
import json
import math
import os
import pickle
import random
import sys
from datetime import datetime

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── NumPy compatibility shim ──────────────────────────────────────────────────

class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

from stable_baselines3.common import save_util

def _patched_json_to_data(json_string, custom_objects=None):
    import base64, warnings
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
from Environments.v3_dalmau import AirspaceEnv, CONFIG

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = os.path.join(
    HERE, 'Runs_saved', 'v3_dalmau',
    'dalmau_8act_obs23_seed18592_20260606_132427',
    'checkpoints', 'ckpt_700000.zip',
)

# ── Grid definition ───────────────────────────────────────────────────────────

N_AC_VALS      = list(range(2, 16))          # 2–15 inclusive
N_DENSITY_BINS = 5
RUNS_PER_CELL  = 3
D_MIN, D_MAX   = 5_000.0, 15_000.0

D_EDGES   = np.linspace(D_MIN, D_MAX, N_DENSITY_BINS + 1)
D_CENTRES = (D_EDGES[:-1] + D_EDGES[1:]) / 2

ACTION_LABELS = ['-60°', '-45°', '-30°', 'hold', '+30°', '+45°', '+60°', 'direct']

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, model, vecnorm, n_ac, d_lo, d_hi):
    """Run one episode with fixed n_ac and density sampled from [d_lo, d_hi]."""
    CONFIG['n_aircraft']  = lambda: n_ac
    CONFIG['density_km2'] = lambda: random.uniform(d_lo, d_hi)

    obs, _ = env.reset()

    area_nm2 = env._polygon_shape.area
    area_km2 = area_nm2 * (1.852 ** 2)
    density  = area_km2 / n_ac

    minx, miny, maxx, maxy = env._polygon_shape.bounds
    diam_nm = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

    cum_reward = 0.0
    los_steps  = 0
    conf_steps = 0
    steps      = 0
    actions    = []

    while True:
        if model is not None:
            norm_obs  = vecnorm.normalize_obs(obs[None])[0] if vecnorm else obs
            action, _ = model.predict(norm_obs, deterministic=True)
        else:
            action = 3  # hold baseline

        obs, reward, terminated, truncated, info = env.step(int(action))
        cum_reward += float(reward)
        steps      += 1
        actions.append(int(action))

        if info.get('los_pairs', 0) > 0:
            los_steps += 1
        n_conf = int((env._urgency_matrix > 0).sum()) // 2 if env._urgency_matrix.size > 0 else 0
        if n_conf > 0:
            conf_steps += 1

        if terminated or truncated:
            break

    act_dist = np.bincount(actions, minlength=8) / max(steps, 1)

    return {
        'n_ac':        n_ac,
        'density':     density,
        'area_km2':    area_km2,
        'diam_nm':     diam_nm,
        'steps':       steps,
        'cum_reward':  cum_reward,
        'mean_reward': cum_reward / max(steps, 1),
        'los_steps':   los_steps,
        'los_rate':    los_steps  / max(steps, 1),
        'conf_rate':   conf_steps / max(steps, 1),
        'safe_rate':   1.0 - los_steps / max(steps, 1),
        'turn_frac':   float(act_dist[[0,1,2,4,5,6]].sum()),
        'hold_frac':   float(act_dist[3]),
        'direct_frac': float(act_dist[7]),
        'act_dist':    act_dist.tolist(),
    }

# ── Plotting ──────────────────────────────────────────────────────────────────

def _heatmap(ax, grid, title, cmap, vmin=None, vmax=None, cbar_label=''):
    """Draw a clean heatmap — colour only, no in-cell text."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('#dddddd')

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(
        masked.T,           # rows = density bins (y), cols = n_ac (x)
        origin='lower',
        aspect='auto',
        cmap=cmap_obj,
        vmin=vmin, vmax=vmax,
        extent=[N_AC_VALS[0] - 0.5, N_AC_VALS[-1] + 0.5, D_MIN, D_MAX],
        interpolation='nearest',
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label, fontsize=8)

    ax.set_xlabel('Aircraft count', fontsize=9)
    ax.set_ylabel('Density (km²/ac)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(N_AC_VALS)
    ax.set_xticklabels(N_AC_VALS, fontsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))

    # Draw cell borders for readability
    ax.set_xticks([x - 0.5 for x in N_AC_VALS] + [N_AC_VALS[-1] + 0.5], minor=True)
    ax.set_yticks(D_EDGES, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.4)
    ax.tick_params(which='minor', length=0)

    return im


def make_figures(grid_mean, grid_std, stem, mode, runs_per_cell):
    n_ep = len(N_AC_VALS) * N_DENSITY_BINS * runs_per_cell
    title_suffix = f'(n={n_ep}, {runs_per_cell} runs/cell, {mode})'

    # ── Figure 1: safety & efficiency ─────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig1.suptitle(f'v3\_dalmau Monte Carlo — safety & efficiency  {title_suffix}',
                  fontsize=12, y=1.01)

    specs = [
        ('los_rate',    'LoS rate',              'RdYlGn_r', 0.0, 0.5,  'fraction of steps'),
        ('safe_rate',   'LoS-free fraction',     'RdYlGn',   0.5, 1.0,  'fraction of steps'),
        ('conf_rate',   'Conflict rate',         'RdYlGn_r', 0.0, 1.0,  'fraction of steps'),
        ('mean_reward', 'Mean reward per step',  'viridis',  None, None, 'reward'),
        ('turn_frac',   'Turn fraction',         'Blues',    0.0, 1.0,  'fraction of actions'),
        ('direct_frac', 'Fly-direct fraction',   'Purples',  0.0, 0.6,  'fraction of actions'),
    ]
    for ax, (key, title, cmap, vmin, vmax, cbar_label) in zip(axes.flat, specs):
        _heatmap(ax, grid_mean[key], title, cmap, vmin, vmax, cbar_label)

    fig1.tight_layout()
    out1 = os.path.join(HERE, f'v3_{stem}_analysis_heatmaps.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  Saved → {out1}')

    # ── Figure 2: per-action fractions ────────────────────────────────────────
    fig2, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig2.suptitle(f'v3\_dalmau Monte Carlo — per-action fractions  {title_suffix}',
                  fontsize=12, y=1.01)
    cmaps = ['RdPu','RdPu','RdPu','Greens','Oranges','Oranges','Oranges','Blues']
    for ai, (ax, label, cmap) in enumerate(zip(axes.flat, ACTION_LABELS, cmaps)):
        key = f'act_{ai}'
        _heatmap(ax, grid_mean[key], f'Action {ai}  {label}', cmap,
                 vmin=0.0, vmax=0.5, cbar_label='fraction of actions')

    fig2.tight_layout()
    out2 = os.path.join(HERE, f'v3_{stem}_analysis_actions.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved → {out2}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default=DEFAULT_MODEL)
    parser.add_argument('--no-policy', action='store_true')
    parser.add_argument('--runs',      type=int, default=RUNS_PER_CELL,
                        help='Episodes per (n_ac, density_bin) cell (default 3)')
    parser.add_argument('--seed',      type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    use_policy = (not args.no_policy) and os.path.exists(args.model)
    model, vecnorm = None, None

    if use_policy:
        print(f'Loading model   : {args.model}')
        model = PPO.load(args.model)
        vecnorm_path = args.model.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            print(f'Loading vecnorm : {vecnorm_path}')
            with open(vecnorm_path, 'rb') as fh:
                vecnorm = _NumpyShim(fh).load()
            vecnorm.set_venv(DummyVecEnv([AirspaceEnv]))
            vecnorm.training    = False
            vecnorm.norm_reward = False
    else:
        print('No policy — running hold baseline')

    stem = (os.path.splitext(os.path.basename(args.model))[0]
            if use_policy else 'baseline_hold')
    mode = (f'policy: {os.path.basename(args.model)}'
            if use_policy else 'hold baseline')

    total = len(N_AC_VALS) * N_DENSITY_BINS * args.runs

    print(f'\n{"═"*65}')
    print(f'  v3_dalmau — Monte Carlo Policy Evaluation')
    print(f'{"═"*65}')
    print(f'  Model        : {os.path.basename(args.model) if use_policy else "none"}')
    print(f'  Grid         : {len(N_AC_VALS)} n_ac × {N_DENSITY_BINS} density bins × {args.runs} runs = {total} episodes')
    print(f'  n_ac range   : {N_AC_VALS[0]}–{N_AC_VALS[-1]}')
    print(f'  density bins : {" | ".join(f"{d/1e3:.0f}k–{D_EDGES[i+1]/1e3:.0f}k" for i,d in enumerate(D_EDGES[:-1]))}')
    print(f'  Seed         : {args.seed}')
    print(f'{"─"*65}')
    print(f'  {"#":>4}  {"n_ac":>4}  {"d_bin":>5}  {"density":>8}  {"steps":>5}  '
          f'{"los":>6}  {"rew/step":>9}  │  {"avg los":>7}  {"avg rew":>9}')
    print(f'  {"─"*4}  {"─"*4}  {"─"*5}  {"─"*8}  {"─"*5}  '
          f'{"─"*6}  {"─"*9}  │  {"─"*7}  {"─"*9}')

    env      = AirspaceEnv()
    episodes = []
    ep_num   = 0

    for n_ac in N_AC_VALS:
        for di, (d_lo, d_hi) in enumerate(zip(D_EDGES[:-1], D_EDGES[1:])):
            for run in range(args.runs):
                ep_num += 1
                r = run_episode(env, model, vecnorm, n_ac, float(d_lo), float(d_hi))
                r['density_bin']    = di
                r['density_centre'] = float(D_CENTRES[di])
                r['run']            = run
                episodes.append(r)

                los_so_far = np.mean([e['los_rate']    for e in episodes])
                rew_so_far = np.mean([e['mean_reward'] for e in episodes])
                print(f'  {ep_num:4d}  {n_ac:4d}  {di:5d}  {r["density"]/1e3:7.1f}k  '
                      f'{r["steps"]:5d}  {r["los_rate"]:6.3f}  {r["mean_reward"]:+9.3f}'
                      f'  │  {los_so_far:7.3f}  {rew_so_far:+9.3f}',
                      flush=True)

    # ── Aggregate into grids ──────────────────────────────────────────────────

    METRICS = ['los_rate', 'safe_rate', 'conf_rate', 'mean_reward',
               'turn_frac', 'hold_frac', 'direct_frac']
    for ai in range(8):
        METRICS.append(f'act_{ai}')

    grid_mean = {m: np.full((len(N_AC_VALS), N_DENSITY_BINS), np.nan) for m in METRICS}
    grid_std  = {m: np.full((len(N_AC_VALS), N_DENSITY_BINS), np.nan) for m in METRICS}

    for ni, n_ac in enumerate(N_AC_VALS):
        for di in range(N_DENSITY_BINS):
            cell = [e for e in episodes if e['n_ac'] == n_ac and e['density_bin'] == di]
            if not cell:
                continue
            for m in METRICS:
                if m.startswith('act_'):
                    ai = int(m.split('_')[1])
                    vals = [e['act_dist'][ai] for e in cell]
                else:
                    vals = [e[m] for e in cell]
                grid_mean[m][ni, di] = np.mean(vals)
                grid_std[m][ni, di]  = np.std(vals)

    # ── Save JSON ─────────────────────────────────────────────────────────────

    json_out = os.path.join(HERE, f'v3_{stem}_mc_results.json')
    payload = {
        'metadata': {
            'model':          args.model if use_policy else None,
            'mode':           mode,
            'timestamp':      datetime.now().isoformat(),
            'seed':           args.seed,
            'runs_per_cell':  args.runs,
            'n_ac_vals':      N_AC_VALS,
            'density_edges':  D_EDGES.tolist(),
            'density_centres': D_CENTRES.tolist(),
            'n_density_bins': N_DENSITY_BINS,
            'total_episodes': total,
            'config': {k: v for k, v in CONFIG.items()
                       if not callable(v)},
        },
        'episodes': [
            {k: (v if not isinstance(v, np.ndarray) else v.tolist())
             for k, v in e.items()}
            for e in episodes
        ],
        'aggregated': {
            m: {
                'mean': grid_mean[m].tolist(),
                'std':  grid_std[m].tolist(),
            }
            for m in METRICS
        },
    }
    with open(json_out, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'\n  JSON saved → {json_out}')

    # ── Figures ───────────────────────────────────────────────────────────────

    print('  Generating figures …')
    make_figures(grid_mean, grid_std, stem, mode, args.runs)

    # ── Summary ───────────────────────────────────────────────────────────────

    los_all = np.array([e['los_rate']    for e in episodes])
    rew_all = np.array([e['mean_reward'] for e in episodes])
    act_all = np.array([e['act_dist']    for e in episodes])

    print(f'\n{"─"*55}')
    print(f'Summary  ({total} episodes, {mode})')
    print(f'{"─"*55}')
    print(f'  LoS rate          {los_all.mean():.4f} ± {los_all.std():.4f}'
          f'  [{los_all.min():.4f} – {los_all.max():.4f}]')
    print(f'  Mean reward/step  {rew_all.mean():+.4f} ± {rew_all.std():.4f}')
    print(f'  LoS-free episodes {(los_all == 0).sum()}/{total}'
          f'  ({100*(los_all==0).mean():.1f}%)')
    print(f'\n  Action breakdown:')
    for label, frac in zip(ACTION_LABELS, act_all.mean(axis=0)):
        print(f'    {label:>7s}  {frac:.3f}  {"█" * int(frac * 40)}')
    print(f'\n  LoS rate by n_ac:')
    for ni, n in enumerate(N_AC_VALS):
        row = grid_mean['los_rate'][ni]
        print(f'    n={n:2d}  avg_los={np.nanmean(row):.4f}  '
              f'by density: {" ".join(f"{v:.3f}" for v in row)}')


if __name__ == '__main__':
    main()
