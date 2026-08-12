"""
Monte-Carlo performance evaluation for a trained v4 policy.

Systematically sweeps a full-factorial grid of operating conditions:
    * number of aircraft   (--n_ac)
    * aircraft density rho  (--density, ac/km^2)

ROUND-ROBIN ordering: one PASS runs every (n_ac x density) cell exactly once
(sweeping aircraft counts low->high, densities low->high). The whole grid is then
repeated --episodes times (e.g. 30 passes). So after pass 1 there is already one
episode in every cell and coverage fills in evenly -- ideal for live monitoring.
The design stays balanced (every cell ends with the same episode count) and seeds
are deterministic and reproducible.

Per-episode metrics are written as ONE ROW PER EPISODE to a CSV (flushed after
every episode, so the file updates live -- tail it or open it in pandas while the
sweep runs). Measured outcomes per episode:
    los_steps        steps with >= 1 loss of separation        ("amount of losses")
    los_pair_steps   sum of LoS pairs over the episode          (LoS intensity)
    actions_nonhold  instructions issued (everything but HOLD)  ("actions taken")
    arrivals         aircraft that exited on-target             ("safe returns")
plus exits, arrival_rate, per-action counts, reward and episode length.

NOTE: density and n_aircraft are the independent variables we vary; LoS / actions /
arrivals are emergent OUTCOMES we measure, not inputs. Sampling the input grid
uniformly is what guarantees the outcome space (including high- and low-LoS regimes)
is well covered.

Run (once you have a checkpoint + its *_vecnorm.pkl next to it):
    python -m Validation.mc_evaluate --model path/to/best_model.zip --episodes 30
    python -m Validation.mc_evaluate --model best_model.zip --baseline   # + HOLD baseline
    python -m Validation.mc_evaluate --quick                             # 2 eps/cell smoke test
"""

import sys, os, argparse, pickle, io, csv, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- NumPy / SB3 compatibility shims (load checkpoints across numpy versions) ----

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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Environments.v4 import AirspaceEnv, CONFIG

ACTION_LABELS = ['L60', 'L45', 'L30', 'HOLD', 'R30', 'R45', 'R60', 'DIR', 'SPDup', 'SPDdn']
HOLD_IDX = 3

CSV_FIELDS = (
    ['policy', 'pass', 'n_ac_target', 'rho', 'area_km2', 'n_ac_realized', 'seed',
     'steps', 'los_steps', 'los_fraction', 'los_pair_steps',
     'actions_total', 'actions_nonhold']
    + [f'act_{lbl}' for lbl in ACTION_LABELS]
    + ['exits', 'arrivals', 'arrival_rate', 'reward_total', 'reward_mean']
)


def fmt_dur(seconds):
    """Human-readable duration: 45s / 12m03s / 1h05m."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}h{m:02d}m'
    if m:
        return f'{m}m{s:02d}s'
    return f'{s}s'

# -- Grid defaults -------------------------------------------------------------
# Both axes are stepped EVENLY and systematically:
#   aircraft : 3, 6, 9, ..., 30                 (step 3 -> 10 levels)
#   density  : 10 evenly spaced rho levels over [1/25000, 1/10000] ac/km^2
# => 10 x 10 = 100 cells, each run for --episodes episodes (balanced design).
DEFAULT_N_AC    = list(range(3, 31, 3))
DEFAULT_DENSITY = list(np.linspace(1 / 25000, 1 / 10000, 10))
DEFAULT_EPISODES = 30


def parse_float_list(text, fallback):
    """Parse a comma list of floats, allowing 'a/b' fraction syntax (e.g. 1/10000)."""
    if not text:
        return fallback
    out = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '/' in tok:
            num, den = tok.split('/')
            out.append(float(num) / float(den))
        else:
            out.append(float(tok))
    return out


def configure_cell(n_ac, rho):
    # The samplers take the env's scenario generator; these two ignore it and pin the cell.
    CONFIG['n_aircraft'] = lambda rng, n=n_ac: n
    CONFIG['rho']        = lambda rng, r=rho:  r


def load_vecnorm(path, venv):
    """Wrap venv in the saved VecNormalize (frozen stats, raw reward) for faithful obs."""
    with open(path, 'rb') as fh:
        vn = _NumpyShim(fh).load()
    vn.set_venv(venv)
    vn.training    = False
    vn.norm_reward = False
    return vn


def run_episode(venv, model, seed):
    """One episode; returns a metrics dict. model=None -> HOLD-only baseline."""
    venv.seed(seed)
    obs = venv.reset()
    try:
        realized = int(venv.get_attr('n_aircraft')[0])
    except Exception:
        realized = None
    los_steps = los_pair_steps = steps = 0
    last_info = {}
    while True:
        if model is None:
            action = np.array([HOLD_IDX])
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, _, done_arr, infos = venv.step(action)
        info = infos[0]
        steps += 1
        pairs = info.get('los_pairs', 0)
        if pairs > 0:
            los_steps += 1
            los_pair_steps += pairs
        if bool(done_arr[0]):
            last_info = info        # truncation step carries the episode summary
            break

    dist = last_info.get('action_distribution', [0] * len(ACTION_LABELS))
    total_actions = int(sum(dist))
    nonhold = total_actions - int(dist[HOLD_IDX]) if len(dist) > HOLD_IDX else total_actions
    return {
        'steps':           steps,
        'los_steps':       los_steps,
        'los_fraction':    los_steps / max(steps, 1),
        'los_pair_steps':  los_pair_steps,
        'actions_total':   total_actions,
        'actions_nonhold': nonhold,
        'act_counts':      [int(c) for c in dist],
        'exits':           int(last_info.get('ep_exits', 0)),
        'arrivals':        int(last_info.get('ep_arrivals', 0)),
        'arrival_rate':    float(last_info.get('ep_arrival_rate', 0.0)),
        'reward_total':    float(last_info.get('ep_reward_total', 0.0)),
        'reward_mean':     float(last_info.get('mean_episode_reward', 0.0)),
        'n_ac_realized':   realized,
    }


def make_venv(model_path, vecnorm_path):
    venv = DummyVecEnv([lambda: AirspaceEnv()])
    model = None
    if model_path:
        if vecnorm_path and os.path.exists(vecnorm_path):
            venv = load_vecnorm(vecnorm_path, venv)
            print(f'loaded VecNormalize obs stats from {vecnorm_path}', flush=True)
        else:
            print(f'WARNING: no vecnorm at {vecnorm_path} -- feeding RAW obs '
                  '(only correct for a norm_obs=False checkpoint)', flush=True)
        model = PPO.load(model_path, device='cpu', custom_objects={
            'observation_space': venv.observation_space,
            'action_space':      venv.action_space,
            'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0,
            'clip_range': lambda _: 0.0,
        })
    return venv, model


def build_tasks(policies, n_ac_grid, density_grid, passes, base_seed):
    """Round-robin, interleaved task list: pass -> cell -> policy. For each scenario
    (pass, cell) every policy is run back-to-back on the SAME seed -- so a trained
    episode and the untrained HOLD episode see identical traffic (paired comparison).
    Pass-major ordering still fills grid coverage evenly across passes."""
    cells = [(ai, n_ac, di, rho)
             for ai, n_ac in enumerate(n_ac_grid)
             for di, rho in enumerate(density_grid)]
    n_cells = len(cells)
    tasks = []
    for rep in range(passes):
        for ci, (ai, n_ac, di, rho) in enumerate(cells):
            seed = base_seed + rep * n_cells + (ai * len(density_grid) + di)
            for policy_name, use_model in policies:
                tasks.append((policy_name, use_model, rep + 1, n_ac, rho, seed, ci + 1, n_cells))
    return tasks


def make_row(task, m):
    policy_name, _use_model, pass_no, n_ac, rho, seed, _ci, _n_cells = task
    return {
        'policy': policy_name, 'pass': pass_no, 'n_ac_target': n_ac, 'rho': rho,
        'area_km2': round(n_ac / rho, 1),
        'n_ac_realized': m.get('n_ac_realized') if m.get('n_ac_realized') is not None else n_ac,
        'seed': seed,
        'steps': m['steps'], 'los_steps': m['los_steps'],
        'los_fraction': round(m['los_fraction'], 5),
        'los_pair_steps': m['los_pair_steps'],
        'actions_total': m['actions_total'], 'actions_nonhold': m['actions_nonhold'],
        **{f'act_{lbl}': c for lbl, c in zip(ACTION_LABELS, m['act_counts'])},
        'exits': m['exits'], 'arrivals': m['arrivals'],
        'arrival_rate': round(m['arrival_rate'], 4),
        'reward_total': round(m['reward_total'], 2),
        'reward_mean': round(m['reward_mean'], 4),
    }


# -- Worker process: one persistent env + model, reused across tasks -----------
_WORKER = {}

def _init_worker(model_path, vecnorm_path):
    venv, model = make_venv(model_path, vecnorm_path)
    _WORKER['venv'], _WORKER['model'] = venv, model

def _run_task(task):
    _policy, use_model, _pass, n_ac, rho, seed, _ci, _nc = task
    configure_cell(n_ac, rho)
    m = run_episode(_WORKER['venv'], _WORKER['model'] if use_model else None, seed)
    return task, m


def run_tasks(writers, tasks, workers, model_path, vecnorm_path):
    """Execute all tasks (serial if workers<=1, else a process pool). Each row is
    routed to its policy's own writer (separate CSV per policy), written and flushed
    live from this single consumer process so every file stays valid."""
    total, done, t0 = len(tasks), 0, time.time()

    def handle(task, m):
        nonlocal done
        policy_name = task[0]
        writer, fh = writers[policy_name]
        writer.writerow(make_row(task, m))
        fh.flush()                      # live update
        done += 1
        elapsed = time.time() - t0
        rate    = done / max(elapsed, 1e-9)
        eta     = (total - done) / max(rate, 1e-9)
        policy_name, _um, pass_no, n_ac, rho, _seed, ci, n_cells = task
        print(f'  [{policy_name:6s}] pass {pass_no:>2} cell {ci:>3}/{n_cells}  '
              f'n={n_ac:>2} rho={rho:.2e}  '
              f'los={m["los_steps"]:>4} acts={m["actions_nonhold"]:>4} arr={m["arrival_rate"]:.2f}  '
              f'| {done:>5}/{total} ({100*done/total:4.1f}%)  '
              f'elapsed {fmt_dur(elapsed)} ETA {fmt_dur(eta)}', flush=True)

    if workers <= 1:
        venv, model = make_venv(model_path, vecnorm_path)
        for task in tasks:
            _policy, use_model, _pass, n_ac, rho, seed, _ci, _nc = task
            configure_cell(n_ac, rho)
            handle(task, run_episode(venv, model if use_model else None, seed))
        venv.close()
    else:
        import multiprocessing as mp
        print(f'  spawning {workers} worker processes...', flush=True)
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(model_path, vecnorm_path)) as pool:
            for task, m in pool.imap_unordered(_run_task, tasks, chunksize=1):
                handle(task, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='best_model.zip', help='PPO checkpoint .zip')
    ap.add_argument('--vecnorm', default=None,
                    help='VecNormalize .pkl (default: <model>_vecnorm.pkl)')
    ap.add_argument('--out', default=None, help='output CSV (default: Validation/mc_results.csv)')
    ap.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                    help='number of full-grid passes (= episodes per cell)')
    ap.add_argument('--n_ac', default=None,
                    help='comma list of aircraft counts (default: 15,18,21,24,27,30)')
    ap.add_argument('--density', default=None,
                    help='comma list of densities ac/km^2, "a/b" allowed '
                         '(default: 1/25000,1/20000,1/15000,1/12500,1/10000)')
    ap.add_argument('--no-baseline', action='store_true',
                    help='skip the HOLD-only NO-policy baseline (included by default)')
    ap.add_argument('--baseline', action='store_true', help=argparse.SUPPRESS)  # deprecated: now default
    ap.add_argument('--seed', type=int, default=1000, help='base seed')
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel worker processes (1 = serial). On a 24-core box try --workers 24')
    ap.add_argument('--quick', action='store_true', help='2 episodes/cell smoke test')
    args = ap.parse_args()

    n_ac_grid    = [int(x) for x in parse_float_list(args.n_ac, DEFAULT_N_AC)]
    density_grid = parse_float_list(args.density, DEFAULT_DENSITY)
    episodes     = 2 if args.quick else args.episodes

    model_path = args.model if os.path.exists(args.model) else None
    if model_path is None:
        print(f'ERROR: model not found at {args.model}. Pass --model <path/to/best_model.zip>.')
        if args.no_baseline:
            sys.exit(1)
        print('Proceeding with the HOLD-only NO-policy baseline only.')
    vecnorm_path = args.vecnorm or (model_path.replace('.zip', '_vecnorm.pkl') if model_path else None)
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mc_results.csv')

    # Trained policy AND the HOLD-only NO-policy baseline (default; --no-baseline to skip).
    # Per scenario (pass, cell) both run on the SAME seed -> paired comparison. Results go
    # to a SEPARATE CSV per policy: <out>_policy.csv and <out>_hold.csv.
    policies = ([('policy', True)] if model_path else []) + ([] if args.no_baseline else [('hold', False)])
    tasks = build_tasks(policies, n_ac_grid, density_grid, episodes, args.seed)
    workers = max(1, min(args.workers, len(tasks)))

    base, ext = os.path.splitext(out_path)
    out_paths = {name: f'{base}_{name}{ext}' for name, _ in policies}

    print(f'grid: {len(n_ac_grid)} aircraft levels x {len(density_grid)} density levels '
          f'x {episodes} passes = {len(n_ac_grid)*len(density_grid)*episodes} episodes/policy')
    print(f'  n_ac     = {n_ac_grid}')
    print(f'  density  = {[f"{d:.2e}" for d in density_grid]}')
    print(f'  policies = {[p for p, _ in policies]}   workers = {workers}')
    print(f'  total    = {len(tasks)} episodes')
    for name, p in out_paths.items():
        print(f'  {name:6s} -> {p}')
    print(flush=True)

    files, writers = [], {}
    try:
        for name, p in out_paths.items():
            fh = open(p, 'w', newline='')
            w  = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            files.append(fh)
            writers[name] = (w, fh)
        run_tasks(writers, tasks, workers, model_path, vecnorm_path)
    finally:
        for fh in files:
            fh.close()

    print('\nDone. Per-episode results:')
    for name, p in out_paths.items():
        print(f'  {name:6s} -> {p}')


if __name__ == '__main__':
    main()
