"""Detailed validation: how do the models resolve conflicts, step by step?

The normal test (Validation/run_test_scenarios.py) records one row of KPIs per episode. This script
flies a SMALL number of test scenarios and records what happens inside them, in four tables:

  steps.csv       one row per decision step (5 s): the focus aircraft and why it is the focus, the
                  selected action and what became of it (hold / repeat / sent), the advisory waiting
                  for execution, and the most urgent intruder of the focus aircraft
  advisories.csv  one row per advisory sent to the ATCO agent: when it was issued and when it was
                  executed, replaced or dropped, the time to LoS at issue and at execution, and its
                  EFFECT: the smallest predicted miss distance (distance at the closest point of
                  approach) to the aircraft in conflict just before and just after execution, the
                  direction of the turn relative to the intruder, and the new conflicts it creates
  conflicts.csv   one row per conflict (a pair with a predicted LoS within t_warn): its start and
                  end, the time to LoS at the start, the first advisory and first execution for one
                  of the two aircraft, the number of advisories, whether it ended in a LoS, and the
                  smallest distance reached
  episodes.csv    the normal episode KPIs, so the detailed runs can be compared with the test round

Nothing in the environment is changed: DetailedEnv wraps the methods of AirspaceEnv and only reads
its state. The models are the furthest-trained checkpoints in MAIN/Models, as in the normal test.

Usage, from the MAIN folder:
    python Detailed_Validation/detailed_validation.py --terminal 1 --terminals 8 --episodes 50 \\
        --n-ac 30 --density 0.0001
    python Detailed_Validation/detailed_validation.py --merge
    python Detailed_Validation/detailed_validation.py --run deterministic 3 deterministic 30 --episodes 5

Every (model, environment) run is its own process, as BlueSky is a process-wide singleton.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import glob
import io
import math
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, os.path.join(MAIN_DIR, 'Validation'))

from Environment.config import (CONFIG, HOLD_ACTION, TURN_DELTAS, SPEED_ACTIONS, OBS_DIM,
                                N_ACTIONS, RETURN_TO_INITIAL_HDG_ACTION, TRAINING_SEEDS,
                                VALIDATION_SEEDS)
import run_test_scenarios as test

MODELS_DIR = os.path.join(MAIN_DIR, 'Models')
RESULTS_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'Detailed_results')
TYPES = ['none', 'deterministic', 'lognormal']
ENVIRONMENTS = [('none', 0), ('deterministic', 30), ('deterministic', 60), ('lognormal', 30)]
TERMINALS = 8
LABEL = {**{i: f'turn_{d:+d}' for i, d in TURN_DELTAS.items()},
         **{i: ('speed_up' if s > 0 else 'speed_down') for i, s in SPEED_ACTIONS.items()},
         HOLD_ACTION: 'hold', RETURN_TO_INITIAL_HDG_ACTION: 'return'}


# -- Loading a model, also across numpy versions -------------------------------------------------

def load_policy(path):
    """As the normal test, with a fallback for checkpoints pickled under a newer numpy: the
    observation and action space are then given explicitly."""
    try:
        return test.load_policy(path)
    except Exception:
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common import save_util

        class _Shim(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)

        original = save_util.json_to_data

        def json_to_data(json_string, custom_objects=None):
            import base64
            import json
            data = {}
            for key, item in json.loads(json_string).items():
                if custom_objects and key in custom_objects:
                    data[key] = custom_objects[key]
                elif isinstance(item, dict) and ':serialized:' in item:
                    try:
                        data[key] = _Shim(io.BytesIO(base64.b64decode(
                            item[':serialized:'].encode()))).load()
                    except Exception:
                        pass
                else:
                    data[key] = item
            return data

        save_util.json_to_data = json_to_data
        try:
            model = PPO.load(path, device='cpu', custom_objects={
                'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0,
                'observation_space': spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
                'action_space': spaces.Discrete(N_ACTIONS)})
        finally:
            save_util.json_to_data = original
        with open(path.replace('.zip', '_vecnorm.pkl'), 'rb') as handle:
            norm = test._Unpickler(handle).load()
        mean = np.asarray(norm.obs_rms.mean, dtype=np.float32)
        std = np.sqrt(np.asarray(norm.obs_rms.var, dtype=np.float32) + norm.epsilon)
        return model, mean, std, float(norm.clip_obs)


# -- The instrumented environment ----------------------------------------------------------------

def make_env_class():
    from Environment import AirspaceEnv
    from Environment.geometry import cpa, heading_drift, ON_ROUTE_DRIFT
    from Environment.traffic import traffic_states

    t_warn, sep = CONFIG['t_warn'], CONFIG['sep_nm']

    class DetailedEnv(AirspaceEnv):
        """AirspaceEnv that writes down what happens, without changing what happens."""

        def __init__(self, *args, **kwargs):
            self.start_logging()
            super().__init__(*args, **kwargs)

        def reset(self, *args, **kwargs):
            self.start_logging()
            return super().reset(*args, **kwargs)

        def start_logging(self):
            self.log_steps, self.log_advisories, self.log_conflicts = [], [], []
            self._open = {}                 # pair -> conflict record
            self._by_advisory = {}          # id(advisory dict) -> advisory record
            self._outcome = None
            self._los_pairs_step = set()

        # -- geometry helpers --------------------------------------------------------------

        def _snapshot(self):
            flying, indices = self._airborne_indices()
            pos, vel = traffic_states(indices) if indices else (np.zeros((0, 2)), np.zeros((0, 2)))
            return flying, pos, vel

        def _pair_geometry(self, flying, pos, vel, cs, own_vel=None):
            """Per other aircraft: distance, t_LoS, distance at CPA, and bearing (deg, + = right)."""
            if cs not in flying or len(flying) < 2:
                return None
            i = flying.index(cs)
            v = vel[i] if own_vel is None else own_vel
            rel_pos, rel_vel = pos - pos[i], vel - v
            dist_sq, tcpa, dcpa_sq, safe_rel, moving = cpa(rel_pos, rel_vel)
            dist = np.sqrt(dist_sq)
            dcpa = np.where(tcpa > 0, np.sqrt(np.maximum(dcpa_sq, 0.0)), dist)
            closing = moving & (tcpa >= 0) & (dcpa < sep)
            tlos = np.where(closing, tcpa - np.sqrt(np.maximum(sep ** 2 - dcpa ** 2, 0.0) /
                                                    np.where(moving, safe_rel, 1.0)), np.inf)
            tlos = np.where(dist < sep, 0.0, tlos)
            hdg = math.atan2(v[0], v[1])
            ahead = rel_pos[:, 0] * math.sin(hdg) + rel_pos[:, 1] * math.cos(hdg)
            right = rel_pos[:, 0] * math.cos(hdg) - rel_pos[:, 1] * math.sin(hdg)
            bearing = np.degrees(np.arctan2(right, ahead))
            mask = np.arange(len(flying)) != i
            return dict(others=[f for k, f in enumerate(flying) if mask[k]], dist=dist[mask],
                        tlos=tlos[mask], dcpa=dcpa[mask], bearing=bearing[mask])

        def _most_urgent(self, geometry):
            if geometry is None or not len(geometry['dist']):
                return None
            k = int(np.argmin(np.where(np.isfinite(geometry['tlos']), geometry['tlos'], 1e9)))
            if not np.isfinite(geometry['tlos'][k]):
                k = int(np.argmin(geometry['dist']))
            return {key: (geometry[key][k] if key != 'others' else geometry['others'][k])
                    for key in geometry}

        # -- decision steps ----------------------------------------------------------------

        def step(self, action):
            cs = self.focus_cs
            reason, row = 'none', self._row_of.get(cs)
            if row is not None:
                worst = float(np.delete(self.urgency[row], row).max()) if len(self.urgency) > 1 else 0.0
                drift = heading_drift(self._aircraft[cs].initial_hdg, float(self._hdg[row]))
                reason = 'los' if worst > 1 else 'conflict' if worst > 0 else \
                    'drift' if drift > ON_ROUTE_DRIFT else 'quiet'
            flying, pos, vel = self._snapshot()
            urgent = self._most_urgent(self._pair_geometry(flying, pos, vel, cs)) if cs else None
            pending = self.atco.advisory
            record = dict(
                step=self._step_count, time_s=self._sim_time_s, focus=cs, focus_reason=reason,
                action=LABEL[int(action)], outcome='hold',
                pending_for=self.atco.cs,
                pending_wait_s=(pending['execute_at_s'] - self._sim_time_s) if pending else np.nan,
                intruder=urgent['others'] if urgent else None,
                intruder_dist_nm=urgent['dist'] if urgent else np.nan,
                intruder_tlos_s=urgent['tlos'] if urgent else np.nan,
                intruder_bearing_deg=urgent['bearing'] if urgent else np.nan,
                n_aircraft=len(flying))
            self._outcome = None
            self._los_pairs_step = set()
            result = super().step(action)
            record['outcome'] = self._outcome or ('hold' if int(action) == HOLD_ACTION else 'none')
            record['los_seconds'] = self._los_seconds_this_step
            self.log_steps.append(record)
            self._track_conflicts()
            if result[3]:
                self._close_all()
            return result

        def _issue_advisory(self, cs, action_idx):
            repeats = self._ep_stats['repeats']
            sent = sum(self._ep_stats['transmitted'])
            held = self.atco.advisory
            super()._issue_advisory(cs, action_idx)
            if self._ep_stats['repeats'] > repeats:
                self._outcome = 'repeat'
            elif sum(self._ep_stats['transmitted']) > sent:
                self._outcome = 'sent'
                if held is not None and id(held) in self._by_advisory:
                    old = self._by_advisory[id(held)]
                    old['fate'] = 'replaced_same' if old['aircraft'] == cs else 'replaced_other'
                    old['ended_s'] = self._sim_time_s
                    old['replaced_by'] = LABEL[action_idx]
                self._new_advisory(cs, action_idx, self.atco.advisory)

        def _new_advisory(self, cs, action_idx, advisory):
            flying, pos, vel = self._snapshot()
            urgent = self._most_urgent(self._pair_geometry(flying, pos, vel, cs))
            record = dict(advisory_id=len(self.log_advisories), aircraft=cs,
                          action=LABEL[action_idx], issued_s=self._sim_time_s,
                          due_s=advisory['execute_at_s'], first_issued_s=advisory['response_start_s'],
                          fate='pending_at_end', ended_s=np.nan, replaced_by=None,
                          issue_tlos_s=urgent['tlos'] if urgent else np.nan,
                          issue_intruder_bearing_deg=urgent['bearing'] if urgent else np.nan)
            self.log_advisories.append(record)
            self._by_advisory[id(advisory)] = record
            for pair, conflict in self._open.items():
                if cs in pair:
                    conflict['n_sent'] += 1
                    conflict['advised'].add(cs)
                    if np.isnan(conflict['first_sent_s']):
                        conflict['first_sent_s'] = self._sim_time_s

        # -- execution and its effect ------------------------------------------------------

        def _execute_due_advisories(self):
            pending, cs = self.atco.advisory, self.atco.cs
            before = None
            if pending is not None and self._sim_time_s >= pending['execute_at_s'] \
                    and self._still_flying(cs):
                flying, pos, vel = self._snapshot()
                before = (flying, pos, vel, self._aircraft[cs].commanded_mach)
            super()._execute_due_advisories()
            if pending is None or self.atco.advisory is not None or id(pending) not in self._by_advisory:
                return
            record = self._by_advisory[id(pending)]
            record['ended_s'] = self._sim_time_s
            if before is None:
                record['fate'] = 'dropped_exit'
                return
            record['fate'] = 'executed'
            flying, pos, vel, mach = before
            geometry = self._pair_geometry(flying, pos, vel, cs)
            i = flying.index(cs)
            speed = float(np.hypot(*vel[i]))
            if 'target_hdg' in pending:
                h = math.radians(pending['target_hdg'])
                new_vel = np.array([speed * math.sin(h), speed * math.cos(h)])
            else:
                # a speed change scales the speed; BlueSky reaches it gradually, this is the end state
                new_vel = vel[i] * pending['target_mach'] / max(mach, 1e-6)
            after = self._pair_geometry(flying, pos, vel, cs, own_vel=new_vel)
            conflict = np.isfinite(geometry['tlos']) & (geometry['tlos'] <= t_warn)
            new_conflict = np.isfinite(after['tlos']) & (after['tlos'] <= t_warn) & ~conflict
            urgent = self._most_urgent(geometry)
            turn = 0.0
            if 'target_hdg' in pending:
                own = math.degrees(math.atan2(vel[i][0], vel[i][1]))
                turn = ((pending['target_hdg'] - own + 180) % 360) - 180
            record.update(
                exec_tlos_s=urgent['tlos'] if urgent else np.nan,
                exec_in_los=bool(urgent is not None and urgent['dist'] < sep),
                n_conflicts_before=int(conflict.sum()),
                dcpa_before_nm=float(geometry['dcpa'][conflict].min()) if conflict.any() else np.nan,
                dcpa_after_nm=float(after['dcpa'][conflict].min()) if conflict.any() else np.nan,
                new_conflicts=int(new_conflict.sum()),
                turn_deg=turn,
                intruder_bearing_deg=urgent['bearing'] if urgent else np.nan,
                # positive: the turn goes away from the intruder's side (intruder right, turn left)
                turn_away=(np.sign(-turn) == np.sign(urgent['bearing'])) if (urgent is not None and turn) else np.nan)
            for pair, c in self._open.items():
                if cs in pair:
                    c['n_exec'] += 1
                    if np.isnan(c['first_exec_s']):
                        c['first_exec_s'] = self._sim_time_s

        def _select_focus(self):
            pending = self.atco.advisory
            super()._select_focus()
            if pending is not None and self.atco.advisory is None and id(pending) in self._by_advisory:
                record = self._by_advisory[id(pending)]
                if record['fate'] == 'pending_at_end':
                    record['fate'], record['ended_s'] = 'dropped_focus', self._sim_time_s

        def _remove_exited_aircraft(self):
            pending = self.atco.advisory
            super()._remove_exited_aircraft()
            if pending is not None and self.atco.advisory is None and id(pending) in self._by_advisory:
                record = self._by_advisory[id(pending)]
                if record['fate'] == 'pending_at_end':
                    record['fate'], record['ended_s'] = 'dropped_exit', self._sim_time_s

        # -- conflicts ---------------------------------------------------------------------

        def _scan_separation(self, index_of=None):
            pairs = super()._scan_separation(index_of)
            self._los_pairs_step |= pairs
            return pairs

        def _track_conflicts(self):
            flying = self._urgency_cs_list
            rows, cols = np.where(self.urgency > 0)
            now = {(flying[i], flying[j]) for i, j in zip(rows, cols) if i < j}
            for pair in now - set(self._open):
                i, j = flying.index(pair[0]), flying.index(pair[1])
                self._open[pair] = dict(
                    a=pair[0], b=pair[1], start_s=self._sim_time_s,
                    start_tlos_s=float(self.t_los[i, j]), end_s=np.nan, los=False,
                    min_dist_nm=np.inf, first_sent_s=np.nan, first_exec_s=np.nan,
                    n_sent=0, n_exec=0, advised=set(), focus_steps=0)
            for pair, c in self._open.items():
                if pair[0] in self._row_of and pair[1] in self._row_of:
                    i, j = self._row_of[pair[0]], self._row_of[pair[1]]
                    c['min_dist_nm'] = min(c['min_dist_nm'],
                                           float(np.hypot(*(self._pos[i] - self._pos[j]))))
                if pair in self._los_pairs_step or (pair[1], pair[0]) in self._los_pairs_step:
                    c['los'] = True
                if self.focus_cs in pair:
                    c['focus_steps'] += 1
            for pair in set(self._open) - now:
                self._close(pair)

        def _close(self, pair):
            c = self._open.pop(pair)
            c['end_s'] = self._sim_time_s
            c['advised'] = '+'.join(sorted(c['advised']))
            self.log_conflicts.append(c)

        def _close_all(self):
            for pair in list(self._open):
                self._close(pair)

    return DetailedEnv


# -- Running -------------------------------------------------------------------------------------

def parse_env(text):
    """'N0' -> ('none', 0), 'D30' -> ('deterministic', 30), 'L45' -> ('lognormal', 45)."""
    law = {'N': 'none', 'D': 'deterministic', 'L': 'lognormal'}[text[0].upper()]
    return law, int(text[1:])


def all_runs(types=TYPES, seeds=TRAINING_SEEDS, environments=ENVIRONMENTS):
    """Ordered environment first, so that dealing the runs out over the terminals in turn gives
    every terminal the same mix of environments, and so about the same workload."""
    return [(t, seed, law, mean) for law, mean in environments for seed in seeds for t in types]


def run_name(model_type, seed, law, mean):
    test_name = 'no_delay' if law == 'none' else f'{law}_{mean}s'
    return f'{test.PREFIX[model_type]}{seed}_{test_name}'


def detailed_run(model_type, seed, law, mean, models, out, episodes):
    folder = os.path.join(out, 'runs', run_name(model_type, seed, law, mean))
    if os.path.exists(os.path.join(folder, 'done')):
        print(f'{os.path.basename(folder)} already done', flush=True)
        return
    steps, checkpoint = test.find_model(models, model_type, seed)
    policy = load_policy(checkpoint)
    env = make_env_class()(delay_mode=law, delay_mean_s=float(mean)) if mean else \
        make_env_class()(delay_mode='none')
    tables = {'steps': [], 'advisories': [], 'conflicts': [], 'episodes': []}
    started = time.time()
    for scenario, scenario_seed in enumerate(VALIDATION_SEEDS[:episodes]):
        observation, _ = env.reset(options={'scenario_seed': scenario_seed})
        while True:
            observation, _, _, truncated, info = env.step(test.choose(policy, observation))
            if truncated:
                break
        tags = dict(model_type=model_type, seed=seed, test_law=law, test_delay=mean,
                    scenario=scenario, scenario_seed=scenario_seed, model_steps=steps)
        for name, rows in (('steps', env.log_steps), ('advisories', env.log_advisories),
                           ('conflicts', env.log_conflicts)):
            tables[name] += [{**tags, **row} for row in rows]
        kpis = {k.removeprefix('ep_'): v for k, v in info.items() if k.startswith('ep_')}
        tables['episodes'].append({**tags, 'n_aircraft': env.n_aircraft, 'rho': env.rho, **kpis})
        rate = (time.time() - started) / (scenario + 1)
        print(f'  {scenario + 1}/{episodes}  {rate:4.1f} s/scenario', flush=True)
    os.makedirs(folder, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(os.path.join(folder, f'{name}.csv'), index=False)
    open(os.path.join(folder, 'done'), 'w').close()


def run_terminal(terminal, terminals, models, out, episodes, airspace, runs):
    share = runs[terminal - 1::terminals]
    print(f'terminal {terminal} of {terminals}: {len(share)} runs -> {out}', flush=True)
    for model_type, seed, law, mean in share:
        command = [sys.executable, os.path.abspath(__file__), '--run', model_type, str(seed), law,
                   str(mean), '--models', models, '--out', out, '--episodes', str(episodes),
                   *airspace]
        subprocess.run(command)


def merge(out):
    for name in ('steps', 'advisories', 'conflicts', 'episodes'):
        paths = sorted(glob.glob(os.path.join(out, 'runs', '*', f'{name}.csv')))
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()
        frame.to_csv(os.path.join(out, f'{name}.csv'), index=False)
        print(f'{name}: {len(frame):,} rows from {len(paths)} runs')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--terminal', type=int)
    parser.add_argument('--terminals', type=int, default=TERMINALS)
    parser.add_argument('--run', nargs=4, metavar=('TYPE', 'SEED', 'LAW', 'MEAN'))
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--models', default=MODELS_DIR)
    parser.add_argument('--out', default=RESULTS_DIR)
    parser.add_argument('--episodes', type=int, default=50,
                        help='test scenarios per run, from the start of the test set (default 50)')
    parser.add_argument('--types', nargs='+', default=TYPES, choices=TYPES)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(TRAINING_SEEDS))
    parser.add_argument('--envs', nargs='+', default=['N0', 'D30', 'D60', 'L30'],
                        help='test environments, e.g. N0 D30 D60 L30 (default all four)')
    parser.add_argument('--n-ac', type=int, default=None)
    parser.add_argument('--density', type=float, default=None)
    args = parser.parse_args()
    out, models = os.path.abspath(args.out), os.path.abspath(args.models)
    airspace = []
    if args.n_ac is not None:
        CONFIG['n_aircraft'] = lambda rng: args.n_ac
        airspace += ['--n-ac', str(args.n_ac)]
    if args.density is not None:
        CONFIG['rho'] = lambda rng: args.density
        airspace += ['--density', repr(args.density)]

    if args.merge:
        merge(out)
    elif args.run:
        model_type, seed, law, mean = args.run
        detailed_run(model_type, int(seed), law, int(mean), models, out, args.episodes)
    elif args.terminal:
        runs = all_runs(args.types, args.seeds, [parse_env(e) for e in args.envs])
        run_terminal(args.terminal, args.terminals, models, out, args.episodes, airspace, runs)
    else:
        parser.error('give --terminal, --run or --merge')


if __name__ == '__main__':
    main()
