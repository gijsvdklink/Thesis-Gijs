# Episode metrics: the counters accumulated during an episode, and the figures derived from them.

import numpy as np

from .config import N_ACTIONS, STEP_DURATION_S


def new_ep_stats():
    return {
        'reward': 0.0,         # -412.7   summed step reward
        'steps': 0,            # 1480     RL steps taken
        'actions': [],         # [3, 3, 5, 7, ...]  one action index per step
        'los_seconds': 0,      # 14       simulated seconds with at least one pair in LoS
        'los_events': 0,       # 3        distinct intrusions (entries, scanned every second)
        'conflicts': 0,        # 47       distinct predicted intrusions within t_warn (entries, per step)
        'flight_s': 0.0,       # 61200.0  airborne time flown by all aircraft
        'exits': 0,            # 21       aircraft that left having actually flown
        'on_route': 0,         # 18       ...of which left within the heading tolerance
        'deviation_nm': 0.0,   # 96.3     ...summed distance from the no-turn exit point
        'flown_nm': 0.0,       # 1742.5   ...summed track length actually flown
        'route_nm': 0.0,       # 1698.0   ...summed straight-line length of the route they were given
        # Drift summed over aircraft and steps; divided by aircraft-steps for a mean angle.
        'drift_deg_sum': 0.0,  # 20450.0  summed |drift| over aircraft-steps
        'drift_samples': 0,    # 17600    aircraft-steps that contributed
        'delay_sum_s': 0.0,    # 2790.0   summed response delay actually served
        'delay_acted': 0,      # 93       instructions the ATCO actually acted on
        'focus_spells': 0,     # 112      times an aircraft became the focus
        'focus_spell_steps': 0,# 1480     steps summed over those spells
        'discarded': 0,        # 19       advisories replaced before they could be flown
        'repeats': 0,          # 240      the same advice re-selected while it was still standing
        'turns': 0,            # 74       turn advisories actually transmitted
        'speeds': 0,           # 38       speed advisories actually transmitted
    }


def episode_summary(stats):
    s = stats
    flight_hours = max(s['flight_s'] / 3600.0, 1e-9)
    acted        = max(s['delay_acted'], 1)
    exits        = s['exits']

    # Advisories TRANSMITTED, counted in _issue_advisory rather than off the action histogram.
    turns, speeds = s['turns'], s['speeds']

    return {
        'mean_episode_reward': s['reward'] / max(s['steps'], 1),
        'ep_reward_total':     s['reward'],
        'ep_length':           s['steps'],
        'ep_los_seconds':      s['los_seconds'],
        'ep_los_fraction':     s['los_seconds'] / max(s['steps'] * STEP_DURATION_S, 1),
        'ep_los_events':       s['los_events'],
        'action_distribution': np.bincount(s['actions'], minlength=N_ACTIONS).tolist(),

        # Traffic-normalised safety: raw LoS counts are not comparable between episodes.
        'ep_flight_hours':      s['flight_s'] / 3600.0,
        'ep_los_events_per_fh': s['los_events'] / flight_hours,
        'ep_conflicts':         s['conflicts'],
        'ep_conflicts_per_fh':  s['conflicts'] / flight_hours,

        # Route keeping over every exit: arrival within arrival_hdg_tol_deg, deviation from the no-turn exit.
        'ep_exit_deviation_nm': s['deviation_nm'] / exits if exits else 0.0,
        # Track flown over the straight route: 1.0 is a perfectly direct crossing.
        'ep_path_ratio':        s['flown_nm'] / s['route_nm'] if s['route_nm'] else 1.0,
        'ep_arrival_rate':      s['on_route'] / exits if exits else 1.0,
        'ep_exits':             exits,

        # Drift from assigned headings and the calls it took; the per-flight-hour rates are the comparable ones.
        'ep_mean_drift_deg':      s['drift_deg_sum'] / max(s['drift_samples'], 1),
        'ep_turns':               turns,
        'ep_speed_changes':       speeds,
        'ep_turns_per_fh':        turns / flight_hours,
        'ep_speed_changes_per_fh': speeds / flight_hours,

        # Diagnostics -- kept in the evaluation CSVs rather than TensorBoard.
        'ep_delay_mean_s':     s['delay_sum_s'] / acted,
        'ep_focus_hold_steps': s['focus_spell_steps'] / max(s['focus_spells'], 1),
        'ep_discarded':        s['discarded'],
        # The advice already standing, selected again: charged as workload, but nothing new is assessed.
        'ep_repeats':          s['repeats'],
    }
