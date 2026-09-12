# Episode metrics: the counters accumulated during an episode, and the figures derived from them.
#
# The reported set is Table 2.4 of the report -- safety, route efficiency, instruction load and
# reward -- broken down by the specific heading and speed advisory issued, so that a change of
# resolution strategy is visible and not only a change of instruction count. Everything reported
# per flight hour is normalised by traffic, so scenarios of different size stay comparable.

from .config import N_ACTIONS, TURN_DELTAS, SPEED_ACTIONS, RETURN_TO_INITIAL_HDG_ACTION

# Advisory index -> the column it is reported in, read off the action layout rather than
# restated here. Hold is absent: it transmits nothing, so there is nothing to count.
ADVISORY_LABELS = {}
for _index, _delta in TURN_DELTAS.items():
    ADVISORY_LABELS[_index] = f'turn_{"p" if _delta > 0 else "m"}{abs(_delta)}'
ADVISORY_LABELS[RETURN_TO_INITIAL_HDG_ACTION] = 'return'
for _index, _sign in SPEED_ACTIONS.items():
    ADVISORY_LABELS[_index] = 'speed_up' if _sign > 0 else 'speed_down'


def new_ep_stats():
    return {
        'reward': 0.0,         # -412.7   summed step reward
        'reward_los': 0.0,     # -310.0   ...of which the LoS term
        'reward_drift': 0.0,   # -42.7    ...of which the drift term
        'reward_work': 0.0,    # -60.0    ...of which the workload term
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
        # Transmitted advisories per action index, so turns and speeds can be read one by one.
        'transmitted': [0] * N_ACTIONS,
    }


def episode_summary(stats):
    s = stats
    flight_hours = max(s['flight_s'] / 3600.0, 1e-9)
    acted        = max(s['delay_acted'], 1)
    exits        = s['exits']

    # Advisories TRANSMITTED, counted in _issue_advisory rather than off the action histogram.
    turns, speeds = s['turns'], s['speeds']

    summary = {
        # LoS and conflicts.
        'ep_los_events_per_fh': s['los_events'] / flight_hours,
        'ep_conflicts_per_fh':  s['conflicts'] / flight_hours,

        # Route efficiency: track flown over the straight route, and the share of aircraft that
        # left within the on-route heading tolerance of the heading they entered on.
        'ep_path_ratio':        s['flown_nm'] / s['route_nm'] if s['route_nm'] else 1.0,
        'ep_on_route_rate':     s['on_route'] / exits if exits else 1.0,

        # Instruction load, by kind and in total.
        'ep_turns_per_fh':         turns / flight_hours,
        'ep_speed_changes_per_fh': speeds / flight_hours,
        'ep_advisories_per_fh':    (turns + speeds) / flight_hours,

        # Reward: the raw episode sum, and the same sum per flight hour so that episodes with
        # different traffic and length stay comparable.
        'ep_reward_total':  s['reward'],
        'ep_reward_per_fh': s['reward'] / flight_hours,

        # The same reward split into its three terms, on the same per-flight-hour scale, so
        # they add up to ep_reward_per_fh and can be read against each other.
        'ep_reward_los_per_fh':   s['reward_los'] / flight_hours,
        'ep_reward_drift_per_fh': s['reward_drift'] / flight_hours,
        'ep_reward_work_per_fh':  s['reward_work'] / flight_hours,

        # Bookkeeping: the denominator behind every rate above, and the episode length.
        'ep_flight_hours': s['flight_s'] / 3600.0,
        'ep_length':       s['steps'],

        # Diagnostics on the delay pipeline itself. Kept because they are the only signal that
        # would show instructions being revised faster than the ATCO can act on them: a mean
        # response far above the nominal delay, or a discard count near the advisory count.
        'ep_delay_mean_s': s['delay_sum_s'] / acted,
        'ep_discarded':    s['discarded'],
        'ep_repeats':      s['repeats'],
    }

    # The specific advisory issued, per flight hour: which turn, and which way the speed went.
    for index, label in ADVISORY_LABELS.items():
        summary[f'ep_{label}_per_fh'] = s['transmitted'][index] / flight_hours

    return summary
