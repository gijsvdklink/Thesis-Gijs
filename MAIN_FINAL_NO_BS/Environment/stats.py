
from .config import N_ACTIONS, TURN_DELTAS, SPEED_ACTIONS, RETURN_TO_INITIAL_HDG_ACTION

ADVISORY_LABELS = {}
for _index, _delta in TURN_DELTAS.items():
    ADVISORY_LABELS[_index] = f'turn_{"p" if _delta > 0 else "m"}{abs(_delta)}'
ADVISORY_LABELS[RETURN_TO_INITIAL_HDG_ACTION] = 'return'
for _index, _sign in SPEED_ACTIONS.items():
    ADVISORY_LABELS[_index] = 'speed_up' if _sign > 0 else 'speed_down'


def new_ep_stats():
    return {
        'reward': 0.0,
        'reward_los': 0.0,
        'reward_drift': 0.0,
        'reward_work': 0.0,
        'steps': 0,
        'actions': [],
        'los_seconds': 0,
        'los_events': 0,
        'conflicts': 0,
        'flight_s': 0.0,
        'exits': 0,
        'on_route': 0,
        'deviation_nm': 0.0,
        'flown_nm': 0.0,
        'route_nm': 0.0,
        'drift_deg_sum': 0.0,
        'drift_samples': 0,
        'delay_sum_s': 0.0,
        'delay_acted': 0,
        'focus_spells': 0,
        'focus_spell_steps': 0,
        'discarded': 0,
        'repeats': 0,
        'turns': 0,
        'speeds': 0,
        'turns_distinct': 0,
        'speeds_distinct': 0,
        'transmitted': [0] * N_ACTIONS,
    }


def episode_summary(stats):
    s = stats
    flight_hours = max(s['flight_s'] / 3600.0, 1e-9)
    acted        = max(s['delay_acted'], 1)
    exits        = s['exits']

    turns, speeds = s['turns_distinct'], s['speeds_distinct']

    sent_turns, sent_speeds = s['turns'], s['speeds']

    summary = {

        'ep_los_events':         s['los_events'],
        'ep_los_events_per_fh':  s['los_events'] / flight_hours,
        'ep_los_seconds':        s['los_seconds'],
        'ep_los_seconds_per_fh': s['los_seconds'] / flight_hours,
        'ep_conflicts':          s['conflicts'],
        'ep_conflicts_per_fh':   s['conflicts'] / flight_hours,

        'ep_flown_nm':          s['flown_nm'],
        'ep_route_nm':          s['route_nm'],
        'ep_path_ratio':        s['flown_nm'] / s['route_nm'] if s['route_nm'] else 1.0,
        'ep_on_route':          s['on_route'],
        'ep_exits':             exits,
        'ep_on_route_rate':     s['on_route'] / exits if exits else 1.0,

        'ep_turns':                turns,
        'ep_turns_per_fh':         turns / flight_hours,
        'ep_speed_changes':        speeds,
        'ep_speed_changes_per_fh': speeds / flight_hours,
        'ep_advisories':           turns + speeds,
        'ep_advisories_per_fh':    (turns + speeds) / flight_hours,

        'ep_turns_sent':                sent_turns,
        'ep_turns_sent_per_fh':         sent_turns / flight_hours,
        'ep_speed_changes_sent':        sent_speeds,
        'ep_speed_changes_sent_per_fh': sent_speeds / flight_hours,
        'ep_advisories_sent':           sent_turns + sent_speeds,
        'ep_advisories_sent_per_fh':    (sent_turns + sent_speeds) / flight_hours,

        'ep_reward_total':  s['reward'],
        'ep_reward_per_fh': s['reward'] / flight_hours,

        'ep_reward_los':          s['reward_los'],
        'ep_reward_los_per_fh':   s['reward_los'] / flight_hours,
        'ep_reward_drift':        s['reward_drift'],
        'ep_reward_drift_per_fh': s['reward_drift'] / flight_hours,
        'ep_reward_work':         s['reward_work'],
        'ep_reward_work_per_fh':  s['reward_work'] / flight_hours,

        'ep_flight_hours': s['flight_s'] / 3600.0,
        'ep_length':       s['steps'],

        'ep_delay_mean_s': s['delay_sum_s'] / acted,
        'ep_discarded':    s['discarded'],
        'ep_repeats':      s['repeats'],
    }

    for index, label in ADVISORY_LABELS.items():
        summary[f'ep_{label}']        = s['transmitted'][index]
        summary[f'ep_{label}_per_fh'] = s['transmitted'][index] / flight_hours

    return summary
