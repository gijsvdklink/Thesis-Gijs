"""
Visualise a trained PPO policy on v2_env.
Runs several episodes in a pygame window.

Usage:
    python -m Visualisation.v2_visualise --model path/to/model.zip
"""

import argparse
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from Environments.v2_env import (
    AirspaceEnv, CONFIG, MAX_AGENTS, OBS_DIM, NM_TO_KM, latlon_to_nm,
)

import bluesky as bs

# ── Visualisation constants ───────────────────────────────────────────────────

N_EPISODES  = 5
WINDOW_SIZE = 850
FPS         = 8

SLOT_COLORS   = [(50,180,50),(50,130,220),(255,140,0),(150,50,200),
                 (0,200,200),(220,200,0),(220,30,30),(160,160,160)]
RED           = (220, 30, 30)
BLACK         = (0, 0, 0)
WHITE         = (255, 255, 255)
ACTION_LABELS = ['-20°', '-10°', '0°', '+10°', '+20°', 'back']


# ── Coordinate view ───────────────────────────────────────────────────────────

class _View:
    def __init__(self, polygon_nm, window_w, window_h):
        padding_km = CONFIG['sep_nm'] * NM_TO_KM * 2.0
        poly_km    = polygon_nm * NM_TO_KM
        x_min = poly_km[:, 0].min() - padding_km
        x_max = poly_km[:, 0].max() + padding_km
        y_min = poly_km[:, 1].min() - padding_km
        y_max = poly_km[:, 1].max() + padding_km
        span  = max(x_max - x_min, y_max - y_min)
        self._center_x  = (x_min + x_max) / 2
        self._center_y  = (y_min + y_max) / 2
        self._px_per_km = min(window_w, window_h) / span
        self._w, self._h = window_w, window_h

    def nm_to_px(self, x_nm, y_nm):
        x_km, y_km = x_nm * NM_TO_KM, y_nm * NM_TO_KM
        px = int((x_km - self._center_x) *  self._px_per_km + self._w / 2)
        py = int((y_km - self._center_y) * -self._px_per_km + self._h / 2)
        return (px, py)

    def latlon_to_px(self, lat, lon):
        nm = latlon_to_nm(CONFIG['center_ll'], lat, lon)
        return self.nm_to_px(nm[0], nm[1])

    def nm_length_to_px(self, length_nm):
        return max(1, int(length_nm * NM_TO_KM * self._px_per_km))


def _draw_dashed_line(surface, color, p1, p2, dash_len=8, gap_len=5):
    dx, dy    = p2[0] - p1[0], p2[1] - p1[1]
    total_len = math.hypot(dx, dy)
    if total_len < 1:
        return
    ux, uy  = dx / total_len, dy / total_len
    x, y    = float(p1[0]), float(p1[1])
    drawn   = 0.0
    drawing = True
    while drawn < total_len:
        seg = min(dash_len if drawing else gap_len, total_len - drawn)
        if drawing:
            pygame.draw.line(surface, color,
                (round(x), round(y)),
                (round(x + ux * seg), round(y + uy * seg)), 1)
        x += ux * seg; y += uy * seg; drawn += seg; drawing = not drawing


def _pad_obs(obs, n_slots):
    """Pad obs from (n_slots, OBS_DIM) to (MAX_AGENTS, OBS_DIM) for the model."""
    out = np.zeros((MAX_AGENTS, OBS_DIM), dtype=np.float32)
    out[:n_slots] = obs
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise trained policy on v2_env')
    parser.add_argument('--model', required=True, metavar='MODEL_PATH',
                        help='Path to the saved model .zip file')
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
    if not os.path.exists(model_path):
        sys.exit(f'Model not found: {model_path}')

    print(f'Loading model : {model_path}')
    model = PPO.load(model_path)

    env = AirspaceEnv()
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('v2_env — trained policy')
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    quit_requested = False

    for episode in range(N_EPISODES):
        obs_stack, _ = env.reset()
        n_agents      = env.n_aircraft
        view          = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px    = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_radius_px = view.nm_length_to_px(CONFIG['sep_nm'] / 2)

        dest_px_by_cs = {cs: view.latlon_to_px(*env._destination_ll[cs])
                         for cs in env._active_callsigns}

        total_rewards = np.zeros(n_agents)
        last_actions  = [2] * n_agents
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True; done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True; done = True

            if quit_requested:
                break

            padded_obs     = _pad_obs(obs_stack, n_agents)
            all_actions, _ = model.predict(padded_obs, deterministic=True)
            actions        = all_actions[:n_agents]
            last_actions   = list(actions)

            obs_stack, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_rewards += rewards
            done   = terminated or truncated
            step  += 1
            if info['los_pairs']:
                los_steps += 1

            for cs in env._active_callsigns:
                if cs not in dest_px_by_cs and cs in env._destination_ll:
                    dest_px_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)

            aircraft_in_los = {cs for pair in bs.traf.cd.lospairs for cs in pair}
            for slot_i, callsign in enumerate(env._slots):
                if callsign is None or callsign not in env._active_callsigns:
                    continue
                idx = bs.traf.id2idx(callsign)
                if idx < 0:
                    continue
                px, py     = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                base_color = SLOT_COLORS[slot_i % len(SLOT_COLORS)]
                color      = RED if callsign in aircraft_in_los else base_color
                dest       = dest_px_by_cs.get(callsign)
                pygame.draw.circle(screen, color, (px, py), sep_radius_px, 1)
                if dest:
                    _draw_dashed_line(screen, color, (px, py), dest)
                pygame.draw.circle(screen, color, (px, py), 5)
                screen.blit(
                    font_sm.render(f'{callsign} {ACTION_LABELS[last_actions[slot_i]]}', True, color),
                    (px + 7, py - 7),
                )

            hud_lines = [
                f'Episode {episode+1}/{N_EPISODES}  ({n_agents} agents)  [trained policy]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Served {env._next_callsign_id}   active={len(env._active_callsigns)}',
                '  '.join(f'S{i}={total_rewards[i]:.0f}' for i in range(n_agents)),
            ]
            for j, line in enumerate(hud_lines):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            legend_y = WINDOW_SIZE - 16 - n_agents * 14
            for i in range(n_agents):
                screen.blit(font_sm.render(f'● Slot {i}', True, SLOT_COLORS[i % len(SLOT_COLORS)]),
                            (8, legend_y + i * 14))
            screen.blit(font_sm.render('● RED = separation violation', True, RED),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(FPS)

        print(f'Episode {episode+1:2d}  n_ac={n_agents}  steps={step:3d}  '
              f'mean_reward={total_rewards.mean():.2f}  LoS-steps={los_steps}  '
              f'served={env._next_callsign_id}')

        if quit_requested:
            break

    pygame.quit()


if __name__ == '__main__':
    main()
