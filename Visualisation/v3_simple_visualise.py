"""
Visualise a trained PPO policy on v3_simple_env (single-agent ATCO, analytical urgency).
Runs several episodes in a pygame window.

The focus aircraft (most urgent by analytical urgency matrix) is highlighted in yellow.
All others use slot colours; aircraft in LoS are shown in red.

Usage:
    python -m Visualisation.v3_simple_visualise --model path/to/model.zip
"""

import argparse
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from Environments.v3_simple_env import SimpleAirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm

import bluesky as bs

# ── Visualisation constants ───────────────────────────────────────────────────

N_EPISODES  = 5
WINDOW_SIZE = 850
FPS         = 8

CYAN   = (  0, 210, 255)   # focus aircraft (most urgent)
ORANGE = (255, 140,   0)   # predicted or active conflict
GREY   = (160, 160, 160)   # no conflict
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)

ACTION_LABELS = ['-20°', '-10°', '0°', '+10°', '+20°', 'WP']


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
        self._cx = (x_min + x_max) / 2
        self._cy = (y_min + y_max) / 2
        self._scale = min(window_w, window_h) / span
        self._w, self._h = window_w, window_h

    def nm_to_px(self, x_nm, y_nm):
        return (int((x_nm * NM_TO_KM - self._cx) *  self._scale + self._w / 2),
                int((y_nm * NM_TO_KM - self._cy) * -self._scale + self._h / 2))

    def latlon_to_px(self, lat, lon):
        nm = latlon_to_nm(CONFIG['center_ll'], lat, lon)
        return self.nm_to_px(nm[0], nm[1])

    def nm_length_to_px(self, length_nm):
        return max(1, int(length_nm * NM_TO_KM * self._scale))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_dashed_line(surface, color, p1, p2, dash_len=8, gap_len=5):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise trained policy on v3_simple_env')
    parser.add_argument('--model', required=True, metavar='MODEL_PATH',
                        help='Path to the saved model .zip file')
    parser.add_argument('--episodes', type=int, default=N_EPISODES,
                        help=f'Number of episodes to run (default {N_EPISODES})')
    parser.add_argument('--fps', type=int, default=FPS,
                        help=f'Playback speed in frames per second (default {FPS})')
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
    if not os.path.exists(model_path):
        sys.exit(f'Model not found: {model_path}')

    print(f'Loading model : {model_path}')
    model = PPO.load(model_path)

    env = SimpleAirspaceEnv()
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('v3_simple_env — trained ATCO policy (analytical urgency)')
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    quit_requested = False

    for episode in range(args.episodes):
        obs, _      = env.reset()
        n_capacity  = env.n_aircraft
        view        = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px  = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_px      = view.nm_length_to_px(CONFIG['sep_nm'] / 2)

        dest_px_by_cs = {cs: view.latlon_to_px(*env._destination_ll[cs])
                         for cs in env._active_callsigns}

        total_reward = 0.0
        last_action  = 2
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True; done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True; done = True

            if quit_requested:
                break

            action, _ = model.predict(obs, deterministic=True)
            last_action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done   = terminated or truncated
            step  += 1
            if info['los_pairs']:
                los_steps += 1

            for cs in env._active_callsigns:
                if cs not in dest_px_by_cs and cs in env._destination_ll:
                    dest_px_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            # ── Draw frame ──────────────────────────────────────────────────

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)

            focus_cs  = env._focus_cs

            # Derive conflict set from the urgency matrix stored by SimpleAirspaceEnv
            conf_set = set()
            U        = env._urgency_matrix
            cs_list  = env._urgency_cs_list
            if U.size > 0:
                row_max = U.max(axis=1)
                conf_set = {cs_list[i] for i in range(len(cs_list)) if row_max[i] > 0}

            for slot_i, callsign in enumerate(env._slots):
                if callsign is None or callsign not in env._active_callsigns:
                    continue
                idx = bs.traf.id2idx(callsign)
                if idx < 0:
                    continue

                px, py   = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                is_focus = callsign == focus_cs

                # Priority: CYAN (focus) > ORANGE (conflict) > GREY (clear)
                color  = CYAN if is_focus else (ORANGE if callsign in conf_set else GREY)
                ring_r = sep_px + 4 if is_focus else sep_px
                width  = 2 if is_focus else 1

                pygame.draw.circle(screen, color, (px, py), ring_r, width)
                if callsign in dest_px_by_cs:
                    draw_dashed_line(screen, color, (px, py), dest_px_by_cs[callsign])
                pygame.draw.circle(screen, color, (px, py), 5)

                label = callsign + (f' {ACTION_LABELS[last_action]} ◄' if is_focus else '')
                screen.blit(font_sm.render(label, True, color), (px + 7, py - 7))

            n_active  = len(env._active_callsigns)
            hud_lines = [
                f'Episode {episode+1}/{args.episodes}  (active={n_active}/{n_capacity})  [v3_simple — analytical urgency]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Focus: {focus_cs}   served={env._next_callsign_id}',
                f'Total reward={total_reward:.1f}   last action={ACTION_LABELS[last_action]}',
            ]
            for j, line in enumerate(hud_lines):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            screen.blit(font_sm.render('● CYAN   = focus aircraft (receiving instruction)', True, CYAN),
                        (8, WINDOW_SIZE - 40))
            screen.blit(font_sm.render('● ORANGE = predicted or active conflict',           True, ORANGE),
                        (8, WINDOW_SIZE - 26))
            screen.blit(font_sm.render('● GREY   = no conflict',                            True, GREY),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(args.fps)

        print(f'Episode {episode+1:2d}  capacity={n_capacity}  served={env._next_callsign_id}  '
              f'steps={step:3d}  total_reward={total_reward:.2f}  LoS-steps={los_steps}')

        if quit_requested:
            break

    pygame.quit()


if __name__ == '__main__':
    main()
