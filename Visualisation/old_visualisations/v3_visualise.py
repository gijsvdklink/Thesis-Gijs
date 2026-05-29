"""
Visualise a trained PPO policy on v3_env (single-agent ATCO).
Runs several episodes in a pygame window.

The focus aircraft (most urgent) is highlighted in yellow.
All others use slot colours; aircraft in LoS are shown in red.

Usage:
    python -m Visualisation.v3_visualise --model path/to/model.zip
"""

import argparse
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from Environments.v3_env import AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm

import bluesky as bs

# ── Visualisation constants ───────────────────────────────────────────────────

N_EPISODES  = 5
WINDOW_SIZE = 850
FPS         = 8

SLOT_COLORS  = [(50,180,50),(50,130,220),(255,140,0),(150,50,200),
                (0,200,200),(220,200,0),(220,30,30),(160,160,160)]
FOCUS_COLOR  = (255, 255,   0)   # yellow — most urgent aircraft
LOS_COLOR    = (220,  30,  30)   # red    — separation violation
BLACK        = (0,     0,   0)
WHITE        = (255, 255, 255)

ACTION_LABELS = ['-20°', '-10°', '0°', '+10°', '+20°', 'back']


# ── Coordinate view ───────────────────────────────────────────────────────────

class _View:
    """Converts NM / lat-lon coordinates to pygame pixel positions."""

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
        px = int((x_nm * NM_TO_KM - self._cx) *  self._scale + self._w / 2)
        py = int((y_nm * NM_TO_KM - self._cy) * -self._scale + self._h / 2)
        return px, py

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
    parser = argparse.ArgumentParser(description='Visualise trained policy on v3_env')
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
    pygame.display.set_caption('v3_env — trained ATCO policy')
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    quit_requested = False

    for episode in range(N_EPISODES):
        obs, _      = env.reset()
        n_capacity  = env.n_aircraft   # slot capacity (fixed per episode)
        view        = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px  = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_px      = view.nm_length_to_px(CONFIG['sep_nm'] / 2)

        dest_px_by_cs = {cs: view.latlon_to_px(*env._destination_ll[cs])
                         for cs in env._active_callsigns}

        total_reward = 0.0
        last_action  = 2    # start with 'hold' label
        step = 0; los_steps = 0; done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True; done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True; done = True

            if quit_requested:
                break

            # obs is 36D ego-centric (focus aircraft + 8 neighbours)
            action, _ = model.predict(obs, deterministic=True)
            last_action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done   = terminated or truncated
            step  += 1
            if info['los_pairs']:
                los_steps += 1

            # Track destinations for newly spawned aircraft
            for cs in env._active_callsigns:
                if cs not in dest_px_by_cs and cs in env._destination_ll:
                    dest_px_by_cs[cs] = view.latlon_to_px(*env._destination_ll[cs])

            # ── Draw frame ──────────────────────────────────────────────────

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, 2)

            los_aircraft = {cs for pair in bs.traf.cd.lospairs for cs in pair}
            focus_cs     = env._focus_cs

            for slot_i, callsign in enumerate(env._slots):
                if callsign is None or callsign not in env._active_callsigns:
                    continue
                idx = bs.traf.id2idx(callsign)
                if idx < 0:
                    continue

                px, py    = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                is_focus  = callsign == focus_cs
                is_los    = callsign in los_aircraft

                # Priority: LoS > focus > normal slot colour
                color  = LOS_COLOR if is_los else (FOCUS_COLOR if is_focus else SLOT_COLORS[slot_i % len(SLOT_COLORS)])
                ring_r = sep_px + 4 if is_focus else sep_px
                width  = 2 if is_focus else 1

                pygame.draw.circle(screen, color, (px, py), ring_r, width)
                if callsign in dest_px_by_cs:
                    draw_dashed_line(screen, color, (px, py), dest_px_by_cs[callsign])
                pygame.draw.circle(screen, color, (px, py), 5)

                label = callsign
                if is_focus:
                    label += f' {ACTION_LABELS[last_action]} ◄'
                screen.blit(font_sm.render(label, True, color), (px + 7, py - 7))

            n_active = len(env._active_callsigns)
            hud_lines = [
                f'Episode {episode+1}/{N_EPISODES}  (active={n_active}/{n_capacity})  [trained ATCO policy]',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Focus: {focus_cs}   served={env._next_callsign_id}   active={n_active}',
                f'Total reward={total_reward:.1f}   last action={ACTION_LABELS[last_action]}',
            ]
            for j, line in enumerate(hud_lines):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            active_slots = [(i, cs) for i, cs in enumerate(env._slots) if cs is not None]
            legend_y = WINDOW_SIZE - 16 - len(active_slots) * 14
            for row, (i, cs) in enumerate(active_slots):
                screen.blit(font_sm.render(f'● Slot {i}: {cs}', True, SLOT_COLORS[i % len(SLOT_COLORS)]),
                            (8, legend_y + row * 14))
            screen.blit(font_sm.render('● YELLOW = focus aircraft (receiving instruction)', True, FOCUS_COLOR),
                        (8, WINDOW_SIZE - 26))
            screen.blit(font_sm.render('● RED = separation violation', True, LOS_COLOR),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(FPS)

        print(f'Episode {episode+1:2d}  capacity={n_capacity}  served={env._next_callsign_id}  '
              f'steps={step:3d}  total_reward={total_reward:.2f}  LoS-steps={los_steps}')

        if quit_requested:
            break

    pygame.quit()


if __name__ == '__main__':
    main()
