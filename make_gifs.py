"""
make_gifs.py — v3 sector visualisation, no policy (aircraft fly direct to destination).

Urgency is taken directly from env._urgency_matrix — no separate computation.
Colour scheme reflects the two ATCO objectives:

  Objective 1 — Safety  (urgency)
    RED    u > 1.0   active loss of separation  (dist < 5 NM)
    ORANGE 0 < u ≤ 1 predicted conflict within 10-min horizon
    GREY               no conflict

  Objective 2 — Efficiency  (drift)
    Circle ring size scales with heading deviation from destination bearing.

  CYAN outline — most urgent aircraft (focus), would receive instruction first.

Output: v3_no_policy.gif
Run:    python make_gifs.py
"""

import argparse
import math
import os
import sys

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pygame
from PIL import Image

import bluesky as bs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Environments.v3_improved import AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm, wrap_to_180

# ── Settings ──────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_NO_POLICY = os.path.join(HERE, 'v3_no_policy.gif')
OUTPUT_POLICY    = os.path.join(HERE, 'v3_policy.gif')
WINDOW_SIZE = 750
FRAME_MS    = 120                # ms per GIF frame
MAX_FRAMES  = 400               # 400 × 120 ms = 48 s of GIF

SEP_NM = float(CONFIG['sep_nm'])

# ── Colours ───────────────────────────────────────────────────────────────────

BLACK      = (  0,   0,   0)
WHITE      = (255, 255, 255)
GREY       = (160, 160, 160)   # no conflict
ORANGE     = (255, 140,   0)   # predicted conflict
RED        = (220,  40,  40)   # active LoS
CYAN       = (  0, 210, 255)   # focus aircraft outline
GREEN_DARK = ( 40, 160,  80)   # on-track label accent


# ── View ──────────────────────────────────────────────────────────────────────

class _View:
    def __init__(self, polygon_nm, w, h):
        pad  = SEP_NM * NM_TO_KM * 2.5
        km   = polygon_nm * NM_TO_KM
        span = max(km[:, 0].max() - km[:, 0].min() + 2 * pad,
                   km[:, 1].max() - km[:, 1].min() + 2 * pad)
        self._cx = (km[:, 0].min() + km[:, 0].max()) / 2
        self._cy = (km[:, 1].min() + km[:, 1].max()) / 2
        self._sc = min(w, h) / span
        self._w, self._h = w, h

    def nm_to_px(self, x_nm, y_nm):
        return (int((x_nm * NM_TO_KM - self._cx) *  self._sc + self._w / 2),
                int((y_nm * NM_TO_KM - self._cy) * -self._sc + self._h / 2))

    def ll_to_px(self, lat, lon):
        nm = latlon_to_nm(CONFIG['center_ll'], lat, lon)
        return self.nm_to_px(float(nm[0]), float(nm[1]))

    def nm_to_px_len(self, nm):
        return max(1, int(nm * NM_TO_KM * self._sc))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _dashed_line(surface, color, p1, p2, dash=7, gap=5):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    total  = math.hypot(dx, dy)
    if total < 1:
        return
    ux, uy = dx / total, dy / total
    x, y   = float(p1[0]), float(p1[1])
    drawn  = 0.0; on = True
    while drawn < total:
        seg = min(dash if on else gap, total - drawn)
        if on:
            pygame.draw.line(surface, color,
                             (round(x), round(y)),
                             (round(x + ux*seg), round(y + uy*seg)), 1)
        x += ux*seg; y += uy*seg; drawn += seg; on = not on


def _urgency_color(u_val):
    """Map urgency score to a display colour."""
    if u_val > 1.0:
        return RED     # active LoS
    if u_val > 0.0:
        return ORANGE  # predicted conflict
    return GREY        # clear


def _blend(c1, c2, t):
    """Linear blend between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ── Frame renderer ────────────────────────────────────────────────────────────

def draw_frame(screen, font, font_sm, env, view, dest_px, los_steps, frame_no):
    screen.fill(WHITE)

    # Sector polygon
    polygon_px = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
    pygame.draw.polygon(screen, (200, 200, 200), polygon_px, 0)   # light fill
    pygame.draw.polygon(screen, BLACK,           polygon_px, 2)

    U       = env._urgency_matrix           # (n, n) urgency matrix from env
    cs_list = env._urgency_cs_list          # matching callsign list

    # Per-aircraft urgency (row-max)
    urg_by_cs = {}
    if U.size > 0:
        row_max = U.max(axis=1)
        for i, cs in enumerate(cs_list):
            urg_by_cs[cs] = float(row_max[i])

    focus_cs = env._focus_cs
    sep_px   = view.nm_to_px_len(SEP_NM / 2)

    for cs in list(env._active_callsigns):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            continue

        lat, lon = bs.traf.lat[idx], bs.traf.lon[idx]
        px, py   = view.ll_to_px(lat, lon)
        hdg_rad  = math.radians(bs.traf.hdg[idx])
        spd_px   = view.nm_to_px_len(bs.traf.gs[idx] / 1852.0 * 90)  # 90s velocity vector

        # Velocity vector
        vx =  math.sin(hdg_rad) * spd_px
        vy = -math.cos(hdg_rad) * spd_px

        u_val  = urg_by_cs.get(cs, 0.0)
        color  = _urgency_color(u_val)

        # Heading deviation from destination → ring thickness reflects efficiency obj
        dest_ll = env._destination_ll.get(cs)
        drift_ring = sep_px
        drift_deg  = 0.0
        if dest_ll is not None:
            dest_bearing, _ = bs.tools.geo.kwikqdrdist(lat, lon,
                                                       float(dest_ll[0]), float(dest_ll[1]))
            cmd_hdg = env._commanded_heading.get(cs, bs.traf.hdg[idx])
            drift_deg = abs(wrap_to_180(dest_bearing - cmd_hdg))
            # Ring thickens when off-track (max 5 px extra at 30° deviation)
            drift_ring = sep_px + int(min(drift_deg / 30.0, 1.0) * 5)

        # Draw: destination line → velocity vector → separation ring → dot
        if cs in dest_px:
            _dashed_line(screen, _blend(color, WHITE, 0.5), (px, py), dest_px[cs])

        pygame.draw.line(screen, color,
                         (px, py), (int(px + vx), int(py + vy)), 2)
        pygame.draw.polygon(screen, color, [
            (int(px + vx), int(py + vy)),
            (int(px + vx - vy * 0.15), int(py + vy + vx * 0.15)),
            (int(px + vx + vy * 0.15), int(py + vy - vx * 0.15)),
        ])

        ring_width = 3 if cs == focus_cs else 1
        pygame.draw.circle(screen, color, (px, py), drift_ring, ring_width)

        if cs == focus_cs:
            pygame.draw.circle(screen, CYAN, (px, py), drift_ring + 3, 2)

        pygame.draw.circle(screen, color, (px, py), 5)

        # Label: callsign + urgency score + drift
        u_str  = f' u={u_val:.2f}' if u_val > 0 else ''
        d_str  = f' {drift_deg:.0f}°off' if drift_deg > 2 else ''
        label  = f'{cs}{u_str}{d_str}' + (' ◄' if cs == focus_cs else '')
        screen.blit(font_sm.render(label, True, color), (px + 7, py - 7))

    # ── HUD ────────────────────────────────────────────────────────────────────

    n_active  = len(env._active_callsigns)
    n_los     = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
    n_conf    = int((U > 0).sum())  // 2 if U.size > 0 else 0
    urg_sum   = float(U.sum()) / 2.0 if U.size > 0 else 0.0

    # Efficiency: mean drift of active aircraft
    drifts = []
    for cs in env._active_callsigns:
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            continue
        dest_ll = env._destination_ll.get(cs)
        if dest_ll is None:
            continue
        dest_bearing, _ = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[idx], bs.traf.lon[idx],
            float(dest_ll[0]), float(dest_ll[1]),
        )
        cmd_hdg = env._commanded_heading.get(cs, bs.traf.hdg[idx])
        drifts.append(abs(wrap_to_180(dest_bearing - cmd_hdg)))
    mean_drift = sum(drifts) / max(len(drifts), 1)

    hud = [
        ('v3 · no policy · 80 000 km² · 450 kts', BLACK),
        (f'T={bs.sim.simt:.0f}s   active={n_active}/{env.n_aircraft}   '
         f'served={env._next_callsign_id}   frame={frame_no}', BLACK),
        (f'[OBJ 1 SAFETY]   LoS pairs={n_los}   conflict pairs={n_conf}'
         f'   Σurgency={urg_sum:.2f}   LoS-steps={los_steps}', RED if n_los > 0 else BLACK),
        (f'[OBJ 2 EFFIC.]   mean drift={mean_drift:.1f}°   '
         f'focus={focus_cs}', GREEN_DARK),
    ]
    for j, (line, col) in enumerate(hud):
        screen.blit(font.render(line, True, col), (8, 8 + j * 15))

    # Legend
    legend = [
        ('● RED    active LoS  (dist < 5 NM)',   RED),
        ('● ORANGE predicted conflict (≤10 min)', ORANGE),
        ('● GREY   no conflict',                  GREY),
        ('◯ CYAN   focus aircraft (most urgent)', CYAN),
    ]
    for j, (txt, col) in enumerate(legend):
        screen.blit(font_sm.render(txt, True, col),
                    (8, WINDOW_SIZE - 10 - (len(legend) - j) * 14))


def capture(surface):
    return Image.fromarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default=None, help='Path to model .zip')
    parser.add_argument('--frames', type=int, default=MAX_FRAMES,
                        help='Number of GIF frames (default %(default)s)')
    parser.add_argument('--fps',    type=int, default=None,
                        help='Override frame delay in ms')
    args = parser.parse_args()
    max_frames = args.frames
    frame_ms   = args.fps if args.fps else FRAME_MS

    # ── Load policy if provided ────────────────────────────────────────────────
    model   = None
    vecnorm = None
    OUTPUT  = OUTPUT_NO_POLICY

    if args.model:
        model_path = args.model
        vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl')
        print(f'Loading model   : {model_path}')
        model = PPO.load(model_path)
        if os.path.exists(vecnorm_path):
            print(f'Loading vecnorm : {vecnorm_path}')
            dummy = DummyVecEnv([AirspaceEnv])
            vecnorm = VecNormalize.load(vecnorm_path, dummy)
            vecnorm.training    = False
            vecnorm.norm_reward = False
        OUTPUT = OUTPUT_POLICY

    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    font    = pygame.font.SysFont('monospace', 11)
    font_sm = pygame.font.SysFont('monospace', 10)

    env    = AirspaceEnv()
    obs, _ = env.reset()
    view   = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
    dest_px = {cs: view.ll_to_px(*env._destination_ll[cs])
               for cs in env._active_callsigns}

    frames    = []
    los_steps = 0

    mode_str = f'policy: {os.path.basename(args.model)}' if model else 'no policy'
    print(f'Recording up to {max_frames} frames  '
          f'(n_aircraft={env.n_aircraft}, max_steps={env._max_steps})')
    print(f'Mode: {mode_str}\n')

    while len(frames) < max_frames:
        if model is not None:
            norm_obs = vecnorm.normalize_obs(obs[None])[0] if vecnorm else obs
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
        else:
            obs, _, terminated, truncated, info = env.step(2)

        if info.get('los_pairs'):
            los_steps += 1

        for cs in env._active_callsigns:
            if cs not in dest_px and cs in env._destination_ll:
                dest_px[cs] = view.ll_to_px(*env._destination_ll[cs])

        draw_frame(screen, font, font_sm, env, view, dest_px, los_steps, len(frames))
        frames.append(capture(screen))

        n = len(frames)
        if n % 50 == 0:
            U      = env._urgency_matrix
            n_los  = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
            n_conf = int((U > 0).sum())   // 2 if U.size > 0 else 0
            print(f'  frame {n:4d}/{max_frames}  T={bs.sim.simt:.0f}s  '
                  f'active={len(env._active_callsigns)}  '
                  f'LoS={n_los}  conf={n_conf}  focus={env._focus_cs}')

        if terminated or truncated:
            print('  Episode ended -- resetting')
            obs, _ = env.reset()
            view    = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
            dest_px = {cs: view.ll_to_px(*env._destination_ll[cs])
                       for cs in env._active_callsigns}
            los_steps = 0

    print(f'\nSaving {len(frames)} frames -> {OUTPUT}')
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=False)
    print(f'Done  ({os.path.getsize(OUTPUT)/1e6:.1f} MB)')
    pygame.quit()


if __name__ == '__main__':
    main()
