"""
make_gifs.py — v3 sector, no policy, pure analytical conflict detection.

All conflict detection is computed from scratch every frame using only
aircraft positions and velocities — no BlueSky CD module used at all.

Colour scheme:
  CYAN   — most urgent aircraft (always one aircraft, even when no conflicts)
  ORANGE — any aircraft involved in a predicted or active conflict
  GREY   — no conflict

Output: v3_no_policy.gif
Run:    python make_gifs.py
"""

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
import Environments.v3_simple_env as _v3_module
from Environments.v3_simple_env import AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm

# ── Settings ──────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
OUTPUT      = os.path.join(HERE, 'v3_simple_no_policy.gif')
WINDOW_SIZE = 700
FRAME_MS    = 100
MAX_FRAMES  = 1200          # 1200 × 100 ms = 2 minutes of GIF playback

CONFIG['density_km2']           = lambda: 4_000.0
CONFIG['max_agents']            = 20
CONFIG['crossings_per_episode'] = 8

SEP_NM  = float(CONFIG['sep_nm'])       # 5.0 NM
HORIZON = float(CONFIG['lookahead_s'])  # 600 s

# ── Colours ───────────────────────────────────────────────────────────────────

GREY   = (160, 160, 160)
CYAN   = (  0, 210, 255)
ORANGE = (255, 140,   0)
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)


# ── View ──────────────────────────────────────────────────────────────────────

class _View:
    def __init__(self, polygon_nm, w, h):
        pad  = SEP_NM * NM_TO_KM * 2.0
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
        return self.nm_to_px(nm[0], nm[1])

    def nm_to_px_len(self, nm):
        return max(1, int(nm * NM_TO_KM * self._sc))


# ── Geometry helpers (no BlueSky CD) ─────────────────────────────────────────

def _pos_nm(idx):
    """Aircraft position as [east_nm, north_nm]."""
    return latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])


def _vel_nm_s(idx):
    """Ground speed as (v_north_nm_s, v_east_nm_s)."""
    spd = bs.traf.gs[idx] / 1852.0          # m/s  →  NM/s
    hdg = math.radians(bs.traf.hdg[idx])    # heading in radians
    return spd * math.cos(hdg), spd * math.sin(hdg)   # (vn, ve)


def _pair_urgency(idx1, idx2):
    """
    Return urgency score for one pair:
      dist < SEP_NM          → 10   (active LoS)
      dCPA < SEP_NM, tCPA<H  → 1/max(tCPA, 1)   (predicted conflict)
      otherwise              → 0
    """
    p1 = _pos_nm(idx1)
    p2 = _pos_nm(idx2)
    dx = p2[0] - p1[0]    # east  difference (NM)
    dy = p2[1] - p1[1]    # north difference (NM)
    dist_sq = dx*dx + dy*dy

    # Active LoS
    if dist_sq < SEP_NM * SEP_NM:
        return 10.0

    # Relative velocity
    vn1, ve1 = _vel_nm_s(idx1)
    vn2, ve2 = _vel_nm_s(idx2)
    dvn = vn2 - vn1
    dve = ve2 - ve1
    rel_v_sq = dvn*dvn + dve*dve

    if rel_v_sq < 1e-12:
        return 0.0   # no relative motion — trajectories never cross

    # Time to CPA:  tCPA = -(r · v_rel) / |v_rel|²
    dot_rv = dx * dve + dy * dvn
    tcpa   = -dot_rv / rel_v_sq

    if tcpa < 0 or tcpa > HORIZON:
        return 0.0   # already past CPA, or too far in the future

    # Distance at CPA
    dcpa_sq = max(0.0, dist_sq - dot_rv * dot_rv / rel_v_sq)
    if dcpa_sq >= SEP_NM * SEP_NM:
        return 0.0   # will miss

    return 1.0 / max(tcpa, 1.0)


# ── Urgency matrix ────────────────────────────────────────────────────────────

def compute_urgency(active_callsigns, destination_ll):
    """
    Build N×N urgency matrix fresh from current geometry.
    Also returns the focus callsign and the set of aircraft in conflict.
    """
    cs_list = [cs for cs in active_callsigns if bs.traf.id2idx(cs) >= 0]
    n = len(cs_list)
    if n == 0:
        return np.zeros((0, 0)), [], None, set()

    bs_idx = [bs.traf.id2idx(cs) for cs in cs_list]
    U = np.zeros((n, n))

    for ii in range(n):
        for jj in range(ii + 1, n):
            score = _pair_urgency(bs_idx[ii], bs_idx[jj])
            U[ii, jj] = U[jj, ii] = score

    row_max  = U.max(axis=1)
    conf_set = {cs_list[i] for i in range(n) if row_max[i] > 0}

    if row_max.max() > 0:
        focus_cs = cs_list[int(np.argmax(row_max))]
    else:
        # No conflicts — pick the aircraft most off-course as fallback
        focus_cs = _most_off_course(cs_list, bs_idx, destination_ll)

    return U, cs_list, focus_cs, conf_set


def _most_off_course(cs_list, bs_idx, destination_ll):
    """Return the callsign with the largest heading deviation from its destination."""
    best_cs, best_drift = None, -1.0
    for cs, idx in zip(cs_list, bs_idx):
        dest_ll = destination_ll.get(cs)
        if dest_ll is None:
            continue
        dest_nm = latlon_to_nm(CONFIG['center_ll'], float(dest_ll[0]), float(dest_ll[1]))
        pos_nm  = _pos_nm(idx)
        dx = dest_nm[0] - pos_nm[0]
        dy = dest_nm[1] - pos_nm[1]
        if dx == 0 and dy == 0:
            continue
        bearing = math.degrees(math.atan2(dx, dy)) % 360   # atan2(east, north)
        diff    = (bearing - bs.traf.hdg[idx] + 180) % 360 - 180
        drift   = (1 - math.cos(math.radians(diff))) / 2
        if drift > best_drift:
            best_drift, best_cs = drift, cs
    return best_cs


# ── Drawing ───────────────────────────────────────────────────────────────────

def _dashed_line(surface, color, p1, p2, dash=8, gap=5):
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


def draw_frame(screen, font, font_sm, env, view, dest_px,
               U, cs_list, focus_cs, conf_set, los_steps):
    screen.fill(WHITE)

    polygon_px = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
    pygame.draw.polygon(screen, BLACK, polygon_px, 2)

    sep_px  = view.nm_to_px_len(SEP_NM / 2)
    idx_map = {cs: i for i, cs in enumerate(cs_list)}

    for cs in list(env._active_callsigns):
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            continue
        px, py   = view.ll_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
        is_focus = (cs == focus_cs)

        color  = CYAN if is_focus else (ORANGE if cs in conf_set else GREY)
        ring_r = sep_px + 4 if is_focus else sep_px

        pygame.draw.circle(screen, color, (px, py), ring_r, 2 if is_focus else 1)
        if cs in dest_px:
            _dashed_line(screen, color, (px, py), dest_px[cs])
        pygame.draw.circle(screen, color, (px, py), 5)

        u_val = float(U[idx_map[cs]].max()) if cs in idx_map else 0.0
        u_str = f' u={u_val:.2f}' if u_val > 0 else ''
        lbl   = f'{cs}{u_str} ◄' if is_focus else f'{cs}{u_str}'
        screen.blit(font_sm.render(lbl, True, color), (px + 7, py - 7))

    n_active = len(env._active_callsigns)
    for j, line in enumerate([
        'v3_simple — no policy   80 000 km²   450 kts',
        f'T={bs.sim.simt:.0f}s   active={n_active}/{env.n_aircraft}   '
        f'LoS={los_steps}   served={env._next_callsign_id}',
        f'focus={focus_cs}   conflicts={len(conf_set)}',
    ]):
        screen.blit(font.render(line, True, BLACK), (8, 8 + j * 15))

    screen.blit(font_sm.render('● CYAN   = most urgent / most off-course', True, CYAN),   (8, WINDOW_SIZE - 40))
    screen.blit(font_sm.render('● ORANGE = predicted or active conflict',   True, ORANGE), (8, WINDOW_SIZE - 26))
    screen.blit(font_sm.render('● GREY   = no conflict',                    True, GREY),  (8, WINDOW_SIZE - 12))


def capture(surface):
    return Image.fromarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    font    = pygame.font.SysFont('monospace', 12)
    font_sm = pygame.font.SysFont('monospace', 10)

    env    = AirspaceEnv()
    obs, _ = env.reset()
    view   = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
    dest_px = {cs: view.ll_to_px(*env._destination_ll[cs])
               for cs in env._active_callsigns}

    frames    = []
    los_steps = 0

    print(f'Recording {MAX_FRAMES} frames (~2 min GIF)  '
          f'(n_aircraft={env.n_aircraft}, max_steps={env._max_steps})')

    while len(frames) < MAX_FRAMES:
        _, _, terminated, truncated, info = env.step(2)   # 2 = hold (0° delta) — no policy

        if info.get('los_pairs'):
            los_steps += 1

        for cs in env._active_callsigns:
            if cs not in dest_px and cs in env._destination_ll:
                dest_px[cs] = view.ll_to_px(*env._destination_ll[cs])

        # Compute urgency entirely from our own geometry
        U, cs_list, focus_cs, conf_set = compute_urgency(
            env._active_callsigns, env._destination_ll
        )

        draw_frame(screen, font, font_sm, env, view, dest_px,
                   U, cs_list, focus_cs, conf_set, los_steps)
        frames.append(capture(screen))

        n = len(frames)
        if n % 100 == 0:
            print(f'  frame {n:4d}/{MAX_FRAMES}  T={bs.sim.simt:.0f}s  '
                  f'active={len(env._active_callsigns)}  '
                  f'conflicts={len(conf_set)}  focus={focus_cs}')

        if terminated or truncated:
            print('  Episode ended — resetting')
            obs, _ = env.reset()
            view    = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
            dest_px = {cs: view.ll_to_px(*env._destination_ll[cs])
                       for cs in env._active_callsigns}
            los_steps = 0

    print(f'\nSaving {len(frames)} frames → {OUTPUT}')
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=False)
    print(f'Done  ({os.path.getsize(OUTPUT)/1e6:.1f} MB)')
    pygame.quit()


if __name__ == '__main__':
    main()
