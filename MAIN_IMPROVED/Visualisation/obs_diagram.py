"""Draw the observation space the way the policy receives it, to check it against a figure.

Everything on the radar is reconstructed from the 27-number observation vector alone -- the
env's internal positions are never read -- so whatever lines up is a real confirmation and not
a redraw of the same state twice.

The frame is the ego frame the observation is written in: the focus ship sits at the centre with
its CURRENT heading pointing up, theta is the bearing clockwise from that, and every intruder's
blue reference arrow is parallel to the ownship heading, so psi is the angle from it.

    python visualisation/obs_diagram.py                     interactive
    python visualisation/obs_diagram.py --save obs.png      render one frame and exit

SPACE pause/run, N single step, R new episode, S save a PNG, ESC/Q quit.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame

from Environment import (AirspaceEnv, CONFIG, OBS_OWNSHIP_LABELS, OBS_INTRUDER_LABELS)
from Environment.config import (N_NEIGHBOURS, EMPTY_RANGE_NM, NO_CONFLICT_S,
                                      KT_PER_MACH)

WIN_W, WIN_H = 1500, 980
CX, CY       = 470, 500          # radar centre
PANEL_X      = 960

BG      = (250, 249, 246)
INK     = (26, 26, 26)
GREY    = (150, 150, 150)
LGREY   = (205, 205, 205)
BLUE    = (74, 144, 226)         # the reference arrow / ownship
OWN_FILL = (150, 200, 240)
ORANGE  = (240, 168, 60)         # intruder with a predicted LoS
GREEN   = (124, 190, 120)        # intruder clear of the horizon
EMPTY   = (200, 200, 200)
PANEL_BG = (243, 242, 238)

TWO_PI = 2.0 * math.pi


# -- geometry helpers ----------------------------------------------------------

def to_screen(bearing_rad, dist_nm, scale):
    """Ego frame: bearing measured clockwise from straight up."""
    return (CX + dist_nm * scale * math.sin(bearing_rad),
            CY - dist_nm * scale * math.cos(bearing_rad))


def bearing_point(cx, cy, bearing_rad, radius_px):
    return (cx + radius_px * math.sin(bearing_rad),
            cy - radius_px * math.cos(bearing_rad))


def arc_points(cx, cy, radius, b_from, b_to, steps=48):
    """Sampled arc between two bearings; drawn by hand so the sweep direction is explicit."""
    return [bearing_point(cx, cy, b_from + (b_to - b_from) * i / steps, radius)
            for i in range(steps + 1)]


def draw_arc(surf, colour, cx, cy, radius, b_from, b_to, width=2):
    pts = arc_points(cx, cy, radius, b_from, b_to)
    if len(pts) > 1:
        pygame.draw.lines(surf, colour, False, pts, width)
    return pts


def dashed(surf, colour, p1, p2, dash=7, gap=6, width=1):
    x1, y1 = p1
    x2, y2 = p2
    total = math.hypot(x2 - x1, y2 - y1)
    if total < 1e-6:
        return
    ux, uy = (x2 - x1) / total, (y2 - y1) / total
    travelled = 0.0
    while travelled < total:
        end = min(travelled + dash, total)
        pygame.draw.line(surf, colour,
                         (x1 + ux * travelled, y1 + uy * travelled),
                         (x1 + ux * end, y1 + uy * end), width)
        travelled = end + gap


def arrow(surf, colour, start, bearing_rad, length_px, width=3, head=10):
    tip = bearing_point(start[0], start[1], bearing_rad, length_px)
    pygame.draw.line(surf, colour, start, tip, width)
    left  = bearing_point(tip[0], tip[1], bearing_rad + math.pi - 0.42, head)
    right = bearing_point(tip[0], tip[1], bearing_rad + math.pi + 0.42, head)
    pygame.draw.polygon(surf, colour, [tip, left, right])
    return tip


def aircraft_glyph(surf, colour, cx, cy, bearing_rad, size=13):
    """A small plane silhouette pointing along `bearing_rad`."""
    def pt(forward, right):
        return (cx + forward * math.sin(bearing_rad) + right * math.cos(bearing_rad),
                cy - forward * math.cos(bearing_rad) + right * math.sin(bearing_rad))
    body = [pt(size, 0), pt(-size * 0.35, size * 0.22), pt(-size, size * 0.16),
            pt(-size * 0.75, 0), pt(-size, -size * 0.16), pt(-size * 0.35, -size * 0.22)]
    wing = [pt(size * 0.2, 0), pt(-size * 0.2, size * 0.95), pt(-size * 0.42, size * 0.95),
            pt(-size * 0.1, 0), pt(-size * 0.42, -size * 0.95), pt(-size * 0.2, -size * 0.95)]
    pygame.draw.polygon(surf, colour, wing)
    pygame.draw.polygon(surf, colour, body)


def label(surf, font, text, pos, colour=INK, centre=True, bg=None):
    img = font.render(text, True, colour)
    rect = img.get_rect()
    if centre:
        rect.center = pos
    else:
        rect.topleft = pos
    if bg is not None:
        pad = surf.subsurface(rect.inflate(6, 2).clip(surf.get_rect())) if False else None
        pygame.draw.rect(surf, bg, rect.inflate(7, 3))
    surf.blit(img, rect)
    return rect


# -- the diagram ---------------------------------------------------------------

def unpack(obs):
    """The observation vector, split exactly as the env packs it."""
    own = {lbl: float(v) for lbl, v in zip(OBS_OWNSHIP_LABELS, obs[:7])}
    intruders = []
    for k in range(N_NEIGHBOURS):
        seg = obs[7 + 5 * k: 12 + 5 * k]
        intruders.append({lbl: float(v) for lbl, v in zip(OBS_INTRUDER_LABELS, seg)})
    return own, intruders


def draw_diagram(screen, fonts, obs, intruder_cs, focus_cs):
    font, small, big, tiny = fonts
    own, intruders = unpack(obs)

    live = [d for d in intruders if d['dist'] < EMPTY_RANGE_NM - 1.0]
    span = max([d['dist'] for d in live], default=40.0) * 1.06
    radar_r = min(CX, CY) - 78
    scale = radar_r / max(span, 12.0)

    # -- range rings, only out to the furthest intruder
    ring_step = 10 if span <= 55 else (20 if span <= 110 else 25)
    r_nm = ring_step
    while r_nm <= span:
        pygame.draw.circle(screen, LGREY, (CX, CY), int(r_nm * scale), 1)
        label(screen, tiny, f'{r_nm} NM',
              bearing_point(CX, CY, math.radians(215), r_nm * scale), LGREY)
        r_nm += ring_step

    # -- ownship heading: straight up by construction, this is the frame's axis
    up = 0.0
    dashed(screen, LGREY, (CX, CY), bearing_point(CX, CY, up, min(CX, CY) - 55), 6, 6, 1)

    # -- the route ("destination"): initial heading sits at -dpsi in the ego frame
    dpsi = own['dpsi']
    b_route = -dpsi
    route_end = bearing_point(CX, CY, b_route, min(CX, CY) - 62)
    dashed(screen, GREY, (CX, CY), route_end, 8, 7, 2)
    label(screen, small, 'Destination (initial hdg)',
          bearing_point(CX, CY, b_route, min(CX, CY) - 30), GREY)

    # -- the commanded heading: h_cmd is measured from the initial heading
    b_cmd = own['h_cmd'] - dpsi
    if abs(own['h_cmd']) > 1e-6:
        cmd_end = bearing_point(CX, CY, b_cmd, 132)
        dashed(screen, (198, 120, 200), (CX, CY), cmd_end, 5, 5, 2)
        label(screen, tiny, 'h_cmd (commanded hdg)',
              bearing_point(CX, CY, b_cmd, 150), (170, 90, 175))

    # -- delta-psi: from the current heading round to the route
    if abs(dpsi) > 0.02:
        draw_arc(screen, (120, 120, 120), CX, CY, 74, up, b_route, 2)
        mid = bearing_point(CX, CY, (up + b_route) / 2, 88)
        label(screen, font, f'Δψ = {math.degrees(dpsi):+.0f}°', mid, (90, 90, 90))

    # -- intruders
    for k, d in enumerate(intruders):
        empty = d['dist'] >= EMPTY_RANGE_NM - 1.0
        if empty:
            continue
        theta = d['theta']
        conflicted = d['tlos'] < NO_CONFLICT_S - 1e-6
        colour = ORANGE if conflicted else GREEN
        ix, iy = to_screen(theta, d['dist'], scale)

        # range line, with d_k set off perpendicular to it so the two never collide
        dashed(screen, INK, (CX, CY), (ix, iy), 8, 6, 1)
        lx, ly = to_screen(theta, d['dist'] * 0.62, scale)
        off = bearing_point(lx, ly, theta + math.pi / 2, 17)
        label(screen, font, f'd{k + 1}', off, INK)
        label(screen, tiny, f'{d["dist"]:.1f} NM',
              bearing_point(lx, ly, theta + math.pi / 2, 33), (120, 120, 120))

        # the protected zone: 5 NM diameter, so 2.5 NM radius
        pz = max(11, int(0.5 * CONFIG['sep_nm'] * scale))
        pygame.draw.circle(screen, colour, (int(ix), int(iy)), pz)

        # theta_k: bearing arc from the ownship heading round to this intruder, each arc
        # riding its own range line so four of them can be told apart
        r_arc = min(max(d['dist'] * scale * 0.34, 46), 165)
        draw_arc(screen, INK, CX, CY, r_arc, up, theta, 2)
        label(screen, font, f'θ{k + 1}',
              bearing_point(CX, CY, (up + theta) / 2, r_arc + 17), INK)

        # the reference arrow: parallel to the OWNSHIP heading, which is what psi is measured from
        arrow(screen, BLUE, (ix, iy), up, 44, 3, 9)

        # the intruder's own velocity, psi from that reference
        psi = d['psi']
        aircraft_glyph(screen, INK, ix, iy, psi, 15)
        v_len = 38 + 20 * (d['vint'] / max(KT_PER_MACH, 1.0))
        v_tip = arrow(screen, INK, (ix, iy), psi, v_len, 3, 9)
        label(screen, font, f'v{k + 1}',
              bearing_point(v_tip[0], v_tip[1], psi, 15), INK)

        # psi_k arc, between the blue reference and the velocity
        draw_arc(screen, INK, ix, iy, 28, up, psi, 2)
        label(screen, font, f'ψ{k + 1}',
              bearing_point(ix, iy, (up + psi) / 2, 39), INK)

        # the name tag goes on the side away from both the velocity arrow and the range line
        cs = intruder_cs[k] if k < len(intruder_cs) else None
        tag = f'Aircraft {k + 1}' + (f'  [{cs}]' if cs else '')
        tag_bearing = psi + math.pi
        tx, ty = bearing_point(ix, iy, tag_bearing, pz + 34)
        label(screen, small, tag, (tx, ty), INK)
        if conflicted:
            label(screen, tiny, f't_los = {d["tlos"]:.0f} s', (tx, ty + 15), (190, 110, 20))

    # -- the focus ship, drawn last so it sits on top
    pz_own = max(10, int(0.5 * CONFIG['sep_nm'] * scale))
    pygame.draw.circle(screen, OWN_FILL, (CX, CY), pz_own)
    arrow(screen, BLUE, (CX, CY), up, 62, 4, 11)
    label(screen, font, 'v_own', (CX - 34, CY - 44), BLUE)
    label(screen, tiny, f'{own["v_own"]:.0f} kt', (CX - 34, CY - 29), BLUE)
    aircraft_glyph(screen, INK, CX, CY, up, 15)
    label(screen, small, f'Focus ship' + (f'  [{focus_cs}]' if focus_cs else ''),
          (CX, CY + pz_own + 18), INK)

    # -- flags that have no geometry
    y = CY + 190
    flags = [('retn_conf', own['retn_conf'], 'returning to initial heading is BLOCKED'),
             ('pending',   own['pending'],   'an advisory is with the ATCO'),
             ('wait_s',    own['wait_s'],    'seconds since it was issued')]
    label(screen, small, 'ownship features with no geometry:', (60, y), (120, 120, 120), False)
    y += 20
    for name, val, meaning in flags:
        on = val > 0.5 if name != 'wait_s' else val > 0
        col = ORANGE if on else (150, 150, 150)
        txt = f'{name} = {val:.0f}' if name != 'wait_s' else f'{name} = {val:.0f}'
        label(screen, small, f'{txt:<16} {meaning if on else ""}', (60, y), col, False)
        y += 18

    # empty slots
    n_empty = sum(1 for d in intruders if d['dist'] >= EMPTY_RANGE_NM - 1.0)
    if n_empty:
        label(screen, small,
              f'{n_empty} of {N_NEIGHBOURS} intruder slots empty '
              f'(padded {EMPTY_RANGE_NM:.0f} NM, {NO_CONFLICT_S:.0f} s)',
              (60, y + 6), (150, 150, 150), False)


def draw_panel(screen, fonts, obs, intruder_cs, focus_cs, step_n, n_ac):
    font, small, big, tiny = fonts
    pygame.draw.rect(screen, PANEL_BG, (PANEL_X, 0, WIN_W - PANEL_X, WIN_H))
    pygame.draw.line(screen, LGREY, (PANEL_X, 0), (PANEL_X, WIN_H), 1)
    x, y = PANEL_X + 22, 22
    label(screen, big, 'OBSERVATION VECTOR', (x, y), INK, False); y += 30
    label(screen, small, f'27 raw values  |  step {step_n}  |  {n_ac} aircraft airborne',
          (x, y), (120, 120, 120), False); y += 30

    own, intruders = unpack(obs)
    label(screen, font, f'ownship  [{focus_cs or "--"}]', (x, y), BLUE, False); y += 22
    units = {'dpsi': 'rad', 'v_own': 'kt', 'h_cmd': 'rad', 'v_cmd': 'kt',
             'retn_conf': '', 'pending': '', 'wait_s': 's'}
    drawn = {'dpsi': 'shown as Δψ', 'v_own': 'shown as v_own', 'h_cmd': 'dashed violet',
             'v_cmd': 'not drawn', 'retn_conf': 'flag', 'pending': 'flag', 'wait_s': 'flag'}
    for i, lbl in enumerate(OBS_OWNSHIP_LABELS):
        v = own[lbl]
        label(screen, small, f'[{i:>2}] {lbl:<10} {v:>9.3f} {units[lbl]:<4} {drawn[lbl]}',
              (x + 8, y), INK, False)
        y += 17
    y += 12

    for k, d in enumerate(intruders):
        empty = d['dist'] >= EMPTY_RANGE_NM - 1.0
        cs = intruder_cs[k] if k < len(intruder_cs) else None
        head = f'intruder {k + 1}' + ('  (empty slot)' if empty else f'  [{cs}]')
        conflicted = (not empty) and d['tlos'] < NO_CONFLICT_S - 1e-6
        col = EMPTY if empty else (ORANGE if conflicted else GREEN)
        label(screen, font, head, (x, y), col, False); y += 20
        for j, lbl in enumerate(OBS_INTRUDER_LABELS):
            v = d[lbl]
            idx = 7 + 5 * k + j
            u = {'dist': 'NM', 'theta': 'rad', 'psi': 'rad', 'vint': 'kt', 'tlos': 's'}[lbl]
            extra = ''
            if lbl in ('theta', 'psi'):
                extra = f'= {math.degrees(v):+7.1f}°'
            label(screen, small, f'[{idx:>2}] {lbl:<6} {v:>9.3f} {u:<3} {extra}',
                  (x + 8, y), EMPTY if empty else INK, False)
            y += 17
        y += 10

    y = WIN_H - 92
    label(screen, small, 'SPACE run/pause   N step   R new episode   S save PNG   ESC quit',
          (x, y), (120, 120, 120), False)
    y += 20
    label(screen, small, 'orange = t_los inside the horizon; green = clear',
          (x, y), (120, 120, 120), False)
    y += 18
    label(screen, small, 'blue arrows are all parallel to the ownship heading',
          (x, y), (120, 120, 120), False)


def render(screen, fonts, obs, intruder_cs, focus_cs, step_n, n_ac):
    screen.fill(BG)
    draw_diagram(screen, fonts, obs, intruder_cs, focus_cs)
    draw_panel(screen, fonts, obs, intruder_cs, focus_cs, step_n, n_ac)


# -- driving the environment ---------------------------------------------------

def interesting(env, obs, want_intruders=4, want_conflict=True):
    """A frame that exercises every annotation: full slots, a predicted LoS, and some drift
    off the route so the delta-psi arc is not degenerate."""
    own, intruders = unpack(obs)
    live = [d for d in intruders if d['dist'] < EMPTY_RANGE_NM - 1.0]
    if len(live) < want_intruders:
        return False
    if want_conflict and not any(d['tlos'] < NO_CONFLICT_S - 1e-6 for d in live):
        return False
    if abs(own['dpsi']) < math.radians(8):
        return False
    # keep the four spread out, so the theta arcs do not sit on top of one another
    bearings = sorted(d['theta'] for d in live)
    if any(abs(b - a) < math.radians(22) for a, b in zip(bearings, bearings[1:])):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=3, help='scenario seed')
    ap.add_argument('--delay', default='none', help='delay mode for the ATCO')
    ap.add_argument('--save', default=None, help='render one frame to this PNG and exit')
    ap.add_argument('--seek', type=int, default=400,
                    help='steps to search for a frame with full slots and a predicted conflict')
    ap.add_argument('--fps', type=int, default=3, help='steps per second when running')
    args = ap.parse_args()

    if args.save:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    env = AirspaceEnv(delay_mode=args.delay, seed=args.seed)
    obs, _ = env.reset(options={'scenario_seed': args.seed})
    rng = np.random.default_rng(args.seed)

    def seek(obs):
        """Fly until a frame shows every annotation. Actions are random rather than HOLD:
        with HOLD nothing ever turns, so dpsi stays 0 and the delta-psi arc never appears."""
        for _ in range(args.seek):
            if interesting(env, obs):
                return obs
            obs, _, _, trunc, _ = env.step(int(rng.integers(0, 10)))
            if trunc:
                obs, _ = env.reset()
        return obs

    obs = seek(obs)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Observation space -- ego frame')
    fonts = (pygame.font.SysFont('dejavusans,arial', 17),
             pygame.font.SysFont('dejavusans,arial', 14),
             pygame.font.SysFont('dejavusans,arial', 20, bold=True),
             pygame.font.SysFont('dejavusans,arial', 12))

    step_n = 0
    render(screen, fonts, obs, env._last_intruder_cs, env.focus_cs, step_n,
           len(env._urgency_cs_list))

    if args.save:
        pygame.image.save(screen, args.save)
        print('wrote', args.save)
        return

    clock = pygame.time.Clock()
    running, paused = True, True
    while running:
        advance = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_n:
                    advance = True
                elif ev.key == pygame.K_s:
                    path = f'obs_diagram_{step_n:04d}.png'
                    pygame.image.save(screen, path)
                    print('wrote', path)
                elif ev.key == pygame.K_r:
                    obs, _ = env.reset()
                    obs = seek(obs)
                    step_n = 0

        if advance or not paused:
            obs, _, _, trunc, _ = env.step(int(rng.integers(0, 10)))
            step_n += 1
            if trunc:
                obs, _ = env.reset()
                step_n = 0

        render(screen, fonts, obs, env._last_intruder_cs, env.focus_cs, step_n,
               len(env._urgency_cs_list))
        pygame.display.flip()
        clock.tick(args.fps if not paused else 30)

    pygame.quit()


if __name__ == '__main__':
    main()
