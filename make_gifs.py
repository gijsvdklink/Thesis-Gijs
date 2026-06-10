"""
make_gifs.py — v3_improved sector visualisation, policy or no-policy.

Colour scheme reflects the two ATCO objectives:

  Objective 1 — Safety  (urgency)
    RED    u > 1.0   active loss of separation  (dist < 5 NM)
    ORANGE 0 < u ≤ 1 predicted conflict within 10-min horizon
    GREY               no conflict

  Objective 2 — Efficiency  (drift)
    Circle ring size scales with heading deviation from destination bearing.

  CYAN outline — focus aircraft (would receive next instruction).

Run:
    python make_gifs.py                        # no policy, random episode
    python make_gifs.py --model path/to/ckpt.zip
"""

import argparse
import io
import math
import os
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pygame
import cv2
from PIL import Image

# ── NumPy compatibility shim ──────────────────────────────────────────────────
# NumPy 2.x renamed numpy._core → numpy.core; this lets old .zip/.pkl files load.

class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

from stable_baselines3.common import save_util

def _patched_json_to_data(json_string, custom_objects=None):
    import json, base64, warnings
    data = {}
    for key, item in json.loads(json_string).items():
        if custom_objects and key in custom_objects:
            data[key] = custom_objects[key]
        elif isinstance(item, dict) and ':serialized:' in item:
            try:
                raw = base64.b64decode(item[':serialized:'].encode())
                data[key] = _NumpyShim(io.BytesIO(raw)).load()
            except Exception as e:
                warnings.warn(f'Could not deserialise {key}: {e}')
        else:
            data[key] = item
    return data

save_util.json_to_data = _patched_json_to_data

# ─────────────────────────────────────────────────────────────────────────────

import bluesky as bs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Environments.v3 import AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm, wrap_to_180

# Default checkpoint to visualise
DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Runs_saved', 'v3',
    'dalmau_8act_obs23_seed18592_20260606_132427',
    'checkpoints', 'ckpt_700000.zip',
)


# ── Settings ──────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 750
FRAME_MS    = 120                # ms per GIF frame
MAX_FRAMES  = 1500               # safety cap — episode ends naturally first

SEP_NM = float(CONFIG['sep_nm'])

# ── Colours ───────────────────────────────────────────────────────────────────

BLACK      = (  0,   0,   0)
WHITE      = (255, 255, 255)
GREY       = (160, 160, 160)   # no conflict
ORANGE     = (255, 140,   0)   # predicted conflict
RED        = (220,  40,  40)   # active LoS
CYAN       = (  0, 210, 255)   # focus aircraft outline
GREEN_DARK = ( 40, 160,  80)   # on-track label accent
GREEN      = ( 50, 200,  80)   # positive reward
BLUE       = ( 80, 140, 220)   # neutral / info
PURPLE     = (160,  60, 200)   # emergency-approaching aircraft
YELLOW     = (220, 200,   0)   # cooldown state


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

def draw_frame(screen, font, font_sm, env, view, dest_px, los_steps, frame_no,
               mode_str='', step_reward=0.0, cum_reward=0.0):
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

    # ── Focus lock state ───────────────────────────────────────────────────────
    clear_steps = CONFIG['focus_clear_steps']
    emerg_u     = CONFIG['focus_emergency_u']
    focus_cur_u = 0.0
    focus_steps_clear = clear_steps
    if focus_cs and focus_cs in cs_list:
        fi = cs_list.index(focus_cs)
        focus_cur_u       = float(U[fi].max()) if U.size > 0 else 0.0
        focus_steps_clear = env._steps_since_urgency.get(focus_cs, clear_steps)

    if focus_cur_u > 1.0:
        lock_str, lock_col = f'COMMITTED — LoS  u={focus_cur_u:.2f}', RED
    elif focus_cur_u > 0.0:
        lock_str, lock_col = f'COMMITTED — conflict  u={focus_cur_u:.2f}', ORANGE
    elif focus_steps_clear < clear_steps:
        lock_str, lock_col = f'COOLDOWN  {focus_steps_clear}/{clear_steps} steps clear', YELLOW
    else:
        lock_str, lock_col = 'FREE — ready to switch', GREEN_DARK

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
            # Cooldown arc: filled sector showing how many clear steps remain
            if focus_steps_clear < clear_steps and focus_cur_u == 0.0:
                frac = focus_steps_clear / clear_steps          # 0 → just cleared, 1 → free
                pygame.draw.circle(screen, YELLOW, (px, py), drift_ring + 6, 1)
                label_cd = f'cool {focus_steps_clear}/{clear_steps}'
                screen.blit(font_sm.render(label_cd, True, YELLOW), (px + 7, py + 6))
        elif cs != focus_cs and u_val >= emerg_u:
            # Emergency-level non-focus aircraft: purple double ring
            pygame.draw.circle(screen, PURPLE, (px, py), drift_ring + 5, 2)
            pygame.draw.circle(screen, PURPLE, (px, py), drift_ring + 8, 1)

        pygame.draw.circle(screen, color, (px, py), 5)

        # Label: callsign + urgency score + drift
        u_str  = f' u={u_val:.2f}' if u_val > 0 else ''
        d_str  = f' {drift_deg:.0f}°off' if drift_deg > 2 else ''
        emg_str = ' ⚡EMERG' if (cs != focus_cs and u_val >= emerg_u) else ''
        label  = f'{cs}{u_str}{d_str}{emg_str}' + (' ◄' if cs == focus_cs else '')
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

    rew_color = GREEN if step_reward > 0 else (RED if step_reward < 0 else BLACK)
    hud = [
        (f'v3 · {mode_str}', BLACK),
        (f'T={bs.sim.simt:.0f}s   active={n_active}/{env.n_aircraft}   '
         f'served={env._next_callsign_id}   frame={frame_no}', BLACK),
        (f'[OBJ 1 SAFETY]   LoS pairs={n_los}   conflict pairs={n_conf}'
         f'   Σurgency={urg_sum:.2f}   LoS-steps={los_steps}', RED if n_los > 0 else BLACK),
        (f'[OBJ 2 EFFIC.]   mean drift={mean_drift:.1f}°', GREEN_DARK),
        (f'[FOCUS]   ac={focus_cs}   {lock_str}', lock_col),
        (f'[REWARD]   step={step_reward:+.3f}   cumulative={cum_reward:+.2f}   '
         f'mean={cum_reward / max(frame_no, 1):+.3f}', rew_color),
    ]
    for j, (line, col) in enumerate(hud):
        screen.blit(font.render(line, True, col), (8, 8 + j * 15))

    # Legend
    legend = [
        ('● RED    active LoS  (dist < 5 NM)',          RED),
        ('● ORANGE predicted conflict (≤10 min)',        ORANGE),
        ('● GREY   no conflict',                         GREY),
        ('◯ CYAN   focus aircraft',                      CYAN),
        ('◯ YELLOW focus in cooldown (N/5 steps clear)', YELLOW),
        ('◯ PURPLE non-focus at emergency (u ≥ 0.8)',   PURPLE),
    ]
    for j, (txt, col) in enumerate(legend):
        screen.blit(font_sm.render(txt, True, col),
                    (8, WINDOW_SIZE - 10 - (len(legend) - j) * 14))


def capture(surface):
    return Image.fromarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, model, vecnorm, use_policy, episode_seed, max_frames,
                frame_ms, mode_str, output_path, screen, font, font_sm):
    obs, _ = env.reset(seed=episode_seed)
    view    = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
    dest_px = {cs: view.ll_to_px(*env._destination_ll[cs])
               for cs in env._active_callsigns}

    frames      = []
    los_steps   = 0
    cum_reward  = 0.0
    step_reward = 0.0

    print(f'  seed={episode_seed}  n_aircraft={env.n_aircraft}  '
          f'max_steps={env._max_steps}  frame cap={max_frames}')

    while len(frames) < max_frames:
        if use_policy:
            norm_obs  = vecnorm.normalize_obs(obs[None])[0] if vecnorm else obs
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, step_reward, terminated, truncated, info = env.step(int(action))
        else:
            obs, step_reward, terminated, truncated, info = env.step(3)

        cum_reward += float(step_reward)
        if info.get('los_pairs', 0) > 0:
            los_steps += 1

        for cs in env._active_callsigns:
            if cs not in dest_px and cs in env._destination_ll:
                dest_px[cs] = view.ll_to_px(*env._destination_ll[cs])

        draw_frame(screen, font, font_sm, env, view, dest_px, los_steps, len(frames),
                   mode_str, float(step_reward), cum_reward)
        frames.append(capture(screen))

        n = len(frames)
        if n % 50 == 0:
            U      = env._urgency_matrix
            n_los  = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
            n_conf = int((U > 0).sum())   // 2 if U.size > 0 else 0
            print(f'    frame {n:4d}  T={bs.sim.simt:.0f}s  '
                  f'LoS={n_los}  conf={n_conf}  focus={env._focus_cs}  '
                  f'r={step_reward:+.3f}  Sr={cum_reward:+.2f}')

        if terminated or truncated:
            print(f'    Episode finished at frame {n}  (LoS-steps={los_steps})')
            break

    print(f'  Saving {len(frames)} frames → {output_path}')
    if output_path.endswith('.mp4'):
        fps_val = round(1000 / frame_ms)
        arr0    = np.array(frames[0])
        h, w    = arr0.shape[:2]
        fourcc  = cv2.VideoWriter_fourcc(*'avc1')   # H.264, broad player support
        writer  = cv2.VideoWriter(output_path, fourcc, fps_val, (w, h))
        if not writer.isOpened():                   # fallback if avc1 not available
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_val, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                       duration=frame_ms, loop=0, optimize=False)
    print(f'  Done  ({os.path.getsize(output_path)/1e6:.1f} MB)\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       default=DEFAULT_MODEL)
    parser.add_argument('--no-policy',   action='store_true')
    parser.add_argument('--mp4',         action='store_true',
                        help='Save as .mp4 instead of .gif')
    parser.add_argument('--frames',      type=int,   default=MAX_FRAMES)
    parser.add_argument('--fps',         type=int,   default=None)
    parser.add_argument('--episodes',    type=int,   default=1,
                        help='Number of episodes to render (default 1)')
    parser.add_argument('--n-aircraft',  type=int,   default=None,
                        help='Pin number of aircraft (overrides CONFIG)')
    parser.add_argument('--density',     type=float, default=None,
                        help='Pin sector density in km²/aircraft (overrides CONFIG)')
    args = parser.parse_args()

    max_frames = args.frames
    frame_ms   = args.fps if args.fps else FRAME_MS
    use_policy = (not args.no_policy) and args.model and os.path.exists(args.model)

    # ── Load policy once ──────────────────────────────────────────────────────
    model   = None
    vecnorm = None

    if use_policy:
        model_path   = args.model
        vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl')
        print(f'Loading model   : {model_path}')
        model = PPO.load(model_path)
        if os.path.exists(vecnorm_path):
            print(f'Loading vecnorm : {vecnorm_path}')
            with open(vecnorm_path, 'rb') as fh:
                vecnorm = _NumpyShim(fh).load()
            vecnorm.set_venv(DummyVecEnv([AirspaceEnv]))
            vecnorm.training    = False
            vecnorm.norm_reward = False

    stem     = os.path.splitext(os.path.basename(args.model))[0] if use_policy else 'no_policy'
    mode_str = f'policy: {os.path.basename(args.model)}' if use_policy else 'no policy'

    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    font    = pygame.font.SysFont('monospace', 11)
    font_sm = pygame.font.SysFont('monospace', 10)

    if args.n_aircraft is not None:
        CONFIG['n_aircraft'] = lambda n=args.n_aircraft: n
    if args.density is not None:
        CONFIG['rho'] = lambda d=args.density: d

    env = AirspaceEnv()

    cond_str = ''
    if args.n_aircraft is not None:
        cond_str += f'_n{args.n_aircraft}'
    if args.density is not None:
        cond_str += f'_d{int(args.density//1000)}k'

    print(f'Mode            : {mode_str}')
    print(f'Episodes        : {args.episodes}\n')

    for ep in range(1, args.episodes + 1):
        episode_seed = int.from_bytes(os.urandom(4), 'big')
        ep_tag       = f'_ep{ep}' if args.episodes > 1 else ''
        ext          = '.mp4' if args.mp4 else '.gif'
        output_path  = os.path.join(HERE, f'v3_{stem}{cond_str}{ep_tag}{ext}')
        print(f'-- Episode {ep}/{args.episodes} ----------------------------')
        run_episode(env, model, vecnorm, use_policy, episode_seed, max_frames,
                    frame_ms, mode_str, output_path, screen, font, font_sm)

    pygame.quit()


if __name__ == '__main__':
    main()
