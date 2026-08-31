"""pygame radar view of a trained policy: dark scope, blips with protected-zone rings, a HUD and a pilot-response panel. SPACE pause, R new episode, ESC/Q quit."""

import argparse, importlib, io, math, os, pickle, random, sys
from collections import deque
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Repo root too, so Environments.* resolves however the script is launched.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- NumPy unpickle shim so SB3 checkpoints load across numpy versions ------------
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
                data[key] = _NumpyShim(io.BytesIO(base64.b64decode(item[':serialized:'].encode()))).load()
            except Exception as e:
                warnings.warn(f'Could not deserialise {key}: {e}')
        else:
            data[key] = item
    return data

save_util.json_to_data = _patched_json_to_data

import pygame
from pygame import gfxdraw
import bluesky as bs
from stable_baselines3 import PPO

from Environments.v4.delays import DELAY_MODES   # for the --delay choices

WIN_W, WIN_H = 1920, 1080         # 16:9 full-HD landscape
CX, CY = 540, 540                 # radar centred in the left 1080x1080 square region
PANEL_X = 1100                    # observation panel starts here (right of the radar square)
MARGIN = 80
DIAG   = math.hypot(WIN_W, WIN_H)
BG     = (6, 14, 10)
GREEN  = (60, 220, 120)
DIM    = (28, 90, 55)
ORANGE = (255, 140, 0)
RED    = (255, 60, 55)
BLUE   = (140, 205, 255)        # light blue -- ownship ring
CYAN   = (90, 220, 255)
WP     = (40, 120, 70)
KT     = 1.94384

ACTION_LABELS = ['L60', 'L45', 'L30', 'HOLD', 'R30', 'R45', 'R60', 'DIR', 'SPD+', 'SPD-']

# Env hooks, bound in main() once the --env module is known.
CONFIG = None
latlon_to_nm = None
OBS_OWNSHIP_LABELS = []
OBS_INTRUDER_LABELS = []


def load_model(path):
    custom = {'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0,
              'clip_range': lambda _: 0.0}
    return PPO.load(path, device='cpu', custom_objects=custom)


def estimate_obs_stats(env, model, warmup=2500):
    """Estimate observation mean/std to stand in for missing VecNormalize stats. Returns (mean, std)."""
    def collect(stats):
        mean, std = stats
        buf, (obs, _) = [], env.reset()
        for _ in range(warmup):
            if mean is None:
                a = env.action_space.sample()
            else:
                a, _ = model.predict(np.clip((obs - mean) / std, -10, 10), deterministic=True)
            obs, _, _, trunc, _ = env.step(a)
            buf.append(obs.copy())
            if trunc:
                obs, _ = env.reset()
        B = np.asarray(buf)
        return B.mean(0), B.std(0) + 1e-8

    return collect(collect((None, None)))


def load_vecnorm_stats(path):
    """Load the real observation mean/std from a saved VecNormalize .pkl."""
    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        vn = _U(f).load()
    if getattr(vn, 'obs_rms', None) is None:
        raise ValueError(f'{path} has no obs_rms (saved with norm_obs=False?)')
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float32)
    std  = np.sqrt(np.asarray(vn.obs_rms.var, dtype=np.float32) + vn.epsilon)
    return mean, std


def fresh_stats():
    return {'cum_r': 0.0, 'last_r': 0.0, 'step_n': 0, 'los': 0,
            'last_action': 3, 'acting_cs': None,
            'counts': [0] * 10, 'recent': deque(maxlen=14),
            # Instruction timeline for the delay panel, read from the env rather than predicted here.
            'pending_seen': {}, 'executed': []}


def reset_episode(env, scenario_seed):
    obs, _ = env.reset(options={'scenario_seed': scenario_seed})
    poly = list(env.polygon) if env.polygon is not None else []
    rng = max(35.0, 1.1 * max((math.hypot(v[0], v[1]) for v in poly), default=60.0))
    scale = (min(CX, CY) - MARGIN) / rng
    prot_px = 0.5 * CONFIG['sep_nm'] * scale         # 2.5 NM protected-zone radius
    return obs, poly, scale, prot_px


def track_advisories(env, st):
    """Mirror the env's outstanding-advisory queue into st and record executions; the delay is not reimplemented here."""
    pending = {cs: ac.pending_advisory for cs, ac in env._aircraft.items()
               if ac.pending_advisory is not None}


    for cs, advisory in st['pending_seen'].items():
        if cs in pending:
            continue
        # The execution time was fixed at issue, so it is exact -- unlike the frame clock's 5 s samples.
        executed_at_s = advisory['execute_at_s']
        st['executed'].append({'cs': cs, 'action': advisory['action'],
                               'issued_at_s': advisory['issued_at_s'],
                               'executed_at_s': executed_at_s,
                               'waited_s': executed_at_s - advisory['issued_at_s']})
    st['pending_seen'] = pending


def policy_step(env, model, obs, deterministic, norm, st, fixed_seq=None, seq_once=False):
    """Advance one RL step and update stats in place; fixed_seq overrides the model, and model=None is HOLD-only."""
    acting_cs = env._focus_cs
    if fixed_seq:
        n = st['step_n']
        a = 3 if (seq_once and n >= len(fixed_seq)) else fixed_seq[n % len(fixed_seq)]
    elif model is None:
        a = 3   # HOLD-only: do-nothing baseline
    else:
        o = obs if norm is None else np.clip((obs - norm[0]) / norm[1], -10, 10)
        action, _ = model.predict(o, deterministic=deterministic)
        a = int(action)
    # Read before the step: the env issues the advisory against this same traffic picture.
    obs, r, _, trunc, _ = env.step(a)
    st['last_r'] = float(r); st['cum_r'] += st['last_r']; st['step_n'] += 1
    st['last_action'] = a; st['acting_cs'] = acting_cs
    st['counts'][a] += 1; st['recent'].appendleft((acting_cs, a))
    if env._los_seconds_this_step:
        st['los'] += 1
    track_advisories(env, st)
    return obs, trunc


def to_screen(x_nm, y_nm, scale):
    return CX + x_nm * scale, CY - y_nm * scale


def urgency(env):
    U, cs_list = env._urgency_matrix, env._urgency_cs_list
    row_max = U.max(axis=1) if U.size else np.zeros(0)
    return {cs: float(row_max[i]) for i, cs in enumerate(cs_list)} if U.size else {}


def aa_ring(surf, cx, cy, r, color):
    cx, cy, r = int(cx), int(cy), int(r)
    if r >= 1:
        gfxdraw.aacircle(surf, cx, cy, r, color)


def dashed_line(surf, color, p1, p2, dash=6, gap=7, width=1):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        pygame.draw.line(surf, color, (x1 + ux * s, y1 + uy * s),
                         (x1 + ux * e, y1 + uy * e), width)
        s += dash + gap


def draw_obs_panel(screen, font, font_hud, obs, focus_cs, intruder_cs):
    """Right-hand panel: the focus aircraft's raw observation vector, labelled."""
    pygame.draw.rect(screen, (8, 16, 12), (PANEL_X, 0, WIN_W - PANEL_X, WIN_H))
    pygame.draw.line(screen, DIM, (PANEL_X, 0), (PANEL_X, WIN_H), 1)
    x, y = PANEL_X + 16, 16
    screen.blit(font_hud.render('OBSERVATION', True, GREEN), (x, y)); y += 26
    screen.blit(font.render(f'ownship: {focus_cs or "--"}', True, DIM), (x, y)); y += 24
    if obs is None:
        return
    n_own = len(OBS_OWNSHIP_LABELS)
    for lbl, val in zip(OBS_OWNSHIP_LABELS, obs[:n_own]):
        col = ORANGE if (lbl == 'retn_conf' and val > 0.5) else GREEN
        screen.blit(font.render(f'{lbl:>9} {val:+.2f}', True, col), (x, y)); y += 16
    y += 8
    rest, n_int = obs[n_own:], len(OBS_INTRUDER_LABELS)
    for k in range(len(rest) // n_int):
        seg = rest[k * n_int:(k + 1) * n_int]
        cs_k = intruder_cs[k] if k < len(intruder_cs) else None
        empty = cs_k is None
        head = f'INTRUDER {k}  (empty)' if empty else f'INTRUDER {k}: {cs_k}'
        screen.blit(font.render(head, True, DIM if empty else CYAN), (x, y)); y += 16
        for lbl, val in zip(OBS_INTRUDER_LABELS, seg):
            screen.blit(font.render(f'{lbl:>9} {val:+.2f}', True,
                                    DIM if empty else GREEN), (x + 10, y)); y += 15
        y += 4


def draw_delay_panel(screen, fonts, env, st, x, y):
    """The pilot-response clock, counted down off the advisory's own execute_at_s rather than estimated."""
    font, font_hud = fonts
    now = env._sim_time_s
    screen.blit(font_hud.render(f'PILOT RESPONSE   t = {now:6.0f} s', True, GREEN), (x, y))
    y += 26
    mode   = env.response_delay.mode
    mean_s = env.response_delay.mean_s
    screen.blit(font.render(f'delay model: {mode}  mean {mean_s:g}s', True, DIM),
                (x, y)); y += 22

    if st['pending_seen']:
        for cs, advisory in st['pending_seen'].items():
            label  = ACTION_LABELS[advisory['action']]
            waited = now - advisory['issued_at_s']
            left   = max(0.0, advisory['execute_at_s'] - now)
            screen.blit(font.render(f'{cs}  {label} transmitted at '
                                    f't={advisory["issued_at_s"]:.0f}s'
                                    f'  (waiting {waited:.0f}s)', True, ORANGE), (x, y))
            y += 20
            screen.blit(font_hud.render(f'EXECUTES IN {left:5.0f} s', True, ORANGE),
                        (x, y)); y += 26
            # Drain bar: full at issue, empty when the pilot acts.
            total = max(advisory['execute_at_s'] - advisory['issued_at_s'], 1e-9)
            w, h = 320, 14
            pygame.draw.rect(screen, DIM, (x, y, w, h), 1)
            pygame.draw.rect(screen, ORANGE, (x + 1, y + 1,
                                              int((w - 2) * (1 - left / total)), h - 2))
            y += h + 10
    else:
        screen.blit(font.render('nothing outstanding', True, DIM), (x, y)); y += 26

    for ev in st['executed'][-4:]:
        screen.blit(font.render(f'{ev["cs"]}  {ACTION_LABELS[ev["action"]]} FLOWN at '
                                f't={ev["executed_at_s"]:.0f}s  after '
                                f'{ev["waited_s"]:.0f}s',
                                True, CYAN), (x, y)); y += 18


def draw_frame(screen, fonts, env, scale, poly, prot_px, st, paused, mode, obs):
    font, font_hud = fonts
    screen.fill(BG)

    if len(poly) >= 3:
        pygame.draw.polygon(screen, DIM, [to_screen(v[0], v[1], scale) for v in poly], 2)

    urg = urgency(env)
    init_hdg_map = {cs: ac.initial_hdg for cs, ac in env._aircraft.items()}
    for cs in env._aircraft:
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            continue
        pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        px, py = to_screen(pos[0], pos[1], scale)
        u = urg.get(cs, 0.0)
        col = RED if u > 1.0 else ORANGE if u > 0.0 else GREEN
        hdg_deg = float(bs.traf.hdg[idx])
        gs_kt = float(bs.traf.tas[idx]) * KT

        # Dashed ray along the INITIAL heading: the line it would have flown had it never turned.
        init_hdg = init_hdg_map.get(cs)
        if init_hdg is not None:
            h0 = math.radians(init_hdg)
            dashed_line(screen, WP, (px, py),
                        (px + math.sin(h0) * DIAG, py - math.cos(h0) * DIAG))

        # 2.5 NM protected-zone ring (two overlapping == LoS); the ownship ring is blue.
        aa_ring(screen, px, py, prot_px, BLUE if cs == env._focus_cs else col)

        h = math.radians(hdg_deg)                          # current-heading leader
        lead = float(bs.traf.tas[idx]) / 1852.0 * 60.0 * scale
        pygame.draw.line(screen, col, (px, py),
                         (px + math.sin(h) * lead, py - math.cos(h) * lead), 2)

        pygame.draw.circle(screen, col, (int(px), int(py)), 2)   # radar blip (no square)
        screen.blit(font.render(f'{cs} {hdg_deg:03.0f}', True, col), (px + 10, py - 18))
        screen.blit(font.render(f'{gs_kt:3.0f}kt', True, col), (px + 10, py - 5))

    al = ACTION_LABELS[st['last_action']]
    tgt = st['acting_cs'] or '--'
    screen.blit(font_hud.render(f'TOTAL REWARD {st["cum_r"]:+.1f}', True, GREEN), (12, 10))
    screen.blit(font.render(f'STEP {st["step_n"]}   R {st["last_r"]:+.2f}   LoS {st["los"]}',
                            True, DIM), (14, 34))
    screen.blit(font.render(f'ACTION  {al:4s} -> {tgt}', True, CYAN), (14, 52))
    screen.blit(font.render(mode, True, DIM), (14, 70))
    if paused:
        screen.blit(font.render('PAUSED', True, ORANGE), (14, 88))

    y0 = WIN_H - 12 - 11 * 16
    screen.blit(font.render('ACTIONS (count)', True, GREEN), (14, y0))
    for i, lbl in enumerate(ACTION_LABELS):
        c = CYAN if i == st['last_action'] else DIM
        screen.blit(font.render(f'{lbl:4s} {st["counts"][i]:4d}', True, c),
                    (14, y0 + 16 * (i + 1)))
    recent = '  '.join(f'{cs}:{ACTION_LABELS[a]}' for cs, a in list(st['recent'])[:6])
    screen.blit(font.render(recent, True, DIM), (150, y0))

    draw_delay_panel(screen, fonts, env, st, 14, 120)

    draw_obs_panel(screen, font, font_hud, obs, env._focus_cs,
                   getattr(env, '_last_intruder_cs', []))


def record_mp4(path, screen, fonts, env, model, obs, poly, scale, prot_px,
               fps, det, norm, mode, fixed_seq=None, max_steps=None, seq_once=False):
    import cv2
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIN_W, WIN_H))
    if not writer.isOpened():
        raise RuntimeError(f'could not open video writer for {path}')
    st = fresh_stats()
    while True:
        obs, trunc = policy_step(env, model, obs, det, norm, st, fixed_seq, seq_once)
        draw_frame(screen, fonts, env, scale, poly, prot_px, st, False, mode, obs)
        frame = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if trunc or (max_steps is not None and st['step_n'] >= max_steps):
            break
    writer.release()
    print(f'saved {path}  ({st["step_n"]} steps, reward {st["cum_r"]:+.1f}, LoS {st["los"]})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='best_model.zip')
    ap.add_argument('--env', default='Environments.v4',
                    help='env module exposing AirspaceEnv / CONFIG / latlon_to_nm')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--n_ac', type=int, default=14, help='aircraft count')
    ap.add_argument('--density', type=float, default=1 / 10000, help='density (ac/km^2)')
    ap.add_argument('--delay', default='none', choices=list(DELAY_MODES),
                    help='action-response delay condition to fly under. Match the delay type the '
                         'model was trained on, or set it deliberately to see how the '
                         'policy copes with pilots it never met.')
    # --delay-first is the old spelling, kept so existing commands still work.
    ap.add_argument('--delay-mean', '--delay-first', dest='delay_mean',
                    type=float, default=None, metavar='S',
                    help='delay magnitude: the MEAN pilot response time in seconds. Match '
                         'the magnitude the model was trained at; default is the env '
                         'default.')
    ap.add_argument('--hold', action='store_true',
                    help='do-nothing policy: always select HOLD (no model loaded)')
    ap.add_argument('--fixed-action', type=int, default=None, metavar='N',
                    help='ignore the model/hold policy and always select action N every '
                         f'step, e.g. 0 = {ACTION_LABELS[0]} (turn -60). For isolating '
                         'the effect of one repeated action.')
    ap.add_argument('--action-seq', default=None, metavar='A,B,C',
                    help='cycle through this comma-separated list of action indices, one '
                         'per step, e.g. "0,3" alternates turn -60 with HOLD. Overrides '
                         '--fixed-action.')
    ap.add_argument('--seq-once', action='store_true',
                    help='play --action-seq / --fixed-action ONCE and then HOLD forever: '
                         'one instruction, then silence. With --n_ac 1 this isolates a '
                         'single pilot response, which is what the delay panel times.')
    ap.add_argument('--steps', type=int, default=None, metavar='N',
                    help='stop after this many RL steps (mp4 only); default: full episode')
    ap.add_argument('--ref_jitter', type=float, default=None, metavar='J',
                    help='exit-point jitter range: ref_jitter ~ uniform(-J, +J); '
                         'J=0.5 gives fully random crossing directions')
    ap.add_argument('--mp4', default=None, metavar='PATH',
                    help='render a single episode to this mp4 file and exit')
    ap.add_argument('--fps', type=int, default=10, help='interactive / mp4 frame rate')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions instead of the argmax (best) policy')
    ap.add_argument('--vecnorm', default='best_model_vecnorm.pkl', metavar='PATH',
                    help='load the real VecNormalize obs stats from this .pkl '
                         '(norm_obs=True models -- the faithful way to normalise). '
                         'Pass "" to disable and feed raw obs.')
    ap.add_argument('--normalize', action='store_true',
                    help='estimate & apply obs normalisation from a warm-up rollout '
                         '(use when no vecnorm .pkl is available)')
    args = ap.parse_args()

    global CONFIG, latlon_to_nm, OBS_OWNSHIP_LABELS, OBS_INTRUDER_LABELS
    envmod = importlib.import_module(args.env)
    AirspaceEnv, CONFIG, latlon_to_nm = envmod.AirspaceEnv, envmod.CONFIG, envmod.latlon_to_nm
    OBS_OWNSHIP_LABELS = getattr(envmod, 'OBS_OWNSHIP_LABELS',
                                 [f'own{i}' for i in range(99)])
    OBS_INTRUDER_LABELS = getattr(envmod, 'OBS_INTRUDER_LABELS',
                                  ['rho', 'theta', 'psi', 'vint', 'tau'])

    # The samplers take the env's scenario generator; these overrides pin the values.
    CONFIG['n_aircraft'] = lambda rng: args.n_ac
    CONFIG['rho']        = lambda rng: args.density
    if args.ref_jitter is not None:
        j = args.ref_jitter
        CONFIG['ref_jitter'] = lambda rng: rng.uniform(-j, j)

    det = not args.stochastic
    fixed_seq = None
    if args.action_seq:
        fixed_seq = [int(t) for t in args.action_seq.split(',') if t.strip() != '']
    elif args.fixed_action is not None:
        fixed_seq = [args.fixed_action]

    if fixed_seq:
        model, norm = None, None      # fixed policy: no model, no obs normalisation
        env = AirspaceEnv(delay_mode=args.delay, delay_mean_s=args.delay_mean)
        labels = ' -> '.join(ACTION_LABELS[a] for a in fixed_seq)
        if args.seq_once:
            mode = f'FIXED once: {labels}, then HOLD'
            print(f'fixed policy: {labels} once, then HOLD every step', flush=True)
        else:
            mode = (f'FIXED action {ACTION_LABELS[fixed_seq[0]]} every step'
                    if len(fixed_seq) == 1 else f'FIXED cycle: {labels}')
            print(f'fixed policy: cycling {labels}', flush=True)
    elif args.hold:
        model, norm = None, None      # do-nothing policy: no model, no obs normalisation
        env = AirspaceEnv(delay_mode=args.delay, delay_mean_s=args.delay_mean)
        mode = 'HOLD-only (do-nothing)'
        print('do-nothing policy: HOLD selected every step', flush=True)
    else:
        model = load_model(args.model)
        env = AirspaceEnv(delay_mode=args.delay, delay_mean_s=args.delay_mean)
        norm = None
        if args.vecnorm:
            norm = load_vecnorm_stats(args.vecnorm)
            print(f'loaded obs normalisation from {args.vecnorm}', flush=True)
        elif args.normalize:
            print('estimating observation normalisation (warm-up rollout)...', flush=True)
            norm = estimate_obs_stats(env, model)
        obs_mode = 'vecnorm' if args.vecnorm else 'est' if norm is not None else 'raw'
        mode = f"obs:{obs_mode}  act:{'argmax' if det else 'sample'}"

    if args.mp4:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('ATC Radar -- policy visualisation')
    fonts = (pygame.font.SysFont('consolas,monospace', 13),
             pygame.font.SysFont('consolas,monospace', 18, bold=True))

    seed = args.seed if args.seed is not None else random.randint(0, 99_999)
    obs, poly, scale, prot_px = reset_episode(env, seed)

    if args.mp4:
        record_mp4(args.mp4, screen, fonts, env, model, obs, poly, scale, prot_px,
                   args.fps, det, norm, mode, fixed_seq, args.steps, args.seq_once)
        pygame.quit()
        return

    clock = pygame.time.Clock()
    paused = False
    st = fresh_stats()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    seed = random.randint(0, 99_999)
                    obs, poly, scale, prot_px = reset_episode(env, seed)
                    st = fresh_stats()

        if not paused:
            obs, trunc = policy_step(env, model, obs, det, norm, st, fixed_seq, args.seq_once)
            if trunc:
                seed = random.randint(0, 99_999)
                obs, poly, scale, prot_px = reset_episode(env, seed)
                st = fresh_stats()

        draw_frame(screen, fonts, env, scale, poly, prot_px, st, paused, mode, obs)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == '__main__':
    main()
