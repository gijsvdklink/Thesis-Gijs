"""
Visualise a trained PPO policy on v3_simple_env.

Usage:
    python -m Visualisation.v3_simple_visualise --model path/to/model.zip
    python -m Visualisation.v3_simple_visualise --model path/to/model.zip --vecnorm path/to/vecnorm.pkl
"""

import argparse, math, os, sys
import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from Environments.v3_simple_env import AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm
import bluesky as bs

N_EPISODES  = 5
WINDOW_SIZE = 850
FPS         = 8

CYAN   = (  0, 210, 255)
ORANGE = (255, 140,   0)
GREY   = (160, 160, 160)
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
RED    = (220,  40,  40)

ACTION_LABELS = ['-30°', '-15°', 'DIRECT', '+15°', '+30°', 'HOLD']


class _View:
    def __init__(self, polygon_nm, w, h):
        pad  = CONFIG['sep_nm'] * NM_TO_KM * 2.0
        km   = polygon_nm * NM_TO_KM
        span = max(km[:, 0].max()-km[:, 0].min()+2*pad,
                   km[:, 1].max()-km[:, 1].min()+2*pad)
        self._cx = (km[:, 0].min()+km[:, 0].max()) / 2
        self._cy = (km[:, 1].min()+km[:, 1].max()) / 2
        self._sc = min(w, h) / span
        self._w, self._h = w, h

    def nm_to_px(self, x, y):
        return (int((x*NM_TO_KM - self._cx)* self._sc + self._w/2),
                int((y*NM_TO_KM - self._cy)*-self._sc + self._h/2))

    def latlon_to_px(self, lat, lon):
        nm = latlon_to_nm(CONFIG['center_ll'], lat, lon)
        return self.nm_to_px(float(nm[0]), float(nm[1]))

    def nm_len_to_px(self, nm):
        return max(1, int(nm * NM_TO_KM * self._sc))


def _dashed_line(surf, color, p1, p2, dash=8, gap=5):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    total  = math.hypot(dx, dy)
    if total < 1: return
    ux, uy = dx/total, dy/total
    x, y   = float(p1[0]), float(p1[1])
    drawn  = 0.0; on = True
    while drawn < total:
        seg = min(dash if on else gap, total-drawn)
        if on:
            pygame.draw.line(surf, color, (round(x), round(y)),
                             (round(x+ux*seg), round(y+uy*seg)), 1)
        x += ux*seg; y += uy*seg; drawn += seg; on = not on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True)
    parser.add_argument('--vecnorm', default=None,
                        help='VecNormalize stats .pkl (required for VecNormalize-trained models)')
    parser.add_argument('--episodes', type=int, default=N_EPISODES)
    parser.add_argument('--fps',      type=int, default=FPS)
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(model_path+'.zip'):
        model_path += '.zip'
    if not os.path.exists(model_path):
        sys.exit(f'Model not found: {model_path}')

    print(f'Loading model : {model_path}')
    model = PPO.load(model_path)

    if args.vecnorm:
        import pickle
        with open(args.vecnorm, 'rb') as f:
            _vn = pickle.load(f)
        _mu  = _vn.obs_rms.mean.copy()
        _std = np.sqrt(_vn.obs_rms.var + _vn.epsilon)
        _clip = float(_vn.clip_obs)
        def preprocess(obs):
            return np.clip((obs - _mu) / _std, -_clip, _clip).astype(np.float32)
        print(f'VecNormalize  : {args.vecnorm}')
    else:
        def preprocess(obs): return obs
        print('VecNormalize  : none')

    env     = AirspaceEnv()
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('v3_simple_env — trained policy')
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    for episode in range(args.episodes):
        obs, _     = env.reset()
        view       = _View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        poly_px    = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_px     = view.nm_len_to_px(CONFIG['sep_nm'] / 2)
        dest_px    = {cs: view.latlon_to_px(*env._destination_ll[cs])
                      for cs in env._active_callsigns}
        total_r    = 0.0
        last_act   = 2
        step = 0; los_steps = 0; done = False

        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit(); return

            action, _ = model.predict(preprocess(obs), deterministic=True)
            last_act  = int(action)
            obs, r, _, truncated, info = env.step(action)
            total_r += r
            done     = truncated
            step    += 1
            if info['los_pairs']:
                los_steps += 1

            for cs in env._active_callsigns:
                if cs not in dest_px and cs in env._destination_ll:
                    dest_px[cs] = view.latlon_to_px(*env._destination_ll[cs])

            # ── Draw ────────────────────────────────────────────────────────

            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, poly_px, 2)

            U       = env._urgency_matrix
            cs_list = env._urgency_cs_list
            conf_set = set()
            los_set  = set()
            if U.size > 0:
                row_max = U.max(axis=1)
                conf_set = {cs_list[i] for i in range(len(cs_list)) if row_max[i] > 0}
                los_set  = {cs_list[i] for i in range(len(cs_list))
                            if row_max[i] > 1.0}

            for cs in env._active_callsigns:
                idx = bs.traf.id2idx(cs)
                if idx < 0: continue
                px, py    = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                is_focus  = cs == env._focus_cs
                color     = (CYAN   if is_focus  else
                             RED    if cs in los_set  else
                             ORANGE if cs in conf_set else GREY)
                ring_r    = sep_px + (4 if is_focus else 0)
                if cs in dest_px:
                    _dashed_line(screen, color, (px, py), dest_px[cs])
                pygame.draw.circle(screen, color, (px, py), ring_r,
                                   2 if is_focus else 1)
                pygame.draw.circle(screen, color, (px, py), 5)
                lbl = cs + (f' {ACTION_LABELS[last_act]} ◄' if is_focus else '')
                screen.blit(font_sm.render(lbl, True, color), (px+7, py-7))

            hud = [
                f'Episode {episode+1}/{args.episodes}  active={len(env._active_callsigns)}',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Focus: {env._focus_cs}   reward={total_r:.1f}   last={ACTION_LABELS[last_act]}',
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8+j*16))

            for j, (txt, col) in enumerate([
                ('● CYAN   focus aircraft', CYAN),
                ('● RED    active LoS',     RED),
                ('● ORANGE conflict',        ORANGE),
                ('● GREY   clear',           GREY),
            ]):
                screen.blit(font_sm.render(txt, True, col),
                            (8, WINDOW_SIZE - 12 - (3-j)*14))

            pygame.display.flip()
            clock.tick(args.fps)

        print(f'Ep {episode+1:2d}  steps={step}  reward={total_r:.1f}  '
              f'LoS-steps={los_steps}')

    pygame.quit()


if __name__ == '__main__':
    main()
