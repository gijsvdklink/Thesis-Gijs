"""
Visualise the trained PPO v3 policy running in AirspaceEnvV3.

    python visualize_policy_v3.py                          # pick from runs/ interactively
    python visualize_policy_v3.py --model path/to/model.zip
    python visualize_policy_v3.py --episodes 20
"""

import os
import sys
import argparse
import math
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from airspace_env_v3 import AirspaceEnvV3, MAX_AGENTS, N_WAVES, NM_TO_KM, CENTER_LL, latlon_to_nm

import bluesky as bs

_SCRIPTS_DIR = os.path.dirname(__file__)


# ── Model selection ───────────────────────────────────────────────────────────

def select_model():
    parser = argparse.ArgumentParser(description='Visualise a trained PPO v3 policy')
    parser.add_argument('--model',    type=str, default=None)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    if args.model:
        return args.model, args.episodes

    runs_dir = os.path.join(_SCRIPTS_DIR, 'runs')
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No runs/ folder at {runs_dir}. Pass --model explicitly.")

    models = sorted(
        [os.path.join(runs_dir, d, 'model.zip')
         for d in os.listdir(runs_dir)
         if os.path.isfile(os.path.join(runs_dir, d, 'model.zip'))],
        reverse=True,
    )
    checkpoints = []
    for d in os.listdir(runs_dir):
        ckpt_dir = os.path.join(runs_dir, d, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir), reverse=True):
                if f.startswith('ppo_v3') and f.endswith('.zip'):
                    checkpoints.append(os.path.join(ckpt_dir, f))
    checkpoints.sort(reverse=True)

    # Flatten: models first, then checkpoints
    all_models = models + checkpoints
    if not all_models:
        raise FileNotFoundError("No model.zip or ppo_v3 checkpoints found. Train first.")

    print("\nAvailable models:")
    for i, m in enumerate(all_models):
        tag = "  <- latest" if i == 0 else ""
        print(f"  [{i}]  {os.path.relpath(m, _SCRIPTS_DIR)}{tag}")

    choice = input(f"\nSelect [0-{len(all_models)-1}]  (Enter = latest): ").strip()
    idx = int(choice) if choice.isdigit() else 0
    print()
    return all_models[idx], args.episodes


# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_PATH, N_EPISODES = select_model()
WINDOW_SIZE = 850
FPS         = 8

ACTION_LABELS = ['-10°', '+10°', '0°', 'back']

BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
RED    = (220,  30,  30)
GREEN  = ( 50, 180,  50)
BLUE   = ( 50, 130, 220)
ORANGE = (255, 140,   0)
GREY   = (160, 160, 160)
PURPLE = (150,  50, 200)
CYAN   = (  0, 200, 200)
YELLOW = (220, 200,   0)

# One colour per slot — up to MAX_AGENTS=8, consistent across wave replacements
SLOT_COLORS = [GREEN, BLUE, ORANGE, PURPLE, CYAN, YELLOW, RED, GREY]
SEP_NM      = 5.0


# ── View helper ───────────────────────────────────────────────────────────────

class View:
    def __init__(self, polygon_nm, w, h):
        poly_km    = polygon_nm * NM_TO_KM
        padding_km = SEP_NM * NM_TO_KM * 1.5
        x_min, x_max = poly_km[:, 0].min(), poly_km[:, 0].max()
        y_min, y_max = poly_km[:, 1].min(), poly_km[:, 1].max()
        span = max((x_max - x_min) + 2*padding_km,
                   (y_max - y_min) + 2*padding_km)
        self._cx  = (x_min + x_max) / 2
        self._cy  = (y_min + y_max) / 2
        self._ppk = min(w, h) / span
        self._w, self._h = w, h

    def nm_to_px(self, x_nm, y_nm):
        x_km, y_km = x_nm * NM_TO_KM, y_nm * NM_TO_KM
        return (int((x_km - self._cx) *  self._ppk + self._w / 2),
                int((y_km - self._cy) * -self._ppk + self._h / 2))

    def latlon_to_px(self, lat, lon):
        nm = latlon_to_nm(CENTER_LL, lat, lon)
        return self.nm_to_px(nm[0], nm[1])

    def nm_to_pixels_len(self, nm):
        return max(1, int(nm * NM_TO_KM * self._ppk))


def draw_dashed_line(surface, color, p1, p2, dash=9, gap=6, width=1):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    total  = math.hypot(dx, dy)
    if total < 1:
        return
    ux, uy = dx/total, dy/total
    x, y   = float(p1[0]), float(p1[1])
    drawn  = 0.0; draw = True
    while drawn < total:
        seg = min(dash if draw else gap, total - drawn)
        if draw:
            pygame.draw.line(surface, color,
                             (round(x), round(y)),
                             (round(x + ux*seg), round(y + uy*seg)), width)
        x += ux*seg; y += uy*seg; drawn += seg; draw = not draw


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = PPO.load(MODEL_PATH)

    env = AirspaceEnvV3()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('PPO v3 — 4-wave dynamic spawning')
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('monospace', 13)
    font_sm = pygame.font.SysFont('monospace', 11)

    episode_stats = []

    for episode in range(N_EPISODES):
        obs_stack, _ = env.reset()
        n_agents      = env.n_aircraft   # actual agents this episode (2–8)

        view          = View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px    = [view.nm_to_px(v[0], v[1]) for v in env.polygon]
        sep_radius_px = view.nm_to_pixels_len(SEP_NM / 2)

        # Track destinations per callsign for rendering
        dest_px_by_cs = {
            cs: view.latlon_to_px(env._dest_ll[cs][0], env._dest_ll[cs][1])
            for cs in env._active
        }

        total_rewards = np.zeros(n_agents)
        last_actions  = [2] * n_agents   # default: hold
        step          = 0
        los_steps     = 0
        spawned, total_ac = env._queue.progress
        done          = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); return

            # Policy: each slot acts on its own observation
            actions = []
            for i in range(n_agents):
                a, _ = model.predict(obs_stack[i], deterministic=True)
                actions.append(int(a))
            last_actions = actions

            obs_stack, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_rewards += rewards
            done = terminated or truncated
            step += 1
            if len(info['los_pairs']) > 0:
                los_steps += 1

            # Update dest_px for any newly spawned aircraft
            for cs in env._active:
                if cs not in dest_px_by_cs and cs in env._dest_ll:
                    dest_px_by_cs[cs] = view.latlon_to_px(
                        env._dest_ll[cs][0], env._dest_ll[cs][1]
                    )

            spawned, total_ac = env._queue.progress

            # ── Render ────────────────────────────────────────────────────────
            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, width=2)

            aircraft_in_los = {cs for pair in bs.traf.cd.lospairs for cs in pair}

            for slot_i, cs in enumerate(env._slots):
                if cs is None or cs not in env._active:
                    continue
                idx = bs.traf.id2idx(cs)
                if idx < 0:
                    continue

                px, py     = view.latlon_to_px(bs.traf.lat[idx], bs.traf.lon[idx])
                base_color = SLOT_COLORS[slot_i % len(SLOT_COLORS)]
                color      = RED if cs in aircraft_in_los else base_color
                d_px       = dest_px_by_cs.get(cs)

                # Sep bubble
                pygame.draw.circle(screen, color, (px, py), sep_radius_px, 1)

                # Dashed line to destination
                if d_px:
                    draw_dashed_line(screen, color, (px, py), d_px)

                # Aircraft dot
                pygame.draw.circle(screen, color, (px, py), 5)

                # Callsign + action label
                label = f"{cs} {ACTION_LABELS[last_actions[slot_i]]}"
                screen.blit(font_sm.render(label, True, color), (px + 7, py - 7))

            # HUD
            r_str = '  '.join(f'S{i}={total_rewards[i]:.0f}' for i in range(n_agents))
            hud = [
                f'Episode {episode+1}/{N_EPISODES}  ({n_agents} agents)',
                f'Step {step}   T={bs.sim.simt:.0f}s   LoS-steps={los_steps}',
                f'Served {spawned}/{total_ac}   queued={env._queue.n_pending}   active={len(env._active)}',
                f'Rewards:  {r_str}',
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            # Slot legend — only show slots active this episode
            legend_y_start = WINDOW_SIZE - 16 - n_agents * 14
            for i in range(n_agents):
                color = SLOT_COLORS[i % len(SLOT_COLORS)]
                screen.blit(font_sm.render(f'● Slot {i}', True, color),
                            (8, legend_y_start + i * 14))
            screen.blit(font_sm.render('● RED = separation violation', True, RED),
                        (8, WINDOW_SIZE - 10))

            pygame.display.flip()
            clock.tick(FPS)

        episode_stats.append({
            'steps':    step,
            'rewards':  total_rewards.copy(),
            'los':      los_steps,
            'n_agents': n_agents,
        })
        mean_r = total_rewards.mean()
        print(f"Episode {episode+1:2d}  n_ac={n_agents}  steps={step:3d}  "
              f"mean_reward={mean_r:7.2f}  LoS-steps={los_steps}  "
              f"served={spawned}/{total_ac}")

    pygame.quit()

    # ── Validation summary ────────────────────────────────────────────────────
    all_mean  = [s['rewards'].mean() for s in episode_stats]
    all_steps = [s['steps']          for s in episode_stats]
    all_los   = [s['los']            for s in episode_stats]
    clean_eps = sum(1 for s in episode_stats if s['los'] == 0)

    print("\n── Validation Summary ──────────────────────────────────────────")
    print(f"  Episodes          : {N_EPISODES}")
    print(f"  Mean reward       : {np.mean(all_mean):7.2f}  ±  {np.std(all_mean):.2f}")
    print(f"  Min / Max reward  : {np.min(all_mean):.2f} / {np.max(all_mean):.2f}")
    print(f"  Mean episode len  : {np.mean(all_steps):.1f} steps")
    print(f"  Total LoS steps   : {sum(all_los)}")
    print(f"  Clean episodes    : {clean_eps}/{N_EPISODES}  "
          f"({100*clean_eps/N_EPISODES:.0f}%  zero LoS)")
    print()
    print("  Per-slot mean rewards (episodes where slot was active):")
    for i in range(MAX_AGENTS):
        agent_rewards = [s['rewards'][i] for s in episode_stats if i < s['n_agents']]
        if not agent_rewards:
            continue
        print(f"    Slot {i}: {np.mean(agent_rewards):7.2f}  ±  {np.std(agent_rewards):.2f}"
              f"  (n={len(agent_rewards)} episodes)")
    print("────────────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()
