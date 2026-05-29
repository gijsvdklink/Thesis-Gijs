"""
Visualise the trained PPO v2 policy running in AirspaceEnvV2.

All 3 aircraft are controlled by the shared policy.
Run from the Scenarios folder:

    python visualize_policy_v2.py                        # pick from runs/ interactively
    python visualize_policy_v2.py --model path/to/model.zip   # use a specific model
    python visualize_policy_v2.py --episodes 20          # run 20 episodes
"""

import os
import sys
import argparse
import importlib
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
sg = importlib.import_module('scenario_generation')

from stable_baselines3 import PPO
from airspace_env_v2 import AirspaceEnvV2, N_AGENTS

import bluesky as bs


# ── Model selection ───────────────────────────────────────────────────────────

def select_model(scripts_dir):
    parser = argparse.ArgumentParser(description='Visualise a trained PPO v2 policy')
    parser.add_argument('--model',    type=str, default=None,
                        help='Path to a model .zip file (default: pick from runs/)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    args = parser.parse_args()

    n_episodes = args.episodes

    if args.model:
        return args.model, n_episodes

    runs_dir = os.path.join(scripts_dir, 'runs')
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No runs/ folder found at {runs_dir}. Pass --model explicitly.")

    models = sorted(
        [os.path.join(runs_dir, d, 'model.zip')
         for d in os.listdir(runs_dir)
         if os.path.isfile(os.path.join(runs_dir, d, 'model.zip'))],
        reverse=True,   # newest first (folders are timestamped YYYYMMDD_HHMMSS)
    )

    if not models:
        raise FileNotFoundError("No model.zip files found in runs/. Train first or pass --model.")

    print("\nAvailable models:")
    for i, m in enumerate(models):
        tag = "  <- latest" if i == 0 else ""
        print(f"  [{i}]  {os.path.relpath(m, scripts_dir)}{tag}")

    choice = input(f"\nSelect [0-{len(models)-1}]  (Enter = latest): ").strip()
    idx = int(choice) if choice.isdigit() else 0
    print()
    return models[idx], n_episodes


# ── Settings ──────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(__file__)
MODEL_PATH, N_EPISODES = select_model(_SCRIPTS_DIR)
WINDOW_SIZE = 800
FPS         = 8

ACTION_LABELS = ['-10°', '+10°', '0°', 'back']

BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
RED    = (220,  30,  30)
GREEN  = ( 50, 180,  50)
BLUE   = ( 50, 130, 220)
ORANGE = (255, 140,   0)
GREY   = (160, 160, 160)

# One colour per aircraft — all three are controlled agents
AC_COLORS = [GREEN, BLUE, ORANGE]


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = PPO.load(MODEL_PATH)

    env = AirspaceEnvV2()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('PPO v2 — all agents controlled')
    clock     = pygame.time.Clock()
    font      = pygame.font.SysFont('monospace', 13)
    font_sm   = pygame.font.SysFont('monospace', 11)

    episode_stats = []

    for episode in range(N_EPISODES):
        obs_stack, _ = env.reset()   # shape (N_AGENTS, obs_dim)

        view          = sg.View(env.polygon, WINDOW_SIZE, WINDOW_SIZE)
        polygon_px    = [view.nm_to_pixels(v[0], v[1]) for v in env.polygon]
        sep_radius_px = view.km_length_to_pixels(sg.SEP_KM / 2)

        dest_px = {
            f'AC{i:02d}': view.latlon_to_pixels(
                env.aircraft_list[i]['dest_ll'][0],
                env.aircraft_list[i]['dest_ll'][1],
            )
            for i in range(N_AGENTS)
        }

        total_rewards = np.zeros(N_AGENTS)
        last_actions  = [0] * N_AGENTS
        step          = 0
        los_steps     = 0
        done          = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); return

            # Each agent acts on its own local observation
            actions = []
            for i in range(N_AGENTS):
                a, _ = model.predict(obs_stack[i], deterministic=True)
                actions.append(int(a))
            last_actions = actions

            obs_stack, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_rewards += rewards
            done = terminated or truncated
            step += 1
            if len(info['los_pairs']) > 0:
                los_steps += 1

            # ── Render ────────────────────────────────────────────────────
            screen.fill(WHITE)
            pygame.draw.polygon(screen, BLACK, polygon_px, width=2)

            aircraft_in_los = {cs for pair in bs.traf.cd.lospairs for cs in pair}

            for i in range(bs.traf.ntraf):
                cs    = bs.traf.id[i]
                ac_i  = int(cs[2:])
                px, py = view.latlon_to_pixels(bs.traf.lat[i], bs.traf.lon[i])
                d_px  = dest_px.get(cs)

                base_color = AC_COLORS[ac_i % len(AC_COLORS)]
                color = RED if cs in aircraft_in_los else base_color

                # Separation bubble
                pygame.draw.circle(screen, color, (px, py), sep_radius_px, 1)

                # Dashed line to destination
                if d_px is not None:
                    sg.draw_dashed_line(screen, color, (px, py), d_px)

                # Aircraft dot
                pygame.draw.circle(screen, color, (px, py), 5)

                # Callsign + last action
                label = f"{cs} {ACTION_LABELS[last_actions[ac_i]]}" if ac_i < len(last_actions) else cs
                screen.blit(font_sm.render(label, True, color), (px + 7, py - 7))

            # HUD
            hud = [
                f'Episode {episode+1}/{N_EPISODES}',
                f'Step {step}   T={bs.sim.simt:.0f}s',
                f'Active={len(info["active"])}   LoS-steps={los_steps}',
                f'R: AC00={total_rewards[0]:.1f}  AC01={total_rewards[1]:.1f}  AC02={total_rewards[2]:.1f}',
            ]
            for j, line in enumerate(hud):
                screen.blit(font.render(line, True, BLACK), (8, 8 + j * 16))

            # Legend
            for i, (color, label) in enumerate(zip(AC_COLORS, ['AC00', 'AC01', 'AC02'])):
                screen.blit(font_sm.render(f'● {label}', True, color),
                            (8, WINDOW_SIZE - 50 + i * 14))
            screen.blit(font_sm.render('● RED = separation violation', True, RED),
                        (8, WINDOW_SIZE - 12))

            pygame.display.flip()
            clock.tick(FPS)

        episode_stats.append({
            'steps':    step,
            'rewards':  total_rewards.copy(),
            'los':      los_steps,
        })
        mean_r = total_rewards.mean()
        print(f"Episode {episode+1:2d}  steps={step:3d}  "
              f"mean_reward={mean_r:6.2f}  LoS-steps={los_steps}")

    pygame.quit()

    # ── Validation summary ────────────────────────────────────────────────────
    all_mean   = [s['rewards'].mean() for s in episode_stats]
    all_steps  = [s['steps']          for s in episode_stats]
    all_los    = [s['los']            for s in episode_stats]
    clean_eps  = sum(1 for s in episode_stats if s['los'] == 0)

    print("\n── Validation Summary ──────────────────────────────────────────")
    print(f"  Episodes          : {N_EPISODES}")
    print(f"  Mean reward       : {np.mean(all_mean):7.3f}  ±  {np.std(all_mean):.3f}")
    print(f"  Min / Max reward  : {np.min(all_mean):.3f} / {np.max(all_mean):.3f}")
    print(f"  Mean episode len  : {np.mean(all_steps):.1f} steps")
    print(f"  Total LoS steps   : {sum(all_los)}")
    print(f"  Clean episodes    : {clean_eps}/{N_EPISODES}  "
          f"({100*clean_eps/N_EPISODES:.0f}%  zero LoS)")
    print()
    print("  Per-agent mean rewards:")
    for i in range(N_AGENTS):
        agent_rewards = [s['rewards'][i] for s in episode_stats]
        print(f"    AC{i:02d}: {np.mean(agent_rewards):6.3f}  ±  {np.std(agent_rewards):.3f}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()
