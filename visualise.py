"""
visualise.py — figures for the v3_improved environment specification.
Covers: psi_cmd geometry, observation space, reward structure,
        action space, decision cycle, and urgency ramp.

Run:  python visualise.py
"""

import math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS, exist_ok=True)

# ── Shared palette ────────────────────────────────────────────────────────────
C_OWN  = '#1A73E8'   # ownship / actual heading
C_CMD  = '#E53935'   # commanded heading
C_DEST = '#2E7D32'   # destination bearing
C_INT  = '#FF6F00'   # intruder
C_CPA  = '#7B1FA2'   # CPA point
C_GREY = '#AAAAAA'
C_BG   = '#F8F8F8'


def _save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {name}')

def _arrow(ax, x0, y0, angle_deg, length, color, lw=2.2, ls='-', zorder=4, ms=16):
    rad = math.radians(angle_deg)
    dx, dy = math.sin(rad) * length, math.cos(rad) * length
    ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, mutation_scale=ms),
                zorder=zorder)

def _arc(ax, a0, a1, r, color, label='', lw=1.6, x0=0, y0=0):
    """Aviation-angle arc (0=N, CW). Draws from a0 to a1."""
    lo, hi = min(a0, a1), max(a0, a1)
    th = np.linspace(math.radians(90 - hi), math.radians(90 - lo), 80)
    ax.plot(x0 + r * np.cos(th), y0 + r * np.sin(th),
            color=color, lw=lw, ls='--')
    mid = math.radians(90 - (lo + hi) / 2)
    ax.text(x0 + (r + 0.08) * math.cos(mid), y0 + (r + 0.08) * math.sin(mid),
            label, color=color, fontsize=9, ha='center', va='center')


# ── 1. psi_cmd geometry ───────────────────────────────────────────────────────

def fig_psi_cmd():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    psi_dest_abs = 15          # destination bearing from north (slight right)
    cases = [
        ( 0,  0,  r'$\delta = 0°$ — direct',        'No offset: fly straight to destination'),
        (+45, 22, r'$\delta = +45°$ — right turn',  'Mid-turn: psi_actual lags behind psi_cmd'),
        (-45,-22, r'$\delta = -45°$ — left turn',   'Mid-turn: psi_actual lags behind psi_cmd'),
    ]

    for ax, (delta, lag, title, subtitle) in zip(axes, cases):
        psi_cmd    = psi_dest_abs + delta
        psi_actual = psi_cmd - lag

        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.5, 2.1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=11, pad=12, fontweight='bold')
        ax.text(0, -0.42, subtitle, fontsize=8.5, ha='center', color='#555555')

        # Faint north reference
        ax.annotate('', xy=(0, 0.35), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#DDDDDD', lw=1.0))
        ax.text(0, 0.38, 'N', color='#CCCCCC', fontsize=7.5, ha='center', va='bottom')

        # Destination star
        dr = math.radians(psi_dest_abs)
        dx_d, dy_d = math.sin(dr) * 1.85, math.cos(dr) * 1.85
        ax.plot(dx_d, dy_d, '*', color=C_DEST, ms=16, zorder=6)
        ax.text(dx_d + 0.1 * np.sign(math.sin(dr) + 0.01),
                dy_d + 0.06, 'Destination', color=C_DEST,
                fontsize=8.5, ha='left' if delta >= 0 else 'right', va='bottom')
        ax.plot([0, dx_d], [0, dy_d], color=C_DEST, lw=0.7, ls=':', alpha=0.4)

        # Three arrows
        _arrow(ax, 0, 0, psi_dest_abs, 1.10, C_DEST,  lw=2.2)
        _arrow(ax, 0, 0, psi_cmd,      0.90, C_CMD,   lw=2.2, ls='dashed')
        _arrow(ax, 0, 0, psi_actual,   0.72, C_OWN,   lw=2.5)

        # Aircraft dot
        ax.plot(0, 0, 'o', color='black', ms=9, zorder=7)
        ax.text(0.07, -0.08, 'aircraft', fontsize=8, color='black')

        # Arcs
        if abs(delta) > 1:
            _arc(ax, psi_dest_abs, psi_cmd, 0.48, C_CMD,
                 f'$\\delta$={delta:+d}°')
        if abs(lag) > 1:
            _arc(ax, psi_cmd, psi_actual, 0.63, C_OWN, 'turn\nprogress')

    legend_patches = [
        mpatches.Patch(color=C_DEST, label=r'$\psi_\mathrm{dest}$ — bearing to destination'),
        mpatches.Patch(color=C_CMD,  label=r'$\psi_\mathrm{cmd}$ — commanded heading ($\psi_\mathrm{dest}+\delta$)'),
        mpatches.Patch(color=C_OWN,  label=r'$\psi_\mathrm{actual}$ — aircraft heading (may lag during turn)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
    fig.suptitle(
        r'Commanded heading geometry: $\psi_\mathrm{cmd} = \psi_\mathrm{dest} + \delta$'
        '\n'
        r'$\psi_\mathrm{dest}$ is recomputed from current position at every step; '
        r'$\delta$ is the agent action offset',
        fontsize=11, y=1.03)
    _save(fig, 'fig_psi_cmd.png')


# ── 2. Observation geometry ───────────────────────────────────────────────────

def fig_obs_geometry():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(-130, 130); ax.set_ylim(-120, 140)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Observation space — ego-centric view (ownship heading = up)\n'
                 '25 features: 5 ownship state + 4 intruders × 5 geometry features',
                 fontsize=10, pad=12)

    # Sector polygon (faint background)
    poly = np.array([[0,110],[85,70],[100,-30],[40,-100],[-60,-90],[-105,30],[-50,100],[0,110]])
    ax.fill(poly[:,0], poly[:,1], alpha=0.07, color=C_OWN)
    ax.plot(poly[:,0], poly[:,1], color=C_OWN, lw=1.2, ls='--', alpha=0.5)

    # d_warn circle (75 NM horizon, scaled to plot)
    th = np.linspace(0, 2*math.pi, 200)
    ax.plot(115*np.cos(th), 115*np.sin(th), color='#EEEEEE', lw=1.0, ls=':')
    ax.text(115, 5, '$d_\\mathrm{warn}$\n75 NM', color='#CCCCCC',
            fontsize=7.5, ha='left', va='center')

    # Ownship at center
    ax.plot(0, 0, 's', color=C_OWN, ms=13, zorder=7)
    ax.text(0, -12, 'ownship', color=C_OWN, ha='center', fontsize=9, fontweight='bold')
    # Heading arrow (up = forward)
    ax.annotate('', xy=(0, 35), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=C_OWN, lw=2.5, mutation_scale=18),
                zorder=6)
    # Destination
    ax.plot(12, 125, '*', color=C_DEST, ms=14, zorder=6)
    ax.text(22, 125, 'destination\n$\\psi_\\mathrm{dest}$', color=C_DEST,
            fontsize=8.5, va='center')

    # Ownship state annotations (obs [0]–[4])
    state_notes = [
        (-55,  30, '[0,1]', '$\\sin\\Delta\\psi,\\cos\\Delta\\psi$\nheading error to dest', 'right'),
        (-55, -15, '[2]',   'turn_progress\n$(\\psi_\\mathrm{cmd}-\\psi_\\mathrm{act})/45°$', 'right'),
        ( 15, -55, '[3]',   'time_to_exit\n$d_\\mathrm{boundary}/(v\\cdot T_\\mathrm{look})$', 'left'),
        ( 15, -80, '[4]',   'speed_offset\n$(M_\\mathrm{cmd}-0.70)/0.12$', 'left'),
    ]
    for x, y, idx, label, ha in state_notes:
        ax.text(x, y, f'{idx}  {label}', fontsize=8, color='#333333', ha=ha, va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F4FF', edgecolor='#CCCCFF', alpha=0.9))

    # Intruders
    intruders = [
        ( 45,  80, C_INT,    'Intruder 1\n(highest urgency)', 1),
        (-55,  55, '#E53935','Intruder 2', 2),
        ( 80,  10, '#FF6F00','Intruder 3', 3),
        (-30, -60, '#888888','Intruder 4\n(diverging → sentinel)', 4),
    ]

    for ix, iy, col, label, slot in intruders:
        # Current position dot
        ax.plot(ix, iy, 'o', color=col, ms=10, zorder=5)
        ax.text(ix + 6, iy + 6, label, color=col, fontsize=8, va='bottom')

        # Position vector (x_norm, y_norm)
        ax.annotate('', xy=(ix, iy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.4,
                                    linestyle='-', mutation_scale=12, alpha=0.6),
                    zorder=3)

        # CPA point (projected)
        if slot < 4:
            tcpa_frac = 0.35 + slot * 0.08
            # velocity direction for intruder (pointing toward ownship roughly)
            vx = -ix * 0.4 * tcpa_frac
            vy = -iy * 0.4 * tcpa_frac
            cpax, cpay = ix + vx, iy + vy
            ax.plot(cpax, cpay, 's', color=col, ms=7, zorder=5, alpha=0.8)
            ax.plot([ix, cpax], [iy, cpay], color=col, lw=1.0, ls=':', alpha=0.6)
            ax.text(cpax + 5, cpay - 5,
                    f'CPA\n$\\hat{{t}}$={tcpa_frac:.2f}',
                    color=col, fontsize=7.5, va='top')
            # CPA offset vector from ownship
            ax.annotate('', xy=(cpax, cpay), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.0,
                                        linestyle='dashed', mutation_scale=10, alpha=0.5),
                        zorder=2)
        else:
            ax.text(ix - 5, iy - 12, '(1, 1, 0, 1, 1)\nsentinel', color=col,
                    fontsize=7.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5',
                              edgecolor='#CCCCCC', alpha=0.9))

    # Feature legend for intruder block
    legend_text = (
        'Per intruder slot [+0…+4]:\n'
        '[+0] $\\Delta\\tilde{x}$ — lateral / 75 NM\n'
        '[+1] $\\Delta\\tilde{y}$ — forward / 75 NM\n'
        '[+2] $\\Delta\\tilde{x}^\\mathrm{CPA}$ — CPA lateral / $d_\\mathrm{sep}$\n'
        '[+3] $\\Delta\\tilde{y}^\\mathrm{CPA}$ — CPA forward / $d_\\mathrm{sep}$\n'
        '[+4] $\\hat{t}^\\mathrm{CPA}$ — $t_\\mathrm{CPA}$ / $T_\\mathrm{look}$'
    )
    ax.text(38, -62, legend_text, fontsize=8, va='top', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8F0',
                      edgecolor=C_INT, alpha=0.95))

    _save(fig, 'fig_obs_geometry.png')


# ── 3. Actions ────────────────────────────────────────────────────────────────

def fig_actions():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                              gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Left: heading actions ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Heading actions (6)\n'
                 r'$\psi_\mathrm{cmd} = \psi_\mathrm{dest} + \delta$',
                 fontsize=10, pad=10)

    # Compass rose
    th = np.linspace(0, 2*math.pi, 200)
    ax.plot(1.35*np.cos(th), 1.35*np.sin(th), color='#EEEEEE', lw=1.0)
    ax.plot(0, 0, 'o', color='black', ms=10, zorder=7)

    psi_dest = 0   # destination straight ahead for clarity
    heading_actions = [
        (-45, '$\\delta=-45°$',  '#1565C0', '-', 1.25),
        (-30, '$\\delta=-30°$',  '#1A73E8', '-', 1.10),
        (  0, 'direct ($\\delta=0$)', C_DEST, '-', 1.10),
        (+30, '$\\delta=+30°$',  C_CMD,     '-', 1.10),
        (+45, '$\\delta=+45°$',  '#B71C1C', '-', 1.25),
    ]

    for delta, label, color, ls, length in heading_actions:
        psi = psi_dest + delta
        rad = math.radians(psi)
        dx, dy = math.sin(rad)*length, math.cos(rad)*length
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                                    linestyle=ls, mutation_scale=16),
                    zorder=4)
        # Label at tip
        offset_x = 0.12 * math.sin(rad)
        offset_y = 0.12 * math.cos(rad)
        ha = 'center'
        if delta < -20: ha = 'right'
        elif delta > 20: ha = 'left'
        ax.text(dx + offset_x, dy + offset_y, label, color=color,
                fontsize=9, ha=ha, va='center')

    # Destination star
    ax.plot(0, 1.58, '*', color=C_DEST, ms=16, zorder=6)
    ax.text(0, 1.72, 'Destination', color=C_DEST, ha='center', fontsize=8.5)

    # Hold action (dashed circle)
    ax.plot(0.65*np.cos(th), 0.65*np.sin(th), color=C_GREY, lw=1.2, ls='--', alpha=0.5)
    ax.text(0.68, 0, 'hold ($\\varnothing$)\nre-issue $\\psi_\\mathrm{cmd}$',
            color=C_GREY, fontsize=8, va='center')

    # Workload cost annotation
    ax.text(0, -1.62,
            'Resolution turn ($|\\delta|>0$): cost $c=1.0$\n'
            'Direct ($\\delta=0$) and hold ($\\varnothing$): cost $c=0$',
            ha='center', fontsize=8.5, color='#444444',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', alpha=0.9))

    # ── Right: speed actions ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 1.4); ax2.set_ylim(-0.3, 1.2)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Speed actions (4)\n'
                  r'$M_\mathrm{cmd} \leftarrow \mathrm{clip}(M_\mathrm{cmd}+\Delta M,\;0.70,\;0.82)$',
                  fontsize=10, pad=10)

    M_min, M_nom, M_max = 0.70, 0.78, 0.82
    def m2x(m): return (m - M_min) / (M_max - M_min)

    bar_y, bar_h = 0.45, 0.10
    # Background bar
    ax2.barh(bar_y, 1.0, height=bar_h, left=0, color='#EEEEEE',
             edgecolor='#CCCCCC', zorder=2)
    # Nominal region
    ax2.barh(bar_y, m2x(M_nom) - m2x(M_min), height=bar_h,
             left=m2x(M_min), color='#E3F2FD', edgecolor='none', zorder=2)

    # Boundaries
    for m, label in [(M_min, '$M=0.70$\n(min)'), (M_nom, '$M=0.78$\nnominal'),
                     (M_max, '$M=0.82$\n(MMO)')]:
        xm = m2x(m)
        ax2.axvline(xm, ymin=0.32, ymax=0.55, color='#555555', lw=1.2, zorder=3)
        ax2.text(xm, bar_y - 0.13, label, ha='center', fontsize=8, color='#444444')

    # Current Mach marker
    M_cur = 0.78
    xc = m2x(M_cur)
    ax2.plot(xc, bar_y + bar_h/2, 'v', color='black', ms=10, zorder=6)
    ax2.text(xc, bar_y + bar_h + 0.04, '$M_\\mathrm{cmd}$', ha='center',
             fontsize=9, color='black')

    # Delta arrows
    speed_actions = [
        (+0.04, '+0.04',  '#B71C1C'),
        (+0.02, '+0.02',  C_CMD),
        (-0.02, '-0.02',  C_OWN),
        (-0.04, '-0.04',  '#1565C0'),
    ]
    arrow_y_start = bar_y + bar_h/2
    for i, (dm, label, color) in enumerate(speed_actions):
        x_start = xc
        x_end   = m2x(M_cur + dm)
        y_row   = 0.80 + i * 0.085
        ax2.annotate('', xy=(x_end, y_row), xytext=(x_start, y_row),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                     mutation_scale=14),
                     zorder=4)
        ax2.plot([x_start, x_start], [arrow_y_start, y_row], color='#CCCCCC',
                 lw=0.8, ls=':', zorder=1)
        side = 'right' if dm > 0 else 'left'
        ax2.text(x_end + (0.04 if dm > 0 else -0.04), y_row,
                 f'$\\Delta M={label}$\ncost $c=0.5$',
                 color=color, fontsize=8, ha=side, va='center')

    _save(fig, 'fig_actions.png')


# ── 4. Reward structure ───────────────────────────────────────────────────────

def fig_reward():
    steps = np.arange(20)
    # Scenario: clean (0-3), conflict onset (4), ongoing (5-9), LoS (7), resolve (10),
    #           cooldown (11-15), clean (16-19)
    sigma = np.array([0,0,0,0, 1,1,1,1,1,1, 0, 0,0,0,0, 0, 0,0,0,0])

    r_conflict = np.zeros(20)
    for t in range(1, 20):
        was, now = sigma[t-1], sigma[t]
        if was == 1 and now == 0:
            r_conflict[t] = +1.0
        elif was == 0 and now == 1:
            r_conflict[t] = -1.0
        elif was == 1 and now == 1:
            r_conflict[t] = -1.0

    los_steps = [7]
    r_los = np.array([-3.0 if t in los_steps else 0.0 for t in steps])

    # r_nav: moderate drift during conflict, near-zero when on track
    r_nav = np.array([-0.05 if sigma[t] == 0 else -0.20 for t in steps])

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={'hspace': 0.05, 'height_ratios': [1, 1.4, 1, 1.4]})

    # Row 0: sigma state
    ax = axes[0]
    ax.step(steps, sigma, where='post', color=C_INT, lw=2.2)
    ax.fill_between(steps, sigma, step='post', alpha=0.15, color=C_INT)
    ax.set_ylabel('$\\sigma_t$', fontsize=10, rotation=0, labelpad=20)
    ax.set_yticks([0, 1]); ax.set_ylim(-0.2, 1.5)
    ax.set_yticklabels(['0\n(clear)', '1\n(conflict)'], fontsize=8)
    ax.set_title('Reward components over an episode (schematic)', fontsize=11, pad=8)
    for spine in ['top','right','bottom']: ax.spines[spine].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    # LoS marker on sigma
    for t in los_steps:
        ax.axvline(t, color=C_CPA, lw=1.5, ls=':', alpha=0.7)
        ax.text(t, 1.35, 'LoS', color=C_CPA, fontsize=8, ha='center')

    # Row 1: r_conflict
    ax = axes[1]
    colors = [C_DEST if v > 0 else (C_CMD if v < 0 else '#DDDDDD') for v in r_conflict]
    ax.bar(steps, r_conflict, color=colors, width=0.7, zorder=3)
    ax.axhline(0, color='#CCCCCC', lw=0.8)
    ax.set_ylabel('$r_\\mathrm{conflict}$', fontsize=10, rotation=0, labelpad=20)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1\n(new/ongoing)', '0', '+1\n(resolved)'], fontsize=8)
    ax.set_ylim(-1.5, 1.5)
    # Annotations
    for t, v, txt in [(4, -1.3, 'new'), (7, -1.3, 'ongoing'), (10, 0.7, 'resolved')]:
        ax.text(t, v, txt, ha='center', fontsize=7.5, color='#444444')
    for spine in ['top','right','bottom']: ax.spines[spine].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    # Row 2: r_LoS
    ax = axes[2]
    ax.bar(steps, r_los, color=C_CPA, width=0.7, alpha=0.85, zorder=3)
    ax.axhline(0, color='#CCCCCC', lw=0.8)
    ax.set_ylabel('$r_\\mathrm{LoS}$', fontsize=10, rotation=0, labelpad=20)
    ax.set_yticks([-3, 0])
    ax.set_ylim(-4, 0.5)
    ax.text(7, -3.6, '$-w_\\mathrm{LoS}=-3$', ha='center', fontsize=8, color=C_CPA)
    for spine in ['top','right','bottom']: ax.spines[spine].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    # Row 3: r_nav
    ax = axes[3]
    nav_colors = [C_INT if sigma[t] else C_OWN for t in steps]
    ax.bar(steps, r_nav, color=nav_colors, width=0.7, alpha=0.8, zorder=3)
    ax.axhline(0, color='#CCCCCC', lw=0.8)
    ax.set_ylabel('$r_\\mathrm{nav}$', fontsize=10, rotation=0, labelpad=20)
    ax.set_ylim(-0.35, 0.05)
    ax.set_xlabel('Step', fontsize=10)
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    ax.text(2,  -0.08, 'small drift\n(on track)', ha='center', fontsize=7.5, color=C_OWN)
    ax.text(7,  -0.26, 'larger drift\n(evasive hdg)', ha='center', fontsize=7.5, color=C_INT)

    # Shared x-axis ticks
    axes[-1].set_xticks(steps)
    axes[-1].set_xticklabels([str(s) for s in steps], fontsize=8)

    # Phase labels across top
    phase_labels = [(1.5, 'Clear'), (7, 'Conflict'), (10.5, 'Cooldown'), (17.5, 'Clear')]
    for x, lbl in phase_labels:
        axes[0].axvline(x - 0.5, color='#EEEEEE', lw=12, zorder=0, solid_capstyle='butt')
        axes[0].text(x, -0.15, lbl, ha='center', fontsize=8, color='#888888')

    _save(fig, 'fig_reward.png')


# ── 5. Decision cycle ─────────────────────────────────────────────────────────

def fig_decision_cycle():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.5, 14.5); ax.set_ylim(-1.2, 3.2)
    ax.axis('off')
    ax.set_title('Decision cycle: one RL step = 10 × 0.5 s BlueSky sub-steps = 5 s simulated',
                 fontsize=10, pad=8)

    # Draw two full step cycles
    for step_i in range(2):
        t0 = step_i * 7.0

        # Sub-step bar
        bar_colors = ['#E3F2FD'] * 9 + ['#BBDEFB']  # last sub-step darker (LoS check)
        for k in range(10):
            ax.barh(1.5, 0.55, height=0.55, left=t0 + 0.5 + k * 0.58,
                    color=bar_colors[k], edgecolor='#90CAF9', linewidth=0.6,
                    zorder=2)
        ax.text(t0 + 0.5 + 4.5 * 0.58, 2.18,
                '10 × 0.5 s sub-steps (physics runs, no new actions)',
                ha='center', fontsize=8.5, color='#1565C0')
        ax.text(t0 + 0.5 + 9.0 * 0.58, 1.05,
                'sub-step 10:\nLoS check',
                ha='center', fontsize=7.5, color='#1565C0')

        # Action applied marker (start of step)
        ax.axvline(t0 + 0.5, color=C_CMD, lw=2.0, ymin=0.35, ymax=0.85, zorder=4)
        ax.text(t0 + 0.5, 2.65, f'Step {step_i+1}\naction applied',
                ha='center', fontsize=8.5, color=C_CMD, fontweight='bold')
        ax.plot(t0 + 0.5, 1.5, 'v', color=C_CMD, ms=9, zorder=5)

        # End-of-step events
        t_end = t0 + 0.5 + 10 * 0.58
        ax.axvline(t_end, color=C_INT, lw=2.0, ymin=0.1, ymax=0.85, zorder=4)
        ax.text(t_end, -0.65,
                'exits processed\nurgency recomputed\nnew focus selected\nreward computed',
                ha='center', fontsize=7.5, color=C_INT,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                          edgecolor=C_INT, alpha=0.9))
        ax.plot(t_end, 1.5, '^', color=C_INT, ms=9, zorder=5)

        # Time axis labels
        ax.text(t0 + 0.5, -1.1, f't = {step_i * 5} s', ha='center',
                fontsize=8.5, color='#444444')
        ax.text(t_end,     -1.1, f't = {(step_i+1) * 5} s', ha='center',
                fontsize=8.5, color='#444444')

    # Obs arrow between steps
    ax.annotate('', xy=(7.5 + 0.5, 1.5), xytext=(7.0 + 0.5 - 0.1, 1.5),
                arrowprops=dict(arrowstyle='->', color=C_DEST, lw=2.0,
                                mutation_scale=14),
                zorder=5)
    ax.text(7.25 + 0.2, 2.0,
            'new obs returned\n(from NEW focus)',
            ha='center', fontsize=8, color=C_DEST,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor=C_DEST, alpha=0.9))

    # Reward arrow
    ax.annotate('', xy=(7.25, 0.7), xytext=(7.25, 1.1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.4,
                                mutation_scale=12))
    ax.text(7.25, 0.4, 'reward $r_t$\n(for OLD focus)', ha='center',
            fontsize=8, color='#555555')

    # Timeline base
    ax.axhline(1.5, color='#DDDDDD', lw=1.0, xmin=0.02, xmax=0.98, zorder=1)

    _save(fig, 'fig_decision_cycle.png')


# ── 6. Urgency ramp ───────────────────────────────────────────────────────────

def fig_urgency():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: urgency vs t_CPA
    ax = axes[0]
    T_WARN, T_LOOK = 600.0, 900.0
    t_cpa = np.linspace(0, T_LOOK + 50, 800)
    u = np.where(t_cpa <= T_WARN,
                 np.clip((T_WARN - t_cpa) / T_WARN, 0, 1),
                 0.0)
    ax.fill_between(t_cpa, u, alpha=0.15, color=C_OWN)
    ax.plot(t_cpa, u, color=C_OWN, lw=2.5,
            label=r'Predicted: $\frac{T_w - t_\mathrm{CPA}}{T_w}$')

    # LoS zone
    ax.axvspan(0, 0.01, alpha=0.3, color=C_CMD)
    ax.text(0.02 * T_LOOK, 1.06, 'LoS zone\n$u \\in [1, 10]$',
            color=C_CMD, fontsize=8.5, ha='left')

    ax.axvline(T_WARN, color=C_GREY, lw=1.2, ls='--')
    ax.text(T_WARN + 10, 0.52, '$T_\\mathrm{warn}=600$\\ s\n(10 min)',
            color=C_GREY, fontsize=8.5)
    ax.axvline(T_LOOK, color='#BBBBBB', lw=1.0, ls=':')
    ax.text(T_LOOK - 20, 0.07, '$T_\\mathrm{look}$', color='#BBBBBB',
            fontsize=8.5, ha='right')

    ax.axhline(0.8, color=C_CPA, lw=1.2, ls='--', alpha=0.7)
    ax.text(20, 0.82, '$u_\\mathrm{emerg}=0.8$\n(emergency override)',
            color=C_CPA, fontsize=8)

    ax.set_xlabel('Time to CPA (s)', fontsize=10)
    ax.set_ylabel('Urgency score $u_{ij}$', fontsize=10)
    ax.set_title('Urgency ramp: predicted conflict', fontsize=10)
    ax.set_xlim(0, T_LOOK + 50); ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # Right: urgency matrix heat-map example (4×4)
    ax2 = axes[1]
    U = np.array([
        [0.00, 0.72, 0.05, 0.00],
        [0.72, 0.00, 0.00, 0.31],
        [0.05, 0.00, 0.00, 0.00],
        [0.00, 0.31, 0.00, 0.00],
    ])
    labels = ['AC01', 'AC02', 'AC03', 'AC04']
    im = ax2.imshow(U, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{U[i,j]:.2f}', ha='center', va='center',
                     fontsize=10, color='white' if U[i,j] > 0.5 else 'black',
                     fontweight='bold')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Urgency matrix $\\mathbf{U}$ (example, 4 aircraft)\n'
                  'Focus $f$ = AC01 (row max = 0.72 with AC02)',
                  fontsize=9)
    # Highlight focus row
    for j in range(4):
        ax2.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 1,
                                     fill=False, edgecolor=C_OWN, lw=2.5))

    _save(fig, 'fig_urgency.png')


# ── 7. Focus commitment timeline ─────────────────────────────────────────────

def fig_focus_commitment():
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True,
                              gridspec_kw={'hspace': 0.05})

    steps = np.arange(25)
    # AC01 in conflict steps 2–12; AC02 emergency at step 10
    u_01 = np.array([0]*2 + [0.6,0.7,0.75,0.8,0.85,0.7,0.6,0.4,0.3] +
                    [0]*14)
    u_02 = np.zeros(25)
    u_02[10] = 0.85; u_02[11] = 0.9; u_02[12:15] = [0.7, 0.5, 0.2]

    focus = np.array(['AC01'] * 25, dtype=object)
    # Switches to AC02 at step 10 (emergency override), then back after cooldown
    focus[10:17] = 'AC02'
    focus[17:]   = 'AC01'

    # Row 0: urgency curves
    ax = axes[0]
    ax.plot(steps, u_01, color=C_OWN, lw=2.2, marker='o', ms=4,
            label='AC01 urgency')
    ax.plot(steps, u_02, color=C_INT, lw=2.2, marker='s', ms=4,
            label='AC02 urgency')
    ax.axhline(0.8, color=C_CPA, lw=1.2, ls='--',
               label='Emergency threshold (0.8)')
    ax.set_ylabel('$u_{fj}$', fontsize=10, rotation=0, labelpad=18)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_title('Focus commitment: AC01 controlled until AC02 reaches emergency urgency',
                 fontsize=10, pad=8)
    for s in ['top','right','bottom']: ax.spines[s].set_visible(False)

    # Row 1: focus identity
    ax2 = axes[1]
    focus_num = np.where(focus == 'AC01', 1, 2)
    ax2.step(steps, focus_num, where='post', color='black', lw=2.0)
    ax2.fill_between(steps, focus_num, 0, step='post', alpha=0.12, color='black')
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['AC01', 'AC02'], fontsize=9)
    ax2.set_ylim(0.3, 2.8)
    ax2.set_ylabel('Focus', fontsize=10, rotation=0, labelpad=20)
    # Shade commitment region
    ax2.axvspan(2, 10, alpha=0.07, color=C_OWN, label='Locked on AC01')
    ax2.axvspan(10, 17, alpha=0.07, color=C_INT, label='Emergency: AC02')
    ax2.axvspan(17, 22, alpha=0.05, color='#AAAAAA', label='Cooldown (5 steps)')
    ax2.legend(fontsize=8, loc='upper right')
    for s in ['top','right','bottom']: ax2.spines[s].set_visible(False)
    ax2.grid(alpha=0.3)
    ax2.text(10.5, 2.55, 'emergency\noverride', color=C_CPA,
             fontsize=7.5, ha='center')

    # Row 2: grace step markers
    ax3 = axes[2]
    grace_steps = [10, 17]
    grace = np.zeros(25)
    for g in grace_steps:
        grace[g] = 1
    ax3.bar(steps, grace, color=C_CPA, width=0.7, alpha=0.8)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['0', 'grace\nstep'], fontsize=8)
    ax3.set_ylim(-0.1, 1.4)
    ax3.set_ylabel('Grace', fontsize=9, rotation=0, labelpad=20)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_xticks(steps)
    ax3.set_xticklabels([str(s) for s in steps], fontsize=8)
    ax3.grid(alpha=0.3)
    for s in ['top','right']: ax3.spines[s].set_visible(False)
    ax3.text(10, 1.15, '$r_\\mathrm{conflict}=0$\n$\\sigma$ re-synced',
             ha='center', fontsize=7.5, color=C_CPA)
    ax3.text(17, 1.15, '$r_\\mathrm{conflict}=0$\n$\\sigma$ re-synced',
             ha='center', fontsize=7.5, color=C_CPA)

    _save(fig, 'fig_focus_commitment.png')


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fig_psi_cmd()
    fig_obs_geometry()
    fig_actions()
    fig_reward()
    fig_decision_cycle()
    fig_urgency()
    fig_focus_commitment()
    print(f'\nAll figures saved to {FIGS}')
