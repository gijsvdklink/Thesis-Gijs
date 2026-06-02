"""
visualise.py — one figure per observation feature of the v3 environment.
Saves all figures to figs/.

Run:  python visualise.py
"""

import math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────

C_OWN   = '#1A73E8'   # ownship / destination
C_CMD   = '#E53935'   # commanded heading
C_ACT   = '#2E7D32'   # actual heading
C_INT   = '#FF6F00'   # intruder
C_CPA   = '#7B1FA2'   # CPA point
C_GREY  = '#AAAAAA'
C_ARC   = '#888888'

def _setup(figsize=(5,5), title=''):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    return fig, ax

def _north(ax, x=0, y=0, length=0.3):
    ax.annotate('', xy=(x, y+length), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=C_GREY, lw=1))
    ax.text(x, y+length+0.03, 'N', color=C_GREY, ha='center', va='bottom', fontsize=8)

def _arrow(ax, angle_deg, length, color, lw=2.2, ls='-', label=None):
    rad = math.radians(angle_deg)
    dx, dy = math.sin(rad)*length, math.cos(rad)*length
    ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls))
    return mpatches.Patch(color=color, label=label) if label else None

def _arc(ax, a0, a1, r, color, label='', lw=1.5):
    """Aviation-angle arc (0=N CW). a0, a1 in degrees."""
    t0 = math.radians(90 - max(a0, a1))
    t1 = math.radians(90 - min(a0, a1))
    th = np.linspace(t0, t1, 80)
    ax.plot(r*np.cos(th), r*np.sin(th), color=color, lw=lw, ls='--')
    mid = (t0+t1)/2
    ax.text((r+0.07)*math.cos(mid), (r+0.07)*math.sin(mid),
            label, color=color, fontsize=9, ha='center', va='center')

def _save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


# ── 1. sin / cos heading error ────────────────────────────────────────────────

def fig_heading_error():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    cases = [
        (0,   r'On course  $\Delta\psi=0°$',   'sin=0.00, cos=1.00'),
        (45,  r'Right  $\Delta\psi=+45°$',      'sin=0.71, cos=0.71'),
        (180, r'Opposite  $\Delta\psi=180°$',   'sin=0.00, cos=−1.00'),
    ]
    for ax, (dpsi, title, vals) in zip(axes, cases):
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, pad=8)

        # Unit circle (faint)
        th = np.linspace(0, 2*math.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color='#EEEEEE', lw=1, zorder=0)

        # Destination arrow (0° = straight ahead in ego frame)
        ax.annotate('', xy=(0, 0.9), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=C_OWN, lw=2.2))
        ax.text(0, 1.0, r'$\psi_\mathrm{dest}$', color=C_OWN,
                ha='center', va='bottom', fontsize=9)

        # Error vector
        rad = math.radians(dpsi)
        sx, cy = math.sin(rad), math.cos(rad)
        ax.annotate('', xy=(sx*0.9, cy*0.9), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=C_CMD, lw=2.2, linestyle='--'))
        ax.text(sx*1.02, cy*1.02, r'$\psi_\mathrm{cmd}$', color=C_CMD,
                ha='center', va='center', fontsize=9)

        # sin / cos projection lines
        if dpsi not in (0, 180):
            ax.plot([sx*0.9, sx*0.9], [0, cy*0.9], color=C_CMD, lw=0.8, ls=':')
            ax.plot([0, sx*0.9], [cy*0.9, cy*0.9], color=C_CMD, lw=0.8, ls=':')
            ax.text(sx*0.9 + 0.05*np.sign(sx), -0.12, f'sin={sx:.2f}',
                    color=C_CMD, fontsize=8, ha='center')
            ax.text(-0.12, cy*0.9, f'cos={cy:.2f}',
                    color=C_CMD, fontsize=8, ha='right', va='center')

        ax.text(0, -1.22, vals, color='#555555', ha='center', fontsize=9)
        ax.plot(0, 0, 'ko', ms=6, zorder=5)

    fig.suptitle('Observation [0,1]: heading error as sin/cos unit vector',
                 fontsize=11, y=1.01)
    _save(fig, 'obs_heading_error.png')


# ── 2. Turn progress ──────────────────────────────────────────────────────────

def fig_turn_progress():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: geometry diagram
    ax = axes[0]
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Geometry: mid-turn state', fontsize=10, pad=8)

    psi_dest, psi_cmd, psi_act = 30, 75, 52
    tp = (psi_cmd - psi_act) / 30.0

    _north(ax)
    _arc(ax, psi_dest, psi_cmd, 0.55, C_ARC, rf'$\Delta\psi$')
    _arc(ax, psi_act,  psi_cmd, 0.72, C_INT, f'turn_prog={tp:.2f}', lw=2)

    patches = [
        _arrow(ax, psi_dest, 0.90, C_OWN, label=r'$\psi_\mathrm{dest}$'),
        _arrow(ax, psi_cmd,  0.78, C_CMD, ls='--', label=r'$\psi_\mathrm{cmd}$'),
        _arrow(ax, psi_act,  0.65, C_ACT, label=r'$\psi_\mathrm{actual}$'),
    ]
    ax.plot(0, 0, 'ko', ms=7, zorder=5)
    ax.legend(handles=[p for p in patches if p], loc='lower left', fontsize=8)

    # Right: turn_progress over time schematic
    ax2 = axes[1]
    steps = np.arange(0, 15)
    # Simulated: command issued at step 0 (30° deviation), aircraft turns at 3°/s, 10s/step
    deviation = np.array([min(30, max(0, 30 - s*3*10/30*10)) for s in steps])
    tp_vals   = deviation / 30.0
    ax2.plot(steps, tp_vals, color=C_INT, lw=2.5, marker='o', ms=5)
    ax2.axhline(0, color='#CCCCCC', lw=1)
    ax2.set_xlabel('Steps after action', fontsize=10)
    ax2.set_ylabel('turn_progress', fontsize=10)
    ax2.set_title('turn_progress decays as aircraft turns\n(30° cmd, ~3°/s turn rate, 10 s/step)',
                  fontsize=9)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)

    fig.suptitle('Observation [2]: turn_progress = clip((ψ_cmd − ψ_actual)/30°, −1, 1)',
                 fontsize=10, y=1.01)
    _save(fig, 'obs_turn_progress.png')


# ── 3. Clearance timer ────────────────────────────────────────────────────────

def fig_clearance_timer():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    steps = np.arange(0, 45)
    # Conflict resolves at step 10, timer then counts up to 30
    timer = np.zeros(len(steps))
    for i, s in enumerate(steps):
        if s <= 10:
            timer[i] = 0.0
        else:
            timer[i] = min((s - 10) / 30.0, 1.0)

    ax.fill_between(steps, timer, alpha=0.15, color=C_OWN)
    ax.plot(steps, timer, color=C_OWN, lw=2.5)
    ax.axvline(10, color=C_INT, lw=1.5, ls='--')
    ax.text(10.5, 0.55, 'conflict\nresolved', color=C_INT, fontsize=9)
    ax.axhline(1.0, color='#CCCCCC', lw=1, ls=':')
    ax.text(44, 1.02, 'safe to\ngo direct', color='#555555', fontsize=8, ha='right')

    ax.fill_betweenx([0, 1], 0, 10, alpha=0.07, color=C_INT)
    ax.text(5, 0.5, 'active\nconflict', color=C_INT, fontsize=9, ha='center')

    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('clearance_timer', fontsize=10)
    ax.set_title('Observation [3]: clearance_timer — steps since last urgency > 0, normalised by 30',
                 fontsize=10)
    ax.set_ylim(-0.05, 1.15); ax.set_xlim(0, 44)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    _save(fig, 'obs_clearance_timer.png')


# ── 4. Time to exit ───────────────────────────────────────────────────────────

def fig_time_to_exit():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Observation [4]: time_to_exit\ndistance to boundary along heading / lookahead',
                 fontsize=10, pad=8)

    # Polygon
    poly = np.array([[0,1],[0.8,0.6],[1,-0.2],[0.3,-1],[-0.6,-0.8],[-1,0.2],[-0.4,0.9],[0,1]])
    ax.fill(poly[:,0], poly[:,1], alpha=0.08, color='#1A73E8')
    ax.plot(poly[:,0], poly[:,1], color='#1A73E8', lw=1.5)

    # Aircraft
    ac = np.array([-0.1, -0.2])
    hdg = 40  # degrees
    rad = math.radians(hdg)
    dx, dy = math.sin(rad), math.cos(rad)
    ax.plot(*ac, 'ko', ms=8, zorder=5)
    ax.text(ac[0]+0.06, ac[1]-0.1, 'aircraft', fontsize=9, color='black')

    # Ray to boundary
    boundary_t = 1.35
    bx, by = ac[0]+dx*boundary_t, ac[1]+dy*boundary_t
    ax.annotate('', xy=(bx, by), xytext=tuple(ac),
                arrowprops=dict(arrowstyle='->', color=C_INT, lw=2, linestyle='--'))
    ax.plot(bx, by, 'o', color=C_INT, ms=8, zorder=5)
    ax.text(bx+0.07, by, 'boundary\nexit point', color=C_INT, fontsize=9, va='center')

    # Distance annotation
    mx, my = ac[0]+dx*boundary_t/2, ac[1]+dy*boundary_t/2
    ax.text(mx+0.12, my, 'd = distance\nalong heading', color=C_INT,
            fontsize=9, ha='left', va='center')

    ax.set_xlim(-1.4, 1.6); ax.set_ylim(-1.3, 1.4)
    _save(fig, 'obs_time_to_exit.png')


# ── 5. Intruder: fwd / right unit vector ─────────────────────────────────────

def fig_intruder_fwd_right():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (int_bearing, label_pos) in zip(axes, [(15, 'right'), (170, 'below')]):
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal'); ax.axis('off')

        own_hdg = 0  # heading north for clarity
        rad_own = math.radians(own_hdg)
        rad_int = math.radians(int_bearing)

        rel   = int_bearing - own_hdg
        fwd   = math.cos(math.radians(rel))
        right = math.sin(math.radians(rel))

        ax.set_title(
            f'Intruder at {int_bearing}° bearing\n'
            f'fwd={fwd:.2f}, right={right:.2f}',
            fontsize=10, pad=8)

        # Ownship heading arrow
        _arrow(ax, own_hdg, 0.85, C_OWN, lw=2.5)
        ax.plot(0, 0, 'o', color=C_OWN, ms=10, zorder=5)
        ax.text(0, -0.12, 'ownship', color=C_OWN, ha='center', fontsize=9)

        # Bearing to intruder
        ix = math.sin(rad_int)*1.0
        iy = math.cos(rad_int)*1.0
        ax.annotate('', xy=(ix, iy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=C_INT, lw=2))
        ax.plot(ix, iy, 'o', color=C_INT, ms=10, zorder=5)
        ax.text(ix*1.12, iy*1.12, 'intruder', color=C_INT,
                ha='center', fontsize=9)

        # fwd projection (along ownship heading = north)
        ax.plot([ix, ix], [0, iy], color=C_CPA, lw=1.2, ls=':')
        ax.plot([0, ix], [iy, iy], color=C_CPA, lw=1.2, ls=':')
        ax.text(ix + 0.08*np.sign(right+0.01), -0.08,
                f'right={right:.2f}', color=C_CPA, fontsize=8.5, ha='center')
        ax.text(0.08, iy*0.5, f'fwd={fwd:.2f}',
                color=C_CPA, fontsize=8.5, va='center')

    axes[0].set_title('Intruder slightly right & ahead\nfwd=+0.97, right=+0.26',
                      fontsize=10, pad=8)
    axes[1].set_title('Intruder almost behind\nfwd=−0.98, right=+0.17',
                      fontsize=10, pad=8)

    fig.suptitle('Observation [+1,+2]: fwd/right unit vector — resolves ahead/behind ambiguity',
                 fontsize=10, y=1.01)
    _save(fig, 'obs_intruder_fwd_right.png')


# ── 6. CPA side ───────────────────────────────────────────────────────────────

def fig_cpa_side():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    cases = [
        (+0.8, 'cpa_side = +0.8\n(intruder passes RIGHT → turn LEFT)'),
        (-0.7, 'cpa_side = −0.7\n(intruder passes LEFT → turn RIGHT)'),
    ]

    for ax, (cpa_val, title) in zip(axes, cases):
        ax.set_xlim(-1.5, 2.2); ax.set_ylim(-1.3, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10, pad=8)

        # Ownship: heading north
        own_pos = np.array([0.0, -0.8])
        ax.plot(*own_pos, 'o', color=C_OWN, ms=10, zorder=5)
        ax.text(own_pos[0]-0.15, own_pos[1]-0.12, 'ownship', color=C_OWN, fontsize=8.5, ha='right')
        ax.annotate('', xy=(own_pos[0], own_pos[1]+0.6), xytext=tuple(own_pos),
                    arrowprops=dict(arrowstyle='->', color=C_OWN, lw=2.2))

        # CPA point
        cpa_pos = np.array([cpa_val * 0.5, own_pos[1] + 0.8])
        ax.plot(*cpa_pos, 's', color=C_CPA, ms=9, zorder=5)
        ax.text(cpa_pos[0], cpa_pos[1]+0.12, 'CPA', color=C_CPA, ha='center', fontsize=9)

        # Sep standard circle at CPA
        th = np.linspace(0, 2*math.pi, 120)
        ax.plot(cpa_pos[0]+0.3*np.cos(th), cpa_pos[1]+0.3*np.sin(th),
                color=C_CPA, lw=1, ls=':', alpha=0.6)

        # Intruder trajectory (crossing)
        sign = np.sign(cpa_val)
        int_start = np.array([cpa_pos[0] - 1.0*sign, cpa_pos[1] - 0.5])
        int_end   = np.array([cpa_pos[0] + 0.6*sign, cpa_pos[1] + 0.4])
        ax.annotate('', xy=tuple(int_end), xytext=tuple(int_start),
                    arrowprops=dict(arrowstyle='->', color=C_INT, lw=2))
        ax.plot(*int_start, 'o', color=C_INT, ms=9, zorder=5)
        ax.text(int_start[0]-0.1, int_start[1]-0.12, 'intruder',
                color=C_INT, fontsize=8.5, ha='center')

        # cpa_side annotation
        ax.annotate('', xy=(cpa_pos[0], cpa_pos[1]),
                    xytext=(own_pos[0], cpa_pos[1]),
                    arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
        ax.text((own_pos[0]+cpa_pos[0])/2, cpa_pos[1]-0.13,
                f'cpa_side={cpa_val:.1f}', color='#555555', ha='center', fontsize=9)

    fig.suptitle('Observation [+3]: cpa_side — lateral offset at closest approach / sep_nm',
                 fontsize=10, y=1.01)
    _save(fig, 'obs_cpa_side.png')


# ── 7. Urgency ramp (vs old 1/tcpa) ──────────────────────────────────────────

def fig_urgency_ramp():
    fig, ax = plt.subplots(figsize=(8, 4))
    tcpa  = np.linspace(0, 300, 500)
    T_WARN = 180.0

    ramp   = np.clip((T_WARN - tcpa) / T_WARN, 0, 1)
    old    = np.where(tcpa > 0, 1.0 / np.maximum(tcpa, 1.0), 1.0)

    ax.plot(tcpa, ramp, color=C_OWN, lw=2.5, label=r'New: linear ramp  $\frac{T_w - t_{CPA}}{T_w}$')
    ax.plot(tcpa, old,  color=C_INT, lw=2.0, ls='--', label=r'Old: $\frac{1}{\max(t_{CPA},1)}$')

    ax.axvline(T_WARN, color=C_ARC, lw=1, ls=':', label=f'$T_w$ = {int(T_WARN)} s')
    ax.axhline(0.5, color='#CCCCCC', lw=0.8, ls=':')
    ax.text(T_WARN/2, 0.52, '90 s out → urgency 0.5\n> efficiency pull (0.2)',
            color=C_OWN, fontsize=8.5, ha='center')

    ax.set_xlabel('Time to CPA (s)', fontsize=10)
    ax.set_ylabel('Urgency score', fontsize=10)
    ax.set_title('Observation [+4] / reward: urgency ramp vs old formulation', fontsize=10)
    ax.set_xlim(0, 300); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    _save(fig, 'obs_urgency_ramp.png')


# ── 8. Urgency density (global) ───────────────────────────────────────────────

def fig_urgency_density():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: soft-OR aggregation across multiple intruders
    ax = axes[0]
    u_vals = np.linspace(0, 1, 200)
    for n, color in [(1,'#1A73E8'), (2,'#E53935'), (3,'#FF6F00'), (4,'#2E7D32')]:
        # n equal threats, each of urgency u
        soft_or = 1 - (1 - u_vals)**n
        ax.plot(u_vals, soft_or, color=color, lw=2, label=f'{n} threat(s)')
    ax.plot([0,1],[0,1], color='#CCCCCC', lw=1, ls=':')
    ax.set_xlabel('Single-pair urgency', fontsize=10)
    ax.set_ylabel('danger_a  (soft-OR)', fontsize=10)
    ax.set_title('Soft-OR aggregation\ndanger_a = 1 − ∏(1 − min(u,1))', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.spines[['top','right']].set_visible(False)

    # Right: urgency_density vs number of conflicts
    ax2 = axes[1]
    n_conf = np.arange(0, 12)
    for u, ls in [(0.3,'--'), (0.6,'-'), (1.0,':')]:
        density = np.minimum(n_conf * u / 10.0, 1.0)
        ax2.plot(n_conf, density, lw=2, ls=ls, label=f'u={u:.1f} per pair')
    ax2.set_xlabel('Number of conflict pairs', fontsize=10)
    ax2.set_ylabel('urgency_density', fontsize=10)
    ax2.set_title('Observation [20]: urgency_density = min(Σu / 10, 1)', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)

    _save(fig, 'obs_urgency_density.png')


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fig_heading_error()
    fig_turn_progress()
    fig_clearance_timer()
    fig_time_to_exit()
    fig_intruder_fwd_right()
    fig_cpa_side()
    fig_urgency_ramp()
    fig_urgency_density()
    print(f'\nAll figures saved to {FIGS}')
