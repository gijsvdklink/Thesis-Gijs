"""Re-render urgency matrix PNGs from extracted data (no simulation required)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def render(matrix, cs_list, focus_cs, output_path):
    # Sort axes AC00, AC01, AC02 …
    order   = sorted(range(len(cs_list)), key=lambda i: cs_list[i])
    cs_list = [cs_list[i] for i in order]
    matrix  = matrix[np.ix_(order, order)]

    n         = len(cs_list)
    focus_idx = cs_list.index(focus_cs) if focus_cs in cs_list else -1

    cmap = LinearSegmentedColormap.from_list('conflict_urg', [
        '#1a7a3c', '#5ab758', '#f5e642', '#f5a623', '#c0392b',
    ])

    disp = np.clip(matrix, 0.0, 1.0).astype(float)
    np.fill_diagonal(disp, np.nan)
    masked = np.ma.array(disp, mask=np.isnan(disp))
    cmap.set_bad('#d8d8d8')

    fig_sz = max(6, n * 0.85)
    fig, ax = plt.subplots(figsize=(fig_sz + 1.8, fig_sz + 0.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f0f0f0')

    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1,
              aspect='equal', interpolation='nearest')

    if focus_idx >= 0:
        for i in range(n):
            for r, c in [(focus_idx, i), (i, focus_idx)]:
                if r == c:
                    continue
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.5, r - 0.5), 1, 1,
                    boxstyle='square,pad=0', linewidth=0,
                    facecolor='#1a6ed8', alpha=0.13))

    fs = max(8, 12 - n)
    for i in range(n):
        for j in range(n):
            u = matrix[i, j]
            if i == j or u <= 0:
                continue
            txt_col = 'white' if u > 0.55 else '#1a1a1a'
            ax.text(j, i, f'{u:.2f}', ha='center', va='center',
                    fontsize=fs, color=txt_col, fontweight='bold')

    if focus_idx >= 0:
        lw, bc = 3.2, '#1a6ed8'
        ax.add_patch(mpatches.FancyBboxPatch(
            (focus_idx - 0.5, -0.5), 1, n,
            boxstyle='square,pad=0', lw=lw, edgecolor=bc, facecolor='none'))
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, focus_idx - 0.5), n, 1,
            boxstyle='square,pad=0', lw=lw, edgecolor=bc, facecolor='none'))

    labels     = [f'{cs} ◀' if cs == focus_cs else cs for cs in cs_list]
    lbl_colors = ['#1a6ed8' if cs == focus_cs else '#222222' for cs in cs_list]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for tl, col in zip(ax.get_xticklabels(), lbl_colors):
        tl.set_color(col)
    for tl, col in zip(ax.get_yticklabels(), lbl_colors):
        tl.set_color(col)

    ax.tick_params(axis='both', length=0)
    ax.set_xlabel('Aircraft  (column)', fontsize=11)
    ax.set_ylabel('Aircraft  (row)',    fontsize=11)
    ax.set_title('Urgency Matrix  (ownship highlighted)', fontsize=13, pad=10)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,
                              norm=plt.Normalize(vmin=0, vmax=1)),
        ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Urgency  u', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


def make_matrix(cs_list, pairs):
    """pairs: list of (cs_i, cs_j, urgency)"""
    idx = {cs: i for i, cs in enumerate(cs_list)}
    n   = len(cs_list)
    m   = np.zeros((n, n))
    for ci, cj, u in pairs:
        m[idx[ci], idx[cj]] = u
        m[idx[cj], idx[ci]] = u
    return m


# ── Step 29 ──────────────────────────────────────────────────────────────────
cs29 = ['AC06', 'AC02', 'AC00', 'AC01', 'AC05', 'AC03', 'AC07', 'AC04']
m29  = make_matrix(cs29, [('AC02', 'AC00', 0.11)])
render(m29, cs29, 'AC00',
       os.path.join(BASE, 'normal.v4_n8_seed1085657196_step29_urgency_matrix.png'))

# ── Step 35 ──────────────────────────────────────────────────────────────────
cs35 = ['AC07', 'AC03', 'AC05', 'AC01', 'AC00', 'AC04', 'AC02', 'AC06']
m35  = make_matrix(cs35, [('AC07', 'AC02', 0.01), ('AC00', 'AC02', 0.19)])
render(m35, cs35, 'AC00',
       os.path.join(BASE, 'normal.v4_n8_seed1085657196_step35_urgency_matrix.png'))

# ── Step 60 ──────────────────────────────────────────────────────────────────
cs60 = ['AC05', 'AC07', 'AC01', 'AC00', 'AC04', 'AC06', 'AC02', 'AC03']
m60  = make_matrix(cs60, [('AC07', 'AC02', 0.35), ('AC00', 'AC02', 0.54)])
render(m60, cs60, 'AC00',
       os.path.join(BASE, 'normal.v4_n8_seed1085657196_step60_urgency_matrix.png'))

# ── Step 88 ──────────────────────────────────────────────────────────────────
cs88 = ['AC04', 'AC07', 'AC01', 'AC06', 'AC03', 'AC02', 'AC00', 'AC05']
m88  = make_matrix(cs88, [('AC07', 'AC02', 0.74), ('AC02', 'AC00', 0.92)])
render(m88, cs88, 'AC00',
       os.path.join(BASE, 'normal.v4_n8_seed1085657196_step88_urgency_matrix.png'))
