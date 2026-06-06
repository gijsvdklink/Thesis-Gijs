"""
scenario_distribution.py — sample 10,000 v3_dalmau scenarios and visualise
the joint distributions of sector size, density, aircraft count, and episode
length.  No BlueSky simulation is run; only the scenario-generation logic from
reset() is replicated (polygon, aircraft count, density draw).
"""

import math
import random
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from polygenerator import random_convex_polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import scale as shapely_scale

# ── Mirror CONFIG from v3_dalmau (scenario-generation fields only) ────────────

N_SAMPLES        = 10_000
SEED             = 42

N_AIRCRAFT_FN    = lambda: random.randint(2, 15)
DENSITY_KM2_FN   = lambda: random.uniform(5_000.0, 15_000.0)
N_VERTICES_FN    = lambda: random.randint(5, 7)
MIN_CIRCULARITY  = 0.65
AC_SPEED_KT      = 450.0   # knots TAS
SIM_DT_S         = 0.5
ACTION_FREQ      = 10      # sub-steps per RL step → 5 s per step
CROSSINGS        = 2.5     # crossings_per_episode
KM_TO_NM         = 1.0 / 1.852
CMAP             = 'viridis'

OUT_PATH = os.path.join(os.path.dirname(__file__), 'scenario_distribution.png')

# ── Polygon helpers (mirror _make_polygon / polygon_diameter) ─────────────────

def _make_polygon(area_km2):
    target_nm2 = area_km2 * KM_TO_NM ** 2
    circ = 0.0
    scaled = None
    for _ in range(1000):
        raw    = ShapelyPolygon(random_convex_polygon(N_VERTICES_FN()))
        scaled = shapely_scale(raw,
                               xfact=math.sqrt(target_nm2 / raw.area),
                               yfact=math.sqrt(target_nm2 / raw.area),
                               origin='centroid')
        circ = 4 * math.pi * scaled.area / scaled.length ** 2
        if circ >= MIN_CIRCULARITY:
            break
    return scaled, float(circ)

def _diameter_nm(poly):
    minx, miny, maxx, maxy = poly.bounds
    return math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

def _max_steps(diam_nm):
    step_s = ACTION_FREQ * SIM_DT_S
    return max(50, round(CROSSINGS * diam_nm / AC_SPEED_KT * 3600.0 / step_s))

# ── Sampling ──────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

n_ac_arr    = np.empty(N_SAMPLES, dtype=int)
density_arr = np.empty(N_SAMPLES)
area_arr    = np.empty(N_SAMPLES)
diam_arr    = np.empty(N_SAMPLES)
ep_len_arr  = np.empty(N_SAMPLES, dtype=int)
circ_arr    = np.empty(N_SAMPLES)

print(f"Sampling {N_SAMPLES:,} v3_dalmau scenarios …")
for i in range(N_SAMPLES):
    n          = N_AIRCRAFT_FN()
    density    = DENSITY_KM2_FN()
    area_km2   = float(n * density)
    poly, circ = _make_polygon(area_km2)
    diam       = _diameter_nm(poly)

    n_ac_arr[i]    = n
    density_arr[i] = density
    area_arr[i]    = area_km2
    diam_arr[i]    = diam
    ep_len_arr[i]  = _max_steps(diam)
    circ_arr[i]    = circ

    if (i + 1) % 2000 == 0:
        print(f"  {i + 1:,} / {N_SAMPLES:,}")

print("Done sampling.\n")

# ── Summary statistics ─────────────────────────────────────────────────────────

def _stat(label, arr, fmt='.1f'):
    print(f"  {label:<22}  mean={arr.mean():{fmt}}  std={arr.std():{fmt}}"
          f"  min={arr.min():{fmt}}  max={arr.max():{fmt}}")

print(f"Summary over {N_SAMPLES:,} scenarios:")
_stat("n_aircraft",          n_ac_arr,    fmt='.2f')
_stat("density (km²/ac)",    density_arr, fmt='.0f')
_stat("area (km²)",          area_arr,    fmt='.0f')
_stat("diameter (NM)",       diam_arr,    fmt='.1f')
_stat("episode length (steps)", ep_len_arr, fmt='.1f')
_stat("circularity",         circ_arr,    fmt='.3f')

# ── Plot helpers ───────────────────────────────────────────────────────────────

def _hex(ax, x, y, xlabel, ylabel, title, bins=35, stat='count'):
    h = ax.hexbin(x, y, gridsize=bins, cmap=CMAP, mincnt=1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(h, ax=ax, label='count', pad=0.02)

def _hist(ax, x, xlabel, title, color, bins=40, integer=False):
    if integer:
        bins = np.arange(x.min() - 0.5, x.max() + 1.5, 1)
    ax.hist(x, bins=bins, color=color, edgecolor='white', linewidth=0.4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(title, fontsize=10)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    f'v3\_dalmau scenario distribution  (N = {N_SAMPLES:,}, seed = {SEED})',
    fontsize=13, y=0.99
)
gs = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── Row 0: input-space joint distributions ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
_hex(ax, n_ac_arr, density_arr / 1e3,
     'Aircraft count', 'Density (10³ km²/ac)',
     'n_ac × density')

ax = fig.add_subplot(gs[0, 1])
_hex(ax, n_ac_arr, area_arr / 1e3,
     'Aircraft count', 'Sector area (10³ km²)',
     'n_ac × sector area')

ax = fig.add_subplot(gs[0, 2])
_hex(ax, density_arr / 1e3, area_arr / 1e3,
     'Density (10³ km²/ac)', 'Sector area (10³ km²)',
     'density × sector area')

ax = fig.add_subplot(gs[0, 3])
_hex(ax, n_ac_arr, diam_arr,
     'Aircraft count', 'Sector diameter (NM)',
     'n_ac × sector diameter')

# ── Row 1: episode length ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
_hex(ax, n_ac_arr, ep_len_arr,
     'Aircraft count', 'Episode length (steps)',
     'n_ac × episode length')

ax = fig.add_subplot(gs[1, 1])
_hex(ax, diam_arr, ep_len_arr,
     'Sector diameter (NM)', 'Episode length (steps)',
     'diameter × episode length')

ax = fig.add_subplot(gs[1, 2])
_hex(ax, area_arr / 1e3, ep_len_arr,
     'Sector area (10³ km²)', 'Episode length (steps)',
     'area × episode length')

ax = fig.add_subplot(gs[1, 3])
_hex(ax, density_arr / 1e3, ep_len_arr,
     'Density (10³ km²/ac)', 'Episode length (steps)',
     'density × episode length')

# ── Row 2: marginal distributions ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
_hist(ax, n_ac_arr, 'Aircraft count', 'Aircraft count', 'steelblue', integer=True)
ax.set_xticks(range(2, 16))

ax = fig.add_subplot(gs[2, 1])
_hist(ax, density_arr / 1e3, 'Density (10³ km²/ac)', 'Density per aircraft', 'darkorange')

ax = fig.add_subplot(gs[2, 2])
_hist(ax, ep_len_arr, 'Episode length (steps)', 'Episode length', 'mediumseagreen')

ax = fig.add_subplot(gs[2, 3])
_hist(ax, circ_arr, 'Circularity', 'Sector circularity\n(4π·area/perimeter²)', 'mediumpurple')
ax.axvline(MIN_CIRCULARITY, color='red', linestyle='--', linewidth=1, label=f'min = {MIN_CIRCULARITY}')
ax.legend(fontsize=8)

plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {OUT_PATH}")
