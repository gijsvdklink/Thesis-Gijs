"""
make_html.py -- HTML canvas visualisation of a v4 episode.

Runs the episode, captures lightweight JSON frame data, then writes a single
self-contained HTML file that plays the animation in any browser.

No pygame / matplotlib / ffmpeg required.

Run:
    python make_html.py
    python make_html.py --model path/to/ckpt.zip --frames 1500
"""

import argparse
import cv2
import io
import json
import math
import os
import pickle
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# -- NumPy shim ---------------------------------------------------------------
class _NumpyShim(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


# Robust unpickler for the VecNormalize .pkl. On top of the numpy._core rename,
# it replaces the numpy.random subtree with a harmless dummy: the saved RNG state
# (PCG64) fails to unpickle across numpy versions, but we never sample at eval --
# only obs_rms (mean/var) is needed to normalise observations. Without this the
# vecnorm silently fails to load and the policy is fed RAW obs, which (since it
# was trained with norm_obs=True) collapses it onto HOLD -> "nothing happens".
class _DummyRNG:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __getstate__(self): return None
    def __reduce__(self): return (_DummyRNG, ())

def _dummy_rng(*a, **k):
    return _DummyRNG()

class _VecnormUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        if module.startswith('numpy.random') or 'bit_generator' in module:
            return _dummy_rng
        return super().find_class(module, name)

from stable_baselines3.common import save_util

def _patched_json_to_data(json_string, custom_objects=None):
    import json as _j, base64, warnings
    data = {}
    for key, item in _j.loads(json_string).items():
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

# -----------------------------------------------------------------------------
import bluesky as bs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium import spaces as gym_spaces

# Select the environment module before it is used (default v4). Pre-scan argv so
# the module-level CONFIG/OBS_DIM below come from the chosen env; e.g.
#   python make_html.py --env test_env --model ...
import importlib
_env_name = 'v4'
for _i, _a in enumerate(sys.argv):
    if _a == '--env' and _i + 1 < len(sys.argv):
        _env_name = sys.argv[_i + 1]
    elif _a.startswith('--env='):
        _env_name = _a.split('=', 1)[1]
_envmod = importlib.import_module(f'Environments.{_env_name}')
AirspaceEnv, CONFIG, NM_TO_KM, latlon_to_nm, wrap_to_180, OBS_DIM = (
    _envmod.AirspaceEnv, _envmod.CONFIG, _envmod.NM_TO_KM,
    _envmod.latlon_to_nm, _envmod.wrap_to_180, _envmod.OBS_DIM)

# Per-env labels for the observation panel (fall back to the v4 ACAS Xu layout:
# 6 ownship states + 5 per-intruder states)
OBS_OWNSHIP_LABELS  = getattr(_envmod, 'OBS_OWNSHIP_LABELS',
                              ['sin Dpsi', 'cos Dpsi', 'v_own', 'turn_prog',
                               'conf_now', 'retn_conf'])
OBS_INTRUDER_LABELS = getattr(_envmod, 'OBS_INTRUDER_LABELS',
                              ['rho', 'theta', 'psi', 'vint', 'tau'])

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

HERE        = os.path.dirname(os.path.abspath(__file__))
MAX_FRAMES  = 1500
SEP_NM      = float(CONFIG['sep_nm'])

# No canonical default checkpoint; pass --model explicitly. If left empty, the
# episode runs with random actions (no-policy preview of the env).
DEFAULT_MODEL = ''

def _checkpoint_n_actions(model_path):
    """Read the discrete action count from a saved policy (action_net rows),
    so older checkpoints with a different action count still load."""
    try:
        import zipfile, io, torch
        with zipfile.ZipFile(model_path) as z:
            sd = torch.load(io.BytesIO(z.read('policy.pth')),
                            map_location='cpu', weights_only=False)
        return int(sd['action_net.weight'].shape[0])
    except Exception:
        return None


# -- Frame data collector -----------------------------------------------------

def _collect_frame(env, reward, cum_reward, los_steps, action=None, obs=None):
    U       = env._urgency_matrix
    cs_list = env._urgency_cs_list
    urg     = {}
    if U.size > 0:
        row_max = U.max(axis=1)
        for i, cs in enumerate(cs_list):
            urg[cs] = float(row_max[i])

    aircraft = []
    for cs in env._active_callsigns:
        idx = bs.traf.id2idx(cs)
        if idx < 0:
            continue
        pos = latlon_to_nm(CONFIG['center_ll'], bs.traf.lat[idx], bs.traf.lon[idx])
        dest_ll = env._destination_ll.get(cs)
        dest = None
        if dest_ll is not None:
            d = latlon_to_nm(CONFIG['center_ll'], float(dest_ll[0]), float(dest_ll[1]))
            dest = [round(float(d[0]), 2), round(float(d[1]), 2)]
        aircraft.append({
            'cs':       cs,
            'x':        round(float(pos[0]), 3),
            'y':        round(float(pos[1]), 3),
            'hdg':      round(float(bs.traf.hdg[idx]), 1),
            'cmd':      round(float(env._commanded_heading.get(cs, bs.traf.hdg[idx])), 1),
            'u':        round(float(urg.get(cs, 0.0)), 3),
            'dest':     dest,
            'focus':    cs == env._focus_cs,
            'sc':       env._steps_since_urgency.get(cs, CONFIG['focus_clear_steps']),
        })

    n_los  = int((U > 1.0).sum()) // 2 if U.size > 0 else 0
    n_conf = int((U > 0).sum())  // 2 if U.size > 0 else 0

    return {
        't':    round(float(bs.sim.simt), 0),
        'r':    round(float(reward), 4),
        'sr':   round(float(cum_reward), 2),
        'los':  n_los,
        'conf': n_conf,
        'lst':  los_steps,
        'a':    action if action is not None else -1,
        'obs':  [round(float(v), 3) for v in obs] if obs is not None else None,
        'ac':   aircraft,
    }

# -- Episode runner -----------------------------------------------------------

def run_episode(env, model, vecnorm, use_policy, seed, max_frames, hold_only=False):
    obs, _ = env.reset(seed=seed)
    polygon = [[round(float(v[0]), 3), round(float(v[1]), 3)] for v in env.polygon]

    frames    = []
    los_steps = 0
    cum_r     = 0.0
    done      = False
    n         = 0

    print(f'  seed={seed}  n_ac={env.n_aircraft}  max_steps={env._max_steps}  '
          f'cap={max_frames if max_frames > 0 else "episode end"}')

    while not done and (max_frames <= 0 or n < max_frames):
        if hold_only:
            action = 3   # HOLD — true no-op, no heading instruction issued
        elif use_policy:
            obs_n = vecnorm.normalize_obs(obs[np.newaxis]) if vecnorm else obs[np.newaxis]
            action = int(model.predict(obs_n, deterministic=True)[0][0])
        else:
            action = int(env.action_space.sample())   # cast off np.int64 -> JSON-serialisable

        obs_used = obs.tolist()   # the observation the agent acted on this step

        obs, reward, terminated, truncated, info = env.step(action)
        done   = terminated or truncated
        cum_r += float(reward)
        n     += 1
        if info.get('los_pairs', 0) > 0:
            los_steps += 1

        frames.append(_collect_frame(env, reward, cum_r, los_steps, action, obs_used))

        if n % 100 == 0:
            print(f'    frame {n:5d}  T={bs.sim.simt:.0f}s  '
                  f'r={reward:+.3f}  Sr={cum_r:+.2f}')

    return polygon, frames, done

# -- HTML template ------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>v4 ATC — {title}</title>
<style>
  body {{ margin:0; background:#060d1a; display:flex; flex-direction:column;
           align-items:center; padding:16px; font-family:monospace; color:#a0b8d8; }}
  canvas {{ border:1px solid #1e3a5c; width:760px; height:760px; }}
  #controls {{ margin-top:10px; display:flex; gap:12px; align-items:center; }}
  button {{ background:#1a2e4a; color:#a0b8d8; border:1px solid #2d5a8a;
             padding:5px 14px; cursor:pointer; border-radius:3px; font-family:monospace; }}
  button:hover {{ background:#253d5e; }}
  label {{ font-size:12px; }}
  input[type=range] {{ width:120px; }}
  #info {{ font-size:12px; color:#6080a8; margin-top:6px; }}
  #wrap {{ display:flex; gap:14px; align-items:flex-start; }}
  #obspanel {{ font-size:12px; line-height:1.45; color:#9fc0e8; background:#0a1428;
               border:1px solid #1e3a5c; border-radius:4px; padding:10px 12px;
               margin:0; white-space:pre; min-width:230px; }}
  #obspanel b {{ color:#c8daf0; }}
</style>
</head>
<body>
<div id="wrap">
<canvas id="c" width="2160" height="2160"></canvas>
<pre id="obspanel">observation</pre>
</div>
<div id="controls">
  <button id="btn">Pause</button>
  <label>Speed <input type="range" id="spd" min="1" max="20" value="{fps}"></label>
  <span id="spd_val">{fps}x</span>
  <label><input type="checkbox" id="loop"> Loop</label>
  <label>Frame <input type="range" id="scrub" min="0" max="{max_frame}" value="0" style="width:200px"></label>
  <span id="fnum">0/{max_frame}</span>
</div>
<div id="info">{mode_str} &nbsp;|&nbsp; {n_ac} aircraft &nbsp;|&nbsp; sep={sep_nm} NM</div>
<script>
const FRAMES   = {frames_json};
const POLYGON  = {polygon_json};
const ENDED    = {ended};      // true when the episode finished within the recording
const SEP_NM   = {sep_nm};
const CLEAR_ST = {focus_clear_steps};
const EMERG_U  = {emerg_u};
const NM2KM    = {nm2km};

// -- Colours
const C = {{
  bg:      '#0a1428', sector: '#111e36', border: '#2d5a8a',
  grey:    '#7090aa', orange: '#ff9820', red:    '#f04040',
  cyan:    '#00d4ff', yellow: '#e8c820', purple: '#b050e0',
  green:   '#40c878', dim:    '#6080a8', text:   '#c8daf0',
  route:   '#2a5a8a', cmd:    '#2a4a6a',
}};

function urgColor(u) {{
  if (u > 1.0) return C.red;
  if (u > 0.0) return C.orange;
  return C.green;
}}

// -- View transform
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const S = W / 750;   // UI scale: geometry tuned at 750px, rendered at 4K

function buildView(polygon) {{
  let xs = polygon.map(p => p[0] * NM2KM);
  let ys = polygon.map(p => p[1] * NM2KM);
  let cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  let cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  let pad = SEP_NM * NM2KM * 3;
  let span = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys)) + 2*pad;
  let sc = Math.min(W, H) / span;
  return {{ cx, cy, sc }};
}}

const view = buildView(POLYGON);

function toXY(x_nm, y_nm) {{
  let xk = x_nm * NM2KM, yk = y_nm * NM2KM;
  return [
    (xk - view.cx) * view.sc + W / 2,
    (view.cy - yk) * view.sc + H / 2,
  ];
}}

function nmToPx(nm) {{ return nm * NM2KM * view.sc; }}

// -- Drawing helpers
function dashedCircle(x, y, r, col, lw=1, dash=[5,5]) {{
  ctx.save();
  ctx.strokeStyle = col;
  ctx.lineWidth   = lw * S;
  ctx.setLineDash(dash.map(d => d * S));
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2*Math.PI); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}}

function dashedLine(x1, y1, x2, y2, col, lw=1, dash=[5,6]) {{
  ctx.save();
  ctx.strokeStyle = col; ctx.lineWidth = lw * S;
  ctx.setLineDash(dash.map(d => d * S));
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}}

// -- Sector polygon
function drawSector() {{
  ctx.beginPath();
  POLYGON.forEach((p, i) => {{
    let [x, y] = toXY(p[0], p[1]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.closePath();
  ctx.fillStyle   = C.sector;
  ctx.fill();
  ctx.strokeStyle = C.border;
  ctx.lineWidth   = 1.5 * S;
  ctx.stroke();
}}

// -- Main draw
function drawFrame(idx) {{
  const f = FRAMES[idx];
  const sepPx    = nmToPx(SEP_NM / 2);   // half-sep radius
  const vecPx    = nmToPx({spd_nms} * 90);  // 90-second track

  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  drawSector();

  // Aircraft
  f.ac.forEach(ac => {{
    let [x, y] = toXY(ac.x, ac.y);
    let col    = urgColor(ac.u);
    let hdgRad = (ac.hdg - 90) * Math.PI / 180;  // canvas: 0=right, so rotate

    // Route line (faint dashed)
    if (ac.dest) {{
      let [dx, dy] = toXY(ac.dest[0], ac.dest[1]);
      dashedLine(x, y, dx, dy, C.route, 1.2, [4, 7]);
    }}

    // Commanded heading line when drifting
    let drift = Math.abs(((ac.hdg - ac.cmd + 540) % 360) - 180);
    if (drift > 2) {{
      let cr = ac.cmd * Math.PI / 180;
      let cLen = nmToPx(SEP_NM * 2.5);
      let cxe = x + Math.sin(cr) * cLen;
      let cye = y - Math.cos(cr) * cLen;
      ctx.save();
      ctx.strokeStyle = C.cmd; ctx.lineWidth = S;
      ctx.setLineDash([3*S, 5*S]);
      ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(cxe,cye); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }}

    // 90-second velocity line (not arrow)
    let hr = ac.hdg * Math.PI / 180;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5 * S;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.sin(hr)*vecPx, y - Math.cos(hr)*vecPx);
    ctx.stroke();

    // Separation ring (half-sep, dashed)
    dashedCircle(x, y, sepPx, col + '88', ac.focus ? 1.5 : 0.8);

    // Focus ring (solid cyan)
    if (ac.focus) {{
      ctx.strokeStyle = C.cyan; ctx.lineWidth = 2 * S;
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x, y, sepPx + sepPx*0.15, 0, 2*Math.PI); ctx.stroke();
      // Cooldown ring
      if (ac.sc < CLEAR_ST && ac.u === 0) {{
        ctx.strokeStyle = C.yellow; ctx.lineWidth = S;
        ctx.setLineDash([2*S, 4*S]);
        ctx.beginPath(); ctx.arc(x, y, sepPx + sepPx*0.3, 0, 2*Math.PI); ctx.stroke();
        ctx.setLineDash([]);
      }}
    }} else if (ac.u >= EMERG_U) {{
      ctx.strokeStyle = C.purple; ctx.lineWidth = 1.2 * S;
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x, y, sepPx + sepPx*0.22, 0, 2*Math.PI); ctx.stroke();
    }}

    // Aircraft dot
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, 5 * S, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 0.8 * S;
    ctx.stroke();

    // Label
    let lbl = ac.cs;
    if (ac.u > 0)   lbl += '  u=' + ac.u.toFixed(2);
    if (drift > 3)  lbl += '  ' + drift.toFixed(0) + 'deg';
    if (ac.focus)   lbl += '  <<';
    ctx.fillStyle   = ac.focus ? C.cyan : col;
    ctx.font        = Math.round(11 * S) + 'px monospace';
    ctx.fillText(lbl, x + 10*S, y - 6*S);
  }});

  // HUD — only reward counter, bottom strip
  ctx.fillStyle = '#060e1e';
  ctx.fillRect(0, H - 34*S, W, 34*S);
  ctx.fillStyle = C.dim;
  ctx.font = Math.round(12 * S) + 'px monospace';
  ctx.fillText('T=' + f.t + 's   frame=' + (idx+1) + '/' + FRAMES.length
               + '   LoS=' + f.los + '   conf=' + f.conf
               + '   LoS-steps=' + f.lst, 10*S, H - 18*S);
  const ACT = ['-60','-45','-30','HOLD','+30','+45','+60','DIRECT','SPD+','SPD-'];
  let rcol = f.r < -1 ? C.red : C.dim;
  ctx.fillStyle = rcol;
  let actStr = (f.a >= 0) ? ('     action=' + ACT[f.a]) : '';
  ctx.fillText('r=' + f.r.toFixed(3) + '   Sr=' + f.sr.toFixed(1) + actStr, 10*S, H - 5*S);

  // End-of-recording banner on the final frame
  if (idx >= FRAMES.length - 1) {{
    ctx.fillStyle = 'rgba(6,14,30,0.85)';
    ctx.fillRect(W/2 - 260*S, 14*S, 520*S, 34*S);
    ctx.fillStyle = ENDED ? C.green : C.yellow;
    ctx.font      = 'bold ' + Math.round(15*S) + 'px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(ENDED ? 'EPISODE END'
                       : 'RECORDING CAP REACHED - EPISODE CONTINUES BEYOND CAPTURE',
                 W/2, 36*S);
    ctx.textAlign = 'left';
  }}

  updateObsPanel(f);
}}

// -- Observation panel (focus aircraft's obs vector the agent acted on)
function fmtv(v) {{ return (v >= 0 ? ' ' : '') + v.toFixed(2); }}
function updateObsPanel(f) {{
  const el = document.getElementById('obspanel');
  if (!f.obs) {{ el.textContent = 'observation: n/a'; return; }}
  const o       = f.obs;
  const ownLbl  = {own_labels_json};
  const intrLbl = {intr_labels_json};
  const nOwn    = ownLbl.length;                       // v4: 6 ownship states
  const perIntr = intrLbl.length;                      // v4: 5 states per intruder
  const nIntr   = Math.floor((o.length - nOwn) / perIntr);
  const ACT = ['-60','-45','-30','HOLD','+30','+45','+60','DIRECT','SPD+','SPD-'];
  const pad = (t, w) => (t + ' '.repeat(w)).slice(0, w);
  let s = 'OBSERVATION  (focus aircraft)\n';
  s += 'action = ' + (f.a >= 0 ? ACT[f.a] : '-') + '\n\n';
  s += 'ownship\n';
  for (let i = 0; i < nOwn; i++) s += '  ' + pad(ownLbl[i] || ('o'+i), 9) + '  ' + fmtv(o[i]) + '\n';
  s += '\nintruders ' + intrLbl.map(l => l.padStart(5).slice(0,5)).join(' ') + '\n';
  for (let k = 0; k < nIntr; k++) {{
    const b = nOwn + k * perIntr;
    let row = '  I' + k + '  ';
    for (let j = 0; j < perIntr; j++) row += ' ' + fmtv(o[b+j]);
    s += row + '\n';
  }}
  el.textContent = s;
}}

// -- Playback
let frameIdx = 0;
let playing  = true;
let delay    = Math.round(1000 / {fps});
let timer    = null;

const btn     = document.getElementById('btn');
const loopBox = document.getElementById('loop');
const spdIn  = document.getElementById('spd');
const spdVal = document.getElementById('spd_val');
const scrub  = document.getElementById('scrub');
const fnum   = document.getElementById('fnum');

function tick() {{
  drawFrame(frameIdx);
  scrub.value = frameIdx;
  fnum.textContent = (frameIdx+1) + '/' + FRAMES.length;
  if (frameIdx >= FRAMES.length - 1) {{
    if (loopBox.checked) {{
      frameIdx = 0;
      if (playing) timer = setTimeout(tick, delay);
      return;
    }}
    playing = false;             // stop on the last frame instead of looping
    btn.textContent = 'Replay';
    return;
  }}
  frameIdx += 1;
  if (playing) timer = setTimeout(tick, delay);
}}

btn.onclick = () => {{
  if (!playing && frameIdx >= FRAMES.length - 1) frameIdx = 0;   // replay
  playing = !playing;
  btn.textContent = playing ? 'Pause' : 'Play';
  if (playing) tick();
}};

spdIn.oninput = () => {{
  delay = Math.round(1000 / parseInt(spdIn.value));
  spdVal.textContent = spdIn.value + 'x';
}};

scrub.oninput = () => {{
  frameIdx = parseInt(scrub.value);
  drawFrame(frameIdx);
  fnum.textContent = (frameIdx+1) + '/' + FRAMES.length;
}};

tick();
</script>
</body>
</html>
"""

# -- cv2 frame renderer (for MP4 export) -------------------------------------

W_MP4, H_MP4 = 2160, 2160        # 4K UHD (square frame)
SC_MP4       = H_MP4 / 900       # geometry was tuned at 900px

# BGR colours matching the HTML palette
_BGR = {
    'bg':     ( 40,  20,  10),
    'sector': ( 55,  30,  17),
    'border': (138,  90,  45),
    'grey':   (170, 144, 112),
    'orange': ( 32, 152, 255),
    'red':    ( 64,  64, 240),
    'cyan':   (255, 212,   0),
    'yellow': ( 32, 200, 232),
    'purple': (224,  80, 176),
    'green':  (120, 200,  64),
    'dim':    (168, 128,  96),
    'text':   (240, 218, 200),
    'route':  (138,  90,  42),
    'cmd':    (106,  74,  42),
}

def _uc_bgr(u):
    if u > 1.0: return _BGR['red']
    if u > 0.0: return _BGR['orange']
    return _BGR['green']


class _ViewCV:
    def __init__(self, polygon, pad_nm=15):
        xs = [p[0] * NM_TO_KM for p in polygon]
        ys = [p[1] * NM_TO_KM for p in polygon]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        span = max(max(xs)-min(xs), max(ys)-min(ys)) + 2 * pad_nm * NM_TO_KM
        self.sc = min(W_MP4, H_MP4) / span
        self.cx, self.cy = cx, cy

    def to_px(self, x_nm, y_nm):
        xk, yk = x_nm * NM_TO_KM, y_nm * NM_TO_KM
        return (int((xk - self.cx) * self.sc + W_MP4 / 2),
                int((self.cy - yk) * self.sc + H_MP4 / 2))

    def nm_px(self, nm):
        return max(1, int(nm * NM_TO_KM * self.sc))


def _dashed_circle_cv(img, cx, cy, r, color, thickness=1, n_segs=40):
    for i in range(n_segs):
        if i % 2 == 0:
            a0 = 2 * math.pi * i / n_segs
            a1 = 2 * math.pi * (i + 1) / n_segs
            p0 = (int(cx + r * math.cos(a0)), int(cy + r * math.sin(a0)))
            p1 = (int(cx + r * math.cos(a1)), int(cy + r * math.sin(a1)))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def render_frame_cv2(frame, polygon, view):
    img = np.zeros((H_MP4, W_MP4, 3), dtype=np.uint8)
    img[:] = _BGR['bg']
    s = SC_MP4   # pixel-value scale relative to the original 900px layout

    # Sector fill
    pts = np.array([view.to_px(p[0], p[1]) for p in polygon], np.int32)
    cv2.fillPoly(img, [pts], _BGR['sector'])
    cv2.polylines(img, [pts], True, _BGR['border'], round(2*s), cv2.LINE_AA)

    sep_px  = view.nm_px(SEP_NM / 2)
    vec_px  = view.nm_px(float(CONFIG['ac_speed']) / 3600.0 * 90)
    clr_st  = CONFIG['focus_clear_steps']
    emg_u   = CONFIG['focus_emergency_u']

    for ac in frame['ac']:
        px, py = view.to_px(ac['x'], ac['y'])
        col    = _uc_bgr(ac['u'])
        hdg    = ac['hdg']

        # Route line (very faint dashed — draw as thin line)
        if ac['dest']:
            dx, dy = view.to_px(ac['dest'][0], ac['dest'][1])
            overlay = img.copy()
            cv2.line(overlay, (px, py), (dx, dy), _BGR['route'], round(s), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Commanded heading line when drifting
        drift = abs(((hdg - ac['cmd'] + 540) % 360) - 180)
        if drift > 2:
            cr   = math.radians(ac['cmd'])
            clen = view.nm_px(SEP_NM * 2.5)
            cex  = int(px + math.sin(cr) * clen)
            cey  = int(py - math.cos(cr) * clen)
            cv2.line(img, (px, py), (cex, cey), _BGR['cmd'], round(s), cv2.LINE_AA)

        # Velocity line (90s track)
        hr  = math.radians(hdg)
        vex = int(px + math.sin(hr) * vec_px)
        vey = int(py - math.cos(hr) * vec_px)
        cv2.line(img, (px, py), (vex, vey), col, round(2*s), cv2.LINE_AA)

        # Separation ring (half-sep, dashed)
        _dashed_circle_cv(img, px, py, sep_px, col, round(s))

        # Focus / cooldown / emergency rings
        if ac['focus']:
            cv2.circle(img, (px, py), sep_px + max(3, sep_px//6),
                       _BGR['cyan'], round(2*s), cv2.LINE_AA)
            if ac['sc'] < clr_st and ac['u'] == 0:
                _dashed_circle_cv(img, px, py, sep_px + max(5, sep_px//4),
                                  _BGR['yellow'], round(s))
        elif ac['u'] >= emg_u:
            cv2.circle(img, (px, py), sep_px + max(4, sep_px//5),
                       _BGR['purple'], round(s), cv2.LINE_AA)

        # Aircraft dot
        cv2.circle(img, (px, py), round(5*s), col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), round(5*s), (200, 200, 200), round(s), cv2.LINE_AA)

        # Label
        parts = [ac['cs']]
        if ac['u'] > 0:   parts.append(f"u={ac['u']:.2f}")
        if drift > 3:      parts.append(f"{drift:.0f}d")
        if ac['focus']:    parts.append('<<')
        lbl_col = _BGR['cyan'] if ac['focus'] else col
        cv2.putText(img, '  '.join(parts), (px + round(10*s), py - round(5*s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35*s, lbl_col, round(s), cv2.LINE_AA)

    # HUD strip
    cv2.rectangle(img, (0, H_MP4 - round(40*s)), (W_MP4, H_MP4), (14, 20, 6), -1)
    hud1 = (f"T={frame['t']:.0f}s   LoS={frame['los']}   conf={frame['conf']}"
            f"   LoS-steps={frame['lst']}")
    hud2 = f"r={frame['r']:+.3f}   Sr={frame['sr']:+.1f}"
    cv2.putText(img, hud1, (round(10*s), H_MP4 - round(22*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38*s, _BGR['dim'], round(s), cv2.LINE_AA)
    hud2_col = _BGR['red'] if frame['r'] < -1 else _BGR['dim']
    cv2.putText(img, hud2, (round(10*s), H_MP4 - round(7*s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38*s, hud2_col, round(s), cv2.LINE_AA)

    return img


def write_mp4(polygon, frames, output_path, fps=8):
    view   = _ViewCV(polygon)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W_MP4, H_MP4))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W_MP4, H_MP4))

    for f in frames:
        img = render_frame_cv2(f, polygon, view)
        writer.write(img)
    writer.release()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'  Saved -> {output_path}  ({size_mb:.1f} MB)')


# -- HTML writer --------------------------------------------------------------

def write_html(polygon, frames, output_path, mode_str, n_ac, fps=8, ended=True):
    spd_nms = float(CONFIG['ac_speed']) / 3600.0
    html = HTML_TEMPLATE.format(
        title             = mode_str,
        ended             = 'true' if ended else 'false',
        mode_str          = mode_str,
        n_ac              = n_ac,
        sep_nm            = SEP_NM,
        focus_clear_steps = CONFIG['focus_clear_steps'],
        emerg_u           = CONFIG['focus_emergency_u'],
        nm2km             = NM_TO_KM,
        spd_nms           = round(spd_nms, 5),
        fps               = fps,
        max_frame         = len(frames) - 1,
        frames_json       = json.dumps(frames, separators=(',', ':')),
        polygon_json      = json.dumps(polygon, separators=(',', ':')),
        own_labels_json   = json.dumps(OBS_OWNSHIP_LABELS,  separators=(',', ':')),
        intr_labels_json  = json.dumps(OBS_INTRUDER_LABELS, separators=(',', ':')),
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'  Saved -> {output_path}  ({size_mb:.1f} MB)')

# -- Urgency matrix visualisation --------------------------------------------

def render_urgency_matrix_png(matrix, cs_list, focus_cs, output_path):
    """Save a styled heatmap of the urgency matrix with the ownship highlighted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    # Sort by callsign so axes always read AC00, AC01, AC02 …
    order    = sorted(range(len(cs_list)), key=lambda i: cs_list[i])
    cs_list  = [cs_list[i] for i in order]
    matrix   = matrix[np.ix_(order, order)]

    n         = len(cs_list)
    focus_idx = cs_list.index(focus_cs) if focus_cs in cs_list else -1

    # Academic green → yellow → orange → red (matches reference style)
    cmap = LinearSegmentedColormap.from_list('conflict_urg', [
        '#1a7a3c',   # u=0.00  deep green  (no threat)
        '#5ab758',   # u=0.25  light green
        '#f5e642',   # u=0.50  yellow
        '#f5a623',   # u=0.75  orange
        '#c0392b',   # u=1.00  red         (near-LoS)
    ])

    # Diagonal → NaN so it renders as grey (masked)
    disp = np.clip(matrix, 0.0, 1.0).astype(float)
    np.fill_diagonal(disp, np.nan)
    masked = np.ma.array(disp, mask=np.isnan(disp))
    cmap.set_bad('#d8d8d8')   # diagonal cell colour

    fig_sz = max(6, n * 0.85)
    fig, ax = plt.subplots(figsize=(fig_sz + 1.8, fig_sz + 0.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f0f0f0')

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1,
                   aspect='equal', interpolation='nearest')

    # Faint blue tint on ownship row and column
    if focus_idx >= 0:
        for i in range(n):
            for r, c in [(focus_idx, i), (i, focus_idx)]:
                if r == c:
                    continue
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.5, r - 0.5), 1, 1,
                    boxstyle='square,pad=0', linewidth=0,
                    facecolor='#1a6ed8', alpha=0.13))

    # Cell text — skip zeros and diagonal; dark/light based on urgency
    fs = max(8, 12 - n)
    for i in range(n):
        for j in range(n):
            u = matrix[i, j]
            if i == j or u <= 0:
                continue
            txt_col = 'white' if u > 0.55 else '#1a1a1a'
            ax.text(j, i, f'{u:.2f}', ha='center', va='center',
                    fontsize=fs, color=txt_col, fontweight='bold')

    # Bold blue border around ownship row and column
    if focus_idx >= 0:
        lw, bc = 3.2, '#1a6ed8'
        ax.add_patch(mpatches.FancyBboxPatch(   # column
            (focus_idx - 0.5, -0.5), 1, n,
            boxstyle='square,pad=0', lw=lw, edgecolor=bc, facecolor='none'))
        ax.add_patch(mpatches.FancyBboxPatch(   # row
            (-0.5, focus_idx - 0.5), n, 1,
            boxstyle='square,pad=0', lw=lw, edgecolor=bc, facecolor='none'))

    # Axis labels — ownship in blue, rest in dark grey
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
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa')

    # White grid lines between cells
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.8)
    ax.tick_params(which='minor', length=0)

    ax.set_title('Urgency Matrix  (ownship highlighted)', color='#111111',
                 fontsize=13, pad=14)
    ax.set_xlabel('Aircraft  (column)', color='#444444', fontsize=10)
    ax.set_ylabel('Aircraft  (row)',    color='#444444', fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Urgency  u', color='#444444', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor('#aaaaaa')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'  Saved -> {output_path}  ({size_mb:.1f} MB)')


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',         default='v4',
                        help='Environment module under Environments/ (default: v4)')
    parser.add_argument('--model',       default=DEFAULT_MODEL)
    parser.add_argument('--no-policy',   action='store_true')
    parser.add_argument('--hold',        action='store_true',
                        help='Always send HOLD (action 3) — true no-op baseline')
    parser.add_argument('--mp4',         action='store_true', help='Also save as .mp4')
    parser.add_argument('--frames',      type=int,   default=0,
                        help='Frame cap (0 = record until the episode ends)')
    parser.add_argument('--fps',         type=int,   default=8)
    parser.add_argument('--episodes',    type=int,   default=1)
    parser.add_argument('--n-aircraft',  type=int,   default=None)
    parser.add_argument('--density',     type=float, default=None)
    parser.add_argument('--circularity', type=float, default=None)
    parser.add_argument('--n-vertices',  type=int,   default=None,
                        help='Override sector polygon vertex count (higher = rounder)')
    parser.add_argument('--crossings',   type=float, default=None,
                        help='Override crossings_per_episode (longer episodes)')
    parser.add_argument('--t_warn',      type=float, default=None,
                        help='Warning horizon in seconds (default: 360 = 6 min)')
    parser.add_argument('--seed',        type=int,   default=None,
                        help='Fix episode seed for reproducibility')
    parser.add_argument('--compare-twarn', action='store_true',
                        help='Save two PNGs comparing T_warn=600s vs T_warn=300s (same scenario)')
    parser.add_argument('--matrix-viz',      action='store_true',
                        help='Save airspace PNG + urgency matrix PNG (8 aircraft, hold actions)')
    parser.add_argument('--matrix-min-conf', type=int,   default=3,
                        help='Min conflict pairs to capture (default 3)')
    parser.add_argument('--matrix-max-conf', type=int,   default=None,
                        help='Max conflict pairs to capture (default: no limit)')
    parser.add_argument('--matrix-min-u',    type=float, default=0.3,
                        help='Min urgency for capture (default 0.3)')
    parser.add_argument('--matrix-frame-at', type=int,   default=None,
                        help='Capture exactly this step number (ignores conf/urgency thresholds)')
    args = parser.parse_args()

    # Apply CONFIG overrides (read by env.reset(), so must be set before any episode)
    if args.n_aircraft  is not None: CONFIG['n_aircraft']           = lambda n=args.n_aircraft: n
    if args.density     is not None: CONFIG['rho']                  = lambda d=args.density: d
    if args.circularity is not None: CONFIG['min_circularity']      = args.circularity
    if args.n_vertices  is not None: CONFIG['n_vertices']           = lambda v=args.n_vertices: v
    if args.crossings   is not None: CONFIG['crossings_per_episode']= args.crossings
    if args.t_warn      is not None: CONFIG['t_warn']               = float(args.t_warn)

    # ------------------------------------------------------------------ #
    # --compare-twarn: two PNGs of the same scenario, different T_warn   #
    # ------------------------------------------------------------------ #
    if args.compare_twarn:
        fixed_seed = args.seed  # None → search across random seeds

        def _run_pair(seed):
            res = {}
            for tw in (600.0, 300.0):
                env_tw = AirspaceEnv(t_warn=tw)
                poly_tw, frames_tw, _ = run_episode(
                    env_tw, None, None, False, seed, 0, hold_only=True)
                res[tw] = (poly_tw, frames_tw)
            return res

        def _best_frame(results):
            """Return (fidx, diff) where diff = conf_600 - conf_300 is maximised,
            preferring frames where conf_600 >= 1 and conf_300 == 0."""
            frames_600 = results[600.0][1]
            frames_300 = results[300.0][1]
            n = min(len(frames_600), len(frames_300))
            # Ideal: 600s sees conflict, 300s does not
            ideal = [(i, frames_600[i]['conf'] - frames_300[i]['conf'])
                     for i in range(n)
                     if frames_600[i]['conf'] >= 1 and frames_300[i]['conf'] == 0]
            if ideal:
                return max(ideal, key=lambda x: x[1])[0], True
            # Fall back: largest gap in conf count
            diffs = [(i, frames_600[i]['conf'] - frames_300[i]['conf'])
                     for i in range(n) if frames_600[i]['conf'] >= 1]
            if diffs:
                return max(diffs, key=lambda x: x[1])[0], False
            return len(frames_600) // 2, False

        # Search up to 8 seeds for the ideal (T600=conflict, T300=clear) frame
        results = None
        seed    = fixed_seed if fixed_seed is not None else int.from_bytes(os.urandom(4), 'big')
        for attempt in range(8):
            print(f'Attempt {attempt+1}: seed={seed}')
            results = _run_pair(seed)
            fidx, is_ideal = _best_frame(results)
            if is_ideal:
                print(f'  -> Good seed found!\n')
                break
            if fixed_seed is not None:
                break   # user pinned a seed; use it regardless
            seed = int.from_bytes(os.urandom(4), 'big')

        frames_600 = results[600.0][1]
        polygon    = results[600.0][0]
        ref_frame  = frames_600[fidx]
        print(f'Target frame : {fidx}  T={ref_frame["t"]}s  '
              f'conf(600s)={ref_frame["conf"]}  '
              f'conf(300s)={results[300.0][1][fidx]["conf"]}')

        view = _ViewCV(polygon)
        for tw in (600.0, 300.0):
            poly, frames = results[tw]
            # Patch cosmetics so both images share the same episode timestamp
            # and the same focus aircraft (determined by the 600s run)
            f = dict(frames[fidx])
            f['t'] = ref_frame['t']          # align HUD time to 600s run
            # Sync focus flag: use 600s run's focus aircraft
            focus_cs_600 = next((ac['cs'] for ac in ref_frame['ac'] if ac['focus']), None)
            f['ac'] = [dict(ac, focus=(ac['cs'] == focus_cs_600)) for ac in f['ac']]
            img  = render_frame_cv2(f, poly, view)
            path = os.path.join(HERE,
                                f'{_env_name}_twarn{int(tw)}_seed{seed}.png')
            cv2.imwrite(path, img)
            print(f'Saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)')
        return

    # ------------------------------------------------------------------ #
    # --matrix-viz: airspace PNG + urgency matrix heatmap                 #
    # ------------------------------------------------------------------ #
    if args.matrix_viz:
        n_ac    = args.n_aircraft if args.n_aircraft is not None else 8
        density = args.density    if args.density    is not None else 1/5000  # high density default
        CONFIG['n_aircraft'] = lambda _n=n_ac: _n
        CONFIG['rho']        = lambda _d=density: _d

        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
        captured = None

        for attempt in range(12):
            print(f'Attempt {attempt+1}: seed={seed}')
            env_mv = AirspaceEnv()
            obs, _ = env_mv.reset(seed=seed)
            polygon = [[round(float(v[0]), 3), round(float(v[1]), 3)]
                       for v in env_mv.polygon]
            done, step = False, 0
            cum_r = 0.0
            best  = None   # (n_conf, u_max, step, capture)

            while not done and step < (args.frames or 1500):
                obs, reward, terminated, truncated, info = env_mv.step(3)  # HOLD
                done   = terminated or truncated
                cum_r += float(reward)
                step  += 1

                U      = env_mv._urgency_matrix
                n_conf = int((U > 0).sum()) // 2 if U.size > 0 else 0
                u_max  = float(U.max()) if U.size > 0 else 0.0

                min_conf   = args.matrix_min_conf
                max_conf   = args.matrix_max_conf if args.matrix_max_conf is not None else 9999
                min_u      = args.matrix_min_u
                frame_at   = args.matrix_frame_at

                hit = (frame_at is not None and step == frame_at and env_mv._focus_cs is not None)
                hit = hit or (frame_at is None
                              and min_conf <= n_conf <= max_conf
                              and u_max >= min_u
                              and env_mv._focus_cs is not None)

                if hit:
                    captured = {
                        'frame':    _collect_frame(env_mv, reward, cum_r, 0, 3,
                                                   obs.tolist()),
                        'matrix':   U.copy(),
                        'cs_list':  list(env_mv._urgency_cs_list),
                        'focus_cs': env_mv._focus_cs,
                        'polygon':  polygon,
                        'seed':     seed,
                    }
                    break
                # Keep best fallback within the conf range
                if n_conf >= 1 and env_mv._focus_cs is not None:
                    if best is None or (n_conf, u_max) > (best[0], best[1]):
                        best = (n_conf, u_max, step, {
                            'frame':    _collect_frame(env_mv, reward, cum_r, 0, 3,
                                                       obs.tolist()),
                            'matrix':   U.copy(),
                            'cs_list':  list(env_mv._urgency_cs_list),
                            'focus_cs': env_mv._focus_cs,
                            'polygon':  polygon,
                            'seed':     seed,
                        })

            if captured is None and best is not None:
                captured = best[3]
                n_conf, u_max, step = best[0], best[1], best[2]

            if captured is not None:
                print(f'  -> frame at step {step}  '
                      f'n_conf={n_conf}  u_max={u_max:.3f}  focus={captured["focus_cs"]}\n')
                break
            seed = int.from_bytes(os.urandom(4), 'big')

        if captured is None:
            print('No suitable frame found after 12 attempts.')
            return

        tag  = f'{_env_name}_n{n_ac}_seed{captured["seed"]}_step{step}'
        view = _ViewCV(captured['polygon'])

        # Airspace PNG
        path_ac = os.path.join(HERE, f'{tag}_airspace.png')
        cv2.imwrite(path_ac, render_frame_cv2(
            captured['frame'], captured['polygon'], view))
        print(f'  Saved -> {path_ac}  ({os.path.getsize(path_ac)/1e6:.1f} MB)')

        # Urgency matrix PNG
        path_mx = os.path.join(HERE, f'{tag}_urgency_matrix.png')
        render_urgency_matrix_png(
            captured['matrix'], captured['cs_list'],
            captured['focus_cs'], path_mx)
        return

    # ------------------------------------------------------------------ #
    # Normal episode mode                                                  #
    # ------------------------------------------------------------------ #
    use_policy = (not args.no_policy) and (not args.hold) and args.model and os.path.exists(args.model)
    model   = None
    vecnorm = None

    env = AirspaceEnv()   # created up front so we can load the model with its exact spaces

    if use_policy:
        print(f'Loading model   : {args.model}')
        # Spaces passed explicitly (checkpoints from other numpy/SB3 versions fail to
        # deserialise them). Obs dim comes from the env; the action count is read from
        # the checkpoint itself, so an older policy with fewer actions (e.g. trained
        # before fly-direct was added) still loads -- the env accepts the subset.
        n_act = _checkpoint_n_actions(args.model) or env.action_space.n
        if n_act != env.action_space.n:
            print(f'  checkpoint has {n_act} actions (env has {env.action_space.n}); '
                  f'loading with Discrete({n_act})')
        model = PPO.load(args.model, custom_objects={
            'observation_space': env.observation_space,
            'action_space':      gym_spaces.Discrete(n_act),
        })
        vn = args.model.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vn):
            print(f'Loading vecnorm : {vn}')
            try:
                with open(vn, 'rb') as fh:
                    vecnorm = _VecnormUnpickler(fh).load()
                vecnorm.set_venv(DummyVecEnv([AirspaceEnv]))
                vecnorm.training    = False
                vecnorm.norm_reward = False
                print(f'  vecnorm loaded (norm_obs={vecnorm.norm_obs})')
            except Exception as e:
                # NOT harmless: policy was trained with norm_obs=True, so raw obs
                # collapse it onto HOLD. Fail loudly rather than silently mislead.
                print(f'  !! vecnorm FAILED to load ({e}) -- policy will see RAW obs!')
                vecnorm = None

    stem     = os.path.splitext(os.path.basename(args.model))[0] if use_policy else ('hold_only' if args.hold else 'no_policy')
    mode_str = f'{os.path.basename(args.model)}' if use_policy else ('HOLD only (no actions)' if args.hold else 'no policy (random)')

    cond = ''
    if args.n_aircraft is not None: cond += f'_n{args.n_aircraft}'
    if args.density    is not None: cond += f'_d{args.density:.0e}'

    print(f'Mode     : {mode_str}')
    print(f'Episodes : {args.episodes}\n')

    for ep in range(1, args.episodes + 1):
        seed    = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
        ep_tag  = f'_ep{ep}' if args.episodes > 1 else ''
        outpath = os.path.join(HERE, f'{_env_name}_{stem}{cond}{ep_tag}.html')
        print(f'-- Episode {ep}/{args.episodes} ---')
        polygon, frames, ended = run_episode(env, model, vecnorm, use_policy, seed, args.frames, hold_only=args.hold)
        write_html(polygon, frames, outpath, mode_str,
                   env.n_aircraft, fps=args.fps, ended=ended)
        if args.mp4:
            mp4_path = outpath.replace('.html', '.mp4')
            print(f'  Rendering MP4 ...')
            write_mp4(polygon, frames, mp4_path, fps=args.fps)

if __name__ == '__main__':
    main()
