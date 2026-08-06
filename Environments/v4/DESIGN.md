# v4 environment -- design notes

How the ATC conflict-resolution environment works, and why each piece is built the way it
is. Organised by design decision rather than by file. Code excerpts are the real source;
line references point at `Environments/v4/`.

- [Architecture](#architecture)
- [Scenarios](#scenarios)
- [Urgency](#urgency)
- [Focus selection](#focus-selection)
- [Observations](#observations)
- [Actions](#actions)
- [Rewards](#rewards)
- [Action-response delays](#action-response-delays)

---

## Architecture

Air traffic control is inherently multi-agent -- every aircraft could act at once -- but
multi-agent RL is hard to train. This environment sidesteps that with an **explicit
ranking mechanism**: each step, one aircraft (the *focus ship*) is selected, and that
aircraft alone produces the observation, receives the action, and generates the reward.
All other traffic keeps flying, and the ranking re-runs every step.

| module | responsibility |
| --- | --- |
| `config.py` | tunables, derived constants, action layout, workload costs |
| `geometry.py` | coordinate transforms, aircraft-state helpers |
| `conflict.py` | separation maths -- time-to-LoS, urgency, return-blocked, LoS check |
| `sector.py` | random sector polygons and entry-route planning |
| `env.py` | the `AirspaceEnv` gymnasium class |

BlueSky handles flight dynamics. `conflict.py` is pure functions with no environment
state, which is what makes the urgency logic testable in isolation.

### Reproducibility

Active aircraft live in a `set` of callsign strings, and Python randomises string hashing
per process, so iterating that set directly gives a different order in every run. That
order feeds urgency tie-breaks, intruder slot assignment and the order exits consume the
RNG -- enough to make two runs at the same seed diverge within ~20 steps. Every such
iteration is therefore `sorted()`. A given `--seed` now reproduces the same trajectories
in any process, which is what lets the three delay arms be compared as a controlled
experiment rather than three differently-seeded runs.

The step loop:

```python
def step(self, action):
    action = int(action)
    self._spawn_due_aircraft()
    self._update_route_headings()

    # The aircraft that HELD the focus when this action was chosen. Reward is charged
    # to it, not to whichever aircraft the focus moves to at the end of the step.
    acting_cs = self._focus_cs
    if acting_cs:
        self._issue_instruction(acting_cs, action)
    self._update_return_to_route_headings()   # re-aim returning aircraft before propagating

    self._advance_simulation()

    self._los_this_step = any_loss_of_separation(self._active_callsigns)
    self._step_count += 1

    self._remove_exited_aircraft()
    self._focus_cs = self._select_focus_aircraft()
    reward         = self._compute_reward(acting_cs, action)
    self._record_step_stats(action, reward)
```

Note that the reward is computed for `acting_cs` -- the aircraft that *held* the focus when
the action was chosen -- not for the newly selected focus. Credit goes to the aircraft that
actually acted.

---

## Scenarios

### Density is the controlled variable, not size

```python
n_ac     = CONFIG['n_aircraft']()
area_km2 = float(n_ac / CONFIG['rho']())
poly = make_sector_polygon(area_km2)
```

```python
    'n_aircraft':            lambda: random.randint(15, 30),            # sampled per episode
    'rho':                   lambda: random.uniform(1/25000, 1/10000),  # sampled per episode; area = n/rho
```

Aircraft count and density are sampled independently per episode, and the sector area is
*derived* from them. Two episodes with very different aircraft counts still present comparable
traffic density, so the policy learns to handle density rather than memorising sector
sizes. Sectors end up roughly 236-528 NM across.

### Sector shape: varied but not degenerate

```python
def make_sector_polygon(area_km2):
    """A random convex polygon of the requested area, centred at the origin (NM frame)."""
    target_nm2 = area_km2 * KM_TO_NM ** 2
    scaled = None
    for _ in range(1000):
        raw   = ShapelyPolygon(random_convex_polygon(CONFIG['n_vertices']()))
        scale = math.sqrt(target_nm2 / raw.area)
        scaled = shapely_scale(raw, xfact=scale, yfact=scale, origin='centroid')
        if _circularity(scaled) >= CONFIG['min_circularity']:
            break
```

6-12 vertices, rejected unless circularity >= 0.7. Varied enough to prevent overfitting to
a single geometry, round enough to avoid slivers where every route is a 10 NM clip of a
corner.

### Routes that reward doing nothing

```python
def plan_entry_route(polygon, sector, n_sectors):
    minx, miny, maxx, maxy = polygon.bounds
    dest_dist = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2) * CONFIG['dest_dist_factor']
    min_chord = CONFIG['min_chord_nm']

    for _ in range(CONFIG['max_placement_tries']):
        t_spawn  = (sector + CONFIG['spawn_jitter']()) / n_sectors
        t_ref    = (t_spawn + 0.5 + CONFIG['ref_jitter']()) % 1.0
        spawn_pt = polygon.exterior.interpolate(t_spawn, normalized=True)
        ref_pt   = polygon.exterior.interpolate(t_ref,   normalized=True)
        if math.hypot(ref_pt.x - spawn_pt.x, ref_pt.y - spawn_pt.y) >= min_chord:
            break

    route_hdg = math.degrees(math.atan2(ref_pt.x - spawn_pt.x,
                                        ref_pt.y - spawn_pt.y)) % 360.0
```

Three decisions in this one function:

**`dest_dist_factor = 20`** places the destination twenty sector-diameters beyond the exit
point. Because it is so far away, the bearing to it barely changes as the aircraft moves  -- 
so *holding a heading keeps you on route*, and "drift" becomes a clean, well-defined
quantity to penalise. Without this, the route bearing would swing as the aircraft
approached its waypoint and the drift penalty would be a moving target.

**`t_ref = t_spawn + 0.5 +/- 0.5`** puts the exit roughly opposite the entry but with enough
jitter to produce fully random crossing directions, not just diameters.

**`min_chord_nm = 15`** rejects entry/exit pairs that land close together. Such an aircraft
would exit within a step or two without ever really flying, polluting the arrival
statistics.

All geometry is in a flat NM frame centred on the equator (`center_ll = (0.0, 0.0)`), where
`cos(lat) = 1`, so there is no projection distortion to correct for.

### Spawns are admitted, not merely placed

```python
def _spawn_is_safe(self, route):
    """Admit a candidate spawn only if it clears (1) a static buffer to all traffic and
    (2) a conflict-free entry: flying its route at cruise it must not reach CPA < sep
    against any active aircraft (current trajectory) within t_warn.

    Conflicts should emerge from geometry evolving, never from spawning into one.
    """
    ...
        if math.hypot(d_east, d_north) < min_spawn_sep:
            return False                                # (1) static buffer
        ...
        tcpa = -(d_east * dv_east + d_north * dv_north) / rel_sq
        if tcpa < 0 or tcpa > horizon:
            continue                                    # diverging or beyond t_warn
        cpa_e, cpa_n = d_east + tcpa * dv_east, d_north + tcpa * dv_north
        if cpa_e ** 2 + cpa_n ** 2 < sep * sep:
            return False                                # (2) predicted LoS within t_warn
```

A candidate must clear a 15 NM static buffer (`sep_nm + buffer_nm`) **and** be
conflict-free on entry. Conflicts therefore emerge from geometry evolving over time, never
from spawning into one -- otherwise the agent would be punished for losses it had no
opportunity to prevent.

### Slots and episode length

There are `n_ac` slots; when an aircraft exits, its slot is refilled with a fresh entry, so
traffic density stays roughly constant for the whole episode. Episode length scales with
the sector rather than being a fixed step count:

```python
self._max_steps = max(50, round(
    CONFIG['crossings_per_episode'] * sector_diam_nm / CONFIG['ac_speed'] * 3600 / step_duration_s
))
```

`crossings_per_episode = 4`, so an episode is about four sector traversals regardless of
how large the sector turned out.

---

## Urgency

The ranking function -- and *only* the ranking function. Urgency never enters the reward.

```python
def urgency_from_states(pos_i, vel_i, pos_j, vel_j):
    """Urgency of the conflict between two aircraft given their planar state (NM, NM/s).

    Returns:
      > 1     active LoS    (1 at the sep boundary -> 10 at zero distance)
      0..1    predicted LoS (0 at t_warn -> 1 at t_los = 0, i.e. intrusion now)
      0       safe / diverging
    """
    sep = CONFIG['sep_nm']
    dist_sq, range_rate, rel_spd_sq = relative_motion(pos_i, vel_i, pos_j, vel_j)

    if dist_sq < sep ** 2:
        return 1.0 + 9.0 * (1.0 - math.sqrt(dist_sq) / sep)

    t_los = time_to_loss_of_separation(dist_sq, range_rate, rel_spd_sq, sep)
    if t_los is None or t_los > CONFIG['t_warn']:
        return 0.0                                       # no intrusion, or beyond the horizon
    return (CONFIG['t_warn'] - t_los) / CONFIG['t_warn']  # t_los in [0, t_warn] -> u in [0, 1]
```

Three branches, no clamps: the branch conditions already bound the result. The LoS branch
runs 1 -> 10 as distance closes to zero, which keeps active losses strictly above every
predicted one.

### Time to *loss of separation*, not to CPA

```python
def time_to_loss_of_separation(dist_sq, range_rate, rel_spd_sq, sep):
    """Seconds until two aircraft first LOSE SEPARATION (distance < sep), from their
    current relative state. range_rate = r.v (negative = converging), rel_spd_sq = |v|^2.

    Returns the LoS-entry time (>= 0), or None if they never intrude (diverging, parallel,
    or miss distance >= sep). This is EARLIER than time-to-CPA: a pair enters the protected
    circle before closest approach, so it is the honest "time until intrusion".
        t_los = tcpa - sqrt((sep^2 - dcpa^2) / |v|^2)
    """
    if rel_spd_sq < 1e-12:
        return None
    tcpa = -range_rate / rel_spd_sq
    if tcpa < 0:
        return None                                    # diverging
    dcpa_sq = max(0.0, dist_sq - range_rate ** 2 / rel_spd_sq)
    if dcpa_sq >= sep * sep:
        return None                                    # miss distance too large: never intrudes
    return tcpa - math.sqrt((sep * sep - dcpa_sq) / rel_spd_sq)
```

The `max(0.0, ...)` here is a genuine float guard: `dist_sq - range_rate**2/rel_spd_sq` can
round marginally negative for nearly collinear pairs, and it feeds a `sqrt`.

### One horizon

`t_warn = 360 s` is simultaneously the urgency scale, the `tlos` observation cap
(`NO_CONFLICT_S`), the spawn-rejection window, and the basis of the emergency threshold. An
earlier design also carried a 900 s `lookahead_s`, which turned out to be inert -- urgency
already clipped to zero past `t_warn` -- and was removed.

---

## Focus selection

Always controlling the most urgent aircraft is the greedy choice per step, but it made the
focus churn and the observations noisy. The rules below trade a little per-step optimality
for a steadier focus.

```python
worst_pair, total_load = self._build_urgency_matrix(flying)

focus_idx      = flying.index(self._focus_cs) if self._focus_cs in flying else -1
focus_urgency  = worst_pair[focus_idx] if focus_idx >= 0 else 0.0
focus_resolved = self._steps_since_urgency.get(self._focus_cs, clear_steps) >= clear_steps
emergency      = worst_pair.max() >= CONFIG['focus_emergency_u']
drift_locked   = focus_urgency == 0 and self._focus_hold_steps < clear_steps

# Candidate: highest worst-pair urgency (tiebreak total_load); ties stay with focus
if worst_pair.max() > 0:
    tied = np.where(worst_pair >= worst_pair.max() - 1e-9)[0]
    if focus_idx >= 0 and np.any(tied == focus_idx):
        best_cs = self._focus_cs
    else:
        best_cs = flying[tied[int(np.argmax(total_load[tied]))]]
else:
    best_cs = self._select_drifter_to_recover(flying)

# Hysteresis: keep the current focus while it is still active, unless fully
# resolved or an emergency forces a switch.
if focus_idx >= 0 and best_cs != self._focus_cs:
    keep = (focus_urgency > 0 or not focus_resolved or drift_locked) and not emergency
    if keep:
        best_cs = self._focus_cs
```

**Selection** uses the row maximum; **ties** use the row sum, so among equally urgent
aircraft the one carrying the most conflicts wins. Because urgency is symmetric, both
members of a conflicting pair share the same row max -- the tie-break fires on *every*
conflict, not just rare edge cases.

**Four stickiness mechanisms** bias towards the incumbent: a tie keeps it, an unresolved
conflict keeps it, 25 s of post-resolution grace keeps it, and `drift_locked` holds a
newly chosen drift focus for 25 s.

**One override.** `emergency` collapses the whole `keep` expression to `False`. It does not
choose the aircraft itself -- it just removes the protection so ordinary selection runs
unopposed. The threshold `focus_emergency_u = 0.67` corresponds to `t_los <= (1 - 0.67) *
360 ~= 120 s`. If the current focus is *itself* part of the emergency pair, it shares the
maximum and the tie-keep clause retains it, so an emergency never interrupts an aircraft
already handling that same emergency.

### Quiet traffic

```python
def _select_drifter_to_recover(self, flying):
    """Pick the most-drifted aircraft that can safely be sent back to its route.

    Score = heading drift, restricted to aircraft whose route heading is currently
    free (route_return_blocked == 0) -- the same predicate the observation reports as
    retn_conf. A drifter that cannot turn back offers the agent nothing to act on, so
    focusing it only burns steps. If every drifted aircraft is blocked the filter is
    dropped, so the rule always returns a focus. A hysteresis margin keeps the current
    focus unless another aircraft scores clearly higher.
    """
    drift = {}
    for cs in sorted(flying):
        idx = bs.traf.id2idx(cs)
        if idx < 0 or cs not in self._route_hdg:
            continue
        drift[cs] = self._heading_drift(cs, idx)

    free = [cs for cs in drift
            if drift[cs] > 0 and not route_return_blocked(cs, flying, self._route_hdg)]
    candidates = free or list(drift)
```

An earlier version scored `drift x clearance`, where clearance ramped with
nearest-neighbour distance. Measured over 1287 quiet steps that proxy selected a
blocked aircraft 12.9% of the time -- marginally *worse* than ignoring clearance
altogether -- because raw distance cannot see a distant converging pair sitting on the
route heading. Replacing it with the `route_return_blocked` predicate dropped blocked focus to
8.1% and step-to-step focus churn from 17.5% to 5.7%.

---

## Observations

27 raw floats, ego-centric from the focus ship, ACAS Xu style. **Nothing is normalised in
the environment** -- values leave as NM, kt, seconds and radians, and
`VecNormalize(norm_obs=True)` learns the scaling during training. This is why the saved
`vecnorm.pkl` is part of the trained artifact and not an optional convenience: feeding raw
values to a policy trained with it produces nonsense.

### Ownship (7)

```python
pending_cmd = self._pending_cmd.get(cs)
wait_s      = self._sim_time_s - pending_cmd['issued_at_s'] if pending_cmd else 0.0

obs = [dpsi_act,
       own_spd * NMS_TO_KT,
       a_cmd,
       v_cmd,
       retn_conf,      # retn_conf ("safe to return"); dummied to 0.0 when ablated
       # Whether an instruction is outstanding -- the honest observable: under a
       # probabilistic delay the controller knows the pilot has not acted yet, but
       # not when they will. Both are constant 0.0 when delay_mode='none'.
       1.0 if pending_cmd else 0.0,
       wait_s]
```

| feature | meaning |
| --- | --- |
| `dpsi` | actual heading error from route (rad) |
| `v_own` | true airspeed (kt) |
| `a_cmd` | **commanded** heading error from route (rad) |
| `v_cmd` | commanded true airspeed (kt) |
| `retn_conf` | 1 when returning to route is blocked |
| `pending` | 1 while an issued instruction awaits execution |
| `wait_s` | seconds since that instruction was issued |

The subtle one is carrying **both** `dpsi` and `a_cmd`. Without delays these are nearly
redundant. With delays they diverge: `a_cmd` is what the controller has asked for, `dpsi`
is what the aircraft is actually doing, and the gap between them *is* the outstanding
instruction.

### Intruders (4 x 5)

```python
if dist_nm < sep:
    tlos = 0.0                             # already inside the protected zone
else:
    ...
    t_los = time_to_loss_of_separation(dist_nm ** 2, range_rate, rel_spd_sq, sep)
    # No upper clip beyond the planning horizon: t_los is unbounded for a
    # near-parallel pair, and letting those values through would dominate
    # the VecNormalize running variance.
    tlos = NO_CONFLICT_S if t_los is None else min(max(0.0, t_los), NO_CONFLICT_S)

...
# fill slots: urgent pairs first (descending urgency), then nearest (ascending distance)
ordered = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0]) \
          + sorted(intruders, key=lambda r: r[1])
```

**Slot ordering** places conflicting aircraft first by descending urgency, then fills the
remainder by ascending distance. Slot 0 is always the worst threat, so each slot carries a
stable positional meaning instead of being an arbitrary list.

**`tlos` is capped at `NO_CONFLICT_S` (= `t_warn`)** for the reason in the comment: a
near-parallel pair has effectively unbounded time-to-LoS, and those values would dominate
the running variance and squash the informative range. The same constant doubles as the
"no conflict" value, which is consistent -- urgency is already zero past that horizon.

**Empty slots** use `EMPTY_RANGE_NM = 1000.0`, deliberately sized to exceed the widest
possible sector so a genuinely distant intruder is never read as an empty slot.

---

## Actions

`Discrete(10)`:

```python
#   0-2, 4-6  heading turns (stack on commanded heading)   3  hold   7  return-to-route
#   8  speed up   9  speed down
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}        # +1/-1 x mach_step on the commanded Mach
N_ACTIONS     = 10
```

```python
# A turn stacks onto the controller's last STATED intent: the outstanding target if
# one exists (the pilot has not acted on it yet), otherwise the flying heading.
if action_idx in SPEED_ACTIONS:
    base = pending['target_mach'] if pending and 'target_mach' in pending \
           else self._commanded_mach.get(cs, CONFIG['ac_mach'])
    mach = base + SPEED_ACTIONS[action_idx] * CONFIG['mach_step']
    cmd['target_mach'] = min(CONFIG['ac_mach_max'], max(CONFIG['ac_mach_min'], mach))
elif action_idx in TURN_DELTAS:
    base = pending['target_hdg'] if pending and 'target_hdg' in pending \
           else self._commanded_heading.get(cs, bs.traf.hdg[idx])
    cmd['target_hdg'] = (base + TURN_DELTAS[action_idx]) % 360
elif action_idx == RETURN_TO_ROUTE_ACTION:
    cmd['return_to_route'] = True
```

**Turns stack on the commanded heading**, i.e. the controller's last stated intent, not the
current flying heading. Two consecutive +30 deg instructions mean +60 deg from what was asked,
which is how a controller reasons -- and it stays coherent when the pilot has not acted yet.

**Return-to-route resolves at execution time**, not at issue: the route heading drifts while the
pilot delays, so "go direct" means direct from where the aircraft *is* when it finally
turns.

**Speed is Mach-based.** The envelope is 0.74-0.82 around a 0.78 nominal with a 0.04 step,
so exactly one instruction takes an aircraft from nominal to either edge -- a realistic
ATC-sized lever rather than a continuous throttle.

---

## Rewards

```python
def _compute_reward(self, acting_cs, action_idx):
    r_los = -CONFIG['w_los'] if self._los_this_step else 0.0

    r_drift = 0.0
    if acting_cs and acting_cs in self._route_hdg:
        idx = bs.traf.id2idx(acting_cs)
        if idx >= 0:
            r_drift = -CONFIG['w_drift'] * self._heading_drift(acting_cs, idx)

    r_work = -CONFIG['w_work'] * ACT_COST[action_idx] if acting_cs else 0.0
    return float(r_los + r_drift + r_work)
```

**Purely negative.** No reward for arriving, no reward for resolving -- only penalties for
losing separation (weight 10, dominant), for being off route, and for transmitting. The
agent minimises harm rather than chasing a bonus, which removes the usual surface for
reward hacking.

**Drift is measured on the executed commanded heading.** `_commanded_heading` is only
updated inside `_execute_due_instructions`, so an instruction the pilot has not yet acted on
earns no drift credit -- the agent cannot collect reward for a turn that has not started.

**Workload is charged at issue time.** The radio call happens whether or not the pilot
complies. Under delays this asymmetry is the entire mechanism: cost lands immediately,
benefit lands 25 s later.

### Sub-additive action costs

```python
# A heading instruction is one radio call, so its cost is SUB-ADDITIVE in the turn it
# commands: a single decisive turn must not cost more than splitting it across several
# instructions, or the policy is rewarded for salami-slicing.

_TURN_30 = 0.5                          # cost anchor: one 30-deg turn

ACT_COST = [
    0.75,                 # 0  turn -60
    0.625,                # 1  turn -45  ((0.75 + 0.5) / 2)
    0.5,                  # 2  turn -30
    0.0,                  # 3  hold (free)
    0.5,                  # 4  turn +30
    0.625,                # 5  turn +45
    0.75,                 # 6  turn +60
    0.25 * _TURN_30,      # 7  return to route (cheap: undoing a deviation)
    0.5  * _TURN_30,      # 8  speed up   (half a 30-deg turn)
    0.5  * _TURN_30,      # 9  speed down
]
```

One 60 deg turn costs 0.75; two 30 deg turns cost 1.0. Splitting a manoeuvre across several radio
calls is therefore always more expensive than committing to it once.

---

## Action-response delays

The experimental variable, and the reason the code separates *issue* from *execute*.

`_issue_instruction` registers the instruction and touches nothing else -- **no BlueSky command
is sent there**. `_execute_due_instructions` is the only place a heading or speed reaches the
simulator:

```python
def _execute_due_instructions(self):
    """Execute every instruction whose deadline has arrived. This is the only place a
    heading/speed instruction reaches BlueSky or updates the commanded state."""
    for cs, cmd in list(self._pending_cmd.items()):
        if cmd['execute_at_s'] > self._sim_time_s:
            continue
        del self._pending_cmd[cs]
        ...
        # Only an EXECUTED advisory earns the shorter delay_next_s for this cycle.
        self._executed_count[cs] = self._executed_count.get(cs, 0) + 1
```

It runs at 1 s resolution inside the 5 s RL step, so 12.5 s means 12.5 s and not "two or
three steps".

### The three modes

```python
def _draw_response_delay(self, executed_n):
    """Seconds until the pilot acts. executed_n = advisories already EXECUTED while this
    aircraft has held the focus, so 0 means the pilot is not engaged yet."""
    if self.delay_mode == 'none':
        return 0.0
    mean = CONFIG['delay_first_s'] if executed_n == 0 else CONFIG['delay_next_s']
    if self.delay_mode == 'deterministic':
        return mean
    # Log-normal parameterised on the MEAN: E[X] = exp(mu + sigma^2/2), so mu below
    # makes E[X] land exactly on `mean` rather than on the median.
    sigma = CONFIG['delay_sigma']
    mu    = math.log(mean) - sigma ** 2 / 2.0
    return min(random.lognormvariate(mu, sigma), CONFIG['delay_max_s'])
```

| mode | behaviour |
| --- | --- |
| `none` | 0 s -- baseline |
| `deterministic` | fixed 25 s, or 12.5 s once engaged |
| `probabilistic` | log-normal on the same means, sigma = 0.4, capped at 120 s |

The log-normal is parameterised on the **mean**, not the median, so the deterministic and
probabilistic arms share an identical expected delay. Comparing them therefore isolates the
effect of *uncertainty* from the effect of delay magnitude. The 120 s cap stops a tail draw
outliving the conflict it was meant to resolve.

**Engagement lasts exactly as long as the aircraft holds the focus.** `_executed_count` is
reset whenever the ownship changes, so a returning aircraft starts from `delay_first_s`
again:

```python
if best_cs != self._focus_cs:
    self._focus_hold_steps = 0
    # New ownship: this pilot is not engaged yet, so their next advisory is a
    # "first" one and earns the full delay_first_s.
    self._executed_count[best_cs] = 0
```

Only *execution* advances the counter -- issuing three advisories while the first is still
pending does not earn the shorter delay.

### Amendments inherit the deadline

```python
if pending is not None:
    # Amendment: the deadline belongs to the pilot's engagement, not to the message.
    # Replace the payload, inherit the clock, draw no new delay, and do not touch
    # _executed_count -- nothing has been executed yet.
    cmd['execute_at_s'] = pending['execute_at_s']
    cmd['issued_at_s']  = pending['issued_at_s']
```

Re-issuing while an instruction is outstanding swaps the payload but keeps both the
original deadline and the original issue time. The clock belongs to the pilot's
engagement, not to the message. Combined with workload charged at issue, this makes
indecision genuinely expensive: three amendments cost three radio calls and still execute
at the original deadline. It is also why `wait_s` keeps counting through amendments  -- 
it always runs against the deadline the pilot is actually working to.

### Cross-arm compatibility

All three arms share the identical 27-feature observation, with `pending` and `wait_s`
constant zero in the baseline. That is what makes a checkpoint from one arm loadable
against another and `Validation/cross_evaluate.py` possible.

It is also the source of that script's caveat: a policy trained with `delay_mode='none'`
never saw `pending = 1`, so its `VecNormalize` statistics give those features near-zero
variance and map them to the clip bound at test time. Its off-diagonal cells therefore mix
a wrong strategy with an out-of-distribution input, and `cross_evaluate.py` prints an OOD
factor per model so the size of that effect is visible.
