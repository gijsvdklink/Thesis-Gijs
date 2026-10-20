# The Gymnasium environment, end to end

How `AirspaceEnv` works: what Gymnasium requires of it, what it takes from BlueSky, how the
focus ship is chosen, which clocks run at which rate, and where PPO fits. Written against the
code in `Environment/`, `Training/train.py` and `Validation/run_test_scenarios.py`.

---

## 1. The shape of the thing

One reinforcement learning agent plays the role of a conflict detection and resolution tool
that advises a single air traffic controller. Every RL step it looks at one aircraft, the
**focus ship**, and issues at most one advisory for it. The advisory does not reach the
aircraft directly: it goes to a modelled controller who needs time to act. BlueSky flies the
aircraft.

```
            ┌──────────── one RL step = 5 simulated seconds ────────────┐
 PPO ──action──▶ AirspaceEnv ──advisory──▶ ATCO (delay τ) ──instruction──▶ BlueSky
  ▲                   │                                                      │
  └── observation, reward ◀──── traffic state (lat, lon, hdg, tas) ◀─────────┘
```

---

## 2. What Gymnasium requires, and what we return

Gymnasium is only a contract. It asks for two spaces and two methods, and it fixes what `step`
returns. It does not simulate anything.

| Required | Ours |
|---|---|
| `observation_space` | `Box(-inf, inf, shape=(29,), float32)` |
| `action_space` | `Discrete(10)` |
| `reset(seed, options)` → `(obs, info)` | `options={'scenario_seed': n}` picks the scenario |
| `step(action)` → `(obs, reward, terminated, truncated, info)` | `terminated` is **always False**; episodes end by `truncated` |

`terminated` would mean the task reached a terminal state, such as a crash. A loss of
separation is penalised, not terminal, so the episode runs to its natural length and only
`truncated` is ever set. On the truncating step `info` carries the whole episode summary
(`ep_los_events`, `ep_conflicts`, `ep_advisories`, `ep_flight_hours`, …) which the evaluation
reads straight into its CSV.

The observation is a flat 29-vector rather than a dict, so a plain `MlpPolicy` fits with no
feature extractor.

---

## 3. Which file owns what

| File | Owns |
|---|---|
| `env.py` | the Gymnasium API, the step cycle, focus selection, observation, reward |
| `traffic.py` | **the only BlueSky bridge**: starts it, converts its state into our frame |
| `atco.py` | the response delay: holding one advisory, revisions, when it fires |
| `geometry.py` | CPA, time to loss of separation, the urgency matrix, lat/lon ↔ NM |
| `sector.py` | the random convex polygon and the entry routes |
| `stats.py` | per-episode counters and the summary dict |
| `config.py` | every tunable number, and the derived constants |

---

## 4. The three clocks

| Clock | Value | Set by |
|---|---|---|
| BlueSky integration step | **1 s** | `CONFIG['sim_dt']`, pushed to BlueSky as `DT 1;FF` |
| RL step | **5 s** = `action_freq × sim_dt` | `CONFIG['action_freq'] = 5` |
| Conflict horizon | **360 s** (6 min) | `CONFIG['t_warn']` |
| Response delay τ | mean **30 s** by default | `CONFIG['delay_mean_s']`, law per run |

So the policy acts once every 5 simulated seconds, and BlueSky advances five times between two
actions. Losses of separation and route drift are scanned **every simulated second**, not once
per RL step, so a two-second intrusion between actions is still counted.

An episode ends when `exits_per_episode × n_aircraft` aircraft have left the sector — 4 per
slot, so 48 exits for 12 aircraft. `max_crossings = 8` sector traversals is a safety cap for
traffic that somehow never leaves. A 12-aircraft episode runs about 1000 RL steps, roughly 16
flight hours.

---

## 5. What we take from BlueSky, and what we send it

Everything crosses in `traffic.py` and in a handful of calls in `env.py`.

**Read** (per RL step, for the aircraft in the sector):

| BlueSky | Units | Becomes |
|---|---|---|
| `bs.traf.lat`, `bs.traf.lon` | degrees | position in east/north **NM** about `center_ll` |
| `bs.traf.hdg` | degrees | heading, used as is |
| `bs.traf.tas` | m/s | speed in **NM/s**, and the velocity vector with the heading |
| `bs.traf.id` | callsigns | the row order everything else indexes by |

`traffic_states(indices)` returns `(pos, vel)` and is the single read path. Nothing else in the
environment knows BlueSky exists.

**Send**:

| Call | When |
|---|---|
| `bs.traf.cre(cs, actype, aclat, aclon, achdg, acspd, acalt)` | an aircraft enters |
| `SPD <cs> <mach>` / `ALT <cs> FL350` | right after creation, so the autopilot holds cruise |
| `HDG <cs> <deg>` / `SPD <cs> <mach>` | an instruction is **executed**, after the delay |
| `bs.traf.delete(idx)` | an aircraft leaves the sector |
| `bs.sim.step()` | once per simulated second |
| `DT 1;FF`, `ASAS OFF` | at reset: our timestep, and BlueSky must not resolve conflicts itself |

`ASAS OFF` matters: BlueSky ships its own conflict resolution, and leaving it on would mean
measuring BlueSky's skill instead of the policy's.

**BlueSky is a process-wide singleton.** `bs.traf` is module state, so a process flies one
environment at a time. Vectorised training works only through `SubprocVecEnv`, where each
worker is a separate process; `DummyVecEnv` with more than one environment would have them
all fighting over the same `bs.traf`.

---

## 6. From traffic state to the focus ship

This happens once per RL step, in `_refresh_traffic_view()` then `_select_focus()`.

**Step 1 — pairwise geometry.** With positions and velocities for all *n* aircraft,
`pairwise()` gives every pair's relative position and velocity, from which `cpa()` gives the
time and distance of closest approach, and `time_to_los()` the time until the 5 NM separation
minimum would be broken (capped at the 360 s horizon).

**Step 2 — the urgency matrix.** `urgency_matrix()` turns that into an *n × n* matrix:

```
urgency = 0                                   no conflict within 360 s
        = (t_warn − t_los) / t_warn ∈ (0, 1]  a conflict is coming; 1 at the moment of intrusion
        = 1 + 9 · (1 − d / 5 NM)   ∈ (1, 10]  separation already lost, growing as they close
```

Symmetric, zero on the diagonal.

**Step 3 — pick one aircraft.** Each aircraft's `worst` is its row maximum. Then, in order:

1. **Hold the incumbent** if it has been conflict-free for fewer than `focus_clear_steps = 5`
   steps *and* nothing anywhere exceeds `focus_emergency_u = 0.67` (about two minutes to
   intrusion). This stops the focus flickering between aircraft.
2. Otherwise, **if anything is in conflict**, take the highest `worst`; ties — and the worst
   pair always ties, since urgency is symmetric — break on the **row sum**, so the aircraft
   caught in more conflicts wins.
3. Otherwise, **the most drifted** aircraft, if its drift exceeds the on-route tolerance of 5°.
4. Otherwise keep the incumbent: a quiet sector still needs someone in the slot.

When the focus moves, any half-formed instruction for the aircraft being released is dropped —
the controller's attention moves with it.

---

## 7. The observation (29 values)

Built for the focus ship only. If there is no controllable aircraft, a neutral vector goes out
instead.

**Ownship, 9 values** (`OBS_OWNSHIP_LABELS`):

| # | Name | Meaning | Unit |
|---|---|---|---|
| 0 | `dpsi` | drift: current heading − initial heading | rad |
| 1 | `v_own` | current speed | kt |
| 2 | `h_cmd` | commanded heading, as offset from initial | rad |
| 3 | `v_cmd` | commanded speed | kt |
| 4 | `retn_conf` | 1 if returning to the initial heading is blocked by a conflict | flag |
| 5 | `pending` | 1 if the controller is holding an advisory | flag |
| 6 | `wait_s` | time since that advisory was issued | s |
| 7 | `pend_hdg` | held heading instruction, as offset from what is flown | rad |
| 8 | `pend_spd` | held speed instruction, as offset | kt |

Fields 5–8 are the **response state**. They are identically zero when `delay_mode='none'`, and
the no-delay models are evaluated with them forced to zero (`--blind`), so a delay-free policy
can never read a signal it was not trained on.

**Intruders, 4 × 5 values**, nearest by urgency then distance (`OBS_INTRUDER_LABELS`):
`dist` [NM], `theta` (relative bearing) [rad], `psi` (relative heading) [rad], `vint` [kt],
`tlos` [s]. Empty slots get a sentinel 1000 NM at 360 s, so the vector is always 29 long.

---

## 8. The action space (10 discrete)

| Index | Action | Cost |
|---|---|---|
| 0–2 | turn −60°, −45°, −30° | 0.75, 0.625, 0.5 |
| 3 | **hold** — nothing is transmitted at all | 0 |
| 4–6 | turn +30°, +45°, +60° | 0.5, 0.625, 0.75 |
| 7 | return to the initial heading | 0.125 |
| 8–9 | speed ±0.02 Mach | 0.25 |

Turns accumulate on the last **executed** heading, not on the current one, so re-selecting the
same turn while one is already pending reproduces the identical target and is recognised as a
repeat rather than a revision. The costs are sub-additive in the turn: two 30° turns cost more
than one 60°, so splitting a manoeuvre is never cheaper.

---

## 9. The response delay

`atco.py` holds **one** advisory at a time.

- **New advisory**: draw τ from the law (`none` → 0, `deterministic` → the mean,
  `lognormal` → mean with `delay_sigma = 0.4`), execute at `now + τ`.
- **Repeat** of the instruction already held: ignored, the standing time keeps running.
- **Revision**: τ is drawn afresh and shortened to `κ = 0.7` of it, because the controller is
  already engaged — but `execute_at = max(standing, now + κτ)`, so a revision can only ever
  *postpone* execution. Without that clamp, changing one's mind would buy an earlier action and
  the policy would learn to churn.
- Polled **every simulated second**. If the aircraft has left the sector the instruction is
  deleted, never flown and never counted as a response served.

---

## 10. The reward

Charged to the aircraft that held the focus when the action was chosen:

```
R = −w_los · (seconds in loss of separation)     w_los   = 10     per second, scanned each second
    −w_drift · Σ (1 − cos Δψ)                    w_drift = 0.1    per second at that drift
    −w_work · ACT_COST[action]                   w_work  = 1      per selected action
```

Two things to note. The workload term is charged on the **action selected**, whether or not it
ever reaches an aircraft — a repeat costs the same as a revision. And the LoS term is summed
over the five seconds of the step, so a long intrusion costs more than a brief one.

---

## 11. Where PPO fits

`Training/train.py`, Stable-Baselines3:

| Setting | Value | Why |
|---|---|---|
| policy | `MlpPolicy` | the observation is a flat 29-vector |
| `n_envs` | 4, via `SubprocVecEnv` | separate processes, so separate BlueSky singletons |
| `n_steps` | 512 per env | rollout = 4 × 512 = 2048 transitions |
| `batch_size` | 2048 | one gradient step per epoch over the whole rollout |
| `n_epochs` | 30 | 30 gradient steps per update |
| everything else | SB3 defaults | γ 0.99, lr 3e-4, GAE λ 0.95, clip 0.2, ent_coef 0 |
| wrappers | `VecMonitor` → `VecNormalize` | observations normalised; **the statistics are part of the model** |

`VecNormalize` is the part that bites. The running mean and variance of the observation live
outside the network, so a checkpoint is useless without its `<name>_vecnorm.pkl`. Evaluation
reproduces the normalisation by hand:

```python
action, _ = model.predict(np.clip((obs - mean) / std, -clip, clip), deterministic=True)
```

`deterministic=True` at evaluation, sampled during training. Episodes are drawn from a seeded
stream; the 500 held-out scenarios (`VALIDATION_SEED = 8`) are excluded from training, so no
model can have met a test scenario however long it trains.

---

## 12. One RL step, in order

```
step(action)
 ├─ _spawn_due_aircraft()            an exited slot refills: bs.traf.cre + SPD + ALT
 ├─ _issue_advisory(focus, action)   action → target heading/Mach → atco.receive()
 ├─ _advance_simulation()            5 × { atco.act_if_ready → HDG/SPD stack commands
 │                                         bs.sim.step()                 ← 1 s of flying
 │                                         scan LoS seconds, scan drift }
 ├─ _remove_exited_aircraft()        score the exit, bs.traf.delete()
 ├─ _refresh_traffic_view()          traffic_states() → pos, vel → urgency matrix
 ├─ _select_focus()                  the rules of §6
 ├─ _compute_reward()                the three terms of §10
 └─ _build_observation()             the 29-vector of §7
```

---

## 13. Things that will catch you out

- **`terminated` is always False.** Filter episodes on `truncated`.
- **A checkpoint without its `_vecnorm.pkl` is unusable**, and silently so: it will run and
  produce nonsense.
- **BlueSky is a singleton.** One environment per process, `SubprocVecEnv` only.
- **`ASAS OFF`** must stay off, or BlueSky resolves the conflicts for you.
- **Advisories issued ≠ advisories executed.** Under delay most issued advisories are replaced
  before they ever reach an aircraft; count the right one for the question you are asking.
- **The workload cost is charged on selection**, not on transmission, so the reward does not
  price the controller's attention.
