# AirspaceEnv — the Gymnasium environment

One PPO agent acts as a CD&R advisory tool. Each RL step it looks at **one** aircraft, the
*focus ship*, and may issue **one** advisory. The advisory does not reach the aircraft: it goes
to a modelled ATCO who executes it after a delay τ. BlueSky flies the traffic.

```
          ┌──────────── one RL step = 5 simulated seconds ────────────┐
PPO ──action──▶ AirspaceEnv ──advisory──▶ ATCO (delay τ) ──HDG/SPD──▶ BlueSky
 ▲                   │                                                   │
 └── observation, reward ◀──── lat, lon, hdg, tas ◀──────────────────────┘
```

| file | owns |
|---|---|
| `Environment/config.py` | every constant: `CONFIG`, action costs, observation labels |
| `Environment/env.py` | the Gymnasium class: step, reset, focus choice, observation, reward |
| `Environment/traffic.py` | the BlueSky bridge: start it, read positions and velocities |
| `Environment/atco.py` | the delay mechanism: one pending advisory, τ, revisions |
| `Environment/geometry.py` | CPA, time-to-LoS, urgency, lat/lon ↔ NM, heading drift |
| `Environment/sector.py` | the random convex sector, entry routes, exit points |
| `Environment/stats.py` | per-episode counters and the end-of-episode summary |
| `Training/train.py` | PPO, `VecNormalize`, checkpoints, the 60 000-episode stop |

---

## 1. Lifecycle: what happens when

**Phase 0 — process start, once.** `AirspaceEnv(delay_mode=..., seed=...)` declares the two
spaces, stores the delay law, seeds `_seed_stream`, and calls `start_bluesky()`:
`bs.init(mode='sim', detached=True)` loads the navigation cache and performance models (the
lines printed at startup, a few seconds), installs a dummy screen, and pushes `DT 1;FF`.
`_reset_episode_state()` then prepares empty containers. **No sector and no aircraft exist
yet** — `step()` before `reset()` would fail on missing scenario state; SB3 always resets first.

**Phase 1 — training start, once.** `train.py` builds `DummyVecEnv([make_env])` (which runs
phase 0), `VecMonitor` around it to record episode length and reward, and `VecNormalize` around
that for the running observation and return statistics. `PPO('MlpPolicy', ...)` builds the 2 × 64
tanh actor and critic, and `learn()` calls `env.reset()` once to obtain `_last_obs`.

**Phase 2 — episode reset, every episode** (`env.py:69`). In order: draw the next scenario seed,
skipping the held-out set → derive `scenario_rng`, `traffic_rng`, `delay_rng` → construct a fresh
`ATCO` → clear BlueSky's command stack, `bs.traf.reset()`, re-push `DT 1;FF` → `_build_sector`
(n_ac, ρ, polygon, `_max_exits`, `_max_steps`) → `_spawn_initial_traffic` (n_ac aircraft, each
re-drawn until separated) → `_refresh_traffic_view` → `_select_focus` → first observation. The
scenario is new every episode; BlueSky's process state is reused, only its traffic is wiped.

**Phase 3 — rollout, 2048 steps.** Per step: normalise the observation with the current running
statistics → policy forward → action → `env.step` (the sequence in *One step, in order*) →
`VecNormalize` updates its statistics and normalises the reward → both go in the buffer. A
rollout spans ~1.4 episodes, so episode boundaries fall inside it.

**Phase 4 — episode end.** On `truncated` the env puts the whole `episode_summary` in `info`.
`DummyVecEnv` stores the final observation as `info['terminal_observation']` and immediately
calls `reset()` (phase 2), so the agent sees no gap; `TimeLimit.truncated` tells PPO to
bootstrap the value of that final state rather than treat it as zero. `LogEpisodes` averages the
summary over the last 100 episodes into TensorBoard and `StopAfterEpisodes` counts it.

**Phase 5 — update, every 2048 steps.** GAE (γ = 0.99, λ = 0.95) over the buffer with the
`dones` flags cutting advantages at episode boundaries, then 30 epochs, each a single gradient
step on the full 2048-sample batch. The environment is idle meanwhile — about 16 % of wall clock.

**Phase 6 — housekeeping.** Every 500 000 steps `Checkpoint` writes `last_model.zip` plus
`last_model_vecnorm.pkl`. At 60 000 episodes `StopAfterEpisodes` returns `False`, `learn()`
returns, and `final_model.*` is written in the `finally` block — so an interrupt also saves.

**Phase 7 — testing, a separate process** (`Validation/run_test_scenarios.py`). The checkpoint
is loaded together with its `vecnorm.pkl`, from which the **frozen** mean, variance and clip are
taken and applied by hand; no `VecNormalize` wrapper and no reward normalisation are used, and
actions are `deterministic=True`. Each of the 500 held-out scenarios is flown with
`reset(options={'scenario_seed': s})` to its natural end, `atco.act_if_ready` is wrapped to
count the instructions actually executed, and one CSV row per scenario is written.

---

## 2. The Gymnasium contract

Two spaces and two methods, nothing more:

- `observation_space = Box(shape=(29,))`, `action_space = Discrete(10)` (`env.py:57`).
  29 = 9 ownship + 4 neighbours × 5.
- `reset(seed, options) -> obs, info`; `options={'scenario_seed': n}` replays an exact
  scenario, which is how validation flies the held-out set (`env.py:69`).
- `step(action) -> obs, reward, terminated, truncated, info` (`env.py:87`).

**`terminated` is always `False`.** An LoS is punished, not fatal, so episodes only run out:
`truncated=True` at the episode end (`env.py:104`). This matters numerically — SB3 bootstraps
the value of a truncated final state instead of treating it as worth zero.

`info` carries `los_seconds`, `focus_cs`, `n_aircraft`, and on the last step the whole
`episode_summary` dict (`stats.py:45`): what the callbacks log and the test runs write to CSV.

---

## 3. Three clocks

| clock | value | from |
|---|---|---|
| BlueSky integration step | 1 s | `sim_dt`, pushed as `DT 1;FF` |
| RL decision step | 5 s | `action_freq` = 5 sub-steps |
| conflict look-ahead | 360 s | `t_warn` |

One `step()` runs the inner loop five times: execute anything due → `bs.sim.step()` → advance
`_sim_time_s` → scan separation (`env.py:359`). Delays are compared against `_sim_time_s`, so a
30 s mean delay is 6 RL steps of waiting.

An episode ends when `exits >= 4 × n_ac`, or on the safety net
`steps >= max(50, 8 × crossing_time / 5 s)` (`env.py:520`). That is ~1 400–1 600 steps, ~2 h of
simulated time.

---

## 4. One scenario

`n_ac ∈ [15, 30]`, density `ρ ∈ [1/20000, 1/10000]` per km², `area = n_ac / ρ`. The sector is a
random convex polygon, 6–12 vertices, rejected unless circularity ≥ 0.7 (`sector.py:27`).

Aircraft are A320s at FL350, Mach 0.78 (0.76–0.80, step 0.02), flying straight lines with no
flight plan: a *route* is just the entry point, the initial heading and the exit the aircraft
would reach if never turned. Traffic is slot-based — `n_ac` slots, one aircraft each. A spawn
needs a chord of ≥ 15 NM through the sector, ≥ 15 NM clearance (5 + 10 buffer) from every
aircraft, and no predicted LoS within 360 s (`env.py:417`). On leaving the polygon an aircraft
is scored, deleted from BlueSky, and its slot re-queued for the next step (`env.py:457`), so the
count stays roughly constant.

**Seeding** (`env.py:485`). The env seed feeds a stream that draws one scenario seed per episode
out of 10⁹, skipping the 500 held-out validation seeds (`config.py:57`). That scenario seed
spawns three independent streams: `scenario_rng` (sector, initial placement), `traffic_rng`
(later entries), `delay_rng` (τ). Training can never see a test scenario, and one seed
reproduces a scenario exactly.

---

## 5. The BlueSky boundary

BlueSky is a process-wide singleton (`bs.traf`, `bs.sim`, the stack), so **one environment per
process**. `start_bluesky()` calls `bs.init(mode='sim', detached=True)` once, swaps in a dummy
screen, and pushes `DT 1;FF` (`traffic.py:21`).

| written | when |
|---|---|
| `cmdstack.clear()`, `bs.traf.reset()`, `DT 1;FF` | reset |
| `bs.traf.cre(cs, A320, lat, lon, hdg, mach, FL350)` + `SPD`, `ALT` | spawn |
| `HDG <cs> <deg>` or `SPD <cs> <mach>` | an advisory executes |
| `bs.traf.delete(idx)` | exit |
| `bs.sim.step()` | every sub-step |

Read back: only `lat`, `lon` (→ NM, flat-Earth about `center_ll`), `tas` (→ NM/s), `hdg`, plus
`id`/`id2idx` to map callsigns to rows (`traffic.py:30`). Urgency, time-to-LoS, drift and
conflicts are computed here, not asked of BlueSky. No altitude changes, no wind.

BlueSky re-indexes its rows on every create and delete, so `_refresh_traffic_view` rebuilds
`_row_of` (callsign → row, callsigns sorted) each step; all matrices use that ordering.

---

## 6. Urgency, and picking the focus ship

`urgency_matrix` (`geometry.py:73`) is n × n over airborne pairs, from the constant-velocity CPA:

- `t_los > 360 s` → 0
- `0 ≤ t_los ≤ 360 s` → `(360 − t_los)/360`, reaching 1.0 at the LoS
- closer than 5 NM → `1 + 9(1 − d/5)`, so 1 → 10 as they converge

Urgency > 0 is a *conflict*; new pairs are counted once (`env.py:595`).
`worst = urgency.max(axis=1)` is each aircraft's worst threat, and `_select_focus`
(`env.py:116`) picks in order:

1. **Keep the incumbent** if its last conflict was fewer than 5 steps ago and nothing exceeds
   urgency 0.67 — hysteresis, so the agent can finish what it started. New aircraft start at the
   limit, so they are never sticky.
2. Else, if any conflict exists: the highest `worst`, tie-broken on the row sum.
3. Else, if an aircraft has drifted more than 5° off its initial heading: the most drifted one.
4. Else keep what is there.

Switching focus **discards** any advisory still pending for the old focus ship
(`atco.release`). The ATCO is a single server: one outstanding instruction at a time.

---

## 7. The observation (29 values)

Built only from the focus ship's view (`env.py:220`); a neutral vector if there is none. Angles
in radians, speeds in knots, distances in NM.

**Ownship (9).** `dpsi` actual − initial heading · `v_own` true airspeed · `h_cmd` commanded −
initial heading · `v_cmd` commanded Mach in kt · `retn_conf` 1 if resuming the initial heading
would conflict within 360 s · `pending` 1 if an advisory awaits execution · `wait_s` how long it
has waited · `pend_hdg`, `pend_spd` what it would change.

The last four are the **response state**, the only channel through which the agent can know the
delay exists. A no-delay model never saw them non-zero, so testing it in a delayed environment
zeroes them to keep its training distribution.

**Per neighbour (4 × 5).** `dist` · `theta` bearing in the ownship frame · `psi` relative
heading · `vint` its speed · `tlos` time to LoS, 0 if already inside 5 NM, clipped to 360.

Slots are filled by urgency descending, then by distance ascending, duplicates removed
(`env.py:292`); unused slots get `[1000, 0, 0, 0, 360]`. The agent thus sees 4 of up to 30
aircraft — the reason this is a POMDP and the returns are noisy.

---

## 8. Ten actions

| idx | advisory | cost |
|---|---|---|
| 0, 6 | turn −60° / +60° relative to the *route* heading | 0.75 |
| 1, 5 | turn −45° / +45° | 0.625 |
| 2, 4 | turn −30° / +30° | 0.5 |
| 3 | hold, transmit nothing | 0 |
| 7 | return to initial heading | 0.125 |
| 8, 9 | Mach +0.02 / −0.02, clipped to [0.76, 0.80] | 0.25 |

Turns are relative to the **initial** heading: the target is
`initial + (commanded − initial) + delta` (`env.py:158`), so turns accumulate against the route
and action 7 undoes them in one go.

Three advisory counts exist, and mixing them up is the easiest way to misread the results
(`env.py:175`, `stats.py:45`):

- `ep_advisories_sent` — everything transmitted to the ATCO.
- `ep_advisories` — only those that changed what that aircraft was last told.
- `delay_acted` — actually executed, reported through `ep_delay_mean_s`.

A repeat of the pending advisory transmits nothing and does not reset the ATCO's timer
(`repeats`); a different one replaces it (`discarded`).

---

## 9. The delay mechanism

`ATCO` (`atco.py`) holds at most one advisory, for one callsign.

- `receive` draws τ: `none` → 0, `deterministic` → the mean, `lognormal` → σ = 0.4 in log space
  with μ = `log(mean) − σ²/2`, so the configured value is the **mean**, not the median.
- `execute_at = now + round(τ)`.
- A **revision** (a different advisory while one is pending) draws τ′ = 0.7 τ
  (`revision_kappa`) and takes `execute_at = max(old execute_at, now + τ′)`: a revision can
  never make the ATCO act sooner than the instruction it replaces, and the measured response
  time keeps running from the first advisory.
- `act_if_ready` fires once `now ≥ execute_at` and drops the advisory if the aircraft has left.
  It is called in the 1 s inner loop, so execution resolves at 1 s granularity.

Only on execution does `HDG`/`SPD` reach BlueSky and `commanded_hdg`/`commanded_mach` change —
so `h_cmd`/`v_cmd` describe what the aircraft is doing, `pend_*` what is still queued.

---

## 10. The reward

```
r = −10.0 · los_seconds   −0.10 · drift_sum   −1.00 · cost(action)
      w_los                 w_drift             w_work
```

- `los_seconds` ∈ {0…5}: of the five sub-steps, how many had the **focus ship** within 5 NM of
  anyone.
- `drift_sum`: `1 − cos(actual − initial heading)` over the five sub-steps, for the **acting
  aircraft only** — ~0 for small deviations, 2 for a reversal.
- `cost`: the table above, charged when an advisory is issued; 0 for hold and when there is no
  focus ship.

Safety and efficiency are therefore both scored on the focus ship, not the sector. The summary
reports the three terms separately per flight hour, where flight hours accumulate as
`n_airborne × 5 s` per step — the normalisation that makes scenarios of different size and
density comparable.

---

## 11. The PPO settings

One environment, SB3 PPO on CPU, the settings as reported: `n_steps = 2048`,
`batch_size = 2048` (the full rollout, so one gradient step per epoch), `n_epochs = 30`, Adam
3e-4 constant, γ 0.99, λ 0.95, clip 0.2, `vf_coef` 0.5, max grad norm 0.5, 2 × 64 tanh actor and
critic. `--n-envs > 1` makes the rollout `n_envs × 2048` and no longer matches this.

`VecNormalize` standardises observations and rewards with running statistics clipped to
[−10, 10]. Those statistics are part of the model: `*_vecnorm.pkl` must travel with every
checkpoint, or the policy sees inputs on the wrong scale.

Runs end at **60 000 episodes** (~8.5 × 10⁷ steps), about 117 steps/s on one core, so 8–9 days.

---

## 12. One step, in order

1. `_spawn_due_aircraft` — fill any empty slot whose timer expired, if a safe entry exists.
2. `_issue_advisory(focus, action)` → ATCO, unless hold, repeat or no focus ship.
3. `_advance_simulation` ×5 — execute due advisories, `bs.sim.step()`, accumulate the focus
   ship's drift, scan for pairs under 5 NM, count new LoS events.
4. `_remove_exited_aircraft` — score, delete, re-queue the slot.
5. `_refresh_traffic_view` — re-read BlueSky, rebuild `_row_of`, urgency, `t_los`,
   `return_blocked`; count new conflicts.
6. `_select_focus` — hysteresis, urgency, drift; release any discarded advisory.
7. `_compute_reward`, `_record_step_stats`, check the episode end, build the new observation.

Consequence of that order: the action at step *t* is chosen for the focus ship selected at the
end of step *t−1*, and the reward at *t* is attributed to that same aircraft.

---

## 13. Things that will catch you out

- **BlueSky is global.** Two envs in one process share `bs.traf`; parallelism means separate
  processes (`SubprocVecEnv`, or the validation worker pool).
- **`--n-envs 4` changes the hyperparameters** — the rollout becomes `n_envs × 2048`. Only
  `--n-envs 1` matches the reported setup.
- **The episode counter is not in the checkpoint.** `--resume` restarts it at zero.
- **Turns are relative to the route**, so action 2 twice is −60° from the route.
- **`t_los` is constant-velocity**, recomputed every step: a look-ahead, not a prediction of
  what the agent will do.
- **The no-delay model's response-state inputs are zeroed at test time**, deliberately.
