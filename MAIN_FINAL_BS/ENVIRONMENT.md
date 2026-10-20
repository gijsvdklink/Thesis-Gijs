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

## 1. Execution model

**Phase 0 · process start.** `AirspaceEnv.__init__` (`env.py:51`) declares
`Box(-inf, inf, (29,), float32)` and `Discrete(10)`, seeds `_seed_stream = Random(seed)`, and
calls `start_bluesky()` (`traffic.py:21`): `bs.init(mode='sim', detached=True)` instantiates the
module-level singletons `bs.traf`, `bs.sim`, `bs.stack` and loads the navdata cache and OpenAP
tables (a few seconds, once per process). `bs.scr` is replaced by a `ScreenIO` subclass whose
`echo` is a no-op, so stack output is discarded. `DT 1;FF` sets the integration step to 1 s and
puts the simulation in fast-forward, i.e. unpaced by wall clock. Scenario attributes
(`_polygon_shape`, `n_aircraft`, `_slots`, `_max_exits`, `_max_steps`) are created in
`_build_sector`, not here, so `step()` before `reset()` raises `AttributeError`.

**Phase 1 · training setup.** `train()` (`train.py:190`) derives one worker seed per env from
`Random(seed)`, then stacks `DummyVecEnv` (one env, in-process, no IPC) →
`VecMonitor` (injects `info['episode']` with length and undiscounted return) → `VecNormalize`
(`obs_rms`: `RunningMeanStd` over the 29 dimensions; `ret_rms`: scalar over discounted returns;
`clip_obs = clip_reward = 10`, γ 0.99). `MlpPolicy` builds two separate trunks,
29 → 64 → 64 → 10 logits and 29 → 64 → 64 → 1 value, tanh. `learn()` → `_setup_learn` performs
the one initial `env.reset()` and stores `_last_obs` (`base_class.py:423`).

**Phase 2 · episode reset** (`env.py:69`).

1. `_new_episode_rngs` (`env.py:485`): rejection-sample a scenario seed from `range(10**9)` until
   it is not in `HELD_OUT`, then `master = Random(seed)` yields three children by
   `getrandbits(64)` — `scenario_rng`, `traffic_rng` (both `random.Random`) and `delay_rng`
   (`np.random.default_rng`, needed for `lognormal`). A fresh `ATCO` is constructed, so the τ
   sequence is a pure function of the scenario seed.
2. `_BsStack.cmdstack.clear()` — commands queued but not yet executed would otherwise fire into
   the new episode — then `bs.traf.reset()` (zeroes the traffic arrays and `ntraf`), then
   `DT 1;FF` re-pushed, since the reset drops it.
3. `_build_sector` (`env.py:520`): `n_ac ~ U{15,30}`, `ρ ~ U(1/20000, 1/10000)` km⁻²,
   `area = n_ac/ρ`; polygon by rejection until circularity `4πA/P² ≥ 0.7`; `shapely.prepare`
   builds the point-in-polygon index used for exit tests; `_max_exits = round(4·n_ac)`;
   `_max_steps = max(50, round(8 · d_bbox/450 kt · 3600 / 5 s))`.
4. `_spawn_initial_traffic`: per slot, up to 50 placement attempts, each requiring a ≥ 15 NM
   chord and passing `_spawn_is_safe` — a vectorised CPA of the candidate against all airborne
   aircraft, rejecting any `d < 15 NM` or `t_los < 360 s`. A slot that fails all attempts is
   deferred 5 steps.
5. `_refresh_traffic_view` reads BlueSky once and builds `_row_of`, `_pos`, `_vel`, `_hdg`,
   `urgency`, `t_los`, `_return_blocked`; `_select_focus` then sets `focus_cs`.

**Phase 3 · rollout, 2048 steps.** Per step: `normalize_obs` applies
`clip((o − μ)/√(σ² + 1e-8), ±10)` with the *current* statistics, which are updated every step
while `training=True`; the policy samples from the categorical over the 10 logits (`ent_coef`
is 0, so exploration comes only from that sampling, not from an entropy bonus); `env.step`
returns the raw reward; `ret_rms` is updated and the reward divided by `√(var + 1e-8)` and
clipped to ±10. The buffer holds `(2048, 1, 29)` observations plus actions, log-probs, values,
rewards and dones.

**Phase 4 · episode boundary.** The env sets `truncated=True` and puts the ~70-key
`episode_summary` in `info`. `DummyVecEnv.step_wait` writes `info['terminal_observation']`, calls
`reset()` inline and returns the *new* episode's first observation, so the rollout has no gap;
`VecNormalize` normalises the stored terminal observation too (`vec_normalize.py:203`). Because
`TimeLimit.truncated` is set and `terminated` is `False`, PPO adds `γ · V(s_terminal)` to that
step's reward (`on_policy_algorithm.py:240`) — the time limit is treated as a cut, not a failure.

**Phase 5 · update, every 2048 steps.** GAE with γ 0.99, λ 0.95:
`δ_t = r_t + γV(s_{t+1})(1−d_t) − V(s_t)`, `A_t = δ_t + γλ(1−d_t)A_{t+1}`, with `d_t` ending the
recursion at episode boundaries. Advantages are standardised over the minibatch, which here is
the whole rollout. 30 epochs × 1 gradient step = 30 Adam steps per rollout, against SB3's
default 320 (10 × 32); the clipped surrogate plus `0.5 · L_VF`, gradients clipped at norm 0.5.

**Phase 6 · housekeeping.** `Checkpoint` writes `last_model.zip` and `last_model_vecnorm.pkl`
every 500 000 steps; `StopAfterEpisodes` returns `False` at 60 000 episodes, which ends
`learn()`; `final_model.*` is written in the `finally` block, so `KeyboardInterrupt` also saves.

**Phase 7 · testing, separate process** (`Validation/run_test_scenarios.py`). The checkpoint's
`vecnorm.pkl` is unpickled through a shim for the renamed numpy modules, and only
`obs_rms.mean`, `obs_rms.var` and `clip_obs` are taken: normalisation is applied by hand and
**frozen**, with no `VecNormalize` wrapper and no reward normalisation. Actions are
`deterministic=True` (argmax, not sampled). `blind=True` zeroes the four response-state indices
before the policy sees them. `atco.act_if_ready` is monkey-patched to count the instructions
that actually execute, which the env itself does not track, and each of the 500 held-out
scenarios runs via `reset(options={'scenario_seed': s})` to its own truncation.

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
critic. There is no option for parallel environments: the rollout is one environment's 2048
steps, which is what the reported batch size assumes.

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
  processes, which is what the validation worker pool does. Training deliberately runs one.
- **The episode counter is not in the checkpoint.** `--resume` restarts it at zero.
- **Turns are relative to the route**, so action 2 twice is −60° from the route.
- **`t_los` is constant-velocity**, recomputed every step: a look-ahead, not a prediction of
  what the agent will do.
- **The no-delay model's response-state inputs are zeroed at test time**, deliberately.
