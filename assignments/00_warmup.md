# Warm-up: getting back into the codebase

You've been away a month. The goal of this file is **not** to teach you Python from
scratch — it's to rebuild the specific mental models you need to read
`Environments/v4/` and think "yes, obviously" instead of "what is this doing".

Every exercise is grounded in your own code. Where you see 📍 you should open the real
file and find the thing.

**Suggested route:** Part 0 → 1 → 2 → 3 → 4 → 5. Budget ~3 hours. Do Part 5 last; it's
the one that actually matters, the rest is scaffolding for it.

Write your scratch answers in `assignments/scratch.py` (gitignored — make it yourself).

---

## Part 0 — Does it still run? (10 min)

Before any theory, confirm the environment still imports and steps.

```python
# assignments/scratch.py
from Environments.v4 import AirspaceEnv

env = AirspaceEnv()
obs, info = env.reset(seed=0)
print(type(obs), obs.shape, obs.dtype)

obs, reward, terminated, truncated, info = env.step(3)   # 3 = hold
print(f"reward={reward:.3f}  terminated={terminated}  truncated={truncated}")
print(info)
```

Run it from the repo root: `python -m assignments.scratch` (or just `python assignments/scratch.py`).

**Q0.1** What is `obs.shape`? Before you run it, predict the number from
📍 [`config.py`](../Environments/v4/config.py) — there's a line that computes it.

<details><summary>Answer</summary>

`(26,)`. From `OBS_DIM = 6 + N_NEIGHBOURS * 5` where `N_NEIGHBOURS = 4`, so `6 + 20 = 26`.
Six ownship features, then four intruder slots of five features each.
</details>

**Q0.2** Run `env.step(3)` twenty times in a loop and print the reward each time. Action 3
is "hold". Why is the reward almost never exactly `0.0`, even though holding is free?

<details><summary>Answer</summary>

`ACT_COST[3] = 0.0`, so the workload term is zero. But `_compute_reward` has three terms:

```python
return float(r_los + r_drift + r_work)
```

`r_drift` is `-w_drift * (1 - cos(hdg_err)) / 2` — it's only zero when the focus aircraft's
*commanded* heading exactly matches its *route* heading. Because `_refresh_route_headings()`
recomputes the route bearing from the live position every step, tiny drift creeps in.
And `r_los` fires `-10.0` on any step where two aircraft are within 5 NM.
</details>

---

## Part 1 — Python core: data structures & functions (45 min)

### 1a. Dicts as the primary state container

Your env stores nearly all per-aircraft state in **dicts keyed by callsign string**
(`'AC00'`, `'AC01'`, …). 📍 `_clear_episode_state()` in
[`env.py`](../Environments/v4/env.py):

```python
self._route_hdg         = {}   # callsign -> live bearing to destination (deg)
self._commanded_heading = {}   # callsign -> heading we last told it to fly
self._commanded_mach    = {}   # callsign -> commanded Mach
self._direct_mode       = {}   # callsign -> bool, is fly-direct active
```

**E1.1** Build this yourself from scratch:

```python
route_hdg = {'AC00': 90.0, 'AC01': 270.0, 'AC02': 45.0}
commanded = {'AC00': 120.0, 'AC02': 45.0}          # note: AC01 missing
```

Write a loop that prints, for every callsign in `route_hdg`, the heading error
`commanded - route`, using `0.0`-drift as the fallback when the callsign is missing from
`commanded`. Do it **without** an `if key in dict` check.

<details><summary>Answer</summary>

```python
for cs, route in route_hdg.items():
    cmd = commanded.get(cs, route)      # fall back to the route itself -> zero error
    print(cs, cmd - route)
```

`.get(key, default)` is the workhorse. Your code uses it constantly, e.g.

```python
cmd_hdg = self._commanded_heading.get(cs, bs.traf.hdg[idx])
```

which means "the heading we commanded, or if we never commanded one, whatever it's
actually flying right now".
</details>

**E1.2** 📍 Find this in `_process_exits()`:

```python
for d in (self._destination_ll, self._ref_ll, self._route_hdg, self._commanded_heading,
          self._commanded_mach, self._direct_mode, self._steps_since_urgency):
    d.pop(cs, None)
```

Explain in one sentence what this does and why the `None` is essential.

<details><summary>Answer</summary>

It iterates over a *tuple of seven dicts* and deletes the exited aircraft's entry from
each one, so a departed callsign leaves no stale state behind. `dict.pop(key)` raises
`KeyError` if the key is absent; `dict.pop(key, None)` returns `None` instead. Since not
every dict is guaranteed to have an entry for `cs`, the default makes it safe.

This is a nice idiom to recognise: **loop over the containers, not the keys.**
</details>

### 1b. Sets

📍 `self._active_callsigns = set()`, with `.add(cs)` in `_spawn_aircraft` and
`.discard(cs)` in `_process_exits`.

**E1.3** Why `.discard(cs)` and not `.remove(cs)`? And why is a `set` the right choice
here rather than a `list`?

<details><summary>Answer</summary>

- `.remove()` raises `KeyError` if absent, `.discard()` is silent. Same safety reasoning
  as `.pop(cs, None)`.
- A set gives O(1) membership testing and automatic deduplication. The code does
  `for cs in self._active_callsigns` and membership checks constantly; order genuinely
  doesn't matter because ordering is always imposed explicitly later (see E1.5).
- ⚠️ Consequence: set iteration order is **not** insertion order. That's why
  `_drift_fallback` iterates `for cs in sorted(flying)` — determinism matters there for
  reproducible seeded runs.
</details>

### 1c. Lists, indexing, and the action layout

📍 [`config.py`](../Environments/v4/config.py):

```python
TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}
ACT_COST      = [0.75, 0.625, 0.5, 0.0, 0.5, 0.625, 0.75, 0.125, 0.25, 0.25]
```

**E1.4** `TURN_DELTAS` is a dict with integer keys 0–6 but **skips 3**. `ACT_COST` is a
list of 10. Why is one a dict and the other a list? What breaks if you make
`TURN_DELTAS` a list?

<details><summary>Answer</summary>

`ACT_COST` is dense — every action 0–9 has a cost, so a list indexed by action is
perfect: `ACT_COST[action_idx]`.

`TURN_DELTAS` is **sparse** — only the six turn actions have a heading delta. Actions 3
(hold), 7 (fly-direct), 8 and 9 (speed) are not turns. The dict lets `_apply_action` ask:

```python
if action_idx in TURN_DELTAS:
```

which is simultaneously a membership test *and* a lookup. As a list you'd need sentinel
values (`None` at index 3, 7, 8, 9) and a separate `is not None` check — more fragile.
</details>

**E1.5** 📍 The intruder-ordering logic in `_get_observation()`:

```python
ordered = sorted([r for r in intruders if r[0] > 0], key=lambda r: -r[0]) \
          + sorted(intruders, key=lambda r: r[1])
```

Each `r` is a tuple `(pair_urgency, dist_nm, feature_list, callsign)`. Describe the
resulting order in words. Why does the second `sorted()` include aircraft already in the
first list, and what stops them appearing twice?

<details><summary>Answer</summary>

**Order:** all aircraft in conflict (urgency > 0), most urgent first; then *every*
aircraft again, nearest first.

The concatenation deliberately over-provides — it's a priority list with duplicates. The
dedup happens in the loop right after:

```python
selected, seen = [], set()
for rec in ordered:
    if len(selected) >= N_NEIGHBOURS:
        break
    if rec[3] not in seen:
        selected.append(rec)
        seen.add(rec[3])
```

`rec[3]` is the callsign; `seen` is a set guarding against re-adding. The effect: fill the
4 slots with conflicts first, then pad with nearest traffic. Note `key=lambda r: -r[0]`
— negating to get descending order, instead of `reverse=True`.
</details>

### 1d. Functions are objects

📍 This is the one that trips people up after a break. In `CONFIG`:

```python
'n_aircraft':   lambda: 30,
'rho':          lambda: 1/2500,
'n_vertices':   lambda: random.randint(6, 12),
'spawn_jitter': lambda: random.uniform(0.1, 0.9),
```

and at the call site in `reset()`:

```python
n_ac     = CONFIG['n_aircraft']()      # note the trailing ()
area_km2 = float(n_ac / CONFIG['rho']())
```

**E1.6** Why are these lambdas instead of plain values? What would break if you wrote
`'n_vertices': random.randint(6, 12)` and `CONFIG['n_vertices']` without the `()`?

<details><summary>Answer</summary>

The dict is built **once at import time**. `random.randint(6, 12)` would be evaluated
once, freezing a single number for the entire training run — every sector would have
identical vertex count. Wrapping it in `lambda:` defers evaluation, so calling
`CONFIG['n_vertices']()` re-rolls the dice **every episode**.

`n_aircraft` and `rho` are currently constant (`lambda: 30`), but keeping them callable
means you can swap in `lambda: random.randint(10, 30)` to randomise density without
touching any call site. That's the real design win: **uniform interface**.

`CONFIG['n_vertices']` without `()` gives you the function object itself — you'd be
passing a `<function>` where an int is expected, and `random_convex_polygon` would
throw a `TypeError`.
</details>

**E1.7** Quick drill, no code from the repo:

```python
fns = [lambda: i for i in range(3)]
print([f() for f in fns])
```

Predict the output before running.

<details><summary>Answer</summary>

`[2, 2, 2]` — not `[0, 1, 2]`. The lambda captures the *variable* `i`, not its value at
creation. By the time they're called, the loop has finished and `i == 2`.

Fix: `[lambda i=i: i for i in range(3)]` (default-arg binding).

Your CONFIG lambdas take no arguments and reference no loop variables, so they sidestep
this entirely — but it's the classic trap and worth having reloaded.
</details>

---

## Part 2 — OOP: classes, inheritance, state (40 min)

### 2a. Inheritance and `super()`

📍 [`env.py`](../Environments/v4/env.py):

```python
class _ScreenDummy(ScreenIO):
    def echo(self, text='', flags=0):
        pass

class AirspaceEnv(gym.Env):
    metadata = {'render_modes': []}
    _EMPTY_SLOT = [1.0, 0.0, 0.0, 0.0, 1.0]

    def __init__(self):
        super().__init__()
        ...
```

**E2.1** `_ScreenDummy` overrides `echo` with `pass`. What pattern is this, and why is it
needed?

<details><summary>Answer</summary>

A **null object** / stub. BlueSky calls `bs.scr.echo(...)` to print to its GUI console.
Running headless in 48 parallel training envs, that output is useless noise and costs
time. Subclassing `ScreenIO` and overriding `echo` to do nothing silences it while
keeping the interface BlueSky expects.

Note the leading underscore: `_ScreenDummy` is a convention meaning "internal, not part
of the public API of this module". Same for all the `self._foo` attributes.
</details>

**E2.2** `metadata` and `_EMPTY_SLOT` are defined at **class level**, not inside
`__init__`. What's the difference, and is `_EMPTY_SLOT` being a mutable list a bug here?

<details><summary>Answer</summary>

Class attributes are shared by all instances; instance attributes (`self.foo = ...`) are
per-object. `_EMPTY_SLOT` is shared across every `AirspaceEnv`.

Mutable class attributes are usually a red flag — if any code did
`env._EMPTY_SLOT.append(0)` it would corrupt every env in the process. Here it's safe
because it's only ever **read**, and only in expressions that build new lists:

```python
obs += selected[k][2] if k < len(selected) else self._EMPTY_SLOT
return np.array([...] + self._EMPTY_SLOT * N_NEIGHBOURS, dtype=np.float32)
```

`list * int` and `list + list` both produce *new* lists. Nothing mutates the original.

⚠️ Worth knowing the adjacent trap though: `list * int` is a **shallow** copy. Here the
elements are floats (immutable), so it's safe. If `_EMPTY_SLOT` were a list of *lists*,
`[[0]] * 4` would give you four references to the **same** inner list, and mutating one
would change all four.
</details>

**E2.3** Why does `_clear_episode_state()` exist as a separate method instead of the code
living directly in `reset()`?

<details><summary>Answer</summary>

It's called from **two** places — `__init__` (line ~66) and `reset()` (line ~109). Without
it you'd either duplicate 25 lines of initialisation or have `__init__` call `reset()`,
which is heavier (it spawns aircraft, builds a polygon) and gives a confusing
"constructor does real work" design.

The docstring says it explicitly: *"shared by `__init__` and reset"*. The payoff is that
every per-episode attribute is declared in exactly one place — so you can read that one
method and know the complete state of an episode.
</details>

### 2b. Module-level mutable state

📍 Top of `env.py`:

```python
_bs_initialized = False

class AirspaceEnv(gym.Env):
    def __init__(self):
        global _bs_initialized
        if not _bs_initialized:
            bs.init(mode='sim', detached=True)
            _bs_initialized = True
```

**E2.4** Why a module-level flag rather than a class attribute or instance attribute? And
why does this still work correctly with `SubprocVecEnv(n_envs=48)`?

<details><summary>Answer</summary>

`bs.init()` sets up BlueSky's **global singleton** simulator state (`bs.traf`, `bs.sim`,
`bs.stack`). Calling it twice in one process is at best wasteful, at worst corrupting. The
flag guards "once per process".

An instance attribute obviously wouldn't work (each env gets its own). A class attribute
would work identically to the module global here — the module global is just the more
conventional spelling for process-level state.

With `SubprocVecEnv` each of the 48 envs lives in its **own OS process**, so each gets a
fresh module import, `_bs_initialized = False`, and its own independent BlueSky singleton.
That's exactly what you want, and it's precisely why `SubprocVecEnv` is used instead of
`DummyVecEnv` — 48 envs in one process would all fight over one `bs.traf`.

**This is the single most important architectural fact about your training setup.**
</details>

### 2c. The `for ... else` construct

📍 In `reset()` — this one catches almost everyone:

```python
for slot in range(n_ac):
    for _ in range(CONFIG['max_placement_tries']):
        ac = place_aircraft(self._polygon_shape, slot, n_ac)
        if self._spawn_ok(ac):
            self._spawn_aircraft(slot, ac)
            break
    else:
        self._pending_spawns[slot] = 5
```

**E2.5** When does the `else:` branch run?

<details><summary>Answer</summary>

**When the inner `for` loop finishes without hitting `break`** — i.e. all 50 placement
attempts failed to find a conflict-free spawn point. It does *not* run when `break` fired.

Read it as `for ... nobreak:`. The logic: "try 50 times to place this aircraft; if none
worked, queue it for a retry in 5 steps via `_pending_spawns`."

Drill to cement it:
```python
for x in [1, 2, 3]:
    if x == 99: break
else:
    print("never broke")     # prints
```
</details>

---

## Part 3 — Gymnasium (35 min)

### 3a. The `Env` contract

A Gymnasium environment is a class with exactly four required pieces:

| Piece | Yours |
|---|---|
| `observation_space` | `spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)` |
| `action_space` | `spaces.Discrete(10)` |
| `reset(seed, options)` | → `(obs, info)` |
| `step(action)` | → `(obs, reward, terminated, truncated, info)` |

**E3.1** `Box` vs `Discrete` — what does each represent, and what would `Box(low=-1,
high=1, shape=(2,))` mean as an *action* space?

<details><summary>Answer</summary>

- `Discrete(n)` = a single integer in `{0, …, n-1}`. Your 10 ATC instructions.
- `Box(low, high, shape)` = a continuous vector of floats. Your 26-float observation.

`Box(low=-1, high=1, shape=(2,))` as an action space would be **continuous control** —
e.g. "turn rate ∈ [-1,1], speed change ∈ [-1,1]". That's a different problem class; PPO
handles both, but with a Gaussian policy head rather than a categorical one.

Your discrete choice is deliberate and defensible: real ATC instructions are discrete
radio calls ("turn right heading 090"), not continuous rates.
</details>

**E3.2 — the big one.** `terminated` vs `truncated`. 📍 Your `step()` ends with:

```python
truncated = self._step_count >= self._max_steps
return self._get_observation(), reward, False, truncated, info
```

`terminated` is **hardcoded `False`**. Why, and why does the distinction matter
*mathematically* for PPO?

<details><summary>Answer</summary>

- **`terminated`** = the episode ended for a reason intrinsic to the MDP — a true
  absorbing state (agent died, goal reached). Future return from that state is genuinely 0.
- **`truncated`** = the episode was cut off artificially (time limit). The underlying
  process would have continued.

Your airspace never "ends". Aircraft exit, new ones spawn, the sector keeps running
forever. There is no terminal state — so `terminated=False` always. You stop after
`_max_steps` purely as a practical training-episode boundary.

**Why it matters:** the TD target is
```
target = r + γ · V(s') · (1 − terminated)
```
On `terminated`, the bootstrap is zeroed. On `truncated`, SB3 still bootstraps `V(s')` —
correctly, because the aircraft are still flying and future reward exists. If you
mistakenly reported `terminated=True` at the time limit, the value function would learn
that every state 300 steps in is worth ~0, poisoning your advantage estimates.

This is the single most common Gymnasium bug and your code gets it right.
</details>

**E3.3** `reset()` has this signature and first lines:

```python
def reset(self, seed=None, options=None):
    effective_seed = seed if seed is not None else CONFIG['seed']
    super().reset(seed=effective_seed)
    if effective_seed is not None:
        random.seed(effective_seed)
        np.random.seed(effective_seed)
```

Why seed **three** separate RNGs?

<details><summary>Answer</summary>

Three independent sources of randomness are in play:

1. `super().reset(seed=...)` seeds `self.np_random`, Gymnasium's own generator (used by
   `action_space.sample()` and anything using `self.np_random`).
2. `random.seed()` — your `sector.py` and `config.py` lambdas use the **stdlib** `random`
   module (`random.randint`, `random.uniform`), a totally separate generator.
3. `np.random.seed()` — `polygenerator` and any numpy-internal sampling.

Miss any one and your "seeded" run isn't reproducible. Since `place_aircraft` leans on
`CONFIG['spawn_jitter']()` → `random.uniform`, #2 is doing real work for you.
</details>

**E3.4** Write a random-policy rollout that reports the mean reward and LoS-step count
over one episode, using only the Gym API:

<details><summary>Answer</summary>

```python
import numpy as np
from Environments.v4 import AirspaceEnv

env = AirspaceEnv()
obs, info = env.reset(seed=1)

rewards, los_steps, done = [], 0, False
while not done:
    action = env.action_space.sample()           # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    los_steps += info['los_pairs'] > 0
    done = terminated or truncated

print(f"steps={len(rewards)}  mean_r={np.mean(rewards):.3f}  los_steps={los_steps}")
print(info)          # on the final step this carries the full episode summary
```

Note `info` on the truncating step is enriched by `_episode_summary()` —
`mean_episode_reward`, `ep_los_steps`, `ep_arrival_rate`, `action_distribution`. That's
the hook your training callbacks read.
</details>

---

## Part 4 — Stable-Baselines3 (40 min)

### 4a. The vectorised-env stack

📍 [`v4_train.py`](../Training/v4_train.py):

```python
venv = make_vec_env(AirspaceEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=seed)
env  = VecNormalize(VecMonitor(venv), norm_obs=True, norm_reward=True,
                    clip_obs=10.0, clip_reward=10.0, gamma=0.99)
```

**E4.1** Three wrappers, nested. Name what each does, and explain why `VecMonitor` is
**inside** `VecNormalize` rather than outside.

<details><summary>Answer</summary>

| Wrapper | Job |
|---|---|
| `SubprocVecEnv` | Runs 48 envs in 48 separate processes, steps them in parallel |
| `VecMonitor` | Records raw episode return + length into `model.ep_info_buffer` |
| `VecNormalize` | Maintains running mean/std, standardises obs and reward |

Order matters enormously. `VecMonitor` sits *inside*, so it sees the **raw, un-normalised**
reward. That's what you want for logging: `-3.42` is interpretable in your reward units,
whereas a normalised `-0.8` drifts in meaning as the running std updates.

Your `BestModelCallback` docstring says exactly this: *"Rewards come from VecMonitor
(inside VecNormalize), so they are un-normalised."* So `self.best` is comparable across
the whole run. If `VecMonitor` were outside, "best model" would be measured on a moving
ruler and the comparison would be meaningless.
</details>

**E4.2 — the trap that will bite you.** The module docstring warns:

> The `*_vecnorm.pkl` saved next to each model holds those stats and MUST be loaded at
> eval/visualisation time — feeding raw observations to a `norm_obs=True` policy collapses it.

Explain *mechanically* why a policy trained with `norm_obs=True` produces garbage on raw
observations.

<details><summary>Answer</summary>

During training the policy network only ever saw `(obs − μ) / σ` — roughly zero-mean,
unit-variance, clipped to ±10. Its first-layer weights are fitted to that scale.

At eval time your raw obs has features on wildly different scales: `rho ∈ [0,1]`,
`theta ∈ [-π, π]`, `v_own ≈ 1.0`, `tau ∈ [0,1]`. Feed those in directly and every
pre-activation lands somewhere the network never trained on. The argmax over action
logits becomes effectively arbitrary — typically the policy collapses to spamming one
action.

The fix, at eval:
```python
env = VecNormalize.load('best_model_vecnorm.pkl', venv)
env.training    = False      # freeze the running stats
env.norm_reward = False      # you want raw reward for reporting
```
Both lines matter: without `training = False` the stats keep updating from eval data.
</details>

**E4.3** Rollout arithmetic. Given `N_ENVS = 48`, `n_steps = 256`, `batch_size = 2048`,
`n_epochs = 10`:

1. How many transitions per rollout?
2. How many gradient steps per rollout?
3. How often does `_on_step` fire relative to `num_timesteps`?

<details><summary>Answer</summary>

1. `256 × 48 = 12,288` transitions per rollout.
2. `12,288 / 2,048 = 6` minibatches, `× 10` epochs = **60 gradient steps** per rollout.
   (Matches the comment in `PPO_KWARGS`.)
3. `_on_step` is called once per **vectorised** step, so once per 48 environment steps.
   `self.num_timesteps` advances by `N_ENVS` each call — which is why
   `ProgressCallback` compares `self.num_timesteps - self.last_print < 10_000` rather
   than counting calls.
</details>

### 4b. Callbacks

📍 `EpisodeStatsCallback._on_step`:

```python
def _on_step(self):
    for info in self.locals.get('infos', []):
        if 'mean_episode_reward' not in info:
            continue
        self.logger.record_mean('episode/mean_reward', info['mean_episode_reward'])
        ...
    return True
```

**E4.4** What is `self.locals`? Why the `if ... not in info: continue` guard? And what
does returning `False` do?

<details><summary>Answer</summary>

- `self.locals` is SB3 handing the callback the **local variables of the collection
  loop** — including `infos`, the list of 48 info dicts from the last vec-step. It's a
  slightly grubby but very useful escape hatch.
- The guard: only the env(s) that *just truncated* have `mean_episode_reward` in their
  info (added by `_episode_summary()`). On a typical step, 47 of 48 dicts lack the key.
  `continue` skips them.
- **`return False` stops training.** Every `_on_step` must return `True` to continue —
  a missing `return` gives `None`, which is falsy, and training halts immediately after
  the first rollout. Classic silent bug.

`record_mean` (rather than `record`) accumulates and logs the mean over the reporting
interval — right choice for a metric arriving from 48 envs at irregular times.
</details>

**E4.5** In `BestModelCallback`, `mean_reward` is computed from `self.model.ep_info_buffer`.
What is that buffer, and what's the subtle issue with using it as a "best model" criterion?

<details><summary>Answer</summary>

`ep_info_buffer` is a `deque` of the last ~100 completed episodes' `{'r': return,
'l': length}`, filled by `VecMonitor`.

The subtlety: it's a **rolling window over recent training episodes**, under the current
(exploring, entropy-regularised) policy — not a clean evaluation. So "best" means "best
recent training performance", which is noisy and slightly optimistic. With
`ent_coef = 0.02` the policy is deliberately stochastic, so it's not measuring the greedy
policy you'd deploy either.

That's a legitimate, cheap choice — a proper `EvalCallback` on held-out seeds costs extra
rollouts. Worth a sentence in your thesis methodology, since a reviewer may ask.
</details>

---

## Part 5 — Capstone: trace one `step()` (45 min)

No new concepts. Open [`env.py`](../Environments/v4/env.py) at `step()` and write out, in
your own words, what happens in order. Do this **on paper or in a comment block** — the
act of writing it is the point.

```python
def step(self, action):
    action = int(action)
    self._process_pending_spawns()
    self._refresh_route_headings()
    acting_cs = self._focus_cs
    if acting_cs:
        self._apply_action(acting_cs, action)
    self._update_direct_headings()
    for _ in range(CONFIG['action_freq']):
        bs.sim.step()
    self._los_this_step = any_los(self._active_callsigns)
    self._step_count += 1
    self._process_exits()
    self._focus_cs = self._select_focus_aircraft()
    reward = self._compute_reward(acting_cs, action)
    ...
```

Answer these as you go:

**C1** Why is `acting_cs` captured into a local variable *before* the action is applied,
and then used again at the very end in `_compute_reward`?

<details><summary>Answer</summary>

Because `self._focus_cs` is **reassigned** two lines before the reward is computed:

```python
self._focus_cs = self._select_focus_aircraft()
reward = self._compute_reward(acting_cs, action)
```

The reward must be attributed to the aircraft that actually *received* the instruction,
not to whichever aircraft the focus has now moved to. Using `self._focus_cs` in
`_compute_reward` would be an off-by-one credit-assignment bug — you'd penalise the new
focus's drift for an action taken by a different aircraft.
</details>

**C2** `for _ in range(CONFIG['action_freq']): bs.sim.step()` — with `action_freq = 5`
and `sim_dt = 1.0`, how much simulated time is one RL step? Why decouple them?

<details><summary>Answer</summary>

5 seconds of simulated time per RL step (5 × 1 s). BlueSky integrates at 1 Hz for
numerical fidelity of the turn dynamics, but the agent only makes a decision every 5 s.

Decoupling gives you: realistic controller cadence (a real ATCO doesn't issue an
instruction every second), 5× fewer decisions per episode of simulated time (cheaper
training, shorter credit-assignment horizon), and physics accuracy that's independent of
the control rate.
</details>

**C3** Trace the fly-direct (action 7) mechanism across three methods: `_apply_action`,
`_update_direct_headings`, `_refresh_route_headings`. Why does it need to be re-issued
every step instead of just once?

<details><summary>Answer</summary>

- `_apply_action(cs, 7)` sets `self._direct_mode[cs] = True` and commands the current
  route heading.
- `_refresh_route_headings()` (top of every step) recomputes `_route_hdg[cs]` as the live
  bearing from the aircraft's **current position** to its far destination.
- `_update_direct_headings()` then re-issues `HDG {cs} {route_hdg}` for every aircraft
  with `_direct_mode == True`.

It must repeat because the route heading is a **live bearing that changes as the aircraft
moves**. Commanding it once would lock in a stale heading; the aircraft would fly a fixed
bearing and never actually converge on the destination. Re-aiming each step makes
fly-direct a persistent *mode* rather than a one-shot instruction — which is why
`_apply_action` sets `_direct_mode[cs] = False` on any manual turn, cancelling the mode.

The destination is placed `dest_dist_factor = 20` sector-diameters away specifically so
this bearing changes slowly and a held heading stays on route.
</details>

**C4** Follow one number end to end: pick intruder feature `tau` in `_get_observation`.
Where does its value come from, what normalises it, and what does `tau = 1.0` mean to the
policy?

<details><summary>Answer</summary>

```python
t_los = time_to_los(dist_nm ** 2, range_rate, rel_spd_sq, sep)
tau   = min(max(0.0, t_los) / t_warn, 1.0) if t_los is not None else 1.0
```

- `time_to_los` (in [`conflict.py`](../Environments/v4/conflict.py)) solves for when the
  pair first breaches the 5 NM circle: `t_los = tcpa − √((sep² − dcpa²)/|v|²)`. It returns
  `None` when diverging, parallel, or the miss distance exceeds `sep`.
- Normalised by `t_warn = 360 s` and clipped to `[0, 1]`.
- **`tau = 1.0` means "no threat"** — either ≥ 6 minutes away, or `None` (never
  intrudes). And `_EMPTY_SLOT = [1.0, 0.0, 0.0, 0.0, 1.0]` sets `rho = 1` (far) and
  `tau = 1` (no threat) for unused slots, so an empty slot is indistinguishable from a
  harmless distant aircraft. That's a deliberate ACAS Xu convention — it means the network
  never needs a separate "slot valid" bit.
- Special case: inside separation, `tau = 0.0` is forced (`if dist_nm < sep`), because
  `time_to_los` isn't meaningful once you're already in LoS.
</details>

**C5** Finally, the focus mechanism. In two or three sentences, explain what problem
`_select_focus_aircraft` solves and why hysteresis (`_focus_hold_steps`,
`focus_clear_steps`, `drift_switch_margin`) is needed at all.

<details><summary>Answer</summary>

You have 30 aircraft but a `Discrete(10)` action space controlling **one** aircraft per
step, so something must decide *which* aircraft the agent is instructing. `_select_focus_aircraft`
picks the aircraft with the highest worst-pair urgency (tiebreak: total urgency burden),
falling back to `_drift_fallback` — the most drifted aircraft in the clearest airspace —
when nothing is in conflict.

Hysteresis stops focus **thrashing**. Without it, two aircraft with near-identical urgency
would alternate every step, each receiving half a manoeuvre, and neither conflict would
resolve. So the current focus is retained while it's still active (`focus_urgency > 0` or
not yet clear for `focus_clear_steps`), and a challenger must beat it by
`drift_switch_margin` to take over. The `emergency` override (`urgency ≥ 0.67`, ~2 min to
CPA) breaks the lock when something genuinely urgent appears elsewhere.

This is arguably the most thesis-relevant design decision in the file — it's the bridge
between a single-agent RL formulation and a multi-aircraft problem.
</details>

---

## Quick reference: your env at a glance

**Observation (26 floats), ego-centric from the focus aircraft**

| Idx | Name | Meaning | Scale |
|---|---|---|---|
| 0 | `dpsi` | actual heading error vs route | rad |
| 1 | `v_own` | own speed / cruise | ~1.0 |
| 2 | `a_cmd` | commanded heading error vs route | rad |
| 3 | `v_cmd` | commanded Mach / nominal | ~1.0 |
| 4 | `retn_conf` | 1 = returning to route is blocked | {0,1} |
| 5 | `in_conf` | 1 = focus has a positive-urgency pair | {0,1} |
| 6–25 | 4 × `[rho, theta, psi, v_int, tau]` | intruder slots | see below |

`rho` = dist/45 NM · `theta` = bearing (rad, + = right) · `psi` = rel heading (rad) ·
`v_int` = speed/cruise · `tau` = t_los/360 s. Empty slot = `[1, 0, 0, 0, 1]`.

**Actions (`Discrete(10)`)**

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| −60° | −45° | −30° | hold | +30° | +45° | +60° | direct | spd+ | spd− |
| 0.75 | 0.625 | 0.5 | 0.0 | 0.5 | 0.625 | 0.75 | 0.125 | 0.25 | 0.25 |

(second row = `ACT_COST`; turn costs are deliberately **sub-additive** so the policy can't
game the reward by salami-slicing one big turn into several small ones)

**Reward** — purely negative:
```
r = −10·1[LoS]  −  1.0·(1−cos(dpsi))/2  −  1.0·ACT_COST[action]
     safety          route drift              controller workload
```

**Key numbers**: 30 aircraft · 5 NM separation · `t_warn` 360 s (45 NM) · 5 s per RL step ·
26-dim obs · 4 neighbours · FL350 · Mach 0.74–0.82.

---

## Where to go next

Once Part 5 feels comfortable:

1. [`Validation/mc_evaluate.py`](../Validation/mc_evaluate.py) — how the trained policy is
   scored (and the `VecNormalize.load` pattern from E4.2 in real use).
2. [`Validation/dalmau_heatmap.py`](../Validation/dalmau_heatmap.py) — the benchmark
   comparison.
3. [`visualisation/visualise.py`](../visualisation/visualise.py) — watch a policy fly;
   the fastest way to build intuition for what it actually learned.
