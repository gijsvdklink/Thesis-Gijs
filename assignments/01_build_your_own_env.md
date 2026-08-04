# Assignment 01 — Gymnasium & SB3 fundamentals

Build a working environment from nothing, train an agent on it, evaluate it. Only the
fundamentals — no tuning, no tricks.

The toy env is a miniature of your ATC env: one aircraft, one intruder, a route to hold,
a separation minimum. ~50 lines instead of 580. Level 7 maps it back to the real thing.

**Format:** skeleton with `TODO`s, then a puzzle, then the solution. Try before you peek.

```bash
mkdir -p assignments/sandbox      # gitignored
```

---

## Level 0 — The mental model (5 min)

```
        ┌──── action ────┐
        │                ▼
   ┌─────────┐      ┌─────────┐
   │  AGENT  │      │   ENV   │
   │  (SB3)  │      │ (yours) │
   └─────────┘      └─────────┘
        ▲                │
        └── obs, reward ─┘
```

| You write | SB3 writes |
|---|---|
| The world's rules and reward | The learning algorithm |
| What the agent sees (`observation_space`) | The neural network |
| What the agent can do (`action_space`) | Gradient updates |
| When an episode ends | Parallelism, logging |

Gymnasium doesn't *do* anything. It's an **interface contract** so any agent can talk to
any environment. That's the whole idea.

---

## Level 1 — Spaces (15 min)

The vocabulary for describing inputs and outputs.

```python
# assignments/sandbox/L1_spaces.py
import numpy as np
from gymnasium import spaces

d = spaces.Discrete(4)                                        # one int in {0,1,2,3}
b = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)   # 3 floats in [-1,1]

print(d.sample(), b.sample())
```

**P1.1** Predict `True`/`False`, then check with `.contains(...)`:

```python
b.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))    # a
b.contains(np.array([0.0, 0.0, 0.0], dtype=np.float64))    # b
b.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))    # c
d.contains(2)                                              # d
d.contains(2.0)                                            # e
```

<details><summary>Answer</summary>

`a` True · `b` **False** · `c` False · `d` True · `e` **False**

The two that matter:
- **(b) dtype must match exactly.** `float64` is not `float32`. This is the #1 cause of
  "my env won't pass `check_env`". Always end with `np.array(..., dtype=np.float32)` —
  📍 exactly what the last line of your `_get_observation` does.
- **(e)** `Discrete` needs an integer, not a float. That's why your real `step()` starts
  with `action = int(action)`.
</details>

**P1.2** Which space would you use for: (i) "turn left / hold / turn right", (ii) a
26-float observation vector, (iii) a continuous turn rate between −1 and 1?

<details><summary>Answer</summary>

(i) `Discrete(3)` — (ii) `Box(shape=(26,))` — (iii) `Box(low=-1, high=1, shape=(1,))`

Your real env is `Discrete(10)` actions, `Box(shape=(26,))` observations. Discrete
actions are the right call for ATC: real instructions are discrete radio calls
("turn right heading 090"), not continuous rates.
</details>

---

## Level 2 — The contract (20 min)

The smallest environment that satisfies Gymnasium. A counter starts at 10; action `1`
decrements it, action `0` waits. Episode ends at zero.

```python
# assignments/sandbox/L2_countdown.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CountdownEnv(gym.Env):
    def __init__(self, start=10):
        super().__init__()
        self.start   = start
        self.counter = start
        self.observation_space = ...     # TODO one float, counter scaled to [0,1]
        self.action_space      = ...     # TODO two choices

    def _obs(self):
        ...                              # TODO shape-(1,) float32 array

    def reset(self, seed=None, options=None):
        ...                              # TODO seed parent, reset counter, return (obs, info)

    def step(self, action):
        ...                              # TODO apply action, reward, return the 5-tuple
```

**P2.1** Fill it in. Reward `+1.0` on reaching zero, `−0.01` otherwise.

<details><summary>Solution</summary>

```python
class CountdownEnv(gym.Env):
    def __init__(self, start=10):
        super().__init__()
        self.start   = start
        self.counter = start
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

    def _obs(self):
        return np.array([self.counter / self.start], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = self.start
        return self._obs(), {}

    def step(self, action):
        if int(action) == 1:
            self.counter -= 1
        terminated = self.counter <= 0
        reward     = 1.0 if terminated else -0.01
        return self._obs(), reward, terminated, False, {}
```

**The three rules of the contract:**
1. `reset` returns a **2-tuple** `(obs, info)`.
2. `step` returns a **5-tuple** `(obs, reward, terminated, truncated, info)`.
3. Call `super().reset(seed=seed)` — it seeds the RNG.
</details>

**P2.2** Check it, then train it:

```python
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

check_env(CountdownEnv())                      # 10 seconds; run it on every new env

model = PPO('MlpPolicy', CountdownEnv(), verbose=1)
model.learn(total_timesteps=20_000)
```

Now break it deliberately and see what `check_env` says: return `float64` from `_obs()`,
then return a 4-tuple from `step()`.

<details><summary>Answer</summary>

`float64` → assertion about dtype mismatch. 4-tuple → unpacking error.

**Lesson: always run `check_env` before training.** Ten seconds, saves an afternoon.
</details>

**P2.3** Confirm it learned to always pick action 1:

```python
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

Why `deterministic=True`?

<details><summary>Answer</summary>

PPO learns a **stochastic** policy — a distribution over actions. The default
(`deterministic=False`) samples from it; `True` takes the argmax.

Stochastic during training (that's exploration), deterministic when evaluating or
visualising. Evaluate stochastically and you're measuring a policy you'd never deploy.
</details>

---

## Level 3 — Build ToyATC (45 min)

**Spec:** 2D plane. Ownship starts at `(−40, 0)` heading east; goal near `(+40, y)`.
An intruder starts near `(+40, y')` flying west — head-on. Both move 1 unit/step.
Heading is **degrees clockwise from north** (90 = east), matching your real code.

- Actions `Discrete(3)`: `0` left 15°, `1` hold, `2` right 15°
- Separation minimum 5, goal tolerance 3, max 300 steps
- Reward: `−10` inside separation, `−(1−cos(bearing_error))/2` for drift,
  `−0.1 × cost` for turning, `+10` on reaching the goal

### Helpers

```python
# assignments/sandbox/L3_toyatc.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap180(deg):
    """Wrap an angle in degrees to (-180, 180]."""
    return (deg + 180) % 360 - 180


def velocity(speed, hdg_deg):
    """Heading (deg clockwise from north) -> (east, north) velocity."""
    h = math.radians(hdg_deg)
    return np.array([speed * math.sin(h), speed * math.cos(h)])
```

**P3.1** Why `(sin, cos)` and not the `(cos, sin)` from school trigonometry?

<details><summary>Answer</summary>

Headings are **compass bearings** — measured clockwise from *north*, not
counter-clockwise from the *x* axis.

Check: 0° (north) → `(sin 0, cos 0) = (0, 1)` ✓. 90° (east) → `(1, 0)` ✓.

📍 Identical to `heading_to_velocity` in [`geometry.py`](../Environments/v4/geometry.py).
It's also why bearings use `atan2(east, north)` — arguments swapped from the usual.
</details>

**P3.2** Design the 5-float observation before writing code. Why should every feature be
**relative to the ownship** rather than absolute world coordinates?

<details><summary>Answer</summary>

```
[0] dist to goal        normalised to [0,1]
[1] bearing error to goal   /180
[2] distance to intruder    normalised
[3] relative bearing to intruder   /180
[4] intruder heading relative to mine  /180
```

**Ego-centric wins because it makes the problem translation- and rotation-invariant.** A
head-on conflict at `(0,0)` and the same conflict at `(30,20)` look completely different
in absolute coordinates but **identical** relative to the ownship. Absolute encoding forces
the network to re-learn the same manoeuvre separately in every region of the map.

That's exactly why your real observation is `(rho, theta, psi, v_int, tau)` per intruder
rather than lat/lon.
</details>

**P3.3** Write it.

```python
class ToyATCEnv(gym.Env):
    AREA, SPEED, TURN_DEG    = 50.0, 1.0, 15.0
    SEP, GOAL_TOL, MAX_STEPS = 5.0, 3.0, 300
    W_SEP, W_DRIFT, W_WORK   = 10.0, 1.0, 0.1
    ACT_COST = [0.5, 0.0, 0.5]          # left, hold, right

    def __init__(self):        ...      # TODO spaces
    def reset(self, seed=None, options=None): ...   # TODO place everything
    def _obs(self):            ...      # TODO the 5 features
    def step(self, action):    ...      # TODO turn, move, reward, done flags
```

<details><summary>Solution</summary>

```python
class ToyATCEnv(gym.Env):
    AREA, SPEED, TURN_DEG    = 50.0, 1.0, 15.0
    SEP, GOAL_TOL, MAX_STEPS = 5.0, 3.0, 300
    W_SEP, W_DRIFT, W_WORK   = 10.0, 1.0, 0.1
    ACT_COST = [0.5, 0.0, 0.5]

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos     = np.array([-40.0, 0.0])
        self.hdg     = 90.0                                  # east
        self.goal    = np.array([40.0, float(rng.uniform(-15, 15))])
        self.int_pos = np.array([40.0, float(rng.uniform(-10, 10))])
        self.int_hdg = 270.0                                 # west, head-on
        self.steps   = 0
        return self._obs(), {}

    def _bearing_to(self, target):
        """Bearing error (deg) from my heading to a target point."""
        d = target - self.pos
        return wrap180(math.degrees(math.atan2(d[0], d[1])) % 360 - self.hdg)

    def _obs(self):
        dist_goal = float(np.linalg.norm(self.goal - self.pos))
        rho       = float(np.linalg.norm(self.int_pos - self.pos))
        return np.array([
            min(dist_goal / (2 * self.AREA), 1.0),
            self._bearing_to(self.goal) / 180.0,
            min(rho / (2 * self.AREA), 1.0),
            self._bearing_to(self.int_pos) / 180.0,
            wrap180(self.int_hdg - self.hdg) / 180.0,
        ], dtype=np.float32)

    def step(self, action):
        action = int(action)
        if action == 0:
            self.hdg = (self.hdg - self.TURN_DEG) % 360
        elif action == 2:
            self.hdg = (self.hdg + self.TURN_DEG) % 360

        self.pos     = self.pos     + velocity(self.SPEED, self.hdg)
        self.int_pos = self.int_pos + velocity(self.SPEED, self.int_hdg)
        self.steps  += 1

        dist_goal = float(np.linalg.norm(self.goal - self.pos))
        sep_dist  = float(np.linalg.norm(self.int_pos - self.pos))
        brg_err   = self._bearing_to(self.goal)

        reward = (-self.W_SEP if sep_dist < self.SEP else 0.0) \
                 - self.W_DRIFT * (1 - math.cos(math.radians(brg_err))) / 2 \
                 - self.W_WORK * self.ACT_COST[action]

        terminated = dist_goal <= self.GOAL_TOL
        if terminated:
            reward += 10.0
        truncated = self.steps >= self.MAX_STEPS

        info = {'sep_dist': sep_dist, 'los': sep_dist < self.SEP}
        return self._obs(), reward, terminated, truncated, info
```

Run `check_env(ToyATCEnv())` before moving on.
</details>

**P3.4 — the most important question in this file.** `ToyATCEnv` has a real `terminated`
(reaching the goal) *and* a `truncated` (300 steps). Your real env has
`terminated=False` **always**. Why?

<details><summary>Answer</summary>

- **`terminated`** = a genuine end state. Nothing follows; future reward is truly zero.
- **`truncated`** = you cut it off artificially. The world would have carried on.

Reaching the goal ends the toy episode — genuinely terminal. But your airspace never ends:
aircraft exit, new ones spawn, the sector keeps running. Stopping at `_max_steps` is a
training convenience, so it's `truncated`.

**Why it matters.** The value target is
```
target = r + γ · V(s') · (1 − terminated)
```
`terminated` zeroes the bootstrap. `truncated` does not — SB3 still uses `V(s')`, correctly,
because the aircraft are still flying.

Get this backwards and you teach the critic that every state near step 300 is worth ≈ 0.
That error spreads backwards through the whole episode and corrupts your advantages —
with no error message anywhere.

It's the most common Gymnasium bug, and your real env gets it right.
</details>

**P3.5** Run a random policy and a hand-coded greedy baseline (always turn toward the
goal, ignore the intruder). Compare goal rate and LoS steps.

<details><summary>Solution</summary>

```python
import numpy as np

def rollout(env, policy, n=20):
    goals, los, rets = 0, [], []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        total, n_los, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r; n_los += info['los']; done = term or trunc
            goals += term
        rets.append(total); los.append(n_los)
    return goals / n, float(np.mean(los)), float(np.mean(rets))

env = ToyATCEnv()
print('random:', rollout(env, lambda o: env.action_space.sample()))
print('greedy:', rollout(env, lambda o: 0 if o[1] < -0.02 else (2 if o[1] > 0.02 else 1)))
```

`obs[1]` is the normalised bearing error: negative = goal is left, so turn left.

Random almost never reaches the goal. Greedy reaches it nearly always but flies straight
through the intruder and eats separation penalties. **That gap is what PPO has to close** —
and having a baseline to beat is how you'll read your real training curves.
</details>

---

## Level 4 — Train it (25 min)

```python
# assignments/sandbox/L4_train.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from L3_toyatc import ToyATCEnv

env   = Monitor(ToyATCEnv())
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200_000)
model.save('toyatc_ppo')
```

**P4.1** Why wrap in `Monitor`? What goes missing without it?

<details><summary>Answer</summary>

`Monitor` records each episode's return and length. Without it,
`rollout/ep_rew_mean` **never appears in the logs at all** — training runs fine, you just
can't see whether it's working.

In your real script the vectorised version `VecMonitor` does this job.
</details>

**P4.2** Watch the `verbose=1` table and find `rollout/ep_rew_mean` and
`train/entropy_loss`. What does each tell you?

<details><summary>Answer</summary>

- **`ep_rew_mean`** — mean episode return over recent episodes. Your "is it learning" number.
- **`entropy_loss`** — negative policy entropy. Starts near `−ln(3) ≈ −1.1` for 3 uniform
  actions and rises toward 0 as the policy commits to choices. If it hits 0 very early,
  the policy collapsed to one action and stopped exploring.
</details>

**P4.3** Go parallel:

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

venv  = VecMonitor(make_vec_env(ToyATCEnv, n_envs=8, vec_env_cls=DummyVecEnv))
model = PPO('MlpPolicy', venv, n_steps=256, batch_size=512, verbose=1)
model.learn(200_000)
```

Two questions: (i) `DummyVecEnv` vs `SubprocVecEnv`? (ii) With `n_envs=8`, does
`learn(200_000)` mean 200k steps *per env*?

<details><summary>Answer</summary>

**(i)** `DummyVecEnv` runs all envs in one process, in a plain loop. `SubprocVecEnv` gives
each env its own process — real parallelism, but with communication overhead. For a fast
toy env the dummy is usually **faster**.

Your real env *must* use `SubprocVecEnv`, and not for speed: BlueSky keeps global state
(`bs.traf`, `bs.sim`). 48 envs in one process would all mutate the *same* traffic array.
Separate processes give each its own BlueSky. 📍 That's what the `_bs_initialized`
module-level flag in `env.py` guards.

**(ii)** No — `total_timesteps` counts transitions across *all* envs. 200k total = 25k per
env. Surprises everyone once.
</details>

---

## Level 5 — VecNormalize: the one trap that will cost you a week (20 min)

```python
from stable_baselines3.common.vec_env import VecNormalize

venv = VecMonitor(make_vec_env(ToyATCEnv, n_envs=8))
env  = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(200_000)

model.save('toyatc_norm')
env.save('toyatc_vecnorm.pkl')       # ← forget this and the model is worthless
```

`VecNormalize` keeps a running mean/std and standardises observations and rewards. Neural
networks train much better on roughly zero-mean, unit-variance inputs.

**P5.1** Prove the failure mode to yourself:

```python
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load('toyatc_norm')

# WRONG — raw observations into a policy trained on normalised ones
raw = VecMonitor(make_vec_env(ToyATCEnv, n_envs=1))
print('no stats:  ', evaluate_policy(model, raw, n_eval_episodes=20))

# RIGHT
fixed = VecNormalize.load('toyatc_vecnorm.pkl',
                          VecMonitor(make_vec_env(ToyATCEnv, n_envs=1)))
fixed.training    = False
fixed.norm_reward = False
print('with stats:', evaluate_policy(model, fixed, n_eval_episodes=20))
```

Why do `training = False` and `norm_reward = False` both matter?

<details><summary>Answer</summary>

- **`training = False`** freezes the running statistics. Left on, evaluating *changes* the
  stats — so your results depend on evaluation order and aren't reproducible.
- **`norm_reward = False`** gives raw rewards in your reported numbers. A normalised
  `−0.8` is uninterpretable; `−34.2` in real reward units is.

You should see a large, obvious gap between the two. **Why:** the policy only ever saw
standardised inputs, so its weights are fitted to that scale. Feed raw values and every
activation lands somewhere the network never trained on — the policy typically collapses to
spamming one action.

📍 This is exactly the warning at the top of [`v4_train.py`](../Training/v4_train.py), and
the easiest way to waste a week wondering why a well-trained model "doesn't work".
</details>

---

## Level 6 — Evaluate and log (20 min)

**P6.1** Evaluate on 50 **held-out** seeds and report domain metrics, not just return.

<details><summary>Solution</summary>

```python
env = ToyATCEnv()
goals, los, rets = 0, [], []
for seed in range(1000, 1050):                 # disjoint from training seeds
    obs, _ = env.reset(seed=seed)
    total, n_los, done = 0.0, 0, False
    while not done:
        a, _ = model.predict(fixed.normalize_obs(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        total += r; n_los += info['los']; done = term or trunc
        goals += term
    rets.append(total); los.append(n_los)

print(f'goal_rate={goals/50:.2f}  los_steps={np.mean(los):.1f}  return={np.mean(rets):.1f}')
```

Two points:
- `fixed.normalize_obs(obs)` — stepping the raw env by hand means applying the
  normalisation yourself.
- **Domain metrics beat scalar return.** `−34.2` says little; "reaches the goal 94% of the
  time with 1.2 LoS steps" is a result you can defend. Same reason your real
  `_episode_summary()` reports `ep_arrival_rate` and `ep_los_steps` separately.
</details>

**P6.2** Write a minimal callback that logs mean separation distance.

<details><summary>Solution</summary>

```python
from stable_baselines3.common.callbacks import BaseCallback

class SepCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'sep_dist' in info:
                self.logger.record_mean('toy/sep_dist', info['sep_dist'])
        return True                        # ← mandatory

model.learn(200_000, callback=SepCallback())
```

Three fundamentals:
- `self.locals` is SB3 handing you its internal loop variables — `infos`, `rewards`, `dones`.
- `self.num_timesteps` advances by `n_envs` per call, never by 1.
- **`return True` is mandatory.** Return `False` and training stops; *forget* the return
  and you get `None`, which is falsy — training halts after one rollout, silently.

Use `record_mean` (not `record`) for per-episode metrics: with 8 envs reporting at
irregular times, `record` logs whichever env happened to be last.
</details>

---

## Level 7 — Bridge to your real env

| ToyATCEnv | `Environments/v4/` |
|---|---|
| `wrap180`, `velocity` | `wrap_to_180`, `heading_to_velocity` — identical |
| `self.pos`, `self.hdg` | `bs.traf.lat/lon/hdg` — BlueSky owns the state |
| `Discrete(3)` | `Discrete(10)` — adds speed and fly-direct |
| 5-float obs | 26-float obs — 6 ownship + 4×5 intruders |
| 1 intruder | 4 most urgent of 30 |
| `sep_dist < SEP` | `any_los()` — all pairs |
| goal reached → `terminated` | **never terminates** — see P3.4 |
| one aircraft | `_select_focus_aircraft` picks one of 30 per step |
| `DummyVecEnv` | `SubprocVecEnv(48)` — BlueSky global state |

**When you can read each right-hand entry and think "that's what I built, plus X", go to
[`00_warmup.md`](00_warmup.md) Part 5.**

---

## Cheat sheet

```python
# --- the contract
class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(lo, hi, shape=(n,), dtype=np.float32)
        self.action_space      = spaces.Discrete(k)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return obs, info                                    # 2-tuple
    def step(self, action):
        return obs, reward, terminated, truncated, info     # 5-tuple

# --- always, before training
check_env(MyEnv())

# --- train
venv  = VecMonitor(make_vec_env(MyEnv, n_envs=8, seed=0))
env   = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(200_000)
model.save('model'); env.save('vecnorm.pkl')      # BOTH, always

# --- load and evaluate
env = VecNormalize.load('vecnorm.pkl', VecMonitor(make_vec_env(MyEnv, n_envs=1)))
env.training, env.norm_reward = False, False
model = PPO.load('model')
evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
```

**Five mistakes that cost the most time**

1. `float64` observations → `check_env` assertion. Always `dtype=np.float32`.
2. Forgetting `env.save('vecnorm.pkl')` → the trained model is unusable.
3. Forgetting `env.training = False` at eval → stats drift, results irreproducible.
4. Callback without `return True` → training silently stops after one rollout.
5. Confusing `terminated` with `truncated` → corrupted value function, no error message.

---

# Appendix — full solution files

Complete, runnable versions of everything above. Drop them in `assignments/sandbox/` and
run in order. **Try each level first** — reading a finished solution teaches far less than
getting there.

Verified against `gymnasium 1.0.0`, `stable-baselines3 2.4.1`, `numpy 1.26.4`.

## `sandbox/L2_countdown.py`

```python
"""The minimum viable Gymnasium environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CountdownEnv(gym.Env):
    def __init__(self, start=10):
        super().__init__()
        self.start   = start
        self.counter = start
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

    def _obs(self):
        return np.array([self.counter / self.start], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = self.start
        return self._obs(), {}

    def step(self, action):
        if int(action) == 1:
            self.counter -= 1
        terminated = self.counter <= 0
        reward     = 1.0 if terminated else -0.01
        return self._obs(), reward, terminated, False, {}


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(CountdownEnv())
    print('check_env passed')

    model = PPO('MlpPolicy', CountdownEnv(), seed=0, verbose=1)
    model.learn(total_timesteps=20_000)

    env, actions = CountdownEnv(), []
    obs, _ = env.reset()
    for _ in range(15):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        actions.append(int(action))
        if term or trunc:
            break
    print('actions:', actions)          # expect all 1s
```

## `sandbox/L3_toyatc.py`

```python
"""ToyATC -- a miniature of the v4 ATC environment.

One ownship, one goal, one head-on intruder. Same conventions as Environments/v4:
headings are degrees clockwise from north, positions are (east, north).
"""

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap180(deg):
    """Wrap an angle in degrees to (-180, 180]."""
    return (deg + 180) % 360 - 180


def velocity(speed, hdg_deg):
    """Heading (deg clockwise from north) -> (east, north) velocity."""
    h = math.radians(hdg_deg)
    return np.array([speed * math.sin(h), speed * math.cos(h)])


class ToyATCEnv(gym.Env):
    metadata = {'render_modes': []}

    AREA, SPEED, TURN_DEG    = 50.0, 1.0, 15.0
    SEP, GOAL_TOL, MAX_STEPS = 5.0, 3.0, 300
    W_SEP, W_DRIFT, W_WORK   = 10.0, 1.0, 0.1
    ACT_COST = [0.5, 0.0, 0.5]          # left, hold, right

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        self.pos = self.goal = self.int_pos = np.zeros(2)
        self.hdg = self.int_hdg = 0.0
        self.steps = 0

    # -- Gym interface ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos     = np.array([-40.0, 0.0])
        self.hdg     = 90.0                                    # east
        self.goal    = np.array([40.0, float(rng.uniform(-15, 15))])
        self.int_pos = np.array([40.0, float(rng.uniform(-10, 10))])
        self.int_hdg = 270.0                                   # west, head-on
        self.steps   = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action == 0:
            self.hdg = (self.hdg - self.TURN_DEG) % 360
        elif action == 2:
            self.hdg = (self.hdg + self.TURN_DEG) % 360

        self.pos     = self.pos     + velocity(self.SPEED, self.hdg)
        self.int_pos = self.int_pos + velocity(self.SPEED, self.int_hdg)
        self.steps  += 1

        dist_goal = float(np.linalg.norm(self.goal - self.pos))
        sep_dist  = float(np.linalg.norm(self.int_pos - self.pos))
        brg_err   = self._bearing_to(self.goal)

        reward = (-self.W_SEP if sep_dist < self.SEP else 0.0) \
                 - self.W_DRIFT * (1 - math.cos(math.radians(brg_err))) / 2 \
                 - self.W_WORK * self.ACT_COST[action]

        terminated = dist_goal <= self.GOAL_TOL
        if terminated:
            reward += 10.0
        truncated = self.steps >= self.MAX_STEPS

        info = {'sep_dist': sep_dist, 'dist_goal': dist_goal, 'los': sep_dist < self.SEP}
        return self._obs(), reward, terminated, truncated, info

    # -- Internals -------------------------------------------------------------

    def _bearing_to(self, target):
        """Bearing error (deg) from my current heading to a target point."""
        d = target - self.pos
        return wrap180(math.degrees(math.atan2(d[0], d[1])) % 360 - self.hdg)

    def _obs(self):
        dist_goal = float(np.linalg.norm(self.goal - self.pos))
        rho       = float(np.linalg.norm(self.int_pos - self.pos))
        return np.array([
            min(dist_goal / (2 * self.AREA), 1.0),
            self._bearing_to(self.goal) / 180.0,
            min(rho / (2 * self.AREA), 1.0),
            self._bearing_to(self.int_pos) / 180.0,
            wrap180(self.int_hdg - self.hdg) / 180.0,
        ], dtype=np.float32)


# -- Baselines -----------------------------------------------------------------

def random_policy(env):
    return lambda obs: env.action_space.sample()


def greedy_policy(obs):
    """Turn toward the goal, ignore the intruder."""
    brg = obs[1]
    return 0 if brg < -0.02 else (2 if brg > 0.02 else 1)


def rollout(env, policy, n=20, seed0=0):
    """Run n episodes, return (goal_rate, mean_los_steps, mean_return)."""
    goals, los, rets = 0, [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        total, n_los, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            n_los += info['los']
            done   = term or trunc
            goals += term
        rets.append(total)
        los.append(n_los)
    return goals / n, float(np.mean(los)), float(np.mean(rets))


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    check_env(ToyATCEnv())
    print('check_env passed')

    env = ToyATCEnv()
    print('random: goal_rate=%.2f  los_steps=%.1f  return=%.1f' % rollout(env, random_policy(env)))
    print('greedy: goal_rate=%.2f  los_steps=%.1f  return=%.1f' % rollout(env, greedy_policy))
```

Running it gives roughly:

```
check_env passed
random: goal_rate=0.00  los_steps=0.3  return=-190.5
greedy: goal_rate=1.00  los_steps=1.6  return=-9.2
```

Read those two lines carefully — they define the problem. Random wanders and never
arrives. Greedy always arrives but flies through the intruder 1.6 steps per episode.
**PPO's job is to keep the goal rate at 1.00 while pushing LoS steps toward 0.** If your
trained agent doesn't beat greedy on LoS steps, something is wrong.

## `sandbox/L4_train.py`

```python
"""Train PPO on ToyATCEnv with VecNormalize. Saves model + normalisation stats."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from L3_toyatc import ToyATCEnv

TOTAL_TIMESTEPS = 300_000
N_ENVS          = 8


class SepCallback(BaseCallback):
    """Log mean separation distance and LoS rate."""

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'sep_dist' in info:
                self.logger.record_mean('toy/sep_dist', info['sep_dist'])
                self.logger.record_mean('toy/los_rate', float(info['los']))
        return True                       # mandatory


def main():
    venv = VecMonitor(make_vec_env(ToyATCEnv, n_envs=N_ENVS,
                                   vec_env_cls=DummyVecEnv, seed=0))
    env  = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, gamma=0.99)

    model = PPO('MlpPolicy', env, n_steps=256, batch_size=512,
                seed=0, verbose=1, tensorboard_log='./tb')
    model.learn(TOTAL_TIMESTEPS, callback=SepCallback())

    model.save('toyatc_ppo')
    env.save('toyatc_vecnorm.pkl')        # BOTH, always
    print('saved toyatc_ppo.zip + toyatc_vecnorm.pkl')


if __name__ == '__main__':
    main()
```

Watch `rollout/ep_rew_mean` climb toward (and past) the greedy baseline's `−9.2`.
`tensorboard --logdir tb` for the curves, including your `toy/los_rate`.

## `sandbox/L5_eval.py`

```python
"""Evaluate the trained policy, and demonstrate the VecNormalize trap."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from L3_toyatc import ToyATCEnv, greedy_policy, rollout

model = PPO.load('toyatc_ppo')

# -- The trap: raw observations into a policy trained on normalised ones -------
raw = VecMonitor(make_vec_env(ToyATCEnv, n_envs=1))
print('WITHOUT stats: %.2f +/- %.2f' % evaluate_policy(model, raw, n_eval_episodes=20))

# -- Correct: load the saved statistics ----------------------------------------
norm = VecNormalize.load('toyatc_vecnorm.pkl',
                         VecMonitor(make_vec_env(ToyATCEnv, n_envs=1)))
norm.training    = False        # freeze the running stats
norm.norm_reward = False        # report raw reward
print('WITH stats:    %.2f +/- %.2f' % evaluate_policy(model, norm, n_eval_episodes=20))

# -- Domain metrics on held-out seeds ------------------------------------------
env = ToyATCEnv()

def ppo_policy(obs):
    action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
    return int(action)

print('\nheld-out seeds 1000-1049')
print('greedy: goal_rate=%.2f  los_steps=%.1f  return=%.1f'
      % rollout(env, greedy_policy, n=50, seed0=1000))
print('ppo:    goal_rate=%.2f  los_steps=%.1f  return=%.1f'
      % rollout(env, ppo_policy, n=50, seed0=1000))
```

Two things to take away:

- The gap between `WITHOUT stats` and `WITH stats` is the whole lesson of Level 5. It is
  large, and it is silent — no exception, no warning, just a bad policy.
- Seeds 1000–1049 are disjoint from training, which is what makes it an honest evaluation.
  Compare against greedy on **domain metrics** (goal rate, LoS steps), not just return.

## Verification note

`L2` and `L3` were run and pass `check_env`; the baseline numbers above are real output.
`L4` and `L5` are provided as-is — run them yourself, since training is the part you
should watch happen.
