"""
DQN training script for CartPoleEnv — compares integrators and timesteps.
Mirrors the structure of train.py for AcrobotEnv.

Key differences from Acrobot:
  - Reward is shaped: 1.0 - |theta| / theta_threshold (closer to upright = better)
  - Termination penalty: -10 when pole falls, to signal failure strongly
  - Solve threshold is 300 (60% of max 500 simulated-time reward)
  - MAX_STEPS = int(500 / dt) scaled to simulated time
  - Reward normalised to simulated-time units: reward *= dt
  - eps_decay scaled without 5x multiplier for faster exploration decay
  - Larger replay buffer (500k) and delayed training start (5000 transitions)

Results saved to results_cartpole/ as:
  - results_cartpole/metrics.json
  - results_cartpole/summary.csv
  - results_cartpole/plots/
"""

import time
import json
import csv
import os
import random
import math
import collections
import numpy as np
import ray

import warnings
warnings.filterwarnings("ignore")

from acrobot import rk4, rk2, feuler, seuler, ieuler, vverlet
from cartpole import CartPoleEnv


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATORS = {
    "rk4":     rk4,
    "rk2":     rk2,
    "feuler":  feuler,
    "seuler":  seuler,
    "ieuler":  ieuler,
    "vverlet": vverlet,
}

TIMESTEPS = [1e-2, 5e-2, 1e-1, 0.5, 1, 1.5, 2]

NUM_EPISODES       = 500
SOLVE_THRESHOLD    = 300
SMOOTH_WINDOW      = 20
SEED               = 42
TERMINATION_PENALTY = -10.0   # large negative reward when pole falls
MIN_BUFFER         = 5000     # don't start training until buffer has this many transitions

THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization helper — numpy float32/int64 not natively serializable
# ─────────────────────────────────────────────────────────────────────────────
def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer — larger capacity than Acrobot
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# DQN (identical to Acrobot train.py)
# ─────────────────────────────────────────────────────────────────────────────
class DQN:
    """Lightweight DQN implemented in pure numpy."""

    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=500,
                 batch_size=64, target_update=50, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.lr         = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)

        self.W  = [he(obs_dim, hidden), he(hidden, hidden), he(hidden, n_actions)]
        self.b  = [np.zeros(hidden), np.zeros(hidden), np.zeros(n_actions)]
        self.tW = [w.copy() for w in self.W]
        self.tb = [b.copy() for b in self.b]
        self.update_count = 0

    def _forward(self, x, W, b):
        h = x
        for i, (w, bi) in enumerate(zip(W, b)):
            h = h @ w + bi
            if i < len(W) - 1:
                h = np.maximum(0, h)
        return h

    def predict(self, s):
        return self._forward(np.atleast_2d(s), self.W, self.b)

    def predict_target(self, s):
        return self._forward(np.atleast_2d(s), self.tW, self.tb)

    def select_action(self, s):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.predict(s)))

    def update(self, replay: ReplayBuffer, min_buffer: int):
        # Wait until buffer has enough diverse experience before training
        if len(replay) < min_buffer:
            return None

        s, a, r, s2, done = replay.sample(self.batch_size)

        q_next  = self.predict_target(s2).max(axis=1)
        targets = r + self.gamma * q_next * (1 - done)

        q_pred  = self.predict(s)
        errors  = q_pred.copy()
        errors[np.arange(len(a)), a] = targets
        loss    = float(np.mean((q_pred - errors) ** 2))

        dL = 2 * (q_pred - errors) / len(a)

        grads_W, grads_b = [], []
        h0 = s
        h1 = np.maximum(0, h0 @ self.W[0] + self.b[0])
        h2 = np.maximum(0, h1 @ self.W[1] + self.b[1])

        hiddens = [h0, h1, h2]
        delta = dL
        for i in reversed(range(len(self.W))):
            grads_W.insert(0, hiddens[i].T @ delta)
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.W[i].T) * (hiddens[i] > 0)

        for i in range(len(self.W)):
            grads_W[i] = np.clip(grads_W[i], -1.0, 1.0)
            grads_b[i] = np.clip(grads_b[i], -1.0, 1.0)
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.tW = [w.copy() for w in self.W]
            self.tb = [b.copy() for b in self.b]

        return loss

    def save(self, path):
        np.savez(path,
                 W0=self.W[0], W1=self.W[1], W2=self.W[2],
                 b0=self.b[0], b1=self.b[1], b2=self.b[2])

    @classmethod
    def load(cls, path, obs_dim, n_actions, hidden=64):
        data  = np.load(path)
        agent = cls(obs_dim, n_actions, hidden=hidden)
        agent.W = [data["W0"], data["W1"], data["W2"]]
        agent.b = [data["b0"], data["b1"], data["b2"]]
        return agent


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for a single configuration
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def train_one(integrator_name, dt, seed=SEED):
    from acrobot import rk4, rk2, feuler, seuler, ieuler, vverlet
    from cartpole import CartPoleEnv
    import math

    TERMINATION_PENALTY = -10.0
    MIN_BUFFER          = 5000

    _integrators = {
        "rk4": rk4, "rk2": rk2, "feuler": feuler,
        "seuler": seuler, "ieuler": ieuler, "vverlet": vverlet,
    }
    integrator_fn = _integrators[integrator_name]

    random.seed(seed)
    np.random.seed(seed)

    env    = CartPoleEnv(integrator=integrator_fn, dt=dt)
    obs, _ = env.reset(seed=seed)
    obs_dim, n_actions = obs.shape[0], env.action_space.n

    eps_decay = int(500 / dt)
    agent     = DQN(obs_dim, n_actions, eps_decay=eps_decay, seed=seed)
    replay    = ReplayBuffer(capacity=500_000)

    episode_rewards   = []
    smoothed_rewards  = []
    episodes_to_solve = None
    nan_events        = 0
    wall_start        = time.perf_counter()

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0

        MAX_STEPS = int(500 / dt)

        for _ in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Simple survival reward: +dt per step (equivalent to +1/step in
            # simulated-time units). No shaping needed — CartPole's dense reward
            # already gives a clear learning signal.
            reward = dt

            # Termination penalty scaled by dt so it stays consistent across
            # all timestep sizes.
            if terminated:
                reward += TERMINATION_PENALTY * dt

            if not np.all(np.isfinite(next_obs)):
                nan_events += 1
                next_obs   = obs.copy()
                terminated = True

            replay.push(obs, action, reward, next_obs, terminated or truncated)

            # Delayed training start — wait for MIN_BUFFER diverse experiences
            agent.update(replay, MIN_BUFFER)

            obs          = next_obs
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(float(total_reward))

        window = episode_rewards[-SMOOTH_WINDOW:]
        smooth = float(np.mean(window))
        smoothed_rewards.append(smooth)

        if episodes_to_solve is None and len(window) == SMOOTH_WINDOW \
                and smooth >= SOLVE_THRESHOLD:
            episodes_to_solve = ep + 1

    wall_time = time.perf_counter() - wall_start

    os.makedirs("weights_cartpole", exist_ok=True)
    weight_path = f"weights_cartpole/{integrator_name}_dt{dt}.npz"
    agent.save(weight_path)

    return {
        "integrator":        integrator_name,
        "dt":                float(dt),
        "episode_rewards":   [float(r) for r in episode_rewards],
        "smoothed_rewards":  [float(r) for r in smoothed_rewards],
        "episodes_to_solve": int(episodes_to_solve) if episodes_to_solve else None,
        "wall_time_s":       round(float(wall_time), 2),
        "nan_events":        int(nan_events),
        "final_smooth":      round(float(smoothed_rewards[-1]), 2),
        "weight_path":       weight_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ray.init(ignore_reinit_error=True)
    os.makedirs("results_cartpole/plots", exist_ok=True)
    all_results = []

    configs = [(name, dt)
               for name in INTEGRATORS
               for dt in TIMESTEPS]

    print(f"Running {len(configs)} configurations x {NUM_EPISODES} episodes each (parallel via Ray).\n")

    pending_map = {
        train_one.remote(name, dt): (name, dt)
        for name, dt in configs
    }

    while pending_map:
        done_refs, _ = ray.wait(list(pending_map.keys()), num_returns=1)
        done_ref = done_refs[0]
        name, dt = pending_map.pop(done_ref)

        try:
            result = ray.get(done_ref)
            print(f"  Done: {name}  dt={dt}  "
                  f"solved_ep={result['episodes_to_solve']}  "
                  f"final_reward={result['final_smooth']}  "
                  f"time={result['wall_time_s']}s  "
                  f"nan={result['nan_events']}")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {name}  dt={dt}  ({e})")

    # JSON serialization — all values already converted to native Python types
    # in train_one return dict, so no extra conversion needed here
    with open("results_cartpole/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    summary_fields = ["integrator", "dt", "episodes_to_solve",
                      "final_smooth", "wall_time_s", "nan_events"]
    with open("results_cartpole/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in summary_fields})

    print("\nResults saved to results_cartpole/metrics.json and results_cartpole/summary.csv")

    try:
        import matplotlib.pyplot as plt

        integrator_names = list(INTEGRATORS.keys())

        by_integrator = {}
        for r in all_results:
            by_integrator.setdefault(r["integrator"], []).append(r)

        for integ_name, runs in by_integrator.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            for r in sorted(runs, key=lambda x: x["dt"]):
                ax.plot(r["smoothed_rewards"],
                        label=f"dt={r['dt']}  "
                              f"(solved={r['episodes_to_solve']}, "
                              f"nan={r['nan_events']})")
            ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label=f"solve threshold ({SOLVE_THRESHOLD})")
            ax.set_title(f"DQN on CartPole — integrator: {integ_name}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results_cartpole/plots/{integ_name}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        for default_dt in TIMESTEPS:
            fig, ax = plt.subplots(figsize=(10, 5))
            for r in all_results:
                if r["dt"] == default_dt:
                    ax.plot(r["smoothed_rewards"], label=r["integrator"])
            ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label=f"solve threshold ({SOLVE_THRESHOLD})")
            ax.set_title(f"Integrator comparison at dt={default_dt}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results_cartpole/plots/comparison_dt_{default_dt}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        heat_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            heat_data[i, j] = r["episodes_to_solve"] if r["episodes_to_solve"] else NUM_EPISODES

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([str(dt) for dt in TIMESTEPS])
        ax.set_yticks(range(len(integrator_names)))
        ax.set_yticklabels(integrator_names)
        ax.set_xlabel("dt")
        ax.set_title("Episodes to solve (lower=better, x=never solved)")
        plt.colorbar(im, ax=ax)
        for i in range(len(integrator_names)):
            for j in range(len(TIMESTEPS)):
                val = heat_data[i, j]
                txt = "-" if np.isnan(val) else (str(int(val)) if val < NUM_EPISODES else "x")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig("results_cartpole/plots/heatmap_solve.png", dpi=150)
        plt.close()
        print("  Plot saved: results_cartpole/plots/heatmap_solve.png")

        nan_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            nan_data[i, j] = r["nan_events"]

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(nan_data, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([str(dt) for dt in TIMESTEPS])
        ax.set_yticks(range(len(integrator_names)))
        ax.set_yticklabels(integrator_names)
        ax.set_xlabel("dt")
        ax.set_title("NaN / instability events (lower=better)")
        plt.colorbar(im, ax=ax)
        for i in range(len(integrator_names)):
            for j in range(len(TIMESTEPS)):
                val = nan_data[i, j]
                txt = "-" if np.isnan(val) else str(int(val))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig("results_cartpole/plots/heatmap_nan.png", dpi=150)
        plt.close()
        print("  Plot saved: results_cartpole/plots/heatmap_nan.png")

        fig, ax = plt.subplots(figsize=(11, 4))
        n_dt = len(TIMESTEPS)
        bar_width = 0.8 / n_dt
        for j, dt in enumerate(TIMESTEPS):
            times = []
            for name in integrator_names:
                match = [r for r in all_results
                         if r["integrator"] == name and r["dt"] == dt]
                times.append(match[0]["wall_time_s"] if match else 0)
            x = np.arange(len(integrator_names))
            ax.bar(x + j * bar_width, times, width=bar_width, label=f"dt={dt}")
        ax.set_xticks(np.arange(len(integrator_names)) + bar_width * (n_dt - 1) / 2)
        ax.set_xticklabels(integrator_names)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Training time per integrator and dt")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results_cartpole/plots/wallclock_time.png", dpi=150)
        plt.close()
        print("  Plot saved: results_cartpole/plots/wallclock_time.png")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()