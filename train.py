"""
DQN training script for AcrobotEnv — compares integrators and timesteps.

Metrics tracked per configuration:
  - Episode reward (smoothed training curve)
  - Episodes to solve (first episode where smoothed reward >= -100)
  - Wall-clock training time
  - Numerical stability (NaN / divergence events)

Usage:
    python train.py

Results are saved to results/  as:
  - results/metrics.json       — raw per-episode data for all configs
  - results/summary.csv        — one row per config with aggregate stats
  - results/plots/             — training curve plots
"""

import time
import json
import csv
import os
import random
import math
import collections
import numpy as np
from numpy import cos, pi, sin
import ray

# ── Optional: suppress gymnasium warnings ────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Import acrobot module ─────────────────────────────────────────────────────
from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, wrap, bound


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATORS = {
    "rk4":    rk4,
    "rk2":    rk2,
    "feuler": feuler,
    "seuler": seuler,
    "ieuler": ieuler,
}

TIMESTEPS = [1e-2, 5e-2, 1e-1, 0.5, 1, 1.5, 2]

NUM_EPISODES    = 500   # training episodes per config
SOLVE_THRESHOLD = -100  # smoothed reward considered "solved"
SMOOTH_WINDOW   = 20    # episodes to average for solve detection
SEED            = 42


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=50_000):
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
# Minimal DQN (numpy-only, no deep learning framework required)
#
# Architecture: two fully-connected hidden layers (64 units, ReLU).
# Trained with SGD + experience replay + target network.
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

        # Weight initialisation (He)
        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)

        self.W = [he(obs_dim, hidden), he(hidden, hidden), he(hidden, n_actions)]
        self.b = [np.zeros(hidden), np.zeros(hidden), np.zeros(n_actions)]
        self.tW = [w.copy() for w in self.W]   # target network weights
        self.tb = [b.copy() for b in self.b]
        self.update_count = 0

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, x, W, b):
        h = x
        for i, (w, bi) in enumerate(zip(W, b)):
            h = h @ w + bi
            if i < len(W) - 1:
                h = np.maximum(0, h)   # ReLU
        return h

    def predict(self, s):
        return self._forward(np.atleast_2d(s), self.W, self.b)

    def predict_target(self, s):
        return self._forward(np.atleast_2d(s), self.tW, self.tb)

    # ── ε-greedy action selection ─────────────────────────────────────────────
    def select_action(self, s):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.predict(s)))

    # ── SGD update on one mini-batch ─────────────────────────────────────────
    def update(self, replay: ReplayBuffer):
        if len(replay) < self.batch_size:
            return None

        s, a, r, s2, done = replay.sample(self.batch_size)

        # Target Q-values
        q_next  = self.predict_target(s2).max(axis=1)
        targets = r + self.gamma * q_next * (1 - done)

        # Current Q-values and loss
        q_pred  = self.predict(s)
        errors  = q_pred.copy()
        errors[np.arange(len(a)), a] = targets
        loss    = float(np.mean((q_pred - errors) ** 2))

        # Backprop through 3-layer net
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

        # Periodically sync target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.tW = [w.copy() for w in self.W]
            self.tb = [b.copy() for b in self.b]

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# ieuler helpers — must be top-level for Ray/pickle serialisation
# ─────────────────────────────────────────────────────────────────────────────
def _numerical_jacobian(f, y, eps=1e-6):
    """Approximate the Jacobian of f at y using forward finite differences."""
    n  = len(y)
    J  = np.zeros((n, n))
    f0 = np.asarray(f(y))
    for j in range(n):
        yp     = y.copy()
        yp[j] += eps
        J[:, j] = (np.asarray(f(yp)) - f0) / eps
    return J


def _make_derivs4(derivs, torque):
    """Return a closure that strips/reattaches the torque dimension."""
    def derivs4(y4):
        return np.asarray(derivs(np.append(y4, torque)))[:4]
    return derivs4


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for a single configuration
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def train_one(integrator_name, dt, seed=SEED):
    # Re-import inside worker — avoids Ray serialising the function object,
    # which breaks ieuler due to its nested closures.
    from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler
    _integrators = {
        "rk4": rk4, "rk2": rk2, "feuler": feuler,
        "seuler": seuler, "ieuler": ieuler,
    }
    integrator_fn = _integrators[integrator_name]

    random.seed(seed)
    np.random.seed(seed)

    env    = AcrobotEnv(integrator=integrator_fn, dt=dt)
    obs, _ = env.reset(seed=seed)
    obs_dim, n_actions = obs.shape[0], env.action_space.n

    eps_decay     = 5*int(100 / dt)  # scale exploration to match steps per episode
    agent  = DQN(obs_dim, n_actions, eps_decay=eps_decay, seed=seed)
    replay = ReplayBuffer()

    episode_rewards   = []
    smoothed_rewards  = []
    episodes_to_solve = None
    nan_events        = 0
    wall_start        = time.perf_counter()

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0

        MAX_STEPS = int(100/dt) 
        for _ in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward *= dt  # normalise reward to simulated-time units

            # Detect numerical instability
            if not np.all(np.isfinite(next_obs)):
                nan_events += 1
                next_obs   = obs.copy()   # recover: stay in place
                terminated = True

            replay.push(obs, action, reward, next_obs, terminated or truncated)
            agent.update(replay)

            obs          = next_obs
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        # Smoothed reward over last SMOOTH_WINDOW episodes
        window = episode_rewards[-SMOOTH_WINDOW:]
        smooth = float(np.mean(window))
        smoothed_rewards.append(smooth)

        # Record first episode that clears the solve threshold
        if episodes_to_solve is None and len(window) == SMOOTH_WINDOW \
                and smooth >= SOLVE_THRESHOLD:
            episodes_to_solve = ep + 1

    wall_time = time.perf_counter() - wall_start

    return {
        "integrator":        integrator_name,
        "dt":                dt,
        "episode_rewards":   episode_rewards,
        "smoothed_rewards":  smoothed_rewards,
        "episodes_to_solve": episodes_to_solve,   # None if never solved
        "wall_time_s":       round(wall_time, 2),
        "nan_events":        nan_events,
        "final_smooth":      round(smoothed_rewards[-1], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: run all configurations and save results
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ray.init(ignore_reinit_error=True)
    os.makedirs("results/plots", exist_ok=True)
    all_results = []

    configs = [(name, dt)
               for name in INTEGRATORS
               for dt in TIMESTEPS]

    print(f"Running {len(configs)} configurations x {NUM_EPISODES} episodes each (parallel via Ray).\n")

    # Dispatch all configs in parallel — use the ref itself as dict key,
    # which relies on Ray's built-in __hash__/__eq__ on object refs.
    pending_map = {
        train_one.remote(name, dt): (name, dt)
        for name, dt in configs
    }

    # Collect results as they complete
    while pending_map:
        done_refs, _ = ray.wait(list(pending_map.keys()), num_returns=1)
        done_ref = done_refs[0]
        name, dt = pending_map.pop(done_ref)

        try:
            result = ray.get(done_ref)
            print(f"  Done: {name}  dt={dt:.0e}  "
                  f"solved_ep={result['episodes_to_solve']}  "
                  f"final_reward={result['final_smooth']}  "
                  f"time={result['wall_time_s']}s  "
                  f"nan={result['nan_events']}")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {name}  dt={dt:.0e}  ({e})")

    # ── Save raw metrics as JSON ──────────────────────────────────────────────
    with open("results/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_fields = ["integrator", "dt", "episodes_to_solve",
                      "final_smooth", "wall_time_s", "nan_events"]
    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in summary_fields})

    print("\nResults saved to results/metrics.json and results/summary.csv")

    # ── Plot training curves (requires matplotlib) ────────────────────────────
    try:
        import matplotlib.pyplot as plt

        integrator_names = list(INTEGRATORS.keys())

        # One plot per integrator — all dt values as separate lines
        by_integrator = {}
        for r in all_results:
            by_integrator.setdefault(r["integrator"], []).append(r)

        for integ_name, runs in by_integrator.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            for r in sorted(runs, key=lambda x: x["dt"]):
                ax.plot(r["smoothed_rewards"],
                        label=f"dt={r['dt']:.0e}  "
                              f"(solved={r['episodes_to_solve']}, "
                              f"nan={r['nan_events']})")
            ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label="solve threshold (-100)")
            ax.set_title(f"DQN on Acrobot — integrator: {integ_name}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results/plots/{integ_name}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        # Cross-integrator comparison — one plot per timestep
        for default_dt in TIMESTEPS:
            fig, ax = plt.subplots(figsize=(10, 5))
            for r in all_results:
                if r["dt"] == default_dt:
                    ax.plot(r["smoothed_rewards"], label=r["integrator"])
            ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label="solve threshold (-100)")
            ax.set_title(f"Integrator comparison at dt={default_dt:.0e}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results/plots/comparison_dt_{default_dt:.0e}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        # Heatmap: episodes_to_solve (integrator x dt)
        heat_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            heat_data[i, j] = r["episodes_to_solve"] if r["episodes_to_solve"] else NUM_EPISODES

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([f"{dt:.0e}" for dt in TIMESTEPS])
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
        plt.savefig("results/plots/heatmap_solve.png", dpi=150)
        plt.close()
        print("  Plot saved: results/plots/heatmap_solve.png")

        # Heatmap: NaN events (integrator x dt)
        nan_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            nan_data[i, j] = r["nan_events"]

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(nan_data, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([f"{dt:.0e}" for dt in TIMESTEPS])
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
        plt.savefig("results/plots/heatmap_nan.png", dpi=150)
        plt.close()
        print("  Plot saved: results/plots/heatmap_nan.png")

        # Bar chart: wall-clock time per integrator across dt values
        fig, ax = plt.subplots(figsize=(11, 4))
        n_dt = len(TIMESTEPS)
        bar_width = 0.8 / n_dt
        for j, dt in enumerate(TIMESTEPS):
            times = []
            names = []
            for name in integrator_names:
                match = [r for r in all_results
                         if r["integrator"] == name and r["dt"] == dt]
                times.append(match[0]["wall_time_s"] if match else 0)
                names.append(name)
            x = np.arange(len(integrator_names))
            ax.bar(x + j * bar_width, times, width=bar_width, label=f"dt={dt:.0e}")
        ax.set_xticks(np.arange(len(integrator_names)) + bar_width * (n_dt - 1) / 2)
        ax.set_xticklabels(integrator_names)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Training time per integrator and dt")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/plots/wallclock_time.png", dpi=150)
        plt.close()
        print("  Plot saved: results/plots/wallclock_time.png")

    except ImportError:
        print("matplotlib not installed — skipping plots. "
              "Run: pip install matplotlib")


if __name__ == "__main__":
    main()