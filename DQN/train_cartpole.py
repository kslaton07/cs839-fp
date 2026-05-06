"""
DQN training script for CartPoleEnv — compares integrators and timesteps.

Key differences from previous version:
  - DQN moved to external module `dqn.py` (Adam optimizer, Huber loss,
    global-norm gradient clipping, hidden=128).
  - MAX_STEPS = int(50 / dt) — gives 50 sim-seconds (was 500). 50s is roughly
    5x the standard CartPole-v1 horizon, achievable by DQN; 500s was 50x and
    much too aggressive.
  - SOLVE_THRESHOLD = 45 — 90% of new max return.
  - CartPole uses a standard DQN-style training signal while debugging:
    +1 per survived decision step, gamma=0.99, and no extra terminal penalty.
  - Logged episode return is still measured in simulated seconds survived, so
    results remain interpretable across dt.

Results saved to results_cartpole/ as:
  - results_cartpole/metrics.json
  - results_cartpole/summary.csv
  - results_cartpole/plots/
"""

import sys
import time
import json
import csv
import os
import random
import collections
import numpy as np
import ray

import warnings
warnings.filterwarnings("ignore")

# Make project-root modules (dqn.py, cartpole.py) importable when running
# this script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartpole import CartPoleEnv
from integrators import rk4, rk2, feuler, seuler, ieuler, vverlet
from dqn import DQN

# ─────────────────────────────────────────────────────────────────────────────
# Output paths — all relative to this script's directory
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = f"{SCRIPT_DIR}/results_cartpole"
PLOTS_DIR   = f"{RESULTS_DIR}/plots"
WEIGHTS_DIR = f"{SCRIPT_DIR}/weights_cartpole"


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

NUM_EPISODES        = 500
SOLVE_THRESHOLD     = 45      # 90% of max return (50 sim-seconds)
SMOOTH_WINDOW       = 20
SEED                = 42


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
# Replay buffer
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
# Training loop for a single configuration
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def train_one(integrator_name, dt, seed=SEED):
    from cartpole import CartPoleEnv
    from integrators import rk4, rk2, feuler, seuler, ieuler, vverlet
    from dqn import DQN

    _integrators = {
        "rk4": rk4,
        "rk2": rk2,
        "feuler": feuler,
        "seuler": seuler,
        "ieuler": ieuler,
        "vverlet": vverlet,
    }

    integrator_fn = _integrators[integrator_name]

    random.seed(seed)
    np.random.seed(seed)

    env = CartPoleEnv(integrator=integrator_fn, dt=dt)
    obs, _ = env.reset(seed=seed)

    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    # Each episode covers ~50 seconds of simulated time.
    MAX_STEPS = int(50 / dt)

    # ~20 full episodes of exploration before going greedy (consistent across dt)
    eps_decay = MAX_STEPS * 20

    agent = DQN(
        obs_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        eps_decay=eps_decay,
        batch_size=128,
        target_update=500,    # slower syncs = more stable Q-targets
        seed=seed,
    )

    replay = ReplayBuffer(capacity=500_000)

    # Collect real buffer diversity before training starts
    MIN_BUFFER = 5000

    # Standard DQN-style CartPole debugging: one optimization attempt per step.
    update_every_steps = 1

    episode_rewards = []
    smoothed_rewards = []
    episodes_to_solve = None
    nan_events = 0

    wall_start = time.perf_counter()

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0

        for step_idx in range(MAX_STEPS):
            action = agent.select_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)

            # Training reward follows the standard CartPole signal.
            # Logged return below remains simulated seconds survived.
            train_reward = 1.0

            if not np.all(np.isfinite(next_obs)):
                nan_events += 1
                next_obs = obs.copy()
                terminated = True

            done = terminated or truncated

            replay.push(obs, action, train_reward, next_obs, done)

            if step_idx % update_every_steps == 0:
                agent.update(replay, MIN_BUFFER)

            obs = next_obs
            total_reward += dt

            if done:
                break

        episode_rewards.append(float(total_reward))

        window = episode_rewards[-SMOOTH_WINDOW:]
        smooth = float(np.mean(window))
        smoothed_rewards.append(smooth)

        if episodes_to_solve is None and len(window) == SMOOTH_WINDOW \
                and smooth >= SOLVE_THRESHOLD:
            episodes_to_solve = ep + 1

    wall_time = time.perf_counter() - wall_start

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weight_path = f"{WEIGHTS_DIR}/{integrator_name}_dt{dt}.npz"
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    all_results = []

    configs = [(name, dt)
               for name in INTEGRATORS
               for dt in TIMESTEPS]

    MAX_PARALLEL = 12

    print(
        f"Running {len(configs)} configurations x {NUM_EPISODES} episodes each "
        f"(Ray, max {MAX_PARALLEL} at a time).\n"
    )

    config_iter = iter(configs)
    pending_map = {}

    # Launch initial batch only
    for _ in range(MAX_PARALLEL):
        try:
            name, dt = next(config_iter)
        except StopIteration:
            break

        ref = train_one.remote(name, dt)
        pending_map[ref] = (name, dt)

    # As each job finishes, launch one more
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

        # Launch next config after one finishes
        try:
            name, dt = next(config_iter)
            ref = train_one.remote(name, dt)
            pending_map[ref] = (name, dt)
        except StopIteration:
            pass

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    summary_fields = ["integrator", "dt", "episodes_to_solve",
                      "final_smooth", "wall_time_s", "nan_events"]
    with open(f"{RESULTS_DIR}/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in summary_fields})

    print(f"\nResults saved to {RESULTS_DIR}/")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        integrator_names = list(INTEGRATORS.keys())

        # Fixed lilac-forward color for each integrator.
        INTEGRATOR_COLORS = {
            "rk4":     "#5b3f8c",
            "rk2":     "#7d5ba6",
            "feuler":  "#9b72cf",
            "seuler":  "#b388d8",
            "ieuler":  "#6d28d9",
            "vverlet": "#c084fc",
        }

        # Fixed lilac-forward color for each timestep.
        DT_COLORS = {
            0.01: "#ede7f6",
            0.05: "#d8c7f0",
            0.1:  "#c7a9e8",
            0.5:  "#b388d8",
            1:    "#9b6bcc",
            1.5:  "#7d5ba6",
            2:    "#5b3f8c",
        }

        PURPLE_CMAP = LinearSegmentedColormap.from_list(
            "custom_lilac",
            ["#f7f4fb", "#d8c7f0", "#b388d8", "#7d5ba6", "#4b2e6f"]
        )

        by_integrator = {}
        for r in all_results:
            by_integrator.setdefault(r["integrator"], []).append(r)

        # ------------------------------------------------------------
        # Plot 1: one plot per integrator, lines colored by timestep.
        # ------------------------------------------------------------
        for integ_name, runs in by_integrator.items():
            fig, ax = plt.subplots(figsize=(10, 5))

            for r in sorted(runs, key=lambda x: x["dt"]):
                ax.plot(
                    r["smoothed_rewards"],
                    color=DT_COLORS[r["dt"]],
                    linewidth=2.0,
                    label=f"dt={r['dt']}  "
                          f"(solved={r['episodes_to_solve']}, "
                          f"nan={r['nan_events']})"
                )

            ax.axhline(
                SOLVE_THRESHOLD,
                color="#2f2438",
                linestyle="--",
                linewidth=0.8,
                label=f"solve threshold ({SOLVE_THRESHOLD})"
            )

            ax.set_title(f"DQN on CartPole: integrator {integ_name}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            fname = f"{PLOTS_DIR}/{integ_name}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        # ------------------------------------------------------------
        # Plot 2: one plot per dt, lines colored by integrator.
        # ------------------------------------------------------------
        for default_dt in TIMESTEPS:
            fig, ax = plt.subplots(figsize=(10, 5))

            for r in all_results:
                if r["dt"] == default_dt:
                    ax.plot(
                        r["smoothed_rewards"],
                        color=INTEGRATOR_COLORS[r["integrator"]],
                        linewidth=2.0,
                        label=r["integrator"]
                    )

            ax.axhline(
                SOLVE_THRESHOLD,
                color="#2f2438",
                linestyle="--",
                linewidth=0.8,
                label=f"solve threshold ({SOLVE_THRESHOLD})"
            )

            ax.set_title(f"Integrator comparison at dt={default_dt}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            fname = f"{PLOTS_DIR}/comparison_dt_{default_dt}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Plot saved: {fname}")

        # ------------------------------------------------------------
        # Heatmap 1: episodes to solve.
        # ------------------------------------------------------------
        heat_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)

        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            heat_data[i, j] = (
                r["episodes_to_solve"]
                if r["episodes_to_solve"]
                else NUM_EPISODES
            )

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(heat_data, aspect="auto", cmap=PURPLE_CMAP)

        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([f"{dt}" for dt in TIMESTEPS])
        ax.set_yticks(range(len(integrator_names)))
        ax.set_yticklabels(integrator_names)
        ax.set_xlabel("dt")
        ax.set_title("Episodes to solve (lower=better, x=never solved)")

        plt.colorbar(im, ax=ax)

        for i in range(len(integrator_names)):
            for j in range(len(TIMESTEPS)):
                val = heat_data[i, j]
                txt = "-" if np.isnan(val) else (
                    str(int(val)) if val < NUM_EPISODES else "x"
                )
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

        plt.tight_layout()
        fname = f"{PLOTS_DIR}/heatmap_solve.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Plot saved: {fname}")

        # ------------------------------------------------------------
        # Heatmap 2: NaN / instability events.
        # ------------------------------------------------------------
        nan_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)

        for r in all_results:
            i = integrator_names.index(r["integrator"])
            j = TIMESTEPS.index(r["dt"])
            nan_data[i, j] = r["nan_events"]

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(nan_data, aspect="auto", cmap=PURPLE_CMAP)

        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([f"{dt}" for dt in TIMESTEPS])
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
        fname = f"{PLOTS_DIR}/heatmap_nan.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Plot saved: {fname}")

        # ------------------------------------------------------------
        # Bar plot: wall-clock time, bars colored by timestep.
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 4))
        n_dt = len(TIMESTEPS)
        bar_width = 0.8 / n_dt

        for j, dt in enumerate(TIMESTEPS):
            times = []

            for name in integrator_names:
                match = [
                    r for r in all_results
                    if r["integrator"] == name and r["dt"] == dt
                ]
                times.append(match[0]["wall_time_s"] if match else 0)

            x = np.arange(len(integrator_names))

            ax.bar(
                x + j * bar_width,
                times,
                width=bar_width,
                color=DT_COLORS[dt],
                label=f"dt={dt}"
            )

        ax.set_xticks(
            np.arange(len(integrator_names)) + bar_width * (n_dt - 1) / 2
        )
        ax.set_xticklabels(integrator_names)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Training time per integrator and dt")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fname = f"{PLOTS_DIR}/wallclock_time.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Plot saved: {fname}")

    except ImportError:
        print("matplotlib not installed, skipping plots.")


if __name__ == "__main__":
    main()