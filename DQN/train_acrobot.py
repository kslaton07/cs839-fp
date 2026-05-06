"""
DQN training script for AcrobotEnv — compares integrators and timesteps.

Metrics tracked per configuration:
  - Episode reward (smoothed training curve)
  - Episodes to solve (first episode where smoothed reward >= -80)
  - Wall-clock training time
  - Numerical stability (NaN / divergence events)

Usage:
    python ./DQN/train_acrobot.py    (run from project root)

All outputs are saved relative to this script's directory:
  - <DQN>/results_acrobot/metrics.json
  - <DQN>/results_acrobot/summary.csv
  - <DQN>/results_acrobot/plots/
  - <DQN>/weights_acrobot/

Changes from previous version:
  - DQN moved to external module `dqn.py` (Adam optimizer, Huber loss, global-norm
    gradient clipping, hidden=128). update() now takes a min_buffer argument.
  - Added MIN_BUFFER constant.
  - Reward scheme unchanged: reward *= dt keeps returns comparable across dt.
"""

import sys
import os
import time
import json
import csv
import random
import collections
import numpy as np
import ray

import warnings
warnings.filterwarnings("ignore")

# Make project-root modules (dqn.py, acrobot.py) importable when running
# this script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet
from dqn import DQN

# ─────────────────────────────────────────────────────────────────────────────
# Output paths — all relative to this script's directory
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = f"{SCRIPT_DIR}/results_acrobot"
PLOTS_DIR   = f"{RESULTS_DIR}/plots"
WEIGHTS_DIR = f"{SCRIPT_DIR}/weights_acrobot"


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

TIMESTEPS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]

NUM_EPISODES    = 500
SOLVE_THRESHOLD = -80
SMOOTH_WINDOW   = 20
SEED            = 42


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=200_000):
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
    from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet
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

    env = AcrobotEnv(integrator=integrator_fn, dt=dt)
    obs, _ = env.reset(seed=seed)

    obs_dim = obs.shape[0]
    n_actions = env.action_space.n

    # Each episode covers ~100 seconds of simulated time.
    MAX_STEPS = int(100 / dt)

    # Scale epsilon decay by number of steps so exploration lasts comparable sim-time.
    eps_decay = MAX_STEPS * 5

    # Warm up replay buffer for the same number of episodes/sim-time across dt.
    WARMUP_EPISODES = 5
    MIN_BUFFER = WARMUP_EPISODES * MAX_STEPS

    # Do DQN updates at a fixed simulation-time interval, not every raw step.
    UPDATE_PERIOD_SIM_TIME = 0.1
    update_every_steps = max(1, int(round(UPDATE_PERIOD_SIM_TIME / dt)))

    agent = DQN(obs_dim, n_actions, eps_decay=eps_decay, seed=seed)
    replay = ReplayBuffer()

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

            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Scale reward so return is comparable across dt.
            reward *= dt

            # Catch numerical instability from simulator/integrator.
            if not np.all(np.isfinite(next_obs)):
                nan_events += 1
                next_obs = obs.copy()
                terminated = True

            done = terminated or truncated

            replay.push(obs, action, reward, next_obs, done)

            # Train only every fixed amount of simulated time.
            if step_idx % update_every_steps == 0:
                agent.update(replay, MIN_BUFFER)

            obs = next_obs
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        window = episode_rewards[-SMOOTH_WINDOW:]
        smooth = float(np.mean(window))
        smoothed_rewards.append(smooth)

        if (
            episodes_to_solve is None
            and len(window) == SMOOTH_WINDOW
            and smooth >= SOLVE_THRESHOLD
        ):
            episodes_to_solve = ep + 1

    wall_time = time.perf_counter() - wall_start

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weight_path = f"{WEIGHTS_DIR}/{integrator_name}_dt{dt}.npz"
    agent.save(weight_path)

    return {
        "integrator": integrator_name,
        "dt": dt,
        "episode_rewards": episode_rewards,
        "smoothed_rewards": smoothed_rewards,
        "episodes_to_solve": episodes_to_solve,
        "wall_time_s": round(wall_time, 2),
        "nan_events": nan_events,
        "final_smooth": round(smoothed_rewards[-1], 2),
        "weight_path": weight_path,
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

            ax.set_title(f"DQN on Acrobot: integrator {integ_name}")
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