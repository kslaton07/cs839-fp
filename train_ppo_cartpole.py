"""
PPO training script for CartPoleEnv — compares integrators and timesteps.
Uses Stable Baselines 3 PPO.

Mirrors the structure of train_PPO_acrobot.py.
Results saved to:
  - results_PPO_cartpole/metrics.json
  - results_PPO_cartpole/summary.csv
  - results_PPO_cartpole/plots/
  - weights_PPO_cartpole/   (.npz files, same format as DQN for visualize.py)

Usage:
    python train_PPO_cartpole.py
"""

import time
import json
import csv
import os
import math
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

TIMESTEPS       = [1e-2, 5e-2, 1e-1, 0.5, 1, 1.5, 2]
NUM_EPISODES    = 500
SOLVE_THRESHOLD = 300   # 60% of max 500 simulated-time reward
SMOOTH_WINDOW   = 20
SEED            = 42

THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helper — same as Acrobot PPO script
# ─────────────────────────────────────────────────────────────────────────────
def extract_and_save_weights(model, path):
    params = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()}
    try:
        W0 = params["mlp_extractor.policy_net.0.weight"].T
        b0 = params["mlp_extractor.policy_net.0.bias"]
        W1 = params["mlp_extractor.policy_net.2.weight"].T
        b1 = params["mlp_extractor.policy_net.2.bias"]
        W2 = params["action_net.weight"].T
        b2 = params["action_net.bias"]
        np.savez(path, W0=W0, W1=W1, W2=W2, b0=b0, b1=b1, b2=b2)
    except KeyError:
        np.savez(path, **{k.replace(".", "_"): v for k, v in params.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for a single configuration
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def train_one(integrator_name, dt, seed=SEED):
    from stable_baselines3 import PPO
    from acrobot import rk4, rk2, feuler, seuler, ieuler, vverlet
    from cartpole import CartPoleEnv
    import math
    import numpy as np
    import time

    THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360

    _integrators = {
        "rk4": rk4, "rk2": rk2, "feuler": feuler,
        "seuler": seuler, "ieuler": ieuler, "vverlet": vverlet,
    }
    integrator_fn = _integrators[integrator_name]

    env = CartPoleEnv(integrator=integrator_fn, dt=dt)

    # Wrap step: shaped reward + normalisation
    original_step = env.step
    def wrapped_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        theta  = obs[2]  # standard ordering [x, x_dot, theta, theta_dot]
        reward = 1.0 - abs(theta) / THETA_THRESHOLD_RADIANS
        if terminated:
            reward += -10.0
        reward *= dt
        return obs, reward, terminated, truncated, info
    env.step = wrapped_step

    MAX_STEPS = int(500 / dt)

    ppo_model = PPO(
        "MlpPolicy", env,
        n_steps=max(MAX_STEPS, 64),
        batch_size=min(64, max(MAX_STEPS, 64)),
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    episode_rewards   = []
    smoothed_rewards  = []
    episodes_to_solve = None
    nan_events        = 0
    wall_start        = time.perf_counter()

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action, _ = ppo_model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            if not np.all(np.isfinite(next_obs)):
                nan_events += 1
                next_obs   = obs.copy()
                terminated = True

            obs           = next_obs
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

        ppo_model.learn(total_timesteps=MAX_STEPS, reset_num_timesteps=False)

    wall_time = time.perf_counter() - wall_start

    os.makedirs("weights_PPO_cartpole", exist_ok=True)
    weight_path = f"weights_PPO_cartpole/{integrator_name}_dt{dt}.npz"
    extract_and_save_weights(ppo_model, weight_path)

    return {
        "integrator":        integrator_name,
        "dt":                float(dt),
        "episode_rewards":   episode_rewards,
        "smoothed_rewards":  smoothed_rewards,
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
    os.makedirs("results_PPO_cartpole/plots", exist_ok=True)
    os.makedirs("weights_PPO_cartpole", exist_ok=True)
    all_results = []

    configs = [(name, dt) for name in INTEGRATORS for dt in TIMESTEPS]
    print(f"Running {len(configs)} configurations x {NUM_EPISODES} episodes (PPO, parallel via Ray).\n")

    pending_map = {train_one.remote(name, dt): (name, dt) for name, dt in configs}

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

    with open("results_PPO_cartpole/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    summary_fields = ["integrator", "dt", "episodes_to_solve",
                      "final_smooth", "wall_time_s", "nan_events"]
    with open("results_PPO_cartpole/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in summary_fields})

    print("\nResults saved to results_PPO_cartpole/")

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
                        label=f"dt={r['dt']}  (solved={r['episodes_to_solve']}, nan={r['nan_events']})")
            ax.axhline(SOLVE_THRESHOLD, color="black", linestyle="--",
                       linewidth=0.8, label=f"solve threshold ({SOLVE_THRESHOLD})")
            ax.set_title(f"PPO on CartPole — integrator: {integ_name}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results_PPO_cartpole/plots/{integ_name}.png"
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
            ax.set_title(f"PPO integrator comparison at dt={default_dt}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(f"Smoothed reward (window={SMOOTH_WINDOW})")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fname = f"results_PPO_cartpole/plots/comparison_dt_{default_dt}.png"
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
        ax.set_title("PPO — Episodes to solve (lower=better, x=never solved)")
        plt.colorbar(im, ax=ax)
        for i in range(len(integrator_names)):
            for j in range(len(TIMESTEPS)):
                val = heat_data[i, j]
                txt = "-" if np.isnan(val) else (str(int(val)) if val < NUM_EPISODES else "x")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig("results_PPO_cartpole/plots/heatmap_solve.png", dpi=150)
        plt.close()
        print("  Plot saved: results_PPO_cartpole/plots/heatmap_solve.png")

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
        ax.set_title("PPO — NaN / instability events (lower=better)")
        plt.colorbar(im, ax=ax)
        for i in range(len(integrator_names)):
            for j in range(len(TIMESTEPS)):
                val = nan_data[i, j]
                txt = "-" if np.isnan(val) else str(int(val))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig("results_PPO_cartpole/plots/heatmap_nan.png", dpi=150)
        plt.close()
        print("  Plot saved: results_PPO_cartpole/plots/heatmap_nan.png")

        fig, ax = plt.subplots(figsize=(11, 4))
        n_dt = len(TIMESTEPS)
        bar_width = 0.8 / n_dt
        for j, dt in enumerate(TIMESTEPS):
            times = []
            for name in integrator_names:
                match = [r for r in all_results if r["integrator"] == name and r["dt"] == dt]
                times.append(match[0]["wall_time_s"] if match else 0)
            x = np.arange(len(integrator_names))
            ax.bar(x + j * bar_width, times, width=bar_width, label=f"dt={dt}")
        ax.set_xticks(np.arange(len(integrator_names)) + bar_width * (n_dt - 1) / 2)
        ax.set_xticklabels(integrator_names)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("PPO — Training time per integrator and dt")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results_PPO_cartpole/plots/wallclock_time.png", dpi=150)
        plt.close()
        print("  Plot saved: results_PPO_cartpole/plots/wallclock_time.png")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()