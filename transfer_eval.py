"""
Transfer evaluation — tests trained policies on the ground truth environment.

For each saved policy (trained on integrator X at dt Y), runs EVAL_EPISODES
evaluation episodes on the ground truth env (RK4 at GT_DT) and records the
mean transfer reward.

Plots:
  - Heatmap: all integrators × all timesteps (transfer reward)
  - One line plot per integrator: transfer reward across all timesteps

Results saved to: {RESULTS_DIR}/plots/transfer_*

Usage:
    python transfer_eval.py          (run from project root)
"""

import os
import sys
import numpy as np
import ray
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from acrobot  import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet
from cartpole import CartPoleEnv
from integrators import rk4, rk2, feuler, seuler, ieuler, vverlet

# ─────────────────────────────────────────────────────────────────────────────
# ★ Configure these two variables to switch experiment
# ─────────────────────────────────────────────────────────────────────────────
ALGO     = "PPO"       # "DQN"  or  "PPO"
ENV_NAME = "acrobot"   # "acrobot"  or  "cartpole"

# Derived paths — matches the naming used by the training scripts
_sfx        = f"PPO_{ENV_NAME}" if ALGO == "PPO" else ENV_NAME
WEIGHTS_DIR = f"{ALGO}/weights_{_sfx}"    # e.g. DQN/weights_acrobot
RESULTS_DIR = f"{ALGO}/results_{_sfx}"    # e.g. DQN/results_acrobot


# ─────────────────────────────────────────────────────────────────────────────
# Sweep settings
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATORS = {
    "rk4":     rk4,
    "rk2":     rk2,
    "feuler":  feuler,
    "seuler":  seuler,
    "ieuler":  ieuler,
    "vverlet": vverlet,
}

TIMESTEPS     = [1e-2, 5e-2, 1e-1, 0.5, 1, 1.5, 2]
GT_DT         = 1e-2    # ground truth integrator timestep
EVAL_EPISODES = 20
SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Ray remote evaluation task
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def evaluate_one(algo, env_name, integ_name, dt, weight_path, gt_dt, eval_episodes, seed):
    from acrobot  import AcrobotEnv, rk4
    from cartpole import CartPoleEnv
    from integrators import rk4
    import numpy as np

    # ── Load policy weights ─────────────────────────────────────────────────
    data = np.load(weight_path)
    if algo == "PPO":
        # Actor network weights only (aW0/aW1/aW2, ab0/ab1/ab2)
        W = [data["aW0"], data["aW1"], data["aW2"]]
        b = [data["ab0"], data["ab1"], data["ab2"]]
    else:
        # DQN Q-network weights (W0/W1/W2, b0/b1/b2)
        W = [data["W0"], data["W1"], data["W2"]]
        b = [data["b0"], data["b1"], data["b2"]]

    def act(obs):
        """Forward pass + greedy action selection (same for DQN and PPO actor)."""
        h = np.atleast_2d(obs).astype(np.float64)
        for i, (w, bi) in enumerate(zip(W, b)):
            h = h @ w + bi
            if i < len(W) - 1:
                h = np.maximum(0, h)   # ReLU
        return int(np.argmax(h, axis=-1)[0])

    # ── Ground truth environment (RK4 at gt_dt) ────────────────────────────
    if env_name == "acrobot":
        env       = AcrobotEnv(integrator=rk4, dt=gt_dt)
        max_steps = int(100 / gt_dt)
    else:
        env       = CartPoleEnv(integrator=rk4, dt=gt_dt)
        max_steps = int(50 / gt_dt)

    rewards = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total  = 0.0
        for _ in range(max_steps):
            obs, reward, terminated, truncated, _ = env.step(act(obs))
            total += reward * gt_dt   # sim-time units
            if terminated or truncated:
                break
        rewards.append(total)

    env.close()
    return {
        "integrator":  integ_name,
        "dt":          float(dt),
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results_list, results_dir, env_name, algo):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    integrator_names = list(INTEGRATORS.keys())
    plots_dir        = f"{results_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # lookup: rewards[integ_name][dt] = mean_reward
    rewards = {n: {} for n in integrator_names}
    for r in results_list:
        rewards[r["integrator"]][r["dt"]] = r["mean_reward"]

    INTEGRATOR_COLORS = {
        "rk4":     "#5b3f8c", "rk2":    "#7d5ba6",
        "feuler":  "#9b72cf", "seuler": "#b388d8",
        "ieuler":  "#6d28d9", "vverlet":"#c084fc",
    }
    PURPLE_CMAP = LinearSegmentedColormap.from_list(
        "custom_lilac", ["#f7f4fb", "#d8c7f0", "#b388d8", "#7d5ba6", "#4b2e6f"])

    # ── Heatmap ───────────────────────────────────────────────────────────────
    heat = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
    for i, name in enumerate(integrator_names):
        for j, dt in enumerate(TIMESTEPS):
            heat[i, j] = rewards[name].get(dt, np.nan)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heat, aspect="auto", cmap=PURPLE_CMAP)
    ax.set_xticks(range(len(TIMESTEPS)))
    ax.set_xticklabels([str(dt) for dt in TIMESTEPS])
    ax.set_yticks(range(len(integrator_names)))
    ax.set_yticklabels(integrator_names)
    ax.set_xlabel("Training dt")
    ax.set_title(f"{algo} on {env_name.capitalize()} — transfer reward on ground truth "
                 f"(RK4 dt={GT_DT})")
    plt.colorbar(im, ax=ax)
    for i in range(len(integrator_names)):
        for j in range(len(TIMESTEPS)):
            val = heat[i, j]
            ax.text(j, i, "-" if np.isnan(val) else f"{val:.1f}",
                    ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fname = f"{plots_dir}/transfer_heatmap.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")

    # ── Per-integrator line plots ─────────────────────────────────────────────
    for integ_name in integrator_names:
        vals = [rewards[integ_name].get(dt, np.nan) for dt in TIMESTEPS]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(TIMESTEPS)), vals,
                marker="o", linewidth=2, markersize=7,
                color=INTEGRATOR_COLORS[integ_name])
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([str(dt) for dt in TIMESTEPS])
        ax.set_xlabel("Training dt")
        ax.set_ylabel("Transfer reward (sim-seconds)")
        ax.set_title(f"{algo} on {env_name.capitalize()} — {integ_name} — "
                     f"transfer across training timesteps\n"
                     f"(evaluated on RK4 dt={GT_DT})")
        ax.grid(alpha=0.3)
        for x, y in enumerate(vals):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
        plt.tight_layout()
        fname = f"{plots_dir}/transfer_{integ_name}.png"
        plt.savefig(fname, dpi=150); plt.close()
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ray.init(ignore_reinit_error=True)
    os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)

    print(f"Transfer eval: algo={ALGO}  env={ENV_NAME}")
    print(f"  Weights: {WEIGHTS_DIR}/")
    print(f"  Results: {RESULTS_DIR}/plots/\n")

    pending = {}
    for integ_name in INTEGRATORS:
        for dt in TIMESTEPS:
            weight_path = f"{WEIGHTS_DIR}/{integ_name}_dt{dt}.npz"
            if not os.path.exists(weight_path):
                print(f"  Skipping {integ_name} dt={dt} — weights not found at {weight_path}")
                continue
            ref = evaluate_one.remote(
                ALGO, ENV_NAME, integ_name, dt, weight_path,
                GT_DT, EVAL_EPISODES, SEED)
            pending[ref] = (integ_name, dt)

    print(f"Dispatched {len(pending)} evaluation tasks via Ray.\n")

    all_results = []
    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        done_ref     = done_refs[0]
        integ_name, dt = pending.pop(done_ref)
        try:
            result = ray.get(done_ref)
            print(f"  Done: {integ_name}  dt={dt}  "
                  f"transfer={result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {integ_name}  dt={dt}  ({e})")

    if all_results:
        try:
            plot_results(all_results, RESULTS_DIR, ENV_NAME, ALGO)
        except ImportError:
            print("matplotlib not installed — skipping plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()