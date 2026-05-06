"""
Transfer evaluation script — tests trained policies on ground truth environment.
Parallelized with Ray for speed.

For each saved policy (trained on integrator X at dt Y), runs N evaluation
episodes on the ground truth environment (RK4 at smallest dt) and records
the transfer reward.

Plots generated (per environment):
  - One heatmap: all integrators x all timesteps (transfer reward)
  - One plot per integrator: transfer reward across all timesteps

Results saved to:
  - results_acrobot/plots/transfer_*
  - results_cartpole/plots/transfer_*

Usage:
    python transfer_eval.py
"""

import os
import numpy as np
import ray

import warnings
warnings.filterwarnings("ignore")

from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet
from cartpole import CartPoleEnv


# ─────────────────────────────────────────────────────────────────────────────
# Config
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
GT_DT         = 1e-2
EVAL_EPISODES = 20
SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Ray remote evaluation task — one task per (env, integrator, dt) config
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
def evaluate_one(env_name, integ_name, dt, weight_path):
    from acrobot import AcrobotEnv, rk4
    from cartpole import CartPoleEnv
    import numpy as np

    GT_DT         = 1e-2
    EVAL_EPISODES = 20
    SEED          = 42

    # Load weights
    data = np.load(weight_path)
    W = [data["W0"], data["W1"], data["W2"]]
    b = [data["b0"], data["b1"], data["b2"]]

    def act(s):
        h = np.atleast_2d(s).astype(np.float64)
        for i, (w, bi) in enumerate(zip(W, b)):
            h = h @ w + bi
            if i < len(W) - 1:
                h = np.maximum(0, h)
        return int(np.argmax(h))

    # Ground truth environment
    if env_name == "acrobot":
        env       = AcrobotEnv(integrator=rk4, dt=GT_DT)
        max_steps = int(100 / GT_DT)
    else:
        env       = CartPoleEnv(integrator=rk4, dt=GT_DT)
        max_steps = int(500 / GT_DT)

    rewards = []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        total  = 0.0
        for _ in range(max_steps):
            action = act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            reward *= GT_DT
            total  += reward
            if terminated or truncated:
                break
        rewards.append(total)

    env.close()
    return {
        "env":         env_name,
        "integrator":  integ_name,
        "dt":          dt,
        "mean_reward": float(np.mean(rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results_list, env_name):
    import matplotlib.pyplot as plt

    integrator_names = list(INTEGRATORS.keys())
    plots_dir        = f"results_{env_name}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Build lookup dict: results[integ_name][dt] = mean_reward
    results = {name: {} for name in integrator_names}
    for r in results_list:
        if r["env"] == env_name:
            results[r["integrator"]][r["dt"]] = r["mean_reward"]

    # ── 1. Heatmap: all integrators x all timesteps ───────────────────────────
    heat_data = np.full((len(integrator_names), len(TIMESTEPS)), np.nan)
    for i, name in enumerate(integrator_names):
        for j, dt in enumerate(TIMESTEPS):
            heat_data[i, j] = results[name].get(dt, np.nan)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heat_data, aspect="auto",
                   cmap="RdYlGn")
    ax.set_xticks(range(len(TIMESTEPS)))
    ax.set_xticklabels([f"{dt}" for dt in TIMESTEPS])
    ax.set_yticks(range(len(integrator_names)))
    ax.set_yticklabels(integrator_names)
    ax.set_xlabel("Training dt")
    ax.set_title(f"{env_name.capitalize()} — transfer reward on ground truth "
                 f"(RK4 dt={GT_DT})")
    plt.colorbar(im, ax=ax)
    for i in range(len(integrator_names)):
        for j in range(len(TIMESTEPS)):
            val = heat_data[i, j]
            txt = "-" if np.isnan(val) else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fname = f"{plots_dir}/transfer_heatmap.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved: {fname}")

    # ── 2. One plot per integrator — transfer reward across all timesteps ─────
    for integ_name in integrator_names:
        rewards = [results[integ_name].get(dt, np.nan) for dt in TIMESTEPS]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(TIMESTEPS)), rewards,
                marker="o", linewidth=2, markersize=7, color="#1D9E75")
        ax.set_xticks(range(len(TIMESTEPS)))
        ax.set_xticklabels([f"{dt}" for dt in TIMESTEPS])
        ax.set_xlabel("Training dt")
        ax.set_ylabel("Transfer reward on ground truth")
        ax.set_title(f"{env_name.capitalize()} — {integ_name} — transfer across timesteps\n"
                     f"(evaluated on RK4 dt={GT_DT})")
        ax.grid(alpha=0.3)
        for x, y in zip(range(len(TIMESTEPS)), rewards):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}", (x, y),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9)
        plt.tight_layout()
        fname = f"{plots_dir}/transfer_{integ_name}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Plot saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ray.init(ignore_reinit_error=True)

    # Build all tasks for both environments
    pending = {}
    for env_name in ["acrobot"]:
        weights_dir = f"weights_{env_name}"
        for integ_name in INTEGRATORS:
            for dt in TIMESTEPS:
                weight_path = f"{weights_dir}/{integ_name}_dt{dt}.npz"
                if not os.path.exists(weight_path):
                    print(f"  Skipping {env_name}/{integ_name} dt={dt} — weights not found")
                    continue
                ref = evaluate_one.remote(env_name, integ_name, dt, weight_path)
                pending[ref] = (env_name, integ_name, dt)

    print(f"Dispatched {len(pending)} evaluation tasks in parallel via Ray.\n")

    # Collect results as they complete
    all_results = []
    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        done_ref = done_refs[0]
        env_name, integ_name, dt = pending.pop(done_ref)
        try:
            result = ray.get(done_ref)
            print(f"  Done: {env_name}  {integ_name}  dt={dt}  "
                  f"transfer_reward={result['mean_reward']:.2f}")
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {env_name}  {integ_name}  dt={dt}  ({e})")

    # Plot results per environment
    try:
        import matplotlib.pyplot as plt
        for env_name in ["acrobot", "cartpole"]:
            print(f"\nPlotting {env_name}...")
            plot_results(all_results, env_name)
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")

    print("\nDone! Transfer evaluation complete.")


if __name__ == "__main__":
    main()