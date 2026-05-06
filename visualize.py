"""
Visualization script — generates GIFs from saved DQN weights.

Loads trained weights from weights_acrobot/ or weights_cartpole/,
runs a greedy episode, and saves a GIF to gifs_acrobot/ or gifs_cartpole/.

Usage:
    # Generate all GIFs for Acrobot:
    python visualize.py --env acrobot

    # Generate all GIFs for CartPole:
    python visualize.py --env cartpole

    # Generate a single GIF:
    python visualize.py --env acrobot --integrator feuler --dt 0.5

Requirements:
    pip install imageio pillow
"""

import os
import argparse
import math
import numpy as np
import imageio

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

TIMESTEPS = [1e-2, 5e-2, 1e-1, 0.5, 1, 1.5, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal DQN forward pass for inference only (no training needed)
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def act(self, s):
        """Greedy action — always pick highest Q-value, no exploration."""
        h = np.atleast_2d(s).astype(np.float64)
        for i, (w, bi) in enumerate(zip(self.W, self.b)):
            h = h @ w + bi
            if i < len(self.W) - 1:
                h = np.maximum(0, h)
        return int(np.argmax(h))

    @staticmethod
    def load(path):
        data = np.load(path)
        W = [data["W0"], data["W1"], data["W2"]]
        b = [data["b0"], data["b1"], data["b2"]]
        return DQNAgent(W, b)


# ─────────────────────────────────────────────────────────────────────────────
# Run one greedy episode and collect RGB frames
# ─────────────────────────────────────────────────────────────────────────────
def collect_frames(env_name, integrator_name, dt, weight_path, max_steps=500):
    integrator_fn = INTEGRATORS[integrator_name]

    if env_name == "acrobot":
        env = AcrobotEnv(integrator=integrator_fn, dt=dt, render_mode="rgb_array")
        max_steps = int(100 / dt)
    else:
        env = CartPoleEnv(integrator=integrator_fn, dt=dt, render_mode="rgb_array")
        max_steps = int(500 / dt)

    agent = DQNAgent.load(weight_path)

    obs, _ = env.reset(seed=0)
    frames = []

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            # Capture final frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    env.close()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Save frames as GIF
# ─────────────────────────────────────────────────────────────────────────────
def save_gif(frames, path, fps=30):
    if not frames:
        print(f"  No frames collected, skipping {path}")
        return
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {path}  ({len(frames)} frames)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["acrobot", "cartpole"], required=True,
                        help="Which environment to visualize")
    parser.add_argument("--integrator", default=None,
                        help="Specific integrator (default: all)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Specific dt value (default: all)")
    parser.add_argument("--fps", type=int, default=30,
                        help="GIF frame rate (default: 30)")
    args = parser.parse_args()

    weights_dir = f"weights_{args.env}"
    gifs_dir    = f"gifs_{args.env}"
    os.makedirs(gifs_dir, exist_ok=True)

    # Build list of configs to visualize
    integrators = [args.integrator] if args.integrator else list(INTEGRATORS.keys())
    timesteps   = [args.dt]         if args.dt         else TIMESTEPS

    total = len(integrators) * len(timesteps)
    print(f"Generating {total} GIFs for {args.env}...\n")

    for integ_name in integrators:
        for dt in timesteps:
            weight_path = f"{weights_dir}/{integ_name}_dt{dt:.0e}.npz"
            gif_path    = f"{gifs_dir}/{integ_name}_dt{dt:.0e}.gif"

            if not os.path.exists(weight_path):
                print(f"  Skipping {integ_name} dt={dt:.0e} — weights not found at {weight_path}")
                continue

            print(f"  Rendering {integ_name}  dt={dt:.0e} ...")
            try:
                frames = collect_frames(args.env, integ_name, dt, weight_path)
                save_gif(frames, gif_path, fps=args.fps)
            except Exception as e:
                print(f"  ERROR: {integ_name} dt={dt:.0e} — {e}")

    print(f"\nDone! GIFs saved to {gifs_dir}/")


if __name__ == "__main__":
    main()