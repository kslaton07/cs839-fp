"""
Visualization script — generates GIFs for selected ACROBOT runs only.

Cases:
    DQN feuler dt=0.05
    DQN rk4    dt=1
    DQN feuler dt=0.01
    PPO feuler dt=0.05
    PPO rk4    dt=1
    PPO rk2    dt=0.1

Output:
    GIFS/<algo>_<integrator>_dt<dt>.gif

Run from the project root:
    python visualize.py

Optional:
    python visualize.py --fps 30 --seed 0
    python visualize.py --max-frames 600
"""

import argparse
import glob
import sys
from pathlib import Path

import imageio
import numpy as np


ROOT = Path(__file__).resolve().parent

# Make imports work whether acrobot.py lives at root, DQN/, or PPO/.
for p in [ROOT, ROOT / "DQN", ROOT / "PPO"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet


INTEGRATORS = {
    "rk4": rk4,
    "rk2": rk2,
    "feuler": feuler,
    "seuler": seuler,
    "ieuler": ieuler,
    "vverlet": vverlet,
}

# The six requested ACROBOT-only visualizations.
CASES = [
    {
        "algo": "DQN",
        "integrator": "feuler",
        "dt": 0.05,
        "weights_dir": ROOT / "DQN" / "weights_acrobot",
    },
    {
        "algo": "DQN",
        "integrator": "rk4",
        "dt": 1.0,
        "weights_dir": ROOT / "DQN" / "weights_acrobot",
    },
    {
        "algo": "DQN",
        "integrator": "feuler",
        "dt": 0.01,
        "weights_dir": ROOT / "DQN" / "weights_acrobot",
    },
    {
        "algo": "PPO",
        "integrator": "feuler",
        "dt": 0.05,
        "weights_dir": ROOT / "PPO" / "weights_PPO_acrobot",
    },
    {
        "algo": "PPO",
        "integrator": "rk4",
        "dt": 1.0,
        "weights_dir": ROOT / "PPO" / "weights_PPO_acrobot",
    },
    {
        "algo": "PPO",
        "integrator": "rk2",
        "dt": 0.1,
        "weights_dir": ROOT / "PPO" / "weights_PPO_acrobot",
    },
]


def dt_tag(dt: float) -> str:
    """Canonical tag used for output GIF names."""
    return f"{dt:g}".replace(".", "p")


def candidate_weight_paths(weights_dir: Path, integrator: str, dt: float):
    """
    Try several common filename formats:
        feuler_dt5e-02.npz
        feuler_dt0.05.npz
        feuler_dt0p05.npz
        feuler_0.05.npz
    plus a final broad glob fallback.
    """
    tags = [
        f"{dt:.0e}",
        f"{dt:g}",
        str(dt),
        f"{dt:g}".replace(".", "p"),
    ]

    names = []
    for tag in tags:
        names.extend(
            [
                f"{integrator}_dt{tag}.npz",
                f"{integrator}_dt_{tag}.npz",
                f"{integrator}_{tag}.npz",
            ]
        )

    paths = [weights_dir / name for name in names]

    patterns = [
        str(weights_dir / f"*{integrator}*dt*{dt:g}*.npz"),
        str(weights_dir / f"*{integrator}*{dt:g}*.npz"),
        str(weights_dir / f"*{integrator}*.npz"),
    ]

    for pattern in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))

    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def find_weight_file(weights_dir: Path, integrator: str, dt: float) -> Path:
    for p in candidate_weight_paths(weights_dir, integrator, dt):
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No weights found for {integrator} dt={dt:g} in {weights_dir}. "
        f"Check filename pattern or update candidate_weight_paths()."
    )


class MLPPolicy:
    """
    Generic NumPy MLP inference wrapper.

    Works for DQN Q-network weights and PPO actor/policy weights as long as the
    saved .npz contains layer arrays in one of the common naming schemes:
        W0,b0,W1,b1,W2,b2
        actor_W0,actor_b0,...
        pi_W0,pi_b0,...
        policy_W0,policy_b0,...

    The action is argmax(output), which is greedy Q for DQN and greedy logits for PPO.
    """

    PREFIXES = ["", "actor_", "pi_", "policy_", "policy_net_", "mlp_"]

    def __init__(self, W, b):
        self.W = W
        self.b = b

    @staticmethod
    def _load_by_prefix(data, prefix):
        W, b = [], []
        i = 0

        while f"{prefix}W{i}" in data and f"{prefix}b{i}" in data:
            W.append(data[f"{prefix}W{i}"])
            b.append(data[f"{prefix}b{i}"])
            i += 1

        return W, b

    @staticmethod
    def _load_fallback_sorted(data):
        keys = list(data.keys())
        w_keys = sorted([k for k in keys if k.lower().startswith("w")])
        b_keys = sorted([k for k in keys if k.lower().startswith("b")])

        if len(w_keys) >= 1 and len(w_keys) == len(b_keys):
            return [data[k] for k in w_keys], [data[k] for k in b_keys]

        return [], []

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=True)

        for prefix in cls.PREFIXES:
            W, b = cls._load_by_prefix(data, prefix)
            if W:
                return cls(W, b)

        W, b = cls._load_fallback_sorted(data)
        if W:
            return cls(W, b)

        raise ValueError(
            f"Could not identify MLP weights in {path}. "
            f"Found keys: {list(data.keys())}. "
            f"Expected W0/b0 style keys, optionally prefixed by actor_, pi_, or policy_."
        )

    def act(self, obs):
        h = np.atleast_2d(obs).astype(np.float64)

        for i, (w, bi) in enumerate(zip(self.W, self.b)):
            h = h @ w + bi

            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)

        return int(np.argmax(h, axis=1)[0])


def collect_frames(integrator_name, dt, weight_path, seed=0, max_frames=None):
    env = AcrobotEnv(
        integrator=INTEGRATORS[integrator_name],
        dt=dt,
        render_mode="rgb_array",
    )

    policy = MLPPolicy.load(weight_path)

    obs, _ = env.reset(seed=seed)
    frames = []

    # Acrobot horizon consistent with your original visualizer.
    max_steps = int(100 / dt)

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            final_frame = env.render()
            if final_frame is not None:
                frames.append(final_frame)
            break

        if max_frames is not None and len(frames) >= max_frames:
            break

    env.close()
    return frames


def save_gif(frames, path: Path, fps: int):
    if not frames:
        print(f"  No frames collected; skipping {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  Saved {path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap to keep large-dt/small-dt GIFs from becoming huge.",
    )
    args = parser.parse_args()

    gifs_dir = ROOT / "GIFS"
    gifs_dir.mkdir(exist_ok=True)

    print("Generating requested ACROBOT GIFs only...\n")

    for case in CASES:
        algo = case["algo"]
        integrator = case["integrator"]
        dt = case["dt"]
        weights_dir = case["weights_dir"]

        print(f"{algo} {integrator} dt={dt:g}")

        try:
            weight_path = find_weight_file(weights_dir, integrator, dt)
            print(f"  Weights: {weight_path}")

            frames = collect_frames(
                integrator_name=integrator,
                dt=dt,
                weight_path=weight_path,
                seed=args.seed,
                max_frames=args.max_frames,
            )

            out_path = gifs_dir / f"{algo.lower()}_{integrator}_dt{dt_tag(dt)}.gif"
            save_gif(frames, out_path, fps=args.fps)

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. GIFs are in: {gifs_dir}")


if __name__ == "__main__":
    main()