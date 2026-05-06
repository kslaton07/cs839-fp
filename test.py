from cartpole import CartPoleEnv
from integrators import rk4
import numpy as np

dt = 0.01
env = CartPoleEnv(integrator=rk4, dt=dt)

for ep in range(10):
    obs, _ = env.reset(seed=ep)
    total_time = 0.0

    for step in range(int(50 / dt)):
        x, x_dot, theta, theta_dot = obs

        # Try this sign first
        action = 0 if theta + 0.5 * theta_dot > 0 else 1

        obs, reward, terminated, truncated, _ = env.step(action)
        total_time += dt

        if terminated or truncated:
            break

    print(ep, total_time)