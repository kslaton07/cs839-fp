"""
CartPole environment modified to accept pluggable integrators and variable dt.
Based on the classic CartPole implementation by Rich Sutton et al.
Modified to match the structure of AcrobotEnv with custom integrator support.

IMPORTANT — state ordering:
  To be compatible with the integrators (which assume positions first, velocities second,
  matching Acrobot's [theta1, theta2, dtheta1, dtheta2] layout), the internal state is:

    [x, theta, x_dot, theta_dot]

  This differs from the classic CartPole ordering [x, x_dot, theta, theta_dot].
  The observation returned to the agent is reordered back to the standard
  [x, x_dot, theta, theta_dot] so the agent sees a consistent interface.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from integrators import rk4, rk2, feuler, seuler, ieuler, vverlet


class CartPoleEnv(Env):
    """
    CartPole environment modified to accept pluggable integrators and variable dt,
    matching the structure of the modified AcrobotEnv.

    Internal state: (x, theta, x_dot, theta_dot)   <- positions first, velocities second
    Observation:    (x, x_dot, theta, theta_dot)    <- standard CartPole ordering

    Actions:
      0 = push cart left  (-force_mag)
      1 = push cart right (+force_mag)

    Reward: +1 for every step the pole stays upright.
    Episode ends if pole angle > +/-12 degrees or cart position > +/-2.4.
    Max episode length handled externally via MAX_STEPS in train script.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Physical constants
    gravity         = 9.8
    masscart        = 1.0
    masspole        = 0.1
    total_mass      = masspole + masscart
    length          = 0.5           # half the pole length
    polemass_length = masspole * length
    force_mag       = 10.0

    # Termination thresholds
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold             = 2.4

    def __init__(self, integrator, dt, render_mode: str | None = None):
        self.integrator  = integrator
        self.dt          = dt
        self.render_mode = render_mode

        self.screen  = None
        self.clock   = None
        self.isopen  = True
        self.state   = None   # internal: [x, theta, x_dot, theta_dot]
        self.steps_beyond_terminated = None

        # Observation space uses standard ordering: [x, x_dot, theta, theta_dot]
        high = np.array(
            [self.x_threshold * 2,
             np.finfo(np.float32).max,
             self.theta_threshold_radians * 2,
             np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

    # ── Dynamics ─────────────────────────────────────────────────────────────
    def _dsdt(self, s_augmented):
        """
        Equations of motion for the CartPole system.

        s_augmented = [x, theta, x_dot, theta_dot, force]
                       ^pos^     ^vel^

        Returns derivatives matching state ordering:
          (x_dot, theta_dot, x_ddot, theta_ddot, 0.0)
           ^d/dt of pos^     ^d/dt of vel^

        This matches the Acrobot convention:
          Acrobot:   [theta1, theta2, dtheta1, dtheta2] -> (dtheta1, dtheta2, ddtheta1, ddtheta2)
          CartPole:  [x,      theta,  x_dot,  theta_dot] -> (x_dot, theta_dot, x_ddot, theta_ddot)

        Reference: https://coneural.org/florian/papers/05_cart_pole.pdf
        """
        force     = s_augmented[-1]
        x         = s_augmented[0]
        theta     = s_augmented[1]
        x_dot     = s_augmented[2]
        theta_dot = s_augmented[3]

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Intermediate quantity
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        # Angular acceleration of pole
        theta_ddot = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        # Linear acceleration of cart
        x_ddot = temp - self.polemass_length * theta_ddot * costheta / self.total_mass

        # Return derivatives in state order: (d_pos1, d_pos2, d_vel1, d_vel2, 0)
        return x_dot, theta_dot, x_ddot, theta_ddot, 0.0

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.state is not None, "Call reset before using CartPoleEnv object."

        force = self.force_mag if action == 1 else -self.force_mag

        # Augment internal state with force — matches Acrobot pattern
        s_augmented = np.append(self.state, force)

        # Integrate one timestep — ns = [x, theta, x_dot, theta_dot]
        ns = self.integrator(self._dsdt, s_augmented, [0, self.dt])

        self.state = ns.astype(np.float32)

        x     = self.state[0]
        theta = self.state[1]

        terminated = bool(
            x     < -self.x_threshold
            or x  >  self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta >  self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return self._get_ob(), reward, terminated, False, {}

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_ob(self):
        """
        Return observation in standard CartPole ordering: [x, x_dot, theta, theta_dot]
        Internal state is [x, theta, x_dot, theta_dot] so we reorder here.
        """
        x, theta, x_dot, theta_dot = self.state
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)

        # Sample in standard ordering [x, x_dot, theta, theta_dot]
        # then reorder to internal [x, theta, x_dot, theta_dot]
        s = self.np_random.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        self.state = np.array([s[0], s[2], s[1], s[3]], dtype=np.float32)

        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()

        return self._get_ob(), {}

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        screen_width  = 600
        screen_height = 400

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale       = screen_width / world_width
        polewidth   = 10.0
        polelen     = scale * (2 * self.length)
        cartwidth   = 50.0
        cartheight  = 30.0

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        if self.state is None:
            return None

        # Internal state: [x, theta, x_dot, theta_dot]
        x     = self.state[0]
        theta = self.state[1]

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx      = x * scale + screen_width / 2.0
        carty      = 100
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in [(l, b), (l, t), (r, t), (r, b)]]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.filled_circle(surf, int(cartx), int(carty + axleoffset), int(polewidth / 2), (129, 132, 203))
        gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False