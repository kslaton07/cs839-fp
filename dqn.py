"""
DQN agent — pure numpy, env-agnostic.

Designed to drop into any training loop that provides (obs_dim, n_actions)
and a ReplayBuffer with .sample(batch_size) returning (s, a, r, s2, done) arrays.

Improvements vs the original train_cartpole.py / train_acrobot.py implementation:
  - Adam optimizer (vs plain SGD): adapts step size per-parameter, much more
    robust to the non-stationary TD targets that DQN produces.
  - Huber loss (vs MSE): bounded gradients on large errors, which helps a lot
    near terminal-state Q-target discontinuities.
  - Global-norm gradient clipping (vs per-element): preserves gradient direction
    while bounding magnitude.
  - Configurable hidden size; default 128 (vs 64).

Interface preserved from the original:
  agent = DQN(obs_dim, n_actions, ...)
  agent.select_action(obs) -> int
  agent.update(replay_buffer, min_buffer) -> float | None
  agent.save(path)
  DQN.load(path, obs_dim, n_actions)
"""

import math
import random
import numpy as np


class DQN:
    """Lightweight DQN in pure numpy with Adam, Huber loss, and global-norm clipping."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 500,
        batch_size: int = 64,
        target_update: int = 50,
        hidden: int = 128,
        huber_delta: float = 1.0,
        grad_clip_norm: float = 10.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)

        self.obs_dim    = obs_dim
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.lr         = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0
        self.hidden     = hidden

        self.huber_delta    = huber_delta
        self.grad_clip_norm = grad_clip_norm

        self.beta1   = adam_beta1
        self.beta2   = adam_beta2
        self.adam_eps = adam_eps
        self.t       = 0  # Adam timestep counter

        # ── Network parameters (3-layer MLP: obs → hidden → hidden → n_actions) ──
        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)

        self.W = [he(obs_dim, hidden), he(hidden, hidden), he(hidden, n_actions)]
        self.b = [np.zeros(hidden), np.zeros(hidden), np.zeros(n_actions)]

        # Target network (frozen copy)
        self.tW = [w.copy() for w in self.W]
        self.tb = [b.copy() for b in self.b]
        self.update_count = 0

        # Adam moment estimates
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, x, W, b):
        h = x
        for i, (w, bi) in enumerate(zip(W, b)):
            h = h @ w + bi
            if i < len(W) - 1:
                h = np.maximum(0, h)
        return h

    def predict(self, s):
        return self._forward(np.atleast_2d(s), self.W, self.b)

    def predict_target(self, s):
        return self._forward(np.atleast_2d(s), self.tW, self.tb)

    # ── Action selection ─────────────────────────────────────────────────────
    def select_action(self, s):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.predict(s)))

    # ── Adam step ────────────────────────────────────────────────────────────
    def _adam_apply(self, params, grads, m_state, v_state):
        """In-place Adam update. params, grads, m_state, v_state are lists of arrays."""
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i in range(len(params)):
            m_state[i] = self.beta1 * m_state[i] + (1 - self.beta1) * grads[i]
            v_state[i] = self.beta2 * v_state[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = m_state[i] / bc1
            v_hat = v_state[i] / bc2
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)

    # ── Global-norm gradient clipping ────────────────────────────────────────
    def _clip_grads_global(self, grads_W, grads_b):
        total_sq = sum((g * g).sum() for g in grads_W) + sum((g * g).sum() for g in grads_b)
        total_norm = math.sqrt(float(total_sq))
        clip_coef = min(1.0, self.grad_clip_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            grads_W = [g * clip_coef for g in grads_W]
            grads_b = [g * clip_coef for g in grads_b]
        return grads_W, grads_b

    # ── Training step ────────────────────────────────────────────────────────
    def update(self, replay, min_buffer: int):
        if len(replay) < min_buffer:
            return None

        s, a, r, s2, done = replay.sample(self.batch_size)
        bs = len(a)

        # ── Forward pass storing intermediate activations for backprop ──────
        z1 = s @ self.W[0] + self.b[0]
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W[1] + self.b[1]
        h2 = np.maximum(0, z2)
        q_pred = h2 @ self.W[2] + self.b[2]      # shape (bs, n_actions)

        # ── TD target (vanilla DQN — uses target net for bootstrap) ─────────
        q_next  = self.predict_target(s2).max(axis=1)
        targets = r + self.gamma * q_next * (1 - done)

        # ── Huber loss on the action-taken column only ──────────────────────
        # Build a "delta" tensor: dL/d(q_pred) = clip(q_pred[i,a]-target[i], -d, d)
        # for the chosen action column, 0 elsewhere. Per-sample average.
        q_pred_a = q_pred[np.arange(bs), a]
        td_error = q_pred_a - targets
        huber_grad_a = np.clip(td_error, -self.huber_delta, self.huber_delta)

        # Loss value (for logging) — exact Huber, not just the gradient form
        abs_err = np.abs(td_error)
        quad_mask = abs_err <= self.huber_delta
        loss_per = np.where(
            quad_mask,
            0.5 * td_error ** 2,
            self.huber_delta * (abs_err - 0.5 * self.huber_delta),
        )
        loss = float(loss_per.mean())

        # Upstream gradient on the q_pred output: zero everywhere except
        # the action-taken column.
        dq = np.zeros_like(q_pred)
        dq[np.arange(bs), a] = huber_grad_a / bs

        # ── Backprop through the MLP ────────────────────────────────────────
        # Layer 3 (output)
        gW2 = h2.T @ dq
        gb2 = dq.sum(axis=0)
        dh2 = dq @ self.W[2].T

        # Layer 2
        dz2 = dh2 * (z2 > 0)
        gW1 = h1.T @ dz2
        gb1 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W[1].T

        # Layer 1
        dz1 = dh1 * (z1 > 0)
        gW0 = s.T @ dz1
        gb0 = dz1.sum(axis=0)

        grads_W = [gW0, gW1, gW2]
        grads_b = [gb0, gb1, gb2]

        # ── Global-norm clip, then Adam ─────────────────────────────────────
        grads_W, grads_b = self._clip_grads_global(grads_W, grads_b)
        self.t += 1
        self._adam_apply(self.W, grads_W, self.m_W, self.v_W)
        self._adam_apply(self.b, grads_b, self.m_b, self.v_b)

        # ── Sync target network ─────────────────────────────────────────────
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.tW = [w.copy() for w in self.W]
            self.tb = [b.copy() for b in self.b]

        return loss

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        np.savez(
            path,
            W0=self.W[0], W1=self.W[1], W2=self.W[2],
            b0=self.b[0], b1=self.b[1], b2=self.b[2],
        )

    @classmethod
    def load(cls, path, obs_dim, n_actions, hidden: int = 128):
        data  = np.load(path)
        agent = cls(obs_dim, n_actions, hidden=hidden)
        agent.W = [data["W0"], data["W1"], data["W2"]]
        agent.b = [data["b0"], data["b1"], data["b2"]]
        # Refresh target net to match loaded weights
        agent.tW = [w.copy() for w in agent.W]
        agent.tb = [b.copy() for b in agent.b]
        return agent