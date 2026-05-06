"""
PPO agent — pure numpy, env-agnostic (discrete action spaces).

On-policy actor-critic with:
  - Clipped surrogate objective (PPO-Clip)
  - Generalized Advantage Estimation (GAE)
  - Entropy bonus to encourage exploration
  - Separate actor and critic MLPs
  - Adam optimizer with global-norm gradient clipping

Usage per episode:
    rollout = RolloutBuffer()
    for each step:
        action, log_prob, value = agent.select_action(obs)
        rollout.push(obs, action, reward, value, log_prob, done)
    last_val = 0.0 if terminated else agent.get_value(last_obs)
    agent.update(rollout, last_value=last_val)
    rollout.clear()
"""

import math
import numpy as np


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Accumulates one episode of transitions for a PPO update."""

    def __init__(self):
        self._obs, self._act, self._rew = [], [], []
        self._val, self._lp, self._done = [], [], []

    def push(self, obs, action, reward, value, log_prob, done):
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._act.append(int(action))
        self._rew.append(float(reward))
        self._val.append(float(value))
        self._lp.append(float(log_prob))
        self._done.append(float(done))

    def get(self):
        return (np.array(self._obs,  dtype=np.float32),
                np.array(self._act,  dtype=np.int64),
                np.array(self._rew,  dtype=np.float32),
                np.array(self._val,  dtype=np.float32),
                np.array(self._lp,   dtype=np.float32),
                np.array(self._done, dtype=np.float32))

    def clear(self):
        self._obs.clear(); self._act.clear(); self._rew.clear()
        self._val.clear(); self._lp.clear(); self._done.clear()

    def __len__(self):
        return len(self._rew)


# ── PPO agent ─────────────────────────────────────────────────────────────────

class PPO:
    """
    Pure-numpy PPO (discrete actions).

    Architecture:
      Actor:  obs → hidden → hidden → n_actions  (logits → softmax)
      Critic: obs → hidden → hidden → 1          (value estimate)

    Both networks use ReLU activations and He initialisation.
    """

    def __init__(
        self,
        obs_dim:        int,
        n_actions:      int,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_eps:       float = 0.2,
        ent_coef:       float = 0.01,
        vf_coef:        float = 0.5,
        n_epochs:       int   = 4,
        mini_batch_size:int   = 64,
        hidden:         int   = 64,
        grad_clip_norm: float = 0.5,
        adam_beta1:     float = 0.9,
        adam_beta2:     float = 0.999,
        adam_eps:       float = 1e-8,
        seed:           int   = 0,
    ):
        rng = np.random.default_rng(seed)

        self.obs_dim         = obs_dim
        self.n_actions       = n_actions
        self.lr              = lr
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.ent_coef        = ent_coef
        self.vf_coef         = vf_coef
        self.n_epochs        = n_epochs
        self.mini_batch_size = mini_batch_size
        self.hidden          = hidden
        self.grad_clip_norm  = grad_clip_norm
        self.beta1           = adam_beta1
        self.beta2           = adam_beta2
        self.adam_eps        = adam_eps
        self.t               = 0   # Adam global step counter

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)

        # Actor weights/biases
        self.actor_W = [he(obs_dim, hidden), he(hidden, hidden), he(hidden, n_actions)]
        self.actor_b = [np.zeros(hidden),    np.zeros(hidden),   np.zeros(n_actions)]

        # Critic weights/biases
        self.critic_W = [he(obs_dim, hidden), he(hidden, hidden), he(hidden, 1)]
        self.critic_b = [np.zeros(hidden),    np.zeros(hidden),   np.zeros(1)]

        # Adam moments over all parameters in a fixed order:
        # [aW0, aW1, aW2, ab0, ab1, ab2, cW0, cW1, cW2, cb0, cb1, cb2]
        all_p = self.actor_W + self.actor_b + self.critic_W + self.critic_b
        self.m = [np.zeros_like(p) for p in all_p]
        self.v = [np.zeros_like(p) for p in all_p]

    # ── Forward passes ────────────────────────────────────────────────────────

    def _actor_fwd(self, x):
        """
        Forward pass through actor.
        Returns (pi, log_pi, zs, hs) where:
          hs[i] = input to layer i  (hs[0] = x)
          zs[i] = pre-activation of layer i
        """
        hs, zs = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.actor_W, self.actor_b)):
            z = h @ W + b
            zs.append(z)
            h = np.maximum(0, z) if i < len(self.actor_W) - 1 else z
            hs.append(h)
        logits = hs[-1]
        # Numerically stable log-softmax
        logits_s = logits - logits.max(axis=-1, keepdims=True)
        exp_l    = np.exp(logits_s)
        pi       = exp_l / exp_l.sum(axis=-1, keepdims=True)
        log_pi   = logits_s - np.log(exp_l.sum(axis=-1, keepdims=True))
        return pi, log_pi, zs, hs

    def _critic_fwd(self, x):
        """
        Forward pass through critic.
        Returns (values, zs, hs) — values has shape (batch,).
        """
        hs, zs = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.critic_W, self.critic_b)):
            z = h @ W + b
            zs.append(z)
            h = np.maximum(0, z) if i < len(self.critic_W) - 1 else z
            hs.append(h)
        return hs[-1].squeeze(-1), zs, hs

    # ── Public inference helpers ──────────────────────────────────────────────

    def select_action(self, obs):
        """
        Sample an action from the current policy.
        Returns (action: int, log_prob: float, value: float).
        """
        x = np.atleast_2d(obs).astype(np.float32)
        pi, log_pi, _, _ = self._actor_fwd(x)
        pi0    = pi[0].clip(1e-8, 1.0); pi0 /= pi0.sum()
        action = int(np.random.choice(self.n_actions, p=pi0))
        value, _, _ = self._critic_fwd(x)
        return action, float(log_pi[0, action]), float(value[0])

    def get_value(self, obs):
        """Scalar value estimate — call for bootstrap at end of non-terminal episode."""
        x = np.atleast_2d(obs).astype(np.float32)
        v, _, _ = self._critic_fwd(x)
        return float(v[0])

    # ── GAE ───────────────────────────────────────────────────────────────────

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalised Advantage Estimation.

        rewards, values, dones: 1-D arrays of length T
        last_value: 0.0 if episode terminated, else V(s_{T+1})
        Returns (advantages, returns) — both shape (T,).
        """
        T   = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v      = last_value if t == T - 1 else values[t + 1]
            non_term    = 1.0 - dones[t]
            delta       = rewards[t] + self.gamma * next_v * non_term - values[t]
            gae         = delta + self.gamma * self.gae_lambda * non_term * gae
            adv[t]      = gae
        return adv, adv + values

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _adam_update(self, all_params, all_grads):
        """In-place Adam step. all_params and all_grads must be same-ordered lists."""
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(all_params, all_grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)

    def _clip_grads(self, grads):
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        if norm > self.grad_clip_norm:
            c = self.grad_clip_norm / (norm + 1e-6)
            return [g * c for g in grads]
        return grads

    # ── Mini-batch gradient step ──────────────────────────────────────────────

    def _update_batch(self, obs, actions, old_log_probs, advantages, returns):
        """One gradient step on a mini-batch. Updates actor and critic in-place."""
        bs = len(obs)

        # ── Actor forward ─────────────────────────────────────────────────────
        pi, log_pi, actor_zs, actor_hs = self._actor_fwd(obs)
        log_prob_new = log_pi[np.arange(bs), actions]      # (bs,)
        ratio        = np.exp(log_prob_new - old_log_probs)  # (bs,)

        # ── PPO clipped policy loss ───────────────────────────────────────────
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -float(np.minimum(surr1, surr2).mean())

        # ── Entropy bonus ─────────────────────────────────────────────────────
        entropy = -(pi * log_pi).sum(axis=1)               # (bs,)

        # ── Critic forward and value loss ─────────────────────────────────────
        values_new, critic_zs, critic_hs = self._critic_fwd(obs)
        value_loss = self.vf_coef * float(((values_new - returns) ** 2).mean())

        # ── Actor backward ────────────────────────────────────────────────────
        # Gradient of policy_loss w.r.t. log_prob_new (PPO clip mask):
        #   zero when ratio is clipped and moving in the beneficial direction
        clip_mask = np.ones(bs, dtype=np.float32)
        clip_mask[(ratio > 1 + self.clip_eps) & (advantages > 0)] = 0.0
        clip_mask[(ratio < 1 - self.clip_eps) & (advantages < 0)] = 0.0
        g_pol = -ratio * advantages * clip_mask / bs        # d(policy_loss)/d(log_p)

        # Gradient w.r.t. logits:
        #   d(log_prob[a])/d(logit[j]) = delta(a,j) - pi[j]
        d_logits  = -pi * g_pol[:, None]
        d_logits[np.arange(bs), actions] += g_pol

        # Entropy contribution:
        #   d(-ent_coef * mean(H)) / d(logit[j]) = ent_coef * pi[j] * (log_pi[j] + H) / N
        d_logits += (self.ent_coef / bs) * pi * (log_pi + entropy[:, None])

        # Backprop through actor MLP
        d_cur = d_logits
        actor_gW, actor_gb = [], []
        for i in reversed(range(len(self.actor_W))):
            actor_gW.insert(0, actor_hs[i].T @ d_cur)
            actor_gb.insert(0, d_cur.sum(0))
            if i > 0:
                d_cur = (d_cur @ self.actor_W[i].T) * (actor_zs[i - 1] > 0)

        # ── Critic backward ───────────────────────────────────────────────────
        # Gradient of vf_coef * mean((V - returns)^2) w.r.t. V
        d_cur = (2 * self.vf_coef * (values_new - returns) / bs)[:, None]  # (bs,1)
        critic_gW, critic_gb = [], []
        for i in reversed(range(len(self.critic_W))):
            critic_gW.insert(0, critic_hs[i].T @ d_cur)
            critic_gb.insert(0, d_cur.sum(0))
            if i > 0:
                d_cur = (d_cur @ self.critic_W[i].T) * (critic_zs[i - 1] > 0)

        # ── Clip and apply Adam ───────────────────────────────────────────────
        all_grads  = actor_gW + actor_gb + critic_gW + critic_gb
        all_grads  = self._clip_grads(all_grads)
        all_params = self.actor_W + self.actor_b + self.critic_W + self.critic_b
        self.t    += 1
        self._adam_update(all_params, all_grads)

        return policy_loss, value_loss, float(entropy.mean())

    # ── Public update ─────────────────────────────────────────────────────────

    def update(self, rollout: RolloutBuffer, last_value: float = 0.0) -> dict:
        """
        Update from one episode's rollout.

        rollout:    RolloutBuffer populated during the episode.
        last_value: 0.0 if terminated, else agent.get_value(last_obs) for bootstrap.
        Returns dict with mean losses for logging.
        """
        if len(rollout) == 0:
            return {}

        obs, actions, rewards, values, log_probs, dones = rollout.get()
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n          = len(obs)
        batch_size = min(self.mini_batch_size, n)
        p_losses, v_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                pl, vl, ent = self._update_batch(
                    obs[b], actions[b], log_probs[b], advantages[b], returns[b])
                p_losses.append(pl); v_losses.append(vl); entropies.append(ent)

        return {
            "policy_loss": float(np.mean(p_losses)),
            "value_loss":  float(np.mean(v_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        np.savez(
            path,
            aW0=self.actor_W[0],  aW1=self.actor_W[1],  aW2=self.actor_W[2],
            ab0=self.actor_b[0],  ab1=self.actor_b[1],  ab2=self.actor_b[2],
            cW0=self.critic_W[0], cW1=self.critic_W[1], cW2=self.critic_W[2],
            cb0=self.critic_b[0], cb1=self.critic_b[1], cb2=self.critic_b[2],
        )

    @classmethod
    def load(cls, path, obs_dim, n_actions, hidden: int = 64):
        d     = np.load(path)
        agent = cls(obs_dim, n_actions, hidden=hidden)
        agent.actor_W  = [d["aW0"], d["aW1"], d["aW2"]]
        agent.actor_b  = [d["ab0"], d["ab1"], d["ab2"]]
        agent.critic_W = [d["cW0"], d["cW1"], d["cW2"]]
        agent.critic_b = [d["cb0"], d["cb1"], d["cb2"]]
        return agent