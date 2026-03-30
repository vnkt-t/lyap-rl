"""Equinox neural networks for PPO + Lyapunov training.

Three separate networks (no shared backbone — avoids gradient interference):

  Policy  pi(s)      : 2x256 MLP -> (mag, az, el) mean + log_std
  Value   V_rl(s)    : 2x256 MLP -> scalar (PPO baseline)
  Lyapunov V_lyap(s) : re-exported from shared/lyapunov.py (2x128, spectral norm)

Policy outputs a diagonal Gaussian: mean from tanh, learned log_std parameter.
Actions are sampled then squashed to [-1, 1] for the env.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from shared.lyapunov import LyapunovNet


class PolicyNet(eqx.Module):
    """Gaussian policy network.

    Outputs action mean (tanh-squashed) and has a learned log_std parameter.
    Action space: (mag, az, el) each in [-1, 1].
    """

    layers: list
    log_std: jnp.ndarray  # (action_dim,) learned log standard deviation

    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 256, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(obs_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
            eqx.nn.Linear(hidden_dim, action_dim, key=k3),
        ]
        self.log_std = jnp.full(action_dim, -0.5)  # init std ~ 0.6

    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: observation vector (obs_dim,)

        Returns:
            (action_mean (action_dim,), log_std (action_dim,))
        """
        x = obs
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        mean = jax.nn.tanh(self.layers[-1](x))
        # Clamp log_std to prevent entropy divergence
        # Range: std in [0.05, 1.0] -> log_std in [-3.0, 0.0]
        log_std = jnp.clip(self.log_std, -3.0, 0.0)
        return mean, log_std


def sample_action(
    policy: PolicyNet,
    obs: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample an action from the policy and compute its log probability.

    Args:
        policy: PolicyNet module
        obs: observation (obs_dim,)
        key: PRNG key

    Returns:
        (action (3,), log_prob scalar)
    """
    mean, log_std = policy(obs)
    std = jnp.exp(log_std)

    # Sample from Gaussian
    noise = jax.random.normal(key, mean.shape)
    action_raw = mean + std * noise

    # Clip to [-1, 1]
    action = jnp.clip(action_raw, -1.0, 1.0)

    # Log probability (diagonal Gaussian)
    log_prob = _gaussian_log_prob(action_raw, mean, log_std)

    return action, log_prob


def action_log_prob(
    policy: PolicyNet,
    obs: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log probability of an action under the policy.

    Args:
        policy: PolicyNet module
        obs: observation (obs_dim,)
        action: action (3,) in [-1, 1]

    Returns:
        Scalar log probability
    """
    mean, log_std = policy(obs)
    return _gaussian_log_prob(action, mean, log_std)


def _gaussian_log_prob(
    x: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    """Log probability under diagonal Gaussian."""
    std = jnp.exp(log_std)
    var = std ** 2
    log_p = -0.5 * jnp.sum(
        ((x - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    return log_p


class ValueNet(eqx.Module):
    """PPO value function (baseline) network."""

    layers: list

    def __init__(self, obs_dim: int, hidden_dim: int = 256, init_value: float = -1.2, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        output_layer = eqx.nn.Linear(hidden_dim, 1, key=k3)
        # Bias the output to match expected early-training returns (~-1.2)
        # so value MSE starts small and GAE advantages are centered
        output_layer = eqx.tree_at(
            lambda l: l.bias, output_layer, jnp.array([init_value])
        )
        self.layers = [
            eqx.nn.Linear(obs_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
            output_layer,
        ]

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass -> scalar value estimate."""
        x = obs
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x).squeeze()


def create_networks(
    obs_dim: int,
    action_dim: int = 3,
    epsilon_target: float = -2.912e8,
    *,
    key: jax.Array,
) -> tuple[PolicyNet, ValueNet, LyapunovNet]:
    """Initialize all three networks.

    Args:
        obs_dim: observation vector dimensionality
        action_dim: action space dimensionality (default 3)
        epsilon_target: target orbital energy for Lyapunov physics prior
        key: PRNG key

    Returns:
        (policy, value, lyapunov) network tuple
    """
    k1, k2, k3 = jax.random.split(key, 3)
    policy = PolicyNet(obs_dim, action_dim, key=k1)
    value = ValueNet(obs_dim, key=k2)
    lyapunov = LyapunovNet(obs_dim, hidden_dim=128, epsilon_target=epsilon_target, key=k3)
    return policy, value, lyapunov
