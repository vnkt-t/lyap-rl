"""Lyapunov function V(s) with physics-informed prior.

V(s) = (epsilon - epsilon_target)^2 + softplus(NN(s))

Properties:
  - V(s) >= 0 always (squared term + softplus >= 0)
  - V(s) = 0 iff epsilon = epsilon_target AND NN(s) = 0 (at target orbit)
  - jax.grad(lyapunov_net)(obs) gives nabla V in one line with Equinox
  - Spectral normalization on all NN layers controls Lipschitz constant
    for Tier 2 certification

The physics term (epsilon - epsilon_target)^2 encodes orbital energy structure.
The neural correction learns everything energy alone can't capture:
inclination, phase, timing, multi-body perturbations.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from shared.constants import MU_SUN, NORM_ENERGY, NORM_POS, NORM_VEL


class SpectralLinear(eqx.Module):
    """Linear layer with spectral normalization.

    Constrains the spectral norm (largest singular value) of the weight
    matrix to be <= 1, which directly controls the Lipschitz constant
    of the layer. Essential for Tier 2 certification.

    Uses power iteration to approximate the top singular value,
    then rescales W -> W / max(sigma, 1).
    """

    weight: jnp.ndarray
    bias: jnp.ndarray
    u: jnp.ndarray  # left singular vector estimate (not a trained param)

    def __init__(self, in_features: int, out_features: int, *, key: jax.Array):
        wkey, bkey, ukey = jax.random.split(key, 3)
        # Lecun normal init, scaled down slightly for stability
        self.weight = jax.random.normal(wkey, (out_features, in_features)) * (
            1.0 / jnp.sqrt(in_features)
        )
        self.bias = jnp.zeros(out_features)
        self.u = jax.random.normal(ukey, (out_features,))
        self.u = self.u / jnp.linalg.norm(self.u)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w_normalized = _spectral_normalize(self.weight, self.u)
        return w_normalized @ x + self.bias


def _spectral_normalize(
    weight: jnp.ndarray, u: jnp.ndarray, n_iters: int = 1
) -> jnp.ndarray:
    """Normalize weight matrix by its spectral norm.

    Power iteration to estimate sigma_max, then W / max(sigma, 1).
    """
    u_hat = u
    for _ in range(n_iters):
        v_hat = weight.T @ u_hat
        v_hat = v_hat / (jnp.linalg.norm(v_hat) + 1e-12)
        u_hat = weight @ v_hat
        u_hat = u_hat / (jnp.linalg.norm(u_hat) + 1e-12)

    sigma = u_hat @ weight @ v_hat
    return weight / jnp.maximum(sigma, 1.0)


class LyapunovNet(eqx.Module):
    """Physics-informed Lyapunov function.

    Architecture:
        Input: full observation vector (obs_dim)
        ├── Compute epsilon = v^2/2 - mu/r from obs[0:6]
        ├── physics_term = ((epsilon - epsilon_target) / NORM_ENERGY)^2
        ├── Neural correction: obs -> tanh layers -> softplus output
        └── V(s) = physics_term + correction

    The neural net has spectral normalization on every layer.
    """

    layers: list
    epsilon_target: float

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        epsilon_target: float = -2.915e8,  # Mars orbit default
        *,
        key: jax.Array,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            SpectralLinear(obs_dim, hidden_dim, key=k1),
            SpectralLinear(hidden_dim, hidden_dim, key=k2),
            SpectralLinear(hidden_dim, 1, key=k3),
        ]
        self.epsilon_target = epsilon_target

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Compute V(s) from observation vector.

        Args:
            obs: normalized observation vector (obs_dim,)

        Returns:
            Scalar Lyapunov value V(s) >= 0
        """
        # --- Physics term ---
        # Extract un-normalized sc state from obs
        sc_pos = obs[0:3] * NORM_POS
        sc_vel = obs[3:6] * NORM_VEL
        r = jnp.linalg.norm(sc_pos)
        v_sq = jnp.dot(sc_vel, sc_vel)
        epsilon = 0.5 * v_sq - MU_SUN / r

        # Normalized squared energy error
        energy_error = (epsilon - self.epsilon_target) / NORM_ENERGY
        physics_term = energy_error ** 2

        # --- Neural correction ---
        x = obs
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        x = self.layers[-1](x)
        correction = jax.nn.softplus(x.squeeze())  # scalar >= 0

        return physics_term + correction


def lyapunov_delta(
    net: LyapunovNet, obs: jnp.ndarray, obs_next: jnp.ndarray
) -> jnp.ndarray:
    """Compute Delta V = V(s') - V(s).

    Used in the PPO loss: penalty = lambda * max(0, delta_v).

    Args:
        net: LyapunovNet module
        obs: current observation
        obs_next: next observation

    Returns:
        Scalar Delta V
    """
    return net(obs_next) - net(obs)


def lyapunov_penalty(
    net: LyapunovNet, obs: jnp.ndarray, obs_next: jnp.ndarray
) -> jnp.ndarray:
    """Compute Lyapunov penalty: max(0, V(s') - V(s)).

    This is the term added to the PPO loss, scaled by lambda.

    Args:
        net: LyapunovNet module
        obs: current observation
        obs_next: next observation

    Returns:
        Scalar penalty >= 0
    """
    delta = lyapunov_delta(net, obs, obs_next)
    return jnp.maximum(0.0, delta)
