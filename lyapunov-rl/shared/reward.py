"""Reward function for trajectory optimization.

The reward combines:
  1. Energy shaping: reward for reducing |epsilon - epsilon_target|
  2. Fuel penalty: penalize thrust usage (proportional to |action|)
  3. Capture bonus: large reward when entering target orbit tolerance
  4. Time penalty: small per-step cost to encourage efficiency

The Lyapunov stability penalty (lambda * max(0, V(s')-V(s))) is NOT here —
it lives in the PPO loss function (train/agent/ppo.py).

All functions are pure JAX — no side effects, vmappable.
"""

import jax.numpy as jnp

from shared.constants import MU_SUN, NORM_ENERGY


def specific_orbital_energy(
    pos: jnp.ndarray, vel: jnp.ndarray, mu: float = MU_SUN
) -> jnp.ndarray:
    """Compute specific orbital energy: epsilon = v^2/2 - mu/r.

    Args:
        pos: heliocentric position (3,) in meters
        vel: heliocentric velocity (3,) in m/s
        mu: gravitational parameter (default: Sun)

    Returns:
        Scalar specific orbital energy (J/kg)
    """
    r = jnp.linalg.norm(pos)
    v_sq = jnp.dot(vel, vel)
    return 0.5 * v_sq - mu / r


def compute_reward(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    action_magnitude: jnp.ndarray,
    epsilon_target: float,
    captured: jnp.ndarray,
    prev_energy_error: jnp.ndarray = None,
) -> jnp.ndarray:
    """Compute per-step reward.

    Uses energy CHANGE (delta shaping) rather than absolute error, so the
    agent gets a clear signal about whether each action helped or hurt.

    Args:
        sc_pos: spacecraft position (3,) in meters
        sc_vel: spacecraft velocity (3,) in m/s
        target_pos: target body position (3,) in meters
        action_magnitude: scalar thrust magnitude (normalized 0-1)
        epsilon_target: target specific orbital energy (J/kg)
        captured: boolean, whether spacecraft is within capture tolerance
        prev_energy_error: previous |epsilon - epsilon_target| / NORM_ENERGY
                           If None, only absolute error is used.

    Returns:
        Scalar reward
    """
    # Current energy error (normalized)
    epsilon = specific_orbital_energy(sc_pos, sc_vel)
    energy_error = jnp.abs(epsilon - epsilon_target) / NORM_ENERGY

    # Delta shaping: reward for REDUCING energy error (positive when improving)
    # Energy delta per step is ~1e-5, so scale by 1000 to make it the dominant signal
    r_energy_delta = jnp.where(
        prev_energy_error is not None,
        (prev_energy_error - energy_error) * 1000.0,
        0.0,
    )

    # Small absolute error penalty to maintain gradient at all times
    r_energy_abs = -0.05 * energy_error

    # Distance to target body
    dist = jnp.linalg.norm(target_pos - sc_pos)
    r_dist = -dist / (3.0e11)

    # Fuel penalty — light enough that thrust is worthwhile
    r_fuel = -0.01 * action_magnitude

    # Time penalty
    r_time = -0.005

    # Capture bonus
    r_capture = 10.0 * captured

    return r_energy_delta + r_energy_abs + 0.05 * r_dist + r_fuel + r_time + r_capture
