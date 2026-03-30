"""Observation vector construction for the trajectory environment.

Layout (all heliocentric, normalized):
  [0:3]   spacecraft position / NORM_POS
  [3:6]   spacecraft velocity / NORM_VEL
  [6:9]   target position / NORM_POS
  [9:12]  target velocity / NORM_VEL
  [12:15] relative position (target - sc) / NORM_POS
  [15:18] relative velocity (target - sc) / NORM_VEL
  [18:]   per-planet relative position to sc (3 per planet) / NORM_POS

Total: 18 + 3 * N_planets

All functions are pure JAX — no side effects, vmappable.
"""

import jax.numpy as jnp

from shared.constants import NORM_POS, NORM_VEL


def build_obs(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    target_vel: jnp.ndarray,
    planet_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Construct the full observation vector.

    Args:
        sc_pos: spacecraft heliocentric position (3,) in meters
        sc_vel: spacecraft heliocentric velocity (3,) in m/s
        target_pos: target body heliocentric position (3,) in meters
        target_vel: target body heliocentric velocity (3,) in m/s
        planet_positions: planet heliocentric positions (N_planets, 3) in meters

    Returns:
        Observation vector (18 + 3*N_planets,)
    """
    rel_pos = target_pos - sc_pos
    rel_vel = target_vel - sc_vel

    # Per-planet relative positions to spacecraft
    planet_rel = (planet_positions - sc_pos[None, :]) / NORM_POS  # (N, 3)

    obs = jnp.concatenate([
        sc_pos / NORM_POS,
        sc_vel / NORM_VEL,
        target_pos / NORM_POS,
        target_vel / NORM_VEL,
        rel_pos / NORM_POS,
        rel_vel / NORM_VEL,
        planet_rel.ravel(),
    ])
    return obs


def obs_dim(n_planets: int) -> int:
    """Return observation vector dimensionality."""
    return 18 + 3 * n_planets


def extract_sc_state(obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract un-normalized spacecraft pos/vel from observation.

    Useful for computing orbital energy inside the Lyapunov forward pass.
    """
    sc_pos = obs[0:3] * NORM_POS
    sc_vel = obs[3:6] * NORM_VEL
    return sc_pos, sc_vel
