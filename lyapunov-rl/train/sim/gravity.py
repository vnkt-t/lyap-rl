"""N-body gravitational acceleration in pure JAX.

All functions are pure, stateless, and vmappable.
No numpy, no side effects, no globals.

Computes gravitational acceleration on a test particle (spacecraft)
due to an arbitrary number of massive bodies (Sun + planets).
"""

import jax
import jax.numpy as jnp


def acceleration_single_body(
    sc_pos: jnp.ndarray,
    body_pos: jnp.ndarray,
    mu: jnp.ndarray,
) -> jnp.ndarray:
    """Gravitational acceleration from one body on the spacecraft.

    a = -mu * (sc_pos - body_pos) / |sc_pos - body_pos|^3

    Args:
        sc_pos: spacecraft position (3,)
        body_pos: body position (3,)
        mu: gravitational parameter, scalar

    Returns:
        Acceleration vector (3,)
    """
    r_vec = sc_pos - body_pos
    r_sq = jnp.dot(r_vec, r_vec)
    # Softened denominator to avoid NaN gradients at r=0
    r_sq_safe = jnp.maximum(r_sq, 1e6)  # ~1 mm minimum distance
    r_inv3 = r_sq_safe ** (-1.5)
    return -mu * r_vec * r_inv3


def total_acceleration(
    sc_pos: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_mus: jnp.ndarray,
) -> jnp.ndarray:
    """Total gravitational acceleration on spacecraft from all bodies.

    Uses vmap over bodies for efficient vectorization.

    Args:
        sc_pos: spacecraft position (3,)
        body_positions: positions of all gravitating bodies (N, 3)
                        First entry should be the Sun at ~origin.
        body_mus: gravitational parameters (N,)

    Returns:
        Total acceleration vector (3,)
    """
    # vmap over bodies: acceleration_single_body(sc_pos, body_i, mu_i)
    accs = jax.vmap(
        acceleration_single_body, in_axes=(None, 0, 0)
    )(sc_pos, body_positions, body_mus)
    return jnp.sum(accs, axis=0)


def acceleration_with_thrust(
    sc_pos: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_mus: jnp.ndarray,
    thrust_vec: jnp.ndarray,
    sc_mass: jnp.ndarray,
) -> jnp.ndarray:
    """Total acceleration including thrust.

    Args:
        sc_pos: spacecraft position (3,)
        body_positions: gravitating body positions (N, 3)
        body_mus: gravitational parameters (N,)
        thrust_vec: thrust force vector (3,) in Newtons
        sc_mass: spacecraft mass, scalar, in kg

    Returns:
        Total acceleration vector (3,)
    """
    grav_acc = total_acceleration(sc_pos, body_positions, body_mus)
    thrust_acc = thrust_vec / sc_mass
    return grav_acc + thrust_acc
