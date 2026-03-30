"""Gravitational force computation for validation.

Numpy-based n-body gravity, used to:
  - Cross-validate JAX gravity (train/sim/gravity.py)
  - Compare force vectors between REBOUND and our implementation
  - Verify conservation properties

This is the reference implementation — correctness over speed.
"""

import numpy as np

from shared.constants import MU_EARTH, MU_MARS, MU_SUN


def gravitational_acceleration(
    pos: np.ndarray,
    source_pos: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Gravitational acceleration from a single body.

    a = -mu * (pos - source_pos) / |pos - source_pos|^3

    Args:
        pos: test particle position (3,)
        source_pos: gravitating body position (3,)
        mu: gravitational parameter of source (m^3/s^2)

    Returns:
        Acceleration vector (3,) in m/s^2
    """
    r_vec = pos - source_pos
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / (r ** 3)


def total_acceleration(
    sc_pos: np.ndarray,
    body_positions: np.ndarray,
    body_mus: np.ndarray,
) -> np.ndarray:
    """Total gravitational acceleration on spacecraft from all bodies.

    Args:
        sc_pos: spacecraft position (3,) in meters
        body_positions: positions of gravitating bodies (N, 3) in meters
                        Should include the Sun.
        body_mus: gravitational parameters (N,) in m^3/s^2

    Returns:
        Total acceleration vector (3,) in m/s^2
    """
    acc = np.zeros(3)
    for i in range(len(body_mus)):
        acc += gravitational_acceleration(sc_pos, body_positions[i], body_mus[i])
    return acc


def compare_forces(
    sc_pos: np.ndarray,
    body_positions: np.ndarray,
    body_mus: np.ndarray,
    jax_acc: np.ndarray,
) -> dict:
    """Compare our gravity computation against a JAX implementation.

    Args:
        sc_pos: spacecraft position (3,)
        body_positions: body positions (N, 3)
        body_mus: gravitational parameters (N,)
        jax_acc: acceleration from JAX implementation (3,)

    Returns:
        dict with:
            ref_acc: reference acceleration (3,)
            jax_acc: JAX acceleration (3,)
            abs_error: absolute error (3,)
            rel_error: scalar relative error
    """
    ref_acc = total_acceleration(sc_pos, body_positions, body_mus)
    abs_error = np.abs(jax_acc - ref_acc)
    ref_mag = np.linalg.norm(ref_acc)
    rel_error = np.linalg.norm(abs_error) / ref_mag if ref_mag > 0 else float("inf")

    return {
        "ref_acc": ref_acc,
        "jax_acc": np.asarray(jax_acc),
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def solar_gravity_at_earth() -> np.ndarray:
    """Sanity check: solar gravity at 1 AU should be ~5.93e-3 m/s^2."""
    earth_pos = np.array([1.496e11, 0.0, 0.0])
    sun_pos = np.array([0.0, 0.0, 0.0])
    return gravitational_acceleration(earth_pos, sun_pos, MU_SUN)
