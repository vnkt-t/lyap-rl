"""Leapfrog (kick-drift-kick) integrator in pure JAX.

Symplectic integrator that preserves the Hamiltonian structure of
orbital mechanics. Second-order accurate, time-reversible.

Kick-drift-kick form:
    v_{1/2} = v_n + (dt/2) * a(x_n)
    x_{n+1} = x_n + dt * v_{1/2}
    v_{n+1} = v_{1/2} + (dt/2) * a(x_{n+1})

All functions are pure JAX — no side effects, vmappable over envs.
"""

import jax
import jax.numpy as jnp

from train.sim.gravity import acceleration_with_thrust, total_acceleration
from shared.constants import DEFAULT_DT, DEFAULT_ISP, G0


def leapfrog_step(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_mus: jnp.ndarray,
    thrust_vec: jnp.ndarray,
    sc_mass: jnp.ndarray,
    dt: float = DEFAULT_DT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One leapfrog step with thrust.

    Thrust is applied as a constant acceleration over the full step.
    Mass decreases according to the Tsiolkovsky equation (impulse approx).

    Args:
        sc_pos: spacecraft position (3,)
        sc_vel: spacecraft velocity (3,)
        body_positions: gravitating body positions (N, 3)
        body_mus: gravitational parameters (N,)
        thrust_vec: thrust force vector (3,) in Newtons
        sc_mass: current spacecraft mass (scalar) in kg
        dt: timestep in seconds

    Returns:
        (new_pos, new_vel, new_mass)
    """
    # Kick 1: half-step velocity update
    acc_n = acceleration_with_thrust(
        sc_pos, body_positions, body_mus, thrust_vec, sc_mass
    )
    vel_half = sc_vel + 0.5 * dt * acc_n

    # Drift: full-step position update
    pos_new = sc_pos + dt * vel_half

    # Kick 2: half-step velocity update at new position
    acc_new = acceleration_with_thrust(
        pos_new, body_positions, body_mus, thrust_vec, sc_mass
    )
    vel_new = vel_half + 0.5 * dt * acc_new

    # Mass loss from thrust: dm = |F| * dt / (Isp * g0)
    thrust_mag = jnp.linalg.norm(thrust_vec)
    dm = thrust_mag * dt / (DEFAULT_ISP * G0)
    mass_new = jnp.maximum(sc_mass - dm, 0.1)

    return pos_new, vel_new, mass_new


def leapfrog_step_ballistic(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_mus: jnp.ndarray,
    dt: float = DEFAULT_DT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One leapfrog step without thrust (ballistic propagation).

    Useful for planet propagation and testing energy conservation.

    Args:
        sc_pos: position (3,)
        sc_vel: velocity (3,)
        body_positions: gravitating body positions (N, 3)
        body_mus: gravitational parameters (N,)
        dt: timestep in seconds

    Returns:
        (new_pos, new_vel)
    """
    acc_n = total_acceleration(sc_pos, body_positions, body_mus)
    vel_half = sc_vel + 0.5 * dt * acc_n

    pos_new = sc_pos + dt * vel_half

    acc_new = total_acceleration(pos_new, body_positions, body_mus)
    vel_new = vel_half + 0.5 * dt * acc_new

    return pos_new, vel_new


def propagate_ballistic(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    body_positions: jnp.ndarray,
    body_mus: jnp.ndarray,
    n_steps: int,
    dt: float = DEFAULT_DT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate ballistic trajectory for n_steps using jax.lax.scan.

    Planet positions are held fixed (adequate for short propagations;
    the env will update them each step).

    Args:
        sc_pos: initial position (3,)
        sc_vel: initial velocity (3,)
        body_positions: gravitating body positions (N, 3)
        body_mus: gravitational parameters (N,)
        n_steps: number of integration steps
        dt: timestep in seconds

    Returns:
        (positions (n_steps+1, 3), velocities (n_steps+1, 3))
    """

    def scan_fn(carry, _):
        pos, vel = carry
        pos_new, vel_new = leapfrog_step_ballistic(
            pos, vel, body_positions, body_mus, dt
        )
        return (pos_new, vel_new), (pos_new, vel_new)

    (_, _), (positions, velocities) = jax.lax.scan(
        scan_fn, (sc_pos, sc_vel), None, length=n_steps
    )

    # Prepend initial state
    positions = jnp.concatenate([sc_pos[None, :], positions], axis=0)
    velocities = jnp.concatenate([sc_vel[None, :], velocities], axis=0)

    return positions, velocities
