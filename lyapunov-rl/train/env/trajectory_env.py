"""Gymnax-style trajectory optimization environment in pure JAX.

Functional API — no classes with mutable state:
    state = reset(key, env_params)
    state, obs, reward, done, info = step(state, action, env_params)

Everything is a pytree. vmap over the batch dimension for parallel envs.
step() returns V(s) and V(s') in info for the Lyapunov penalty in PPO.

Planet positions use Keplerian circular orbits (fast, adequate for training).
REBOUND handles real n-body physics in validation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from shared.constants import (
    AU,
    DEFAULT_DT,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_MAX_THRUST,
    DEFAULT_SC_MASS,
    EPSILON_MARS,
    MU_SUN,
    NORM_POS,
    NORM_VEL,
    PLANET_A,
    PLANET_MU,
)
from shared.constants import NORM_ENERGY as _NORM_ENERGY
from shared.obs import build_obs
from shared.reward import compute_reward
from train.sim.gravity import acceleration_with_thrust
from train.sim.integrator import leapfrog_step


# ---------------------------------------------------------------------------
# State and params as frozen pytrees
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Full environment state — pure data, no methods."""
    sc_pos: jnp.ndarray       # (3,) spacecraft heliocentric position, m
    sc_vel: jnp.ndarray       # (3,) spacecraft heliocentric velocity, m/s
    sc_mass: jnp.ndarray      # () current mass, kg
    time: jnp.ndarray         # () elapsed time, seconds
    step_count: jnp.ndarray   # () integer step counter
    planet_phases: jnp.ndarray # (N_planets,) initial orbital phases, radians
    prev_energy_error: jnp.ndarray  # () previous |epsilon - epsilon_target| / NORM_ENERGY
    done: jnp.ndarray         # () boolean


class EnvParams(NamedTuple):
    """Environment configuration — constant across an episode."""
    dt: float = DEFAULT_DT
    max_steps: int = DEFAULT_EPISODE_STEPS
    max_thrust: float = DEFAULT_MAX_THRUST
    sc_mass_init: float = DEFAULT_SC_MASS
    epsilon_target: float = EPSILON_MARS
    # Target orbit semi-major axis (for position-based capture check)
    target_a: float = 2.279e11  # Mars
    # Capture tolerance
    capture_radius: float = 1e10       # 10 million km position tolerance
    capture_energy_tol: float = 5e6    # J/kg energy tolerance
    # Which planets to include (indices into PLANET_A / PLANET_MU)
    # Default: Venus(1), Earth(2), Mars(3)
    planet_indices: tuple = (1, 2, 3)


# ---------------------------------------------------------------------------
# Planet kinematics (Keplerian circular orbits)
# ---------------------------------------------------------------------------

def _planet_omega(a: jnp.ndarray) -> jnp.ndarray:
    """Mean angular velocity for circular orbit: omega = sqrt(mu/a^3)."""
    return jnp.sqrt(MU_SUN / a ** 3)


def planet_positions_at_time(
    planet_a: jnp.ndarray,
    planet_phases: jnp.ndarray,
    time: jnp.ndarray,
) -> jnp.ndarray:
    """Compute planet heliocentric positions at a given time.

    Circular orbit approximation in the ecliptic plane.

    Args:
        planet_a: semi-major axes (N,)
        planet_phases: initial orbital phases (N,) in radians
        time: elapsed time, scalar, in seconds

    Returns:
        Positions (N, 3) in meters
    """
    omega = _planet_omega(planet_a)
    theta = planet_phases + omega * time
    x = planet_a * jnp.cos(theta)
    y = planet_a * jnp.sin(theta)
    z = jnp.zeros_like(x)
    return jnp.stack([x, y, z], axis=-1)


def planet_velocities_at_time(
    planet_a: jnp.ndarray,
    planet_phases: jnp.ndarray,
    time: jnp.ndarray,
) -> jnp.ndarray:
    """Compute planet heliocentric velocities (circular orbit).

    v = omega * a * [-sin(theta), cos(theta), 0]

    Args:
        planet_a: semi-major axes (N,)
        planet_phases: initial orbital phases (N,) in radians
        time: elapsed time, scalar, in seconds

    Returns:
        Velocities (N, 3) in m/s
    """
    omega = _planet_omega(planet_a)
    theta = planet_phases + omega * time
    vx = -omega * planet_a * jnp.sin(theta)
    vy = omega * planet_a * jnp.cos(theta)
    vz = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, vz], axis=-1)


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def action_to_thrust(action: jnp.ndarray, max_thrust: float) -> jnp.ndarray:
    """Convert policy output (mag, az, el) to thrust force vector.

    Policy outputs are assumed to be in [-1, 1] (tanh output).
      - mag: thrust magnitude fraction, clipped to [0, 1]
      - az: azimuth angle, scaled to [0, 2*pi]
      - el: elevation angle, scaled to [-pi/2, pi/2]

    Args:
        action: (3,) array [mag, az, el] in [-1, 1]
        max_thrust: maximum thrust in Newtons

    Returns:
        Thrust force vector (3,) in Newtons
    """
    mag = jnp.clip((action[0] + 1.0) / 2.0, 0.0, 1.0) * max_thrust
    az = (action[1] + 1.0) * jnp.pi   # [0, 2*pi]
    el = action[2] * (jnp.pi / 2.0)    # [-pi/2, pi/2]

    fx = mag * jnp.cos(el) * jnp.cos(az)
    fy = mag * jnp.cos(el) * jnp.sin(az)
    fz = mag * jnp.sin(el)
    return jnp.array([fx, fy, fz])


# ---------------------------------------------------------------------------
# Core env functions
# ---------------------------------------------------------------------------

def _get_body_arrays(
    planet_a: jnp.ndarray,
    planet_phases: jnp.ndarray,
    planet_mus: jnp.ndarray,
    time: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get body positions and mus (Sun + planets) at a given time.

    Returns:
        body_positions: (1 + N_planets, 3) — Sun first, then planets
        body_mus: (1 + N_planets,)
    """
    p_pos = planet_positions_at_time(planet_a, planet_phases, time)
    sun_pos = jnp.zeros((1, 3))
    body_positions = jnp.concatenate([sun_pos, p_pos], axis=0)
    body_mus = jnp.concatenate([jnp.array([MU_SUN]), planet_mus])
    return body_positions, body_mus


def _build_obs_from_state(
    state: EnvState,
    planet_a: jnp.ndarray,
    planet_mus: jnp.ndarray,
) -> jnp.ndarray:
    """Build observation vector from env state."""
    p_pos = planet_positions_at_time(planet_a, state.planet_phases, state.time)
    p_vel = planet_velocities_at_time(planet_a, state.planet_phases, state.time)

    # Target = last planet (Mars by default)
    target_pos = p_pos[-1]
    target_vel = p_vel[-1]

    return build_obs(state.sc_pos, state.sc_vel, target_pos, target_vel, p_pos)


def _check_capture(
    sc_pos: jnp.ndarray,
    sc_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    epsilon_target: float,
    capture_radius: float,
    capture_energy_tol: float,
) -> jnp.ndarray:
    """Check if spacecraft has been captured by target orbit.

    Capture = within position tolerance AND energy tolerance.
    """
    dist = jnp.linalg.norm(sc_pos - target_pos)
    epsilon = 0.5 * jnp.dot(sc_vel, sc_vel) - MU_SUN / jnp.linalg.norm(sc_pos)
    energy_err = jnp.abs(epsilon - epsilon_target)

    pos_ok = dist < capture_radius
    energy_ok = energy_err < capture_energy_tol
    return pos_ok & energy_ok


def reset(key: jax.Array, params: EnvParams = EnvParams()) -> tuple[EnvState, jnp.ndarray]:
    """Reset environment to initial conditions.

    Spacecraft starts near Earth with slight random perturbation.
    Planet phases are randomized for diverse initial configurations.

    Args:
        key: PRNG key
        params: environment parameters

    Returns:
        (initial_state, initial_obs)
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Earth position + small random offset
    earth_a = 1.496e11
    earth_phase = jax.random.uniform(k1, (), minval=0.0, maxval=2.0 * jnp.pi)
    sc_pos = jnp.array([
        earth_a * jnp.cos(earth_phase),
        earth_a * jnp.sin(earth_phase),
        0.0,
    ])
    # Add small perturbation (up to 1% of AU in each axis)
    sc_pos = sc_pos + jax.random.normal(k2, (3,)) * (0.01 * AU)

    # Circular velocity + small perturbation
    r = jnp.linalg.norm(sc_pos)
    v_circ = jnp.sqrt(MU_SUN / r)
    # Velocity perpendicular to position (circular orbit direction)
    v_dir = jnp.array([-sc_pos[1], sc_pos[0], 0.0])
    v_dir = v_dir / jnp.linalg.norm(v_dir)
    sc_vel = v_circ * v_dir + jax.random.normal(k3, (3,)) * (0.01 * v_circ)

    # Random planet phases
    n_planets = len(params.planet_indices)
    planet_phases = jax.random.uniform(
        k1, (n_planets,), minval=0.0, maxval=2.0 * jnp.pi
    )

    # Initial energy error for delta reward shaping
    r = jnp.linalg.norm(sc_pos)
    v_sq = jnp.dot(sc_vel, sc_vel)
    epsilon_init = 0.5 * v_sq - MU_SUN / r
    init_energy_error = jnp.abs(epsilon_init - params.epsilon_target) / _NORM_ENERGY

    state = EnvState(
        sc_pos=sc_pos,
        sc_vel=sc_vel,
        sc_mass=jnp.array(params.sc_mass_init),
        time=jnp.array(0.0),
        step_count=jnp.array(0, dtype=jnp.int32),
        planet_phases=planet_phases,
        prev_energy_error=init_energy_error,
        done=jnp.array(False),
    )

    planet_a = PLANET_A[jnp.array(params.planet_indices)]
    planet_mus = PLANET_MU[jnp.array(params.planet_indices)]
    obs = _build_obs_from_state(state, planet_a, planet_mus)

    return state, obs


def step(
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams = EnvParams(),
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Take one environment step.

    Args:
        state: current environment state
        action: (3,) policy output [mag, az, el] in [-1, 1]
        params: environment parameters

    Returns:
        (next_state, obs, reward, done, info)
        info contains 'v_s' and 'v_s_next' placeholders (filled by training loop
        after passing obs through LyapunovNet), plus 'captured' and 'truncated'.
    """
    planet_idx = jnp.array(params.planet_indices)
    planet_a = PLANET_A[planet_idx]
    planet_mus = PLANET_MU[planet_idx]

    # Convert action to thrust vector
    thrust_vec = action_to_thrust(action, params.max_thrust)
    thrust_mag = jnp.linalg.norm(thrust_vec)

    # Get gravitating bodies at current time
    body_positions, body_mus = _get_body_arrays(
        planet_a, state.planet_phases, planet_mus, state.time
    )

    # Integrate one leapfrog step
    new_pos, new_vel, new_mass = leapfrog_step(
        state.sc_pos, state.sc_vel,
        body_positions, body_mus,
        thrust_vec, state.sc_mass,
        params.dt,
    )

    new_time = state.time + params.dt
    new_step = state.step_count + 1

    # Target planet position at new time (last planet = target)
    target_pos = planet_positions_at_time(planet_a, state.planet_phases, new_time)[-1]

    # Check termination
    captured = _check_capture(
        new_pos, new_vel, target_pos,
        params.epsilon_target, params.capture_radius, params.capture_energy_tol,
    )
    truncated = new_step >= params.max_steps
    done = captured | truncated

    # Current energy error for delta shaping
    new_epsilon = 0.5 * jnp.dot(new_vel, new_vel) - MU_SUN / jnp.linalg.norm(new_pos)
    new_energy_error = jnp.abs(new_epsilon - params.epsilon_target) / _NORM_ENERGY

    # Reward (with delta shaping)
    reward = compute_reward(
        new_pos, new_vel, target_pos,
        thrust_mag / params.max_thrust,  # normalized action magnitude
        params.epsilon_target,
        captured,
        prev_energy_error=state.prev_energy_error,
    )

    # Build next state
    next_state = EnvState(
        sc_pos=new_pos,
        sc_vel=new_vel,
        sc_mass=new_mass,
        time=new_time,
        step_count=new_step,
        planet_phases=state.planet_phases,
        prev_energy_error=new_energy_error,
        done=done,
    )

    obs = _build_obs_from_state(next_state, planet_a, planet_mus)

    info = {
        "captured": captured,
        "truncated": truncated,
        "thrust_mag": thrust_mag,
        "sc_mass": new_mass,
        "epsilon": new_epsilon,
        "energy_error": new_energy_error,
    }

    return next_state, obs, reward, done, info
