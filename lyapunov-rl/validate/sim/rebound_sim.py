"""REBOUND MERCURIUS reference simulation.

Provides a high-fidelity n-body simulation using REBOUND's MERCURIUS
integrator (symplectic for distant encounters, IAS15 for close ones).

Used for:
  - Cross-validating the JAX leapfrog integrator
  - Running trained policies in full n-body physics
  - Generating ground-truth trajectories for experiments

This module is validation-only — never imported in the training loop.
REBOUND is CPU-only and not JAX-compatible.
"""

import numpy as np

try:
    import rebound
except ImportError:
    rebound = None  # Allow import on systems without rebound (e.g., Colab GPU)

from shared.constants import (
    AU,
    DAY,
    DEFAULT_DT,
    DEFAULT_ISP,
    DEFAULT_SC_MASS,
    G0,
    M_EARTH,
    M_JUPITER,
    M_MARS,
    M_SUN,
    M_VENUS,
)


def _require_rebound():
    if rebound is None:
        raise ImportError(
            "rebound is not installed. Install with: pip install rebound"
        )


def create_solar_system(
    planets: tuple[str, ...] = ("earth", "mars"),
) -> "rebound.Simulation":
    """Create a REBOUND simulation with the Sun and specified planets.

    Uses MERCURIUS integrator with default settings.
    Planets are initialized on circular orbits at their semi-major axes.

    Args:
        planets: tuple of planet names to include

    Returns:
        Configured rebound.Simulation
    """
    _require_rebound()

    sim = rebound.Simulation()
    sim.integrator = "mercurius"
    sim.dt = DEFAULT_DT
    sim.units = ("m", "s", "kg")

    # Sun at origin
    sim.add(m=M_SUN)

    planet_data = {
        "venus": {"m": M_VENUS, "a": 1.082e11},
        "earth": {"m": M_EARTH, "a": 1.496e11},
        "mars": {"m": M_MARS, "a": 2.279e11},
        "jupiter": {"m": M_JUPITER, "a": 7.785e11},
    }

    for name in planets:
        data = planet_data[name]
        sim.add(m=data["m"], a=data["a"])

    sim.move_to_com()
    return sim


def add_spacecraft(
    sim: "rebound.Simulation",
    pos: np.ndarray,
    vel: np.ndarray,
    mass: float = DEFAULT_SC_MASS,
) -> int:
    """Add a spacecraft particle to the simulation.

    Args:
        sim: REBOUND simulation
        pos: initial heliocentric position (3,) in meters
        vel: initial heliocentric velocity (3,) in m/s
        mass: spacecraft mass in kg (effectively zero for gravity)

    Returns:
        Index of the spacecraft particle in sim.particles
    """
    sim.add(
        m=mass,
        x=pos[0], y=pos[1], z=pos[2],
        vx=vel[0], vy=vel[1], vz=vel[2],
    )
    return sim.N - 1


def apply_thrust(
    sim: "rebound.Simulation",
    sc_index: int,
    thrust_vec: np.ndarray,
    dt: float = DEFAULT_DT,
    sc_mass: float = DEFAULT_SC_MASS,
) -> None:
    """Apply an impulsive thrust to the spacecraft.

    Converts thrust force to delta-v and applies instantaneously.
    For low-thrust, this is applied once per timestep (impulse approximation).

    Args:
        sim: REBOUND simulation
        sc_index: spacecraft particle index
        thrust_vec: thrust force vector (3,) in Newtons
        dt: timestep over which thrust is applied
        sc_mass: current spacecraft mass in kg
    """
    dv = thrust_vec * dt / sc_mass
    p = sim.particles[sc_index]
    p.vx += dv[0]
    p.vy += dv[1]
    p.vz += dv[2]


def get_particle_state(
    sim: "rebound.Simulation", index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and velocity of a particle.

    Returns heliocentric coordinates (relative to particle 0 = Sun).

    Args:
        sim: REBOUND simulation
        index: particle index

    Returns:
        (position (3,), velocity (3,)) in SI units
    """
    p = sim.particles[index]
    sun = sim.particles[0]
    pos = np.array([p.x - sun.x, p.y - sun.y, p.z - sun.z])
    vel = np.array([p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz])
    return pos, vel


def get_all_planet_positions(
    sim: "rebound.Simulation", n_planets: int
) -> np.ndarray:
    """Extract heliocentric positions of all planets (indices 1..n_planets).

    Args:
        sim: REBOUND simulation
        n_planets: number of planets (excluding Sun)

    Returns:
        Positions array (n_planets, 3) in meters
    """
    positions = np.zeros((n_planets, 3))
    sun = sim.particles[0]
    for i in range(n_planets):
        p = sim.particles[i + 1]
        positions[i] = [p.x - sun.x, p.y - sun.y, p.z - sun.z]
    return positions


def propagate(
    sim: "rebound.Simulation",
    n_steps: int,
    sc_index: int,
    thrust_fn=None,
    dt: float = DEFAULT_DT,
    sc_mass: float = DEFAULT_SC_MASS,
    isp: float = DEFAULT_ISP,
) -> dict:
    """Propagate the simulation forward, optionally applying thrust.

    Args:
        sim: REBOUND simulation
        n_steps: number of timesteps to integrate
        sc_index: spacecraft particle index
        thrust_fn: callable(step, pos, vel, planet_positions) -> thrust_vec (3,) in N
                   If None, spacecraft is ballistic.
        dt: timestep in seconds
        sc_mass: initial spacecraft mass in kg
        isp: specific impulse in seconds (for mass tracking)

    Returns:
        dict with keys:
            positions: (n_steps+1, 3) spacecraft positions
            velocities: (n_steps+1, 3) spacecraft velocities
            times: (n_steps+1,) simulation times
            mass: (n_steps+1,) spacecraft mass (accounting for fuel use)
            planet_positions: (n_steps+1, n_planets, 3) planet positions
    """
    n_planets = sim.N - 2  # exclude Sun and spacecraft

    positions = np.zeros((n_steps + 1, 3))
    velocities = np.zeros((n_steps + 1, 3))
    times = np.zeros(n_steps + 1)
    mass_history = np.zeros(n_steps + 1)
    planet_pos_history = np.zeros((n_steps + 1, n_planets, 3))

    # Record initial state
    pos, vel = get_particle_state(sim, sc_index)
    positions[0] = pos
    velocities[0] = vel
    times[0] = sim.t
    mass_history[0] = sc_mass
    planet_pos_history[0] = get_all_planet_positions(sim, n_planets)

    current_mass = sc_mass

    for step in range(n_steps):
        # Apply thrust if provided
        if thrust_fn is not None:
            planet_positions = get_all_planet_positions(sim, n_planets)
            thrust_vec = thrust_fn(step, pos, vel, planet_positions)
            apply_thrust(sim, sc_index, thrust_vec, dt, current_mass)

            # Track mass loss: dm = |F| * dt / (Isp * g0)
            thrust_mag = np.linalg.norm(thrust_vec)
            dm = thrust_mag * dt / (isp * G0)
            current_mass = max(current_mass - dm, 0.1)  # floor to avoid zero mass

        # Integrate one step
        sim.integrate(sim.t + dt)

        # Record state
        pos, vel = get_particle_state(sim, sc_index)
        positions[step + 1] = pos
        velocities[step + 1] = vel
        times[step + 1] = sim.t
        mass_history[step + 1] = current_mass
        planet_pos_history[step + 1] = get_all_planet_positions(sim, n_planets)

    return {
        "positions": positions,
        "velocities": velocities,
        "times": times,
        "mass": mass_history,
        "planet_positions": planet_pos_history,
    }


def earth_departure_state() -> tuple[np.ndarray, np.ndarray]:
    """Standard Earth departure: 1 AU on +x axis, circular velocity on +y.

    Returns:
        (position (3,), velocity (3,)) in SI units
    """
    r = 1.496e11  # 1 AU
    v = np.sqrt(1.32712440018e20 / r)  # circular velocity at 1 AU
    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v, 0.0])
    return pos, vel
