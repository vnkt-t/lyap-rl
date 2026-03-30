"""Bridge between trained JAX policy and REBOUND validation.

Loads a trained Equinox policy checkpoint, wraps it to produce thrust
vectors from REBOUND simulation state, and runs full n-body evaluation.

This is the critical validation layer: policy trained in simplified JAX env,
evaluated in high-fidelity REBOUND physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from shared.constants import (
    AU,
    DEFAULT_DT,
    DEFAULT_MAX_THRUST,
    DEFAULT_SC_MASS,
    EPSILON_MARS,
    MU_SUN,
    NORM_POS,
    NORM_VEL,
    PLANET_A,
    PLANET_MU,
)
from shared.obs import build_obs, obs_dim
from shared.reward import specific_orbital_energy
from train.agent.networks import PolicyNet, ValueNet, create_networks
from shared.lyapunov import LyapunovNet
from train.env.trajectory_env import action_to_thrust
from validate.sim.rebound_sim import (
    add_spacecraft,
    create_solar_system,
    get_all_planet_positions,
    get_particle_state,
    propagate,
)


def load_policy(
    checkpoint_prefix: str,
    obs_dim: int,
    epsilon_target: float = EPSILON_MARS,
) -> tuple[PolicyNet, ValueNet, LyapunovNet]:
    """Load trained networks from Equinox checkpoint files.

    Expects files: {prefix}_policy.eqx, {prefix}_value.eqx, {prefix}_lyapunov.eqx

    Args:
        checkpoint_prefix: path prefix (e.g., "checkpoints/step_1000")
        obs_dim: observation dimensionality
        epsilon_target: target orbital energy

    Returns:
        (policy, value_net, lyap_net)
    """
    key = jax.random.PRNGKey(0)  # dummy key for skeleton init
    policy, value_net, lyap_net = create_networks(
        obs_dim, epsilon_target=epsilon_target, key=key
    )

    policy = eqx.tree_deserialise_leaves(f"{checkpoint_prefix}_policy.eqx", policy)
    value_net = eqx.tree_deserialise_leaves(f"{checkpoint_prefix}_value.eqx", value_net)
    lyap_net = eqx.tree_deserialise_leaves(f"{checkpoint_prefix}_lyapunov.eqx", lyap_net)

    return policy, value_net, lyap_net


class PolicyBridge:
    """Wraps a JAX policy for use as a REBOUND thrust function.

    Converts REBOUND state (numpy arrays) to normalized observations,
    runs the policy forward pass, and converts actions back to thrust vectors.
    """

    def __init__(
        self,
        policy: PolicyNet,
        lyap_net: LyapunovNet | None = None,
        planets: tuple[str, ...] = ("venus", "earth", "mars"),
        max_thrust: float = DEFAULT_MAX_THRUST,
        epsilon_target: float = EPSILON_MARS,
        deterministic: bool = True,
    ):
        self.policy = policy
        self.lyap_net = lyap_net
        self.planets = planets
        self.max_thrust = max_thrust
        self.epsilon_target = epsilon_target
        self.deterministic = deterministic

        # Track Lyapunov values across steps
        self.v_history: list[float] = []
        self.epsilon_history: list[float] = []

    def __call__(
        self,
        step_idx: int,
        sc_pos: np.ndarray,
        sc_vel: np.ndarray,
        planet_positions: np.ndarray,
    ) -> np.ndarray:
        """Thrust function compatible with rebound_sim.propagate().

        Args:
            step_idx: current simulation step
            sc_pos: spacecraft position (3,)
            sc_vel: spacecraft velocity (3,)
            planet_positions: planet positions (N, 3)

        Returns:
            Thrust force vector (3,) in Newtons
        """
        # Target = last planet (Mars by default)
        target_pos = planet_positions[-1]

        # Approximate target velocity (circular orbit)
        r_target = np.linalg.norm(target_pos)
        v_circ = np.sqrt(MU_SUN / r_target)
        target_vel = np.array([
            -target_pos[1] / r_target * v_circ,
            target_pos[0] / r_target * v_circ,
            0.0,
        ])

        # Build observation
        obs = build_obs(
            jnp.array(sc_pos),
            jnp.array(sc_vel),
            jnp.array(target_pos),
            jnp.array(target_vel),
            jnp.array(planet_positions),
        )

        # Policy forward pass
        mean, log_std = self.policy(obs)

        if self.deterministic:
            action = mean
        else:
            key = jax.random.PRNGKey(step_idx)
            action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
            action = jnp.clip(action, -1.0, 1.0)

        # Convert to thrust
        thrust = action_to_thrust(action, self.max_thrust)

        # Track Lyapunov value
        if self.lyap_net is not None:
            v = float(self.lyap_net(obs))
            self.v_history.append(v)

        epsilon = float(specific_orbital_energy(
            jnp.array(sc_pos), jnp.array(sc_vel)
        ))
        self.epsilon_history.append(epsilon)

        return np.array(thrust)


def evaluate_policy(
    policy: PolicyNet,
    lyap_net: LyapunovNet | None = None,
    planets: tuple[str, ...] = ("venus", "earth", "mars"),
    sc_pos0: np.ndarray | None = None,
    sc_vel0: np.ndarray | None = None,
    n_steps: int = 9600,  # ~400 days at 1hr steps
    dt: float = DEFAULT_DT,
    max_thrust: float = DEFAULT_MAX_THRUST,
    epsilon_target: float = EPSILON_MARS,
    deterministic: bool = True,
) -> dict:
    """Run a full evaluation episode in REBOUND.

    Args:
        policy: trained policy network
        lyap_net: optional Lyapunov network for tracking V(s)
        planets: planets to include in simulation
        sc_pos0: initial spacecraft position (None = Earth departure)
        sc_vel0: initial spacecraft velocity (None = circular at Earth)
        n_steps: number of simulation steps
        dt: timestep
        max_thrust: maximum thrust in Newtons
        epsilon_target: target orbital energy
        deterministic: use mean action (True) or sample (False)

    Returns:
        dict with trajectories, Lyapunov history, capture info
    """
    from validate.sim.rebound_sim import earth_departure_state

    if sc_pos0 is None or sc_vel0 is None:
        sc_pos0, sc_vel0 = earth_departure_state()

    # Create simulation
    sim = create_solar_system(planets=planets)
    sc_idx = add_spacecraft(sim, sc_pos0, sc_vel0)

    # Create policy bridge
    bridge = PolicyBridge(
        policy, lyap_net, planets, max_thrust, epsilon_target, deterministic
    )

    # Propagate
    result = propagate(sim, n_steps, sc_idx, thrust_fn=bridge, dt=dt)

    # Check capture at each step
    target_a = 2.279e11  # Mars
    capture_radius = 1e10
    capture_energy_tol = 5e6

    captured_step = None
    for i in range(len(result["positions"])):
        pos = result["positions"][i]
        vel = result["velocities"][i]
        # Approximate target position at this time
        # (planet_positions are in result)
        target_pos = result["planet_positions"][i, -1]  # last planet = target
        dist = np.linalg.norm(pos - target_pos)
        eps = 0.5 * np.dot(vel, vel) - MU_SUN / np.linalg.norm(pos)
        if dist < capture_radius and abs(eps - epsilon_target) < capture_energy_tol:
            captured_step = i
            break

    result["v_history"] = np.array(bridge.v_history) if bridge.v_history else np.array([])
    result["epsilon_history"] = np.array(bridge.epsilon_history)
    result["captured"] = captured_step is not None
    result["captured_step"] = captured_step
    result["epsilon_target"] = epsilon_target

    return result
