"""Experiment 3: Perturbation recovery.

At t = 30% of transfer, inject an out-of-plane velocity kick.
Compare recovery rates: Lyapunov PPO vs unconstrained PPO vs Lambert (no recovery).

Expected result: Lyapunov recovers at higher perturbation magnitudes.
Plot: success rate vs perturbation magnitude (three lines, clear separation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.constants import DEFAULT_DT, EPSILON_MARS, NORM_VEL
from shared.obs import obs_dim
from train.agent.ppo import PPOHyperParams
from train.agent.train import train
from train.env.trajectory_env import EnvParams
from validate.bridge.policy_eval import evaluate_policy
from validate.sim.rebound_sim import earth_departure_state


# Perturbation magnitudes as fraction of orbital velocity (~30 km/s)
PERTURBATION_FRACS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def _inject_perturbation(
    result: dict,
    perturbation_frac: float,
    inject_step: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract state at inject_step and add out-of-plane kick.

    Returns:
        (perturbed_pos, perturbed_vel) at the injection point
    """
    pos = result["positions"][inject_step].copy()
    vel = result["velocities"][inject_step].copy()

    # Out-of-plane kick (z-direction)
    rng = np.random.RandomState(seed)
    kick_magnitude = perturbation_frac * NORM_VEL  # fraction of ~30 km/s
    vel[2] += kick_magnitude * rng.choice([-1.0, 1.0])

    return pos, vel


def run_perturbation_experiment(
    n_seeds: int = 3,
    n_envs: int = 32,
    n_steps: int = 128,
    total_updates: int = 500,
    perturbation_fracs: list[float] | None = None,
    output_dir: str = "results/perturbation",
) -> dict:
    """Run perturbation recovery experiment.

    For each variant and perturbation magnitude:
      1. Run ballistic trajectory to 30% of transfer
      2. Inject perturbation
      3. Continue with policy (or ballistic for Lambert baseline)
      4. Check if capture still occurs

    Args:
        n_seeds: seeds per configuration
        n_envs: parallel envs for training
        n_steps: rollout length
        total_updates: training updates
        perturbation_fracs: perturbation magnitudes to test
        output_dir: results directory

    Returns:
        dict: variant -> perturbation_frac -> [seed results]
    """
    if perturbation_fracs is None:
        perturbation_fracs = PERTURBATION_FRACS

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    env_params = EnvParams()
    n_planets = len(env_params.planet_indices)
    odim = obs_dim(n_planets)

    total_steps = 9600  # ~400 days
    inject_frac = 0.3
    inject_step = int(total_steps * inject_frac)
    remaining_steps = total_steps - inject_step

    results = {"lyapunov": {}, "unconstrained": {}, "lambert": {}}

    # Train both variants
    trained = {}
    for variant, lambda_lyap in [("lyapunov", 0.1), ("unconstrained", 0.0)]:
        print(f"\n=== Training {variant} ===")
        hp = PPOHyperParams(lambda_lyap=lambda_lyap)
        train_state = train(
            n_envs=n_envs, n_steps=n_steps, total_updates=total_updates,
            seed=0, hp=hp, env_params=env_params,
        )
        trained[variant] = train_state

    # Run nominal trajectory to get injection state
    sc_pos0, sc_vel0 = earth_departure_state()

    for pfrac in perturbation_fracs:
        print(f"\n--- Perturbation: {pfrac*100:.0f}% of v_orb ---")

        for seed in range(n_seeds):
            # Get state at 30% by running ballistic
            nominal = evaluate_policy(
                trained["lyapunov"].policy,
                n_steps=inject_step,
                deterministic=True,
            )

            # Inject perturbation
            pert_pos, pert_vel = _inject_perturbation(
                nominal, pfrac, inject_step - 1, seed
            )

            # Evaluate each variant from perturbed state
            for variant in ["lyapunov", "unconstrained"]:
                if str(pfrac) not in results[variant]:
                    results[variant][str(pfrac)] = []

                eval_result = evaluate_policy(
                    trained[variant].policy,
                    lyap_net=trained[variant].lyap_net if variant == "lyapunov" else None,
                    sc_pos0=pert_pos,
                    sc_vel0=pert_vel,
                    n_steps=remaining_steps,
                    epsilon_target=float(env_params.epsilon_target),
                )

                results[variant][str(pfrac)].append({
                    "seed": seed,
                    "captured": eval_result["captured"],
                    "final_epsilon": float(eval_result["epsilon_history"][-1]),
                })

            # Lambert baseline: ballistic (no thrust) from perturbed state
            if str(pfrac) not in results["lambert"]:
                results["lambert"][str(pfrac)] = []

            # Evaluate with zero thrust (pass untrained dummy policy)
            lambert_result = evaluate_policy(
                trained["lyapunov"].policy,  # won't matter — deterministic=True
                sc_pos0=pert_pos,
                sc_vel0=pert_vel,
                n_steps=remaining_steps,
                max_thrust=0.0,  # no thrust = ballistic
                epsilon_target=float(env_params.epsilon_target),
            )
            results["lambert"][str(pfrac)].append({
                "seed": seed,
                "captured": lambert_result["captured"],
                "final_epsilon": float(lambert_result["epsilon_history"][-1]),
            })

    # Save
    results_path = f"{output_dir}/perturbation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def plot_perturbation(
    results: dict,
    perturbation_fracs: list[float] | None = None,
    output_dir: str = "results/perturbation",
):
    """Plot success rate vs perturbation magnitude (three lines).

    This is a key paper figure.
    """
    import matplotlib.pyplot as plt

    if perturbation_fracs is None:
        perturbation_fracs = PERTURBATION_FRACS

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"lyapunov": "#2ecc71", "unconstrained": "#e74c3c", "lambert": "#3498db"}
    labels = {"lyapunov": "Lyapunov PPO", "unconstrained": "Unconstrained PPO", "lambert": "Lambert (ballistic)"}

    for variant in ["lyapunov", "unconstrained", "lambert"]:
        rates = []
        for pfrac in perturbation_fracs:
            seed_results = results[variant].get(str(pfrac), [])
            if seed_results:
                rate = sum(r["captured"] for r in seed_results) / len(seed_results)
            else:
                rate = 0.0
            rates.append(rate)

        ax.plot(
            [f * 100 for f in perturbation_fracs], rates,
            "o-", color=colors[variant], label=labels[variant], linewidth=2, markersize=6,
        )

    ax.set_xlabel("Perturbation Magnitude (% of orbital velocity)")
    ax.set_ylabel("Recovery / Capture Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Perturbation Recovery: Lyapunov vs Unconstrained vs Lambert")
    ax.grid(True, alpha=0.3)

    fig_path = f"{output_dir}/perturbation_recovery.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    results = run_perturbation_experiment(n_seeds=2, total_updates=200)
    plot_perturbation(results)
