"""Experiment 1: Convergence speed comparison.

Lyapunov-augmented PPO vs unconstrained PPO (same architecture, no Lyapunov loss).
Measures: capture rate vs training steps, reward curves, fuel efficiency.

Expected result: Lyapunov-augmented reaches X% capture rate in Y% fewer steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.constants import EPSILON_MARS
from shared.obs import obs_dim
from train.agent.ppo import PPOHyperParams
from train.agent.train import train
from train.env.trajectory_env import EnvParams
from validate.bridge.policy_eval import evaluate_policy


def run_convergence_experiment(
    n_seeds: int = 5,
    n_envs: int = 32,
    n_steps: int = 128,
    total_updates: int = 500,
    eval_every: int = 50,
    output_dir: str = "results/convergence",
) -> dict:
    """Run convergence comparison between Lyapunov and unconstrained PPO.

    Trains both variants across multiple seeds, evaluating periodically
    in REBOUND to measure capture rate.

    Args:
        n_seeds: number of random seeds per variant
        n_envs: parallel environments
        n_steps: rollout length
        total_updates: total training updates
        eval_every: evaluate every N updates
        output_dir: directory for results

    Returns:
        dict with training curves and evaluation results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    env_params = EnvParams()
    n_planets = len(env_params.planet_indices)
    odim = obs_dim(n_planets)

    results = {"lyapunov": [], "unconstrained": []}

    for variant, lambda_lyap in [("lyapunov", 0.1), ("unconstrained", 0.0)]:
        print(f"\n=== Training {variant} (lambda={lambda_lyap}) ===")

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            hp = PPOHyperParams(lambda_lyap=lambda_lyap)

            # Train
            checkpoint_dir = f"{output_dir}/{variant}_seed{seed}"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

            train_state = train(
                n_envs=n_envs,
                n_steps=n_steps,
                total_updates=total_updates,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                hp=hp,
                env_params=env_params,
            )

            # Evaluate final policy in REBOUND
            eval_result = evaluate_policy(
                train_state.policy,
                lyap_net=train_state.lyap_net if lambda_lyap > 0 else None,
                epsilon_target=float(env_params.epsilon_target),
            )

            seed_result = {
                "seed": seed,
                "captured": eval_result["captured"],
                "captured_step": eval_result["captured_step"],
                "final_epsilon": float(eval_result["epsilon_history"][-1]),
                "fuel_used": float(eval_result["mass"][0] - eval_result["mass"][-1]),
            }
            results[variant].append(seed_result)
            print(f"  Captured: {seed_result['captured']}, "
                  f"Fuel: {seed_result['fuel_used']:.1f} kg")

    # Save results
    results_path = f"{output_dir}/convergence_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def plot_convergence(results: dict, output_dir: str = "results/convergence"):
    """Plot convergence comparison figure.

    Creates figure with:
      - Capture rate vs training steps
      - Mean reward curves
      - Fuel efficiency comparison
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Capture rate bar chart
    ax = axes[0]
    for i, variant in enumerate(["lyapunov", "unconstrained"]):
        captures = [r["captured"] for r in results[variant]]
        rate = sum(captures) / len(captures)
        ax.bar(i, rate, label=variant, color=["#2ecc71", "#e74c3c"][i])
    ax.set_ylabel("Capture Rate")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Lyapunov PPO", "Unconstrained PPO"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Capture Rate After Training")

    # Fuel usage
    ax = axes[1]
    for i, variant in enumerate(["lyapunov", "unconstrained"]):
        fuels = [r["fuel_used"] for r in results[variant]]
        ax.bar(i, np.mean(fuels), yerr=np.std(fuels),
               label=variant, color=["#2ecc71", "#e74c3c"][i], capsize=5)
    ax.set_ylabel("Fuel Used (kg)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Lyapunov PPO", "Unconstrained PPO"])
    ax.set_title("Fuel Efficiency")

    plt.tight_layout()
    fig_path = f"{output_dir}/convergence.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    results = run_convergence_experiment(n_seeds=3, total_updates=200)
    plot_convergence(results)
