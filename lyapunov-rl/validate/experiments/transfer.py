"""Experiment 2: Zero-shot transfer across target planets.

Train on Earth->Mars, test on Earth->Venus and Earth->Jupiter
without retraining. Lyapunov policy should generalize because V(s)
encodes orbital energy structure, not route memory.

Expected result: Lyapunov policy transfers; unconstrained does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.constants import EPSILON_JUPITER, EPSILON_MARS, EPSILON_VENUS
from shared.obs import obs_dim
from train.agent.ppo import PPOHyperParams
from train.agent.train import train
from train.env.trajectory_env import EnvParams
from validate.bridge.policy_eval import evaluate_policy


TRANSFER_TARGETS = {
    "mars": {
        "epsilon_target": EPSILON_MARS,
        "target_a": 2.279e11,
        "planets": ("venus", "earth", "mars"),
    },
    "venus": {
        "epsilon_target": EPSILON_VENUS,
        "target_a": 1.082e11,
        "planets": ("venus", "earth", "mars"),
    },
    "jupiter": {
        "epsilon_target": EPSILON_JUPITER,
        "target_a": 7.785e11,
        "planets": ("venus", "earth", "mars", "jupiter"),
    },
}


def run_transfer_experiment(
    n_seeds: int = 3,
    n_envs: int = 32,
    n_steps: int = 128,
    total_updates: int = 500,
    output_dir: str = "results/transfer",
) -> dict:
    """Train on Mars, evaluate on Mars/Venus/Jupiter.

    Args:
        n_seeds: random seeds per variant
        n_envs: parallel environments
        n_steps: rollout length
        total_updates: training updates
        output_dir: results directory

    Returns:
        Nested dict: variant -> target -> [seed results]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    env_params = EnvParams()  # trains on Mars by default
    n_planets = len(env_params.planet_indices)
    odim = obs_dim(n_planets)

    results = {"lyapunov": {}, "unconstrained": {}}

    for variant, lambda_lyap in [("lyapunov", 0.1), ("unconstrained", 0.0)]:
        print(f"\n=== Training {variant} on Earth->Mars ===")

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            hp = PPOHyperParams(lambda_lyap=lambda_lyap)

            train_state = train(
                n_envs=n_envs,
                n_steps=n_steps,
                total_updates=total_updates,
                seed=seed,
                hp=hp,
                env_params=env_params,
            )

            # Evaluate on each target
            for target_name, target_cfg in TRANSFER_TARGETS.items():
                if target_name not in results[variant]:
                    results[variant][target_name] = []

                # For Jupiter, need more steps (longer transfer)
                eval_steps = 9600 if target_name != "jupiter" else 19200

                eval_result = evaluate_policy(
                    train_state.policy,
                    lyap_net=train_state.lyap_net if lambda_lyap > 0 else None,
                    planets=target_cfg["planets"],
                    n_steps=eval_steps,
                    epsilon_target=float(target_cfg["epsilon_target"]),
                )

                seed_result = {
                    "seed": seed,
                    "captured": eval_result["captured"],
                    "captured_step": eval_result["captured_step"],
                    "final_epsilon": float(eval_result["epsilon_history"][-1]),
                    "epsilon_target": float(target_cfg["epsilon_target"]),
                    "fuel_used": float(eval_result["mass"][0] - eval_result["mass"][-1]),
                }
                results[variant][target_name].append(seed_result)
                print(f"  {target_name}: captured={seed_result['captured']}")

    # Save
    results_path = f"{output_dir}/transfer_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def plot_transfer(results: dict, output_dir: str = "results/transfer"):
    """Plot transfer comparison: capture rate per target per variant."""
    import matplotlib.pyplot as plt

    targets = list(TRANSFER_TARGETS.keys())
    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (variant, color) in enumerate([("lyapunov", "#2ecc71"), ("unconstrained", "#e74c3c")]):
        rates = []
        for t in targets:
            captures = [r["captured"] for r in results[variant][t]]
            rates.append(sum(captures) / len(captures) if captures else 0)
        ax.bar(x + i * width, rates, width, label=variant.capitalize(), color=color)

    ax.set_ylabel("Capture Rate")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([t.capitalize() for t in targets])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Zero-Shot Transfer: Train on Mars, Test on Others")

    fig_path = f"{output_dir}/transfer.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    results = run_transfer_experiment(n_seeds=2, total_updates=200)
    plot_transfer(results)
