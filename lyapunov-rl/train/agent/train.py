"""Training loop for Lyapunov-augmented PPO.

Entry point for training. Handles:
  - Vectorized rollout collection (vmapped env)
  - GAE computation
  - PPO update epochs with minibatching
  - W&B logging
  - Orbax checkpoint saving

Usage:
    python -m train.agent.train [--n_envs 32] [--n_steps 128] [--total_updates 1000]
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx

from shared.constants import EPSILON_MARS
from shared.lyapunov import lyapunov_penalty
from shared.obs import obs_dim
from train.env.trajectory_env import EnvParams, EnvState, reset, step
from train.agent.networks import create_networks, sample_action
from train.agent.ppo import (
    Batch,
    PPOHyperParams,
    TrainState,
    compute_gae,
    create_train_state,
    update_step,
)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    train_state: TrainState,
    env_states: EnvState,
    env_params: EnvParams,
    key: jax.Array,
    n_steps: int,
    hp: PPOHyperParams = PPOHyperParams(),
) -> tuple[EnvState, Batch, dict]:
    """Collect a rollout of n_steps across all parallel environments.

    Args:
        train_state: current training state (networks)
        env_states: batched env states (n_envs,)
        env_params: env parameters
        key: PRNG key
        n_steps: rollout length
        hp: hyperparameters (needed for lambda_lyap)

    Returns:
        (final_env_states, batch, rollout_metrics)
    """
    policy = train_state.policy
    value_net = train_state.value_net
    n_envs = env_states.sc_pos.shape[0]

    # Pre-allocate storage
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_rewards = []
    all_dones = []
    all_values = []
    all_obs_next = []

    # Build initial obs
    # We need obs from current states — rebuild them
    obs = _batch_obs(env_states, env_params)

    for t in range(n_steps):
        key, k_act = jax.random.split(key)

        # Value estimate
        values = jax.vmap(value_net)(obs)

        # Sample actions
        act_keys = jax.random.split(k_act, n_envs)
        actions, log_probs = jax.vmap(sample_action, in_axes=(None, 0, 0))(
            policy, obs, act_keys
        )

        # Env step
        env_states, obs_next, rewards, dones, infos = jax.vmap(
            step, in_axes=(0, 0, None)
        )(env_states, actions, env_params)

        # Augment reward: subtract Lyapunov penalty (connects stability to policy)
        # r_aug = r_env - lambda * max(0, V(s') - V(s))
        lyap_penalties = jax.vmap(lyapunov_penalty, in_axes=(None, 0, 0))(
            train_state.lyap_net, obs, obs_next
        )
        rewards_aug = rewards - hp.lambda_lyap * lyap_penalties

        all_obs.append(obs)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_rewards.append(rewards_aug)
        all_dones.append(dones)
        all_values.append(values)
        all_obs_next.append(obs_next)

        # Auto-reset done envs
        key, k_reset = jax.random.split(key)
        env_states, obs_next = _auto_reset(env_states, obs_next, dones, k_reset, env_params)
        obs = obs_next

    # Stack into arrays: (n_steps, n_envs, ...)
    all_obs = jnp.stack(all_obs)
    all_actions = jnp.stack(all_actions)
    all_log_probs = jnp.stack(all_log_probs)
    all_rewards = jnp.stack(all_rewards)
    all_dones = jnp.stack(all_dones)
    all_values = jnp.stack(all_values)
    all_obs_next = jnp.stack(all_obs_next)

    # Bootstrap value for last step
    final_obs = _batch_obs(env_states, env_params)
    bootstrap_values = jax.vmap(value_net)(final_obs)
    # Append bootstrap: (n_steps+1, n_envs)
    all_values_extended = jnp.concatenate([all_values, bootstrap_values[None]], axis=0)

    # GAE per environment, then flatten
    def gae_single_env(rewards, values, dones):
        return compute_gae(rewards, values, dones)

    advantages, returns = jax.vmap(gae_single_env, in_axes=(1, 1, 1), out_axes=1)(
        all_rewards, all_values_extended, all_dones
    )

    # Flatten (n_steps, n_envs) -> (n_steps * n_envs,)
    def flatten(x):
        return x.reshape(-1, *x.shape[2:])

    batch = Batch(
        obs=flatten(all_obs),
        actions=flatten(all_actions),
        log_probs_old=flatten(all_log_probs),
        returns=flatten(returns),
        advantages=flatten(advantages),
        obs_next=flatten(all_obs_next),
    )

    rollout_metrics = {
        "rollout/mean_reward": jnp.mean(all_rewards),
        "rollout/mean_episode_return": jnp.mean(jnp.sum(all_rewards, axis=0)),
        "rollout/done_frac": jnp.mean(all_dones),
        "rollout/reward_std": jnp.std(all_rewards),
        "rollout/reward_min": jnp.min(all_rewards),
        "rollout/reward_max": jnp.max(all_rewards),
        "rollout/mean_advantage": jnp.mean(advantages),
        "rollout/std_advantage": jnp.std(advantages),
        "rollout/mean_return": jnp.mean(returns),
    }

    return env_states, batch, rollout_metrics


def _batch_obs(env_states: EnvState, env_params: EnvParams) -> jnp.ndarray:
    """Rebuild observation vectors from batched env states.

    This is needed because env states don't store the obs directly.
    """
    from shared.obs import build_obs
    from shared.constants import PLANET_A, PLANET_MU
    from train.env.trajectory_env import planet_positions_at_time, planet_velocities_at_time

    planet_idx = jnp.array(env_params.planet_indices)
    planet_a = PLANET_A[planet_idx]
    planet_mus = PLANET_MU[planet_idx]

    def single_obs(sc_pos, sc_vel, planet_phases, t):
        p_pos = planet_positions_at_time(planet_a, planet_phases, t)
        p_vel = planet_velocities_at_time(planet_a, planet_phases, t)
        target_pos = p_pos[-1]
        target_vel = p_vel[-1]
        return build_obs(sc_pos, sc_vel, target_pos, target_vel, p_pos)

    return jax.vmap(single_obs)(
        env_states.sc_pos, env_states.sc_vel,
        env_states.planet_phases, env_states.time,
    )


def _auto_reset(
    env_states: EnvState,
    obs: jnp.ndarray,
    dones: jnp.ndarray,
    key: jax.Array,
    env_params: EnvParams,
) -> tuple[EnvState, jnp.ndarray]:
    """Reset environments that are done, keep others unchanged."""
    n_envs = dones.shape[0]
    keys = jax.random.split(key, n_envs)

    fresh_states, fresh_obs = jax.vmap(reset, in_axes=(0, None))(keys, env_params)

    # Select fresh or existing based on done flag
    def select(fresh, current, done):
        return jnp.where(done, fresh, current)

    new_states = jax.tree.map(
        lambda f, c: jax.vmap(select)(f, c, dones), fresh_states, env_states
    )
    new_obs = jax.vmap(select)(fresh_obs, obs, dones)

    return new_states, new_obs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_envs: int = 32,
    n_steps: int = 128,
    total_updates: int = 1000,
    seed: int = 0,
    use_wandb: bool = False,
    checkpoint_dir: str | None = None,
    hp: PPOHyperParams = PPOHyperParams(),
    env_params: EnvParams = EnvParams(),
):
    """Main training loop.

    Args:
        n_envs: number of parallel environments
        n_steps: rollout length per update
        total_updates: total number of PPO updates
        seed: random seed
        use_wandb: whether to log to W&B
        checkpoint_dir: directory for Orbax checkpoints (None = no saving)
        hp: PPO hyperparameters
        env_params: environment parameters
    """
    key = jax.random.PRNGKey(seed)

    # W&B
    if use_wandb:
        import wandb
        wandb.init(
            project="lyapunov-rl",
            config={
                "n_envs": n_envs,
                "n_steps": n_steps,
                "total_updates": total_updates,
                **hp._asdict(),
            },
        )

    # Initialize
    n_planets = len(env_params.planet_indices)
    odim = obs_dim(n_planets)
    print(f"Obs dim: {odim}, N_planets: {n_planets}, N_envs: {n_envs}")

    key, k_net, k_env = jax.random.split(key, 3)
    policy, value_net, lyap_net = create_networks(
        odim, epsilon_target=env_params.epsilon_target, key=k_net
    )
    train_state, optimizers = create_train_state(policy, value_net, lyap_net, hp)

    # Initialize envs
    env_keys = jax.random.split(k_env, n_envs)
    env_states, _ = jax.vmap(reset, in_axes=(0, None))(env_keys, env_params)

    # Checkpoint setup
    if checkpoint_dir is not None:
        import orbax.checkpoint as ocp
        checkpointer = ocp.StandardCheckpointer()

    print(f"Starting training: {total_updates} updates, {n_envs} envs, {n_steps} steps/rollout")
    print(f"Lyapunov lambda: {hp.lambda_lyap}, clip_eps: {hp.clip_eps}, lr: {hp.lr}")

    t_start = time.time()

    for update in range(total_updates):
        key, k_rollout, k_update = jax.random.split(key, 3)

        # Collect rollout (with Lyapunov-augmented rewards)
        env_states, batch, rollout_metrics = collect_rollout(
            train_state, env_states, env_params, k_rollout, n_steps, hp,
        )

        # PPO update epochs
        batch_size = n_envs * n_steps
        minibatch_size = batch_size // hp.n_minibatches

        for epoch in range(hp.n_epochs):
            k_update, k_perm = jax.random.split(k_update)
            perm = jax.random.permutation(k_perm, batch_size)

            for mb in range(hp.n_minibatches):
                mb_idx = perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                minibatch = Batch(
                    obs=batch.obs[mb_idx],
                    actions=batch.actions[mb_idx],
                    log_probs_old=batch.log_probs_old[mb_idx],
                    returns=batch.returns[mb_idx],
                    advantages=batch.advantages[mb_idx],
                    obs_next=batch.obs_next[mb_idx],
                )
                train_state, update_metrics = update_step(
                    train_state, minibatch, optimizers, hp,
                )

        # Logging
        if update % 10 == 0:
            elapsed = time.time() - t_start
            sps = (update + 1) * n_envs * n_steps / elapsed
            r_mean = float(rollout_metrics['rollout/mean_reward'])
            r_min = float(rollout_metrics['rollout/reward_min'])
            r_max = float(rollout_metrics['rollout/reward_max'])
            adv_std = float(rollout_metrics['rollout/std_advantage'])
            l_pol = float(update_metrics['loss/policy'])
            l_val = float(update_metrics['loss/value'])
            l_lyap = float(update_metrics['loss/lyapunov'])
            entropy = float(update_metrics['loss/entropy'])
            log_str = (
                f"[{update:4d}/{total_updates}] "
                f"r={r_mean:+.4f} [{r_min:+.3f},{r_max:+.3f}] "
                f"adv_std={adv_std:.3f} "
                f"pol={l_pol:.3f} val={l_val:.3f} lyap={l_lyap:.5f} "
                f"ent={entropy:.2f} SPS={sps:.0f}"
            )
            print(log_str)

            if use_wandb:
                import wandb
                wandb.log({
                    **{k: float(v) for k, v in rollout_metrics.items()},
                    **{k: float(v) for k, v in update_metrics.items()},
                    "perf/sps": sps,
                    "perf/update": update,
                })

        # Checkpoint
        if checkpoint_dir is not None and (update + 1) % 100 == 0:
            ckpt_path = f"{checkpoint_dir}/step_{update + 1}"
            ckpt_data = {
                "policy": eqx.tree_serialise_leaves(None, train_state.policy),
                "value": eqx.tree_serialise_leaves(None, train_state.value_net),
                "lyapunov": eqx.tree_serialise_leaves(None, train_state.lyap_net),
            }
            # For now, just save with eqx directly
            eqx.tree_serialise_leaves(f"{ckpt_path}_policy.eqx", train_state.policy)
            eqx.tree_serialise_leaves(f"{ckpt_path}_value.eqx", train_state.value_net)
            eqx.tree_serialise_leaves(f"{ckpt_path}_lyapunov.eqx", train_state.lyap_net)
            print(f"  Saved checkpoint to {ckpt_path}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete. {total_updates} updates in {elapsed:.1f}s")
    print(f"Average SPS: {total_updates * n_envs * n_steps / elapsed:.0f}")

    if use_wandb:
        import wandb
        wandb.finish()

    return train_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Lyapunov-PPO agent")
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_lyap", type=float, default=0.1)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    args = parser.parse_args()

    hp = PPOHyperParams(
        lr=args.lr,
        lambda_lyap=args.lambda_lyap,
        clip_eps=args.clip_eps,
    )

    train(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_updates=args.total_updates,
        seed=args.seed,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
        hp=hp,
    )


if __name__ == "__main__":
    main()
