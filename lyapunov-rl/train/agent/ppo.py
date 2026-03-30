"""Custom PPO with Lyapunov stability penalty.

Standard PPO-clip loss augmented with:
    L_total = L_policy + c_vf * L_value + lambda_lyap * L_lyapunov - c_ent * H

where L_lyapunov = mean(max(0, V(s') - V(s))) penalizes Lyapunov increase.

All functions are pure JAX. Training state is an explicit pytree.
Uses optax for optimizer state management.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from shared.lyapunov import LyapunovNet, lyapunov_penalty
from train.agent.networks import PolicyNet, ValueNet, action_log_prob


# ---------------------------------------------------------------------------
# Trajectory batch (collected from rollouts)
# ---------------------------------------------------------------------------

class Batch(NamedTuple):
    """A batch of transitions for PPO update."""
    obs: jnp.ndarray           # (T, obs_dim)
    actions: jnp.ndarray       # (T, action_dim)
    log_probs_old: jnp.ndarray # (T,) log probs under collection policy
    returns: jnp.ndarray       # (T,) discounted returns (GAE-based)
    advantages: jnp.ndarray    # (T,) GAE advantages
    obs_next: jnp.ndarray      # (T, obs_dim) for Lyapunov delta


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation.

    Args:
        rewards: (T,) per-step rewards
        values: (T+1,) value estimates (last entry is bootstrap value)
        dones: (T,) episode termination flags
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (advantages (T,), returns (T,))
    """
    T = rewards.shape[0]

    def scan_fn(gae, t):
        # Scan backwards from T-1 to 0
        idx = T - 1 - t
        delta = rewards[idx] + gamma * values[idx + 1] * (1.0 - dones[idx]) - values[idx]
        gae = delta + gamma * lam * (1.0 - dones[idx]) * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(scan_fn, jnp.array(0.0), jnp.arange(T))
    advantages = jnp.flip(advantages_rev)
    returns = advantages + values[:-1]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss functions
# ---------------------------------------------------------------------------

def policy_loss(
    policy: PolicyNet,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    log_probs_old: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_eps: float = 0.2,
) -> jnp.ndarray:
    """Clipped PPO policy loss (negative because we minimize).

    Args:
        policy: current policy network
        obs: (T, obs_dim)
        actions: (T, action_dim)
        log_probs_old: (T,) log probs from rollout policy
        advantages: (T,) normalized advantages
        clip_eps: PPO clipping epsilon

    Returns:
        Scalar policy loss
    """
    # Compute new log probs for all transitions
    log_probs_new = jax.vmap(action_log_prob, in_axes=(None, 0, 0))(
        policy, obs, actions
    )
    ratio = jnp.exp(log_probs_new - log_probs_old)

    # Normalize advantages
    adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Clipped surrogate
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -jnp.mean(jnp.minimum(surr1, surr2))


def value_loss(
    value_net: ValueNet,
    obs: jnp.ndarray,
    returns: jnp.ndarray,
) -> jnp.ndarray:
    """Value function MSE loss.

    Args:
        value_net: value network
        obs: (T, obs_dim)
        returns: (T,) target returns

    Returns:
        Scalar MSE loss
    """
    values = jax.vmap(value_net)(obs)
    return jnp.mean((values - returns) ** 2)


def lyapunov_loss(
    lyap_net: LyapunovNet,
    obs: jnp.ndarray,
    obs_next: jnp.ndarray,
) -> jnp.ndarray:
    """Lyapunov stability penalty: mean(max(0, V(s') - V(s))).

    Args:
        lyap_net: Lyapunov network
        obs: (T, obs_dim)
        obs_next: (T, obs_dim)

    Returns:
        Scalar penalty >= 0
    """
    penalties = jax.vmap(lyapunov_penalty, in_axes=(None, 0, 0))(
        lyap_net, obs, obs_next
    )
    return jnp.mean(penalties)


def entropy_bonus(
    policy: PolicyNet,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian entropy bonus for exploration.

    H = 0.5 * sum(log(2*pi*e*std^2)) = 0.5 * sum(1 + log(2*pi) + 2*log_std)

    Args:
        policy: policy network
        obs: (T, obs_dim) — only used to get log_std (which is state-independent)

    Returns:
        Scalar mean entropy
    """
    # log_std is a parameter, not state-dependent, but call forward for consistency
    _, log_std = policy(obs[0])
    entropy = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std)
    return entropy


# ---------------------------------------------------------------------------
# Combined loss and update step
# ---------------------------------------------------------------------------

class PPOHyperParams(NamedTuple):
    """PPO hyperparameters."""
    clip_eps: float = 0.2
    c_vf: float = 1.0           # value loss coefficient (high — value must learn fast)
    c_ent: float = 0.001        # entropy bonus coefficient (low — log_std handles exploration)
    lambda_lyap: float = 0.1    # Lyapunov penalty coefficient
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 3           # PPO epochs per rollout
    n_minibatches: int = 4      # minibatches per epoch


def total_loss(
    policy: PolicyNet,
    value_net: ValueNet,
    lyap_net: LyapunovNet,
    batch: Batch,
    hp: PPOHyperParams,
) -> tuple[jnp.ndarray, dict]:
    """Combined PPO + Lyapunov loss.

    Returns:
        (scalar loss, dict of component losses for logging)
    """
    l_policy = policy_loss(
        policy, batch.obs, batch.actions, batch.log_probs_old,
        batch.advantages, hp.clip_eps,
    )
    l_value = value_loss(value_net, batch.obs, batch.returns)
    l_lyap = lyapunov_loss(lyap_net, batch.obs, batch.obs_next)
    h = entropy_bonus(policy, batch.obs)

    loss = l_policy + hp.c_vf * l_value + hp.lambda_lyap * l_lyap - hp.c_ent * h

    metrics = {
        "loss/total": loss,
        "loss/policy": l_policy,
        "loss/value": l_value,
        "loss/lyapunov": l_lyap,
        "loss/entropy": h,
    }
    return loss, metrics


class TrainState(NamedTuple):
    """Mutable training state — passed through update steps."""
    policy: PolicyNet
    value_net: ValueNet
    lyap_net: LyapunovNet
    opt_state_policy: optax.OptState
    opt_state_value: optax.OptState
    opt_state_lyap: optax.OptState


def create_train_state(
    policy: PolicyNet,
    value_net: ValueNet,
    lyap_net: LyapunovNet,
    hp: PPOHyperParams = PPOHyperParams(),
) -> tuple[TrainState, tuple]:
    """Initialize training state with separate optimizers for each network.

    Returns:
        (train_state, (opt_policy, opt_value, opt_lyap)) — optimizers needed for updates
    """
    opt_policy = optax.chain(
        optax.clip_by_global_norm(hp.max_grad_norm),
        optax.adam(hp.lr),
    )
    opt_value = optax.chain(
        optax.clip_by_global_norm(hp.max_grad_norm),
        optax.adam(hp.lr),
    )
    opt_lyap = optax.chain(
        optax.clip_by_global_norm(hp.max_grad_norm),
        optax.adam(hp.lr),
    )

    state = TrainState(
        policy=policy,
        value_net=value_net,
        lyap_net=lyap_net,
        opt_state_policy=opt_policy.init(eqx.filter(policy, eqx.is_array)),
        opt_state_value=opt_value.init(eqx.filter(value_net, eqx.is_array)),
        opt_state_lyap=opt_lyap.init(eqx.filter(lyap_net, eqx.is_array)),
    )
    return state, (opt_policy, opt_value, opt_lyap)


def update_step(
    train_state: TrainState,
    batch: Batch,
    optimizers: tuple,
    hp: PPOHyperParams = PPOHyperParams(),
) -> tuple[TrainState, dict]:
    """One PPO update step across all three networks.

    Computes gradients of the total loss w.r.t. each network separately,
    then applies optimizer updates.

    Args:
        train_state: current training state
        batch: transition batch
        optimizers: (opt_policy, opt_value, opt_lyap)
        hp: hyperparameters

    Returns:
        (updated_train_state, metrics_dict)
    """
    policy, value_net, lyap_net = train_state.policy, train_state.value_net, train_state.lyap_net
    opt_policy, opt_value, opt_lyap = optimizers

    # --- Policy gradient ---
    @eqx.filter_grad
    def policy_grad_fn(p):
        l_p = policy_loss(p, batch.obs, batch.actions, batch.log_probs_old,
                          batch.advantages, hp.clip_eps)
        h = entropy_bonus(p, batch.obs)
        return l_p - hp.c_ent * h

    grads_policy = policy_grad_fn(policy)
    updates_p, new_opt_p = opt_policy.update(
        grads_policy, train_state.opt_state_policy, eqx.filter(policy, eqx.is_array)
    )
    new_policy = eqx.apply_updates(policy, updates_p)

    # --- Value gradient ---
    @eqx.filter_grad
    def value_grad_fn(v):
        return value_loss(v, batch.obs, batch.returns)

    grads_value = value_grad_fn(value_net)
    updates_v, new_opt_v = opt_value.update(
        grads_value, train_state.opt_state_value, eqx.filter(value_net, eqx.is_array)
    )
    new_value = eqx.apply_updates(value_net, updates_v)

    # --- Lyapunov gradient ---
    @eqx.filter_grad
    def lyap_grad_fn(l):
        return lyapunov_loss(l, batch.obs, batch.obs_next)

    grads_lyap = lyap_grad_fn(lyap_net)
    updates_l, new_opt_l = opt_lyap.update(
        grads_lyap, train_state.opt_state_lyap, eqx.filter(lyap_net, eqx.is_array)
    )
    new_lyap = eqx.apply_updates(lyap_net, updates_l)

    # Compute metrics for logging
    _, metrics = total_loss(new_policy, new_value, new_lyap, batch, hp)

    new_state = TrainState(
        policy=new_policy,
        value_net=new_value,
        lyap_net=new_lyap,
        opt_state_policy=new_opt_p,
        opt_state_value=new_opt_v,
        opt_state_lyap=new_opt_l,
    )
    return new_state, metrics
