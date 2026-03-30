# Project Memory — Code Repo

Project: Lyapunov RL Trajectory Optimization
Code repo: ~/Documents/lyapunov-rl
Research vault: ~/Documents/obsidian-trajectory/obsidian-trajectory

## Current Position

Build complete + training fixes applied. Colab-ready.
Next: run notebooks/train_colab.ipynb on T4 with 2048 envs.

Training confirmed working on Mac (300 updates, 32 envs):
  - Reward: -0.068 → -0.014 (improving)
  - Entropy: 2.75 → 0.66 (converging)
  - Policy loss: stable < 0.1 after update 200

Three training bugs fixed (2026-03-29):
  1. Value net output bias (0.05 → -1.2 at init, matching return targets)
  2. Lyapunov penalty applied as reward shaping during rollout (not isolated to lyap_net only)
  3. log_std clamped to [-3, 0] to prevent entropy divergence

Colab artifacts:
  - notebooks/train_colab.ipynb — full training notebook
  - requirements.txt — pinned dependency versions

## File Structure

```
lyapunov-rl/
├── shared/
│   ├── constants.py       ✅ Physical constants, unit conversions
│   ├── obs.py             ✅ Observation vector construction (18 + N_planet dims)
│   ├── reward.py          ✅ Reward function
│   ├── lyapunov.py        ✅ V(s) = (ε-ε_target)² + softplus(NN(s)), spectral norm
│   └── certify.py         ✅ Lipschitz bounds, grid verification (Tier 2)
├── train/
│   ├── sim/
│   │   ├── integrator.py  ✅ JAX leapfrog (cross-validated vs REBOUND)
│   │   └── gravity.py     ✅ N-body forces in pure JAX (vmapped)
│   ├── env/
│   │   └── trajectory_env.py  ✅ Gymnax-style, vmapped, JIT'd
│   └── agent/
│       ├── networks.py    ✅ Policy 2x256, Value 2x256, Lyapunov 2x128
│       ├── ppo.py         ✅ PPO-clip + Lyapunov penalty + entropy
│       └── train.py       ✅ Training loop, W&B, checkpoints
├── validate/
│   ├── sim/
│   │   ├── rebound_sim.py ✅ REBOUND MERCURIUS reference simulation
│   │   └── forces.py      ✅ Force validation
│   ├── bridge/
│   │   └── policy_eval.py ✅ Load policy, run in REBOUND, track V(s)
│   └── experiments/
│       ├── convergence.py ✅ Exp 1: Lyapunov vs unconstrained convergence
│       ├── transfer.py    ✅ Exp 2: Zero-shot Mars→Venus/Jupiter
│       └── perturbation.py ✅ Exp 3: Recovery after out-of-plane kick
├── CC-Session-Logs/
└── CLAUDE.md
```

## Code Conventions

- Pure JAX everywhere in train/ — no numpy, no side effects, everything must vmap
- Equinox modules for all networks — jax.grad(net)(state) must work
- Spectral normalization on Lyapunov net layers from the start
- step() returns V(s) and V(s') alongside (next_state, obs, reward, done)
- REBOUND is validation only — never in the training loop
- All functions take explicit state, no globals

## Dependencies

jax, jaxlib, equinox, optax, orbax-checkpoint, rebound (local only), numpy, matplotlib, wandb

## Dev Environment

- M4 Mac: dev, debug, small test runs (8-32 envs, 500-2k steps/sec), REBOUND validation
- Colab Pro T4: real training (2048 envs, 50-200k steps/sec), HP sweeps
- Orbax checkpoints save to Google Drive from Colab, pull to M4

## Key Decisions

<!-- /preserve appends here: YYYY-MM-DD | decision | reasoning -->

## Pending Tasks

<!-- /compress appends here -->
