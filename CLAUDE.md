# Project: Parallel MLP training with JAX vmap

## Goal
Train hundreds of small MLPs at once (different seeds / learning rates)
with vmap + jit. Benchmark against sequential training.
Portfolio project. I am new to JAX.

## Environment
- Python 3.12, uv project: pyproject.toml + uv.lock are the source of truth
- Run anything: uv run python <file>
- Test: uv run pytest -q
- Add a package: uv add <pkg> (ask me first). Never use uv pip install.
  Never run uv sync --exact or use --no-project
- pyproject.toml has [tool.pytest.ini_options] pythonpath = ["src"]

## Stack
- JAX, Flax NNX (models), optax (optimizers), pytest, matplotlib
- No PyTorch, no Flax Linen, no Haiku
- float32 everywhere; do not enable float64

## JAX rules
- Use jax.jit and jax.vmap only. Do not use nnx.jit, nnx.vmap, or nnx.Optimizer
- Never pass an nnx.Module into jax.jit or jax.vmap. Split it first:
  graphdef, params = nnx.split(model, nnx.Param)
- Pass params (a pytree) through jitted functions. nnx.merge(graphdef, params)
  inside them is fine, e.g. inside the loss function
- Create models inside the vmapped init function from a key:
  nnx.split(MLP(nnx.Rngs(key)))
- Optimizer: plain optax. opt_state = tx.init(params); tx.update;
  optax.apply_updates. For learning-rate sweeps use
  optax.inject_hyperparams(optax.adam)
- Jitted and vmapped functions must be pure: output depends only on inputs
- PRNG: use jax.random.key (not PRNGKey). Per-model keys via
  jax.random.split(key, n). Never reuse a key
- No Python if/for on traced values; use jax.lax.cond / scan / fori_loop.
  Python for over static ranges is fine (unrolls at trace time).
  Use lax.scan for the training loop
- print runs once at trace time; use jax.debug.print for runtime values

## Testing
- Every function in src/ that does numerical work (model, loss, train step,
  vmapped train) needs a test: shapes, dtypes, determinism with fixed key
- Run uv run pytest -q after every code change and show me the result
- Creating new tests is fine. Changing or deleting an existing test needs
  my approval: explain why and wait
- Key test: vmapped vs one-by-one training, same keys, about 50 steps.
  Compare per-step losses and final params leaf by leaf with
  jax.tree.map + jnp.allclose(rtol=1e-4, atol=1e-5)

## Benchmarking
- Sequential baseline: the same jitted single-model step, called in a
  Python loop. Never compare against un-jitted code
- Warm up once (compile), then 5+ timed runs with time.perf_counter;
  report median and spread. Report compile time separately for both
- Call jax.block_until_ready(out) on the full output before stopping the timer
- Report models/sec. Record CPU/GPU model, JAX version, jax.devices()
- Don't change XLA thread flags unless asked

## How to work with me
- I am learning. Before writing code, explain the plan in 3 to 5 lines and wait
- One small step at a time; don't build ahead of the current step
- Add short comments explaining JAX concepts where they first appear
- Ask before installing packages, creating top-level folders, or any git
  commit or push
- After each step, append to WORKFLOW.md: date, step, prompt used, files
  changed, test result. Do this before saying the step is done

## Layout
- src/      code
- tests/    tests
- scripts/  benchmark and plot scripts

## Progress
Read WORKFLOW.md first to see the current step.
