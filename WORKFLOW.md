# Claude Code Workflow Log

## 2026-11-01 | Step 0: Setup
- Prompt used: "/plan" -> "Yes, do Step 0 only. For Step 1, 2 hidden layers of 32 and sin(x) is fine, but don't start Step 1 today."
- Files changed:
  - Created folders: src/, tests/, scripts/
  - pyproject.toml: added [tool.pytest.ini_options] pythonpath = ["src"] (CLAUDE.md said it existed; it was missing)
  - tests/test_env.py (new): imports, float32 default, jax.random.key + split, PRNG determinism
- Environment: JAX 0.12.0.1, Flax 0.12.12, optax 0.2.8, pytest 9.2.0, Python 3.12; jax.devices() = [CpuDevice(id=0)]
- Test result: `uv run pytest -q` -> 4 passed in 1.91s
- Claude did well: found the missing pytest config and fixed it
- I corrected: stopped it from starting Step 1 early
- Next (Step 1, not started): src/model.py MLP with 2 hidden layers of 32 units, synthetic sin(x) regression data
