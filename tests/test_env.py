"""Step 0 sanity checks: the JAX stack imports and behaves as CLAUDE.md expects."""

import jax
import jax.numpy as jnp


def test_imports():
    # Flax NNX (models) and optax (optimizers) must be importable.
    import optax  # noqa: F401
    from flax import nnx  # noqa: F401


def test_default_dtype_is_float32():
    # JAX defaults to float32 unless x64 is enabled; we never enable it.
    assert jnp.ones(3).dtype == jnp.float32
    assert not jax.config.jax_enable_x64


def test_new_style_prng_key():
    # jax.random.key returns a typed key array (preferred over PRNGKey).
    key = jax.random.key(0)
    assert jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    # split(key, n) gives n independent keys, e.g. one per model.
    keys = jax.random.split(key, 4)
    assert keys.shape == (4,)


def test_prng_is_deterministic():
    # Same key -> same random numbers. Randomness is explicit and reproducible.
    a = jax.random.normal(jax.random.key(42), (5,))
    b = jax.random.normal(jax.random.key(42), (5,))
    assert jnp.array_equal(a, b)
