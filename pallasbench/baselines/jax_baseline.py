"""Pure JAX reference implementations for PallasBench tasks.

Each function uses only jax.numpy / jax.lax / jax.nn — no Pallas.
These serve as correctness references and performance baselines.
"""

from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Level 1: Single operators
# ---------------------------------------------------------------------------

@jax.jit
def jax_relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(x, 0)


@jax.jit
def jax_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x)


@jax.jit
def jax_softmax(x: jax.Array) -> jax.Array:
    return jax.nn.softmax(x, axis=-1)


@jax.jit
def jax_layernorm(x: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


@jax.jit
def jax_rmsnorm(x: jax.Array) -> jax.Array:
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)


@jax.jit
def jax_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    return x @ y


@jax.jit
def jax_reduce_sum(x: jax.Array) -> jax.Array:
    return jnp.sum(x, axis=-1)


@jax.jit
def jax_embedding_lookup(table: jax.Array, indices: jax.Array) -> jax.Array:
    return table[indices]


# ---------------------------------------------------------------------------
# Level 2: Fusion patterns
# ---------------------------------------------------------------------------

@jax.jit
def jax_matmul_relu(x: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.maximum(x @ w, 0)


@jax.jit
def jax_rmsnorm_residual(x: jax.Array, residual: jax.Array) -> jax.Array:
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    normed = x / jnp.sqrt(ms + 1e-5)
    return normed + residual


@jax.jit
def jax_swiglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return gate * up


# ---------------------------------------------------------------------------
# Level 3: Architecture components
# ---------------------------------------------------------------------------

@jax.jit
def jax_flash_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v
