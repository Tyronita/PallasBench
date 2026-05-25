"""Level 3: Multi-Head Attention via Pallas.

Provenance: jax-ml/jax pallas/ops/tpu/flash_attention.py
             AI-Hypercomputer/maxtext splash attention training kernel
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _mha_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(jnp.float32(d_k))
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    o_ref[...] = weights @ v


def pallas_multi_head_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    n_heads, seq_len, d_head = q.shape

    return pl.pallas_call(
        _mha_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(n_heads,),
        in_specs=[
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
    )(q, k, v)


pallas_kernel = pallas_multi_head_attention
task_name = "multi_head_attention"
input_shapes = [(8, 256, 64), (8, 256, 64), (8, 256, 64)]
category = "attention"
level = 3
