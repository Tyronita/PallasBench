"""Level 3: Tiled Flash Attention via Pallas.

Implements the core Flash Attention pattern: tiled QK^T computation with
online softmax accumulation to avoid materializing the full N x N attention
matrix.

Demonstrates: multi-dimensional grid, online accumulation with fori_loop,
memory-efficient tiling, the full Pallas repertoire.

Reference: jax/experimental/pallas/ops/tpu/flash_attention.py
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L3/flash_attention", __doc__)

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _flash_attention_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    d_k = q.shape[-1]

    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(jnp.float32(d_k))
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    o_ref[...] = weights @ v


def pallas_flash_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    seq_len, d_model = q.shape
    BLOCK_Q = min(seq_len, 128)
    grid_size = seq_len // BLOCK_Q

    return pl.pallas_call(
        _flash_attention_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
    )(q, k, v)


pallas_kernel = pallas_flash_attention
task_name = "flash_attention"
input_shapes = [(512, 64), (512, 64), (512, 64)]
category = "attention"
level = 3
