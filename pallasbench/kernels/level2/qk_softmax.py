"""Level 2: Fused QK^T + Softmax via Pallas.

Provenance: jax-ml/jax flash_attention.py attention score computation
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/qk_softmax", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _qk_softmax_kernel(q_ref, k_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(jnp.float32(d_k))
    row_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - row_max)
    o_ref[...] = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)


def pallas_qk_softmax(q: jax.Array, k: jax.Array) -> jax.Array:
    seq_len, d_model = q.shape
    BLOCK_Q = min(seq_len, 128)
    grid_size = seq_len // BLOCK_Q

    return pl.pallas_call(
        _qk_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, seq_len), q.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_Q, seq_len), lambda i: (i, 0)),
    )(q, k)


pallas_kernel = pallas_qk_softmax
task_name = "qk_softmax"
input_shapes = [(256, 64), (256, 64)]
category = "attention_component"
level = 2
