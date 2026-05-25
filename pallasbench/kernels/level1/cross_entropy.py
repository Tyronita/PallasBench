"""Level 1: Row-wise cross-entropy loss via Pallas.

Provenance: openxla/tokamax linear_softmax_cross_entropy_loss pattern
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/cross_entropy", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _cross_entropy_kernel(logits_ref, labels_ref, o_ref):
    logits = logits_ref[...]
    labels = labels_ref[...]
    row_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - row_max
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    o_ref[...] = -jnp.sum(labels * log_probs, axis=-1)


def pallas_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    n_rows = logits.shape[0]
    n_cols = logits.shape[1]
    block_rows = min(128, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _cross_entropy_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), logits.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(logits, labels)


pallas_kernel = pallas_cross_entropy
task_name = "cross_entropy"
input_shapes = [(1024, 512), (1024, 512)]
category = "loss"
level = 1
