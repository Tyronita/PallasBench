"""Level 1: Row-wise softmax via Pallas.

Demonstrates: reductions within a block, numerical stability (max subtraction),
multi-pass pattern (max -> subtract -> exp -> sum -> divide).
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/softmax", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    row_max = jnp.max(x, axis=-1, keepdims=True)
    x_safe = x - row_max
    exp_x = jnp.exp(x_safe)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    o_ref[...] = exp_x / sum_exp


def pallas_softmax(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_softmax
task_name = "softmax"
input_shapes = [(2048, 2048)]
category = "softmax"
level = 1
