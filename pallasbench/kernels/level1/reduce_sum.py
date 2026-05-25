"""Level 1: Row-wise sum reduction via Pallas.

Demonstrates: reduction along an axis, row-parallel BlockSpec.
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/reduce_sum", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _reduce_sum_kernel(x_ref, o_ref):
    o_ref[...] = jnp.sum(x_ref[...], axis=-1)


def pallas_reduce_sum(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    block_rows = min(256, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _reduce_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(x)


pallas_kernel = pallas_reduce_sum
task_name = "reduce_sum"
input_shapes = [(4096, 2048)]
category = "reduce"
level = 1
