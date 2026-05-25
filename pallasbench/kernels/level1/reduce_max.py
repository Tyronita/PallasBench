"""Level 1: Row-wise max reduction via Pallas.

Provenance: jnp.max reduction, used in softmax numerics and argmax patterns
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _reduce_max_kernel(x_ref, o_ref):
    o_ref[...] = jnp.max(x_ref[...], axis=-1)


def pallas_reduce_max(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    block_rows = min(256, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _reduce_max_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(x)


pallas_kernel = pallas_reduce_max
task_name = "reduce_max"
input_shapes = [(4096, 2048)]
category = "reduce"
level = 1
