"""Level 1: Layer normalization via Pallas.

Demonstrates: mean/variance reduction, epsilon stability, row-parallel tiling.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _layernorm_kernel(x_ref, o_ref):
    x = x_ref[...]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    o_ref[...] = (x - mean) / jnp.sqrt(var + 1e-5)


def pallas_layernorm(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    block_rows = min(128, n_rows)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _layernorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_layernorm
task_name = "layernorm"
input_shapes = [(2048, 1024)]
category = "normalization"
level = 1
