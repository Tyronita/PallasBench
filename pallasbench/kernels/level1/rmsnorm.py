"""Level 1: RMS normalization via Pallas.

Demonstrates: squared-mean reduction, rsqrt pattern.
Inspired by pallas-forge's RMSNorm kernel (3.44x over XLA).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _rmsnorm_kernel(x_ref, o_ref):
    x = x_ref[...]
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    o_ref[...] = x * jnp.rsqrt(ms + 1e-5)


def pallas_rmsnorm(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    block_rows = min(128, n_rows)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_rmsnorm
task_name = "rmsnorm"
input_shapes = [(2048, 1024)]
category = "normalization"
level = 1
