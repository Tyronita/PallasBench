"""Level 1: Batched matrix multiplication via Pallas.

Provenance: jnp.matmul with batch dims, used in multi-head attention
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _batched_matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]


def pallas_batched_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    batch, m, k = x.shape
    _, _, n = y.shape

    return pl.pallas_call(
        _batched_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((batch, m, n), x.dtype),
        grid=(batch,),
        in_specs=[
            pl.BlockSpec((1, m, k), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, k, n), lambda b: (b, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, m, n), lambda b: (b, 0, 0)),
    )(x, y)


pallas_kernel = pallas_batched_matmul
task_name = "batched_matmul"
input_shapes = [(8, 256, 256), (8, 256, 256)]
category = "matmul"
level = 1
