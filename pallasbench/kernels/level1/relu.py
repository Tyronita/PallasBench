"""Level 1: Elementwise ReLU via Pallas.

Demonstrates: basic pallas_call, grid, BlockSpec, program_id.
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/relu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _relu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.maximum(x, 0)


def pallas_relu(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_relu
task_name = "relu"
input_shapes = [(4096, 4096)]
category = "activation"
level = 1
