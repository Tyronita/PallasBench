"""Level 1: Elementwise multiplication via Pallas.

Provenance: jnp.multiply, used in gating, scaling, and attention weights
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/multiply", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _multiply_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * y_ref[...]


def pallas_multiply(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _multiply_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
        ],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x, y)


pallas_kernel = pallas_multiply
task_name = "multiply"
input_shapes = [(4096, 4096), (4096, 4096)]
category = "elementwise"
level = 1
