"""Level 1: Elementwise addition via Pallas.

Provenance: jnp.add, fundamental binary op in residual connections
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/add", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def pallas_add(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
        ],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x, y)


pallas_kernel = pallas_add
task_name = "add"
input_shapes = [(4096, 4096), (4096, 4096)]
category = "elementwise"
level = 1
