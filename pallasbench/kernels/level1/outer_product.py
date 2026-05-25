"""Level 1: Outer product via Pallas.

Provenance: jnp.outer, rank-1 update pattern used in Evoformer/AlphaFold
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/outer_product", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _outer_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = x[:, None] * y[None, :]


def pallas_outer_product(x: jax.Array, y: jax.Array) -> jax.Array:
    m = x.shape[0]
    n = y.shape[0]

    return pl.pallas_call(
        _outer_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m,), lambda i: (0,)),
            pl.BlockSpec((n,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((m, n), lambda i: (0, 0)),
    )(x, y)


pallas_kernel = pallas_outer_product
task_name = "outer_product"
input_shapes = [(1024,), (1024,)]
category = "matmul"
level = 1
