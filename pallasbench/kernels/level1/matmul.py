"""Level 1: Tiled matrix multiplication via Pallas.

Demonstrates: 2D grid, BlockSpec with K-dimension accumulation,
multi-block tiling pattern from the Pallas quickstart.
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/matmul", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]


def pallas_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = y.shape
    bm = min(512, m)
    bn = min(512, n)
    grid = (m // bm, n // bn)

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, y)


pallas_kernel = pallas_matmul
task_name = "matmul"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul"
level = 1
