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
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, y)


pallas_kernel = pallas_matmul
task_name = "matmul"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul"
level = 1
