"""Level 2: Fused MatMul + ReLU via Pallas.

Demonstrates: operator fusion — single pallas_call replaces matmul + relu,
avoiding a round-trip through HBM between the two ops.
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/matmul_relu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_relu_kernel(x_ref, w_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...] @ w_ref[...], 0)


def pallas_matmul_relu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_relu
task_name = "matmul_relu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
