"""Level 2: Fused MatMul + ReLU via Pallas.

Demonstrates: operator fusion — single pallas_call replaces matmul + relu,
avoiding a round-trip through HBM between the two ops.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_relu_kernel(x_ref, w_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...] @ w_ref[...], 0)


def pallas_matmul_relu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    bm = min(512, m)
    bn = min(512, n)
    grid = (m // bm, n // bn)

    return pl.pallas_call(
        _matmul_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_relu
task_name = "matmul_relu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
