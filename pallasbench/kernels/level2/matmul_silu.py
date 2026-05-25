"""Level 2: Fused MatMul + SiLU via Pallas.

Provenance: openxla/tokamax gated_linear_unit uses SiLU gate path
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_silu_kernel(x_ref, w_ref, o_ref):
    z = x_ref[...] @ w_ref[...]
    o_ref[...] = z / (1.0 + jnp.exp(-z))


def pallas_matmul_silu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    bm = min(512, m)
    bn = min(512, n)
    grid = (m // bm, n // bn)

    return pl.pallas_call(
        _matmul_silu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_silu
task_name = "matmul_silu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
