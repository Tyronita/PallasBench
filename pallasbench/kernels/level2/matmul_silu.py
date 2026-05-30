"""Level 2: Fused MatMul + SiLU via Pallas.

Provenance: openxla/tokamax gated_linear_unit uses SiLU gate path
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/matmul_silu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_silu_kernel(x_ref, w_ref, o_ref):
    z = x_ref[...] @ w_ref[...]
    o_ref[...] = z / (1.0 + jnp.exp(-z))


def pallas_matmul_silu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_silu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_silu
task_name = "matmul_silu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
