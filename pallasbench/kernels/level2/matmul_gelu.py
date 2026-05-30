"""Level 2: Fused MatMul + GELU via Pallas.

Provenance: keras-team/keras-io define_custom_kernel guide FusedDense pattern
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/matmul_gelu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_gelu_kernel(x_ref, w_ref, o_ref):
    z = x_ref[...] @ w_ref[...]
    o_ref[...] = z * 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (z + 0.044715 * z ** 3)
    ))


def pallas_matmul_gelu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_gelu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_gelu
task_name = "matmul_gelu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
