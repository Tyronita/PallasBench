"""Level 2: Fused MatMul + GELU via Pallas.

Provenance: keras-team/keras-io define_custom_kernel guide FusedDense pattern
"""

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
    bm = min(512, m)
    bn = min(512, n)
    grid = (m // bm, n // bn)

    return pl.pallas_call(
        _matmul_gelu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_gelu
task_name = "matmul_gelu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2
