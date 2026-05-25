"""Level 2: Fused Linear + Bias + ReLU via Pallas.

Provenance: keras-team FusedDense pattern, standard MLP first layer
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _linear_bias_relu_kernel(x_ref, w_ref, b_ref, o_ref):
    z = x_ref[...] @ w_ref[...] + b_ref[...]
    o_ref[...] = jnp.maximum(z, 0)


def pallas_linear_bias_relu(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    bm = min(512, m)

    return pl.pallas_call(
        _linear_bias_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec((bm, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((n,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((bm, n), lambda i: (i, 0)),
    )(x, w, b)


pallas_kernel = pallas_linear_bias_relu
task_name = "linear_bias_relu"
input_shapes = [(1024, 1024), (1024, 2048), (2048,)]
category = "matmul_activation"
level = 2
