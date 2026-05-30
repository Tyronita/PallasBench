"""Level 1: Elementwise sigmoid via Pallas.

Provenance: jax.nn.sigmoid, used in loss functions and gating
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/sigmoid", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _sigmoid_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = 1.0 / (1.0 + jnp.exp(-x))


def pallas_sigmoid(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _sigmoid_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_sigmoid
task_name = "sigmoid"
input_shapes = [(4096, 4096)]
category = "activation"
level = 1
