"""Level 1: Elementwise GELU via Pallas.

Demonstrates: transcendental functions (tanh, erf) inside kernels.
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/gelu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _gelu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x * 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)
    ))


def pallas_gelu(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _gelu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_gelu
task_name = "gelu"
input_shapes = [(4096, 4096)]
category = "activation"
level = 1
