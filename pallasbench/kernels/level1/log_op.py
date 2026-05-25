"""Level 1: Elementwise log via Pallas.

Provenance: jnp.log, used in cross-entropy loss and log-softmax
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _log_kernel(x_ref, o_ref):
    o_ref[...] = jnp.log(x_ref[...] + 1e-7)


def pallas_log(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _log_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_log
task_name = "log"
input_shapes = [(4096, 4096)]
category = "elementwise"
level = 1
