"""Level 1: SiLU (Swish) activation via Pallas.

Provenance: jax.nn.silu / openxla/tokamax gated_linear_unit uses SiLU gate
"""

from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/silu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _silu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x / (1.0 + jnp.exp(-x))


def pallas_silu(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _silu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_silu
task_name = "silu"
input_shapes = [(4096, 4096)]
category = "activation"
level = 1
