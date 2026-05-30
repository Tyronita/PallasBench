"""Level 1: Elementwise clamp via Pallas.

Provenance: jnp.clip, used in gradient clipping and activation clamping
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/clamp", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _clamp_kernel(x_ref, o_ref):
    o_ref[...] = jnp.clip(x_ref[...], -1.0, 1.0)


def pallas_clamp(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _clamp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_clamp
task_name = "clamp"
input_shapes = [(4096, 4096)]
category = "elementwise"
level = 1
