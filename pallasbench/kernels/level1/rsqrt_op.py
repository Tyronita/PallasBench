"""Level 1: Elementwise reciprocal square root via Pallas.

Provenance: jax.lax.rsqrt, critical in normalization layers
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/rsqrt", __doc__)

import jax
from jax.experimental import pallas as pl


def _rsqrt_kernel(x_ref, o_ref):
    o_ref[...] = jax.lax.rsqrt(x_ref[...] + 1e-5)


def pallas_rsqrt(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _rsqrt_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_rsqrt
task_name = "rsqrt"
input_shapes = [(4096, 4096)]
category = "elementwise"
level = 1
