"""Level 1: Elementwise exp via Pallas.

Provenance: jnp.exp, core transcendental used in softmax and loss functions
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/exp", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _exp_kernel(x_ref, o_ref):
    o_ref[...] = jnp.exp(x_ref[...])


def pallas_exp(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _exp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_exp
task_name = "exp"
input_shapes = [(4096, 4096)]
category = "elementwise"
level = 1
