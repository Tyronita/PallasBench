"""Level 1: Row-wise log-softmax via Pallas.

Provenance: jax.nn.log_softmax, critical for cross-entropy loss computation
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _log_softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    row_max = jnp.max(x, axis=-1, keepdims=True)
    shifted = x - row_max
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    o_ref[...] = shifted - log_sum_exp


def pallas_log_softmax(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    block_rows = min(128, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _log_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_log_softmax
task_name = "log_softmax"
input_shapes = [(2048, 2048)]
category = "softmax"
level = 1
