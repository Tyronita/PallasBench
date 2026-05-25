"""Level 1: Mean squared error loss via Pallas.

Provenance: standard regression loss, (pred - target)^2 reduced per row
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _mse_kernel(pred_ref, target_ref, o_ref):
    diff = pred_ref[...] - target_ref[...]
    o_ref[...] = jnp.mean(diff * diff, axis=-1)


def pallas_mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    n_rows = pred.shape[0]
    n_cols = pred.shape[1]
    block_rows = min(256, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _mse_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), pred.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(pred, target)


pallas_kernel = pallas_mse_loss
task_name = "mse_loss"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss"
level = 1
