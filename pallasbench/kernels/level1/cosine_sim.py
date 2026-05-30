"""Level 1: Row-wise cosine similarity via Pallas.

Provenance: standard similarity metric for embeddings and retrieval
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/cosine_sim", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _cosine_sim_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    dot = jnp.sum(x * y, axis=-1)
    norm_x = jnp.sqrt(jnp.sum(x * x, axis=-1))
    norm_y = jnp.sqrt(jnp.sum(y * y, axis=-1))
    o_ref[...] = dot / (norm_x * norm_y + 1e-8)


def pallas_cosine_sim(x: jax.Array, y: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _cosine_sim_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(x, y)


pallas_kernel = pallas_cosine_sim
task_name = "cosine_sim"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss"
level = 1
