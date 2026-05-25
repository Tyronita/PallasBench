"""Level 1: Embedding table lookup via Pallas.

Demonstrates: gather-style indexing, integer index handling,
non-contiguous memory access patterns.
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/embedding_lookup", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _embedding_kernel(table_ref, idx_ref, o_ref):
    idx = idx_ref[...]
    o_ref[...] = table_ref[idx, :]


def pallas_embedding_lookup(
    table: jax.Array, indices: jax.Array
) -> jax.Array:
    seq_len = indices.shape[0]
    embed_dim = table.shape[1]

    return pl.pallas_call(
        _embedding_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, embed_dim), table.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(table.shape, lambda i: (0, 0)),
            pl.BlockSpec(indices.shape, lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((seq_len, embed_dim), lambda i: (0, 0)),
    )(table, indices)


pallas_kernel = pallas_embedding_lookup
task_name = "embedding_lookup"
input_shapes = [(32000, 768), (512,)]
category = "index"
level = 1
