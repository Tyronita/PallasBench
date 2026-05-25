"""Level 3: Triangle Multiplicative Update via Pallas.

Implements the core triangular multiplicative update from AlphaFold2/3's
Evoformer / Pairformer: for each pair (i,j), aggregate information from
all intermediate positions k via element-wise product of edges (i,k)
and (k,j), enabling triplet reasoning for 3D structure consistency.

This is the computational heart of protein structure prediction and
is the most expensive operation in the Pairformer stack.

Provenance: google-deepmind/alphafold3 Pairformer triangle multiplication
             "Triangle Multiplication Is All You Need" (arXiv:2510.18870)
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L3/triangle_update", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _triangle_update_kernel(pair_ref, mask_ref, o_ref):
    pair = pair_ref[...]
    mask = mask_ref[...]
    n, _, c = pair.shape

    left_proj = pair * mask[:, :, None]
    right_proj = pair * mask[:, :, None]

    # Triangle update: out[i,j] = sum_k left[i,k] * right[k,j]
    # This is effectively a batched matmul over the channel dimension
    left_t = left_proj.transpose(2, 0, 1)
    right_t = right_proj.transpose(2, 0, 1)
    update = jnp.sum(left_t[:, :, :, None] * right_t[:, None, :, :], axis=2)
    update = update.transpose(1, 2, 0)

    o_ref[...] = pair + update


def pallas_triangle_update(pair: jax.Array, mask: jax.Array) -> jax.Array:
    n, _, c = pair.shape

    return pl.pallas_call(
        _triangle_update_kernel,
        out_shape=jax.ShapeDtypeStruct(pair.shape, pair.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(pair.shape, lambda i: (0, 0, 0)),
            pl.BlockSpec(mask.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(pair.shape, lambda i: (0, 0, 0)),
    )(pair, mask)


pallas_kernel = pallas_triangle_update
task_name = "triangle_update"
input_shapes = [(64, 64, 32), (64, 64)]
category = "genomics"
level = 3
