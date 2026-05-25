"""Level 2: Pairwise Euclidean distance matrix via Pallas.

Computes the N x N distance matrix from N points in d dimensions.
Core operation in structural biology (AlphaFold distance maps),
molecular dynamics (neighbor lists), and genomics (phylogenetics).

Provenance: google-deepmind/alphafold3 pair representation distance maps
             JAX-MD (arxiv:1912.04232) pairwise distance computation
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _pairwise_dist_kernel(x_ref, o_ref):
    x = x_ref[...]
    n = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]
    o_ref[...] = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)


def pallas_pairwise_distance(x: jax.Array) -> jax.Array:
    n, d = x.shape

    return pl.pallas_call(
        _pairwise_dist_kernel,
        out_shape=jax.ShapeDtypeStruct((n, n), x.dtype),
        grid=(1,),
        in_specs=[pl.BlockSpec((n, d), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((n, n), lambda i: (0, 0)),
    )(x)


pallas_kernel = pallas_pairwise_distance
task_name = "pairwise_distance"
input_shapes = [(256, 32)]
category = "genomics"
level = 2
