"""Level 2: Fused Sigmoid + Binary Cross-Entropy via Pallas.

Provenance: standard binary classification loss fusion
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _sigmoid_bce_kernel(logits_ref, targets_ref, o_ref):
    logits = logits_ref[...]
    targets = targets_ref[...]
    max_val = jnp.maximum(-logits, 0.0)
    loss = max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
    o_ref[...] = loss - targets * logits + targets * loss


def pallas_sigmoid_bce(logits: jax.Array, targets: jax.Array) -> jax.Array:
    n = logits.shape[0]
    block_size = min(1024, n)
    grid_size = n // block_size

    return pl.pallas_call(
        _sigmoid_bce_kernel,
        out_shape=jax.ShapeDtypeStruct(logits.shape, logits.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
            pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
        ],
        out_specs=pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
    )(logits, targets)


pallas_kernel = pallas_sigmoid_bce
task_name = "sigmoid_bce"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss_fusion"
level = 2
