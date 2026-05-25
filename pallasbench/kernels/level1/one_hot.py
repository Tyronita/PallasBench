"""Level 1: One-hot encoding via Pallas.

Provenance: jax.nn.one_hot, used in cross-entropy label preparation
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L1/one_hot", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _one_hot_kernel(indices_ref, o_ref):
    idx = indices_ref[...]
    num_classes = o_ref.shape[-1]
    o_ref[...] = (jnp.arange(num_classes) == idx[..., None]).astype(jnp.float32)


def pallas_one_hot(indices: jax.Array) -> jax.Array:
    seq_len = indices.shape[0]
    num_classes = 1024

    return pl.pallas_call(
        _one_hot_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, num_classes), jnp.float32),
        grid=(1,),
        in_specs=[pl.BlockSpec((seq_len,), lambda i: (0,))],
        out_specs=pl.BlockSpec((seq_len, num_classes), lambda i: (0, 0)),
    )(indices)


pallas_kernel = pallas_one_hot
task_name = "one_hot"
input_shapes = [(512,)]
category = "index"
level = 1
