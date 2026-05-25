"""Level 1: Nucleotide one-hot encoding via Pallas.

Encodes integer-encoded DNA sequences (A=0, C=1, G=2, T=3) into
4-channel one-hot representation used by genomics models (Enformer, etc).

Provenance: google-deepmind/deepmind-research Enformer
             DNA sequence input encoding (one-hot 4-channel)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _nucleotide_onehot_kernel(seq_ref, o_ref):
    seq = seq_ref[...]
    num_classes = 4
    o_ref[...] = (jnp.arange(num_classes) == seq[..., None]).astype(jnp.float32)


def pallas_nucleotide_onehot(seq: jax.Array) -> jax.Array:
    seq_len = seq.shape[0]

    return pl.pallas_call(
        _nucleotide_onehot_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, 4), jnp.float32),
        grid=(1,),
        in_specs=[pl.BlockSpec((seq_len,), lambda i: (0,))],
        out_specs=pl.BlockSpec((seq_len, 4), lambda i: (0, 0)),
    )(seq)


pallas_kernel = pallas_nucleotide_onehot
task_name = "nucleotide_onehot"
input_shapes = [(4096,)]
category = "genomics"
level = 1
