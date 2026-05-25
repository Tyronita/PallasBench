"""Level 2: Position Weight Matrix (PWM) motif scanning via Pallas.

Scans a one-hot encoded DNA sequence with a PWM (4 x motif_len) to find
transcription factor binding sites. This is a 1D "convolution" over the
4-channel nucleotide representation — the core operation in genomics
motif discovery tools (MEME, PWMScan, Enformer conv tower).

Provenance: google-deepmind/deepmind-research Enformer conv tower
             PWMScan (Bioinformatics 2018) motif scanning pattern
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _pwm_scan_kernel(seq_onehot_ref, pwm_ref, o_ref):
    seq = seq_onehot_ref[...]
    pwm = pwm_ref[...]
    motif_len = pwm.shape[1]
    seq_len = seq.shape[0]
    out_len = seq_len - motif_len + 1

    scores = jnp.zeros(out_len, dtype=seq.dtype)
    for pos in range(motif_len):
        scores = scores + jnp.sum(seq[pos:pos + out_len, :] * pwm[:, pos][None, :], axis=-1)
    o_ref[:out_len] = scores


def pallas_pwm_scan(seq_onehot: jax.Array, pwm: jax.Array) -> jax.Array:
    seq_len = seq_onehot.shape[0]
    motif_len = pwm.shape[1]
    out_len = seq_len - motif_len + 1

    return pl.pallas_call(
        _pwm_scan_kernel,
        out_shape=jax.ShapeDtypeStruct((out_len,), seq_onehot.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(seq_onehot.shape, lambda i: (0, 0)),
            pl.BlockSpec(pwm.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((out_len,), lambda i: (0,)),
    )(seq_onehot, pwm)


pallas_kernel = pallas_pwm_scan
task_name = "pwm_scan"
input_shapes = [(1024, 4), (4, 12)]
category = "genomics"
level = 2
