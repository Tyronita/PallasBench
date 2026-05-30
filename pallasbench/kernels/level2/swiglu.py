"""Level 2: Fused SwiGLU activation via Pallas.

SwiGLU: gate = silu(x @ W_gate), up = x @ W_up, output = gate * up.
Demonstrates: multi-input fusion, gated activation, silu transcendental.
Inspired by pallas-forge's SwiGLU kernel.
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/swiglu", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _swiglu_kernel(x_ref, w_gate_ref, w_up_ref, o_ref):
    x = x_ref[...]
    gate = x @ w_gate_ref[...]
    gate = gate / (1.0 + jnp.exp(-gate))  # silu
    up = x @ w_up_ref[...]
    o_ref[...] = gate * up


def pallas_swiglu(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array
) -> jax.Array:
    m, k = x.shape
    _, n = w_gate.shape
    BLOCK_M = min(m, 128)

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, n), lambda i: (i, 0)),
    )(x, w_gate, w_up)


pallas_kernel = pallas_swiglu
task_name = "swiglu"
input_shapes = [(512, 1024), (1024, 2048), (1024, 2048)]
category = "mlp_fusion"
level = 2
