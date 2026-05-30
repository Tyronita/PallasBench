"""Level 3: Full Gated MLP Block (SwiGLU) via Pallas.

Provenance: openxla/tokamax gated_linear_unit
             AI-Hypercomputer/maxtext Llama/Gemma MLP blocks
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L3/gated_mlp", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _gated_mlp_kernel(x_ref, w_gate_ref, w_up_ref, w_down_ref, o_ref):
    x = x_ref[...]
    gate = x @ w_gate_ref[...]
    gate = gate / (1.0 + jnp.exp(-gate))
    up = x @ w_up_ref[...]
    hidden = gate * up
    o_ref[...] = hidden @ w_down_ref[...]


def pallas_gated_mlp(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
    m, d_model = x.shape
    _, d_ff = w_gate.shape
    BLOCK_M = min(m, 128)

    return pl.pallas_call(
        _gated_mlp_kernel,
        out_shape=jax.ShapeDtypeStruct((m, d_model), x.dtype),
        grid=(m // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, d_model), lambda i: (i, 0)),
            pl.BlockSpec((d_model, d_ff), lambda i: (0, 0)),
            pl.BlockSpec((d_model, d_ff), lambda i: (0, 0)),
            pl.BlockSpec((d_ff, d_model), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, d_model), lambda i: (i, 0)),
    )(x, w_gate, w_up, w_down)


pallas_kernel = pallas_gated_mlp
task_name = "gated_mlp"
input_shapes = [(256, 512), (512, 1024), (512, 1024), (1024, 512)]
category = "mlp"
level = 3
