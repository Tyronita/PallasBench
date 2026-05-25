"""Level 2: Fused GeGLU activation via Pallas.

Provenance: variant of gated linear unit, used in PaLM/Gemma MLPs
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _geglu_kernel(x_ref, w_gate_ref, w_up_ref, o_ref):
    x = x_ref[...]
    gate = x @ w_gate_ref[...]
    gate = gate * 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (gate + 0.044715 * gate ** 3)
    ))
    up = x @ w_up_ref[...]
    o_ref[...] = gate * up


def pallas_geglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w_gate.shape
    bm = min(256, m)

    return pl.pallas_call(
        _geglu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm,),
        in_specs=[
            pl.BlockSpec((bm, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((bm, n), lambda i: (i, 0)),
    )(x, w_gate, w_up)


pallas_kernel = pallas_geglu
task_name = "geglu"
input_shapes = [(512, 1024), (1024, 2048), (1024, 2048)]
category = "mlp_fusion"
level = 2
