"""Level 2: Fused RMSNorm + Residual Add via Pallas.

Demonstrates: two-input fusion, norm + elementwise add in one kernel.
Inspired by pallas-forge's 3.44x speedup over XLA for this pattern.
"""


from pallasbench.provenance import describe_task as _describe_task

__doc__ = _describe_task("L2/rmsnorm_residual", __doc__)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _rmsnorm_residual_kernel(x_ref, residual_ref, o_ref):
    x = x_ref[...]
    residual = residual_ref[...]
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(ms + 1e-5)
    o_ref[...] = normed + residual


def pallas_rmsnorm_residual(
    x: jax.Array, residual: jax.Array
) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    block_rows = min(128, n_rows)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _rmsnorm_residual_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x, residual)


pallas_kernel = pallas_rmsnorm_residual
task_name = "rmsnorm_residual"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "norm_residual"
level = 2
