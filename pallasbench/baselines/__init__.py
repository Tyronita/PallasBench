"""Pure JAX reference implementations (baselines) for all PallasBench tasks."""

from pallasbench.baselines.jax_baseline import (
    jax_relu,
    jax_gelu,
    jax_softmax,
    jax_layernorm,
    jax_rmsnorm,
    jax_matmul,
    jax_reduce_sum,
    jax_embedding_lookup,
    jax_matmul_relu,
    jax_rmsnorm_residual,
    jax_swiglu,
    jax_flash_attention,
)
