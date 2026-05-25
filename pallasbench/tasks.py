"""Task registry: maps task names to kernel/baseline pairs with metadata."""

from __future__ import annotations

from pallasbench.kernels.level1 import (
    relu, gelu, softmax, layernorm, rmsnorm, matmul, reduce_sum, embedding_lookup,
)
from pallasbench.kernels.level2 import matmul_relu, rmsnorm_residual, swiglu
from pallasbench.kernels.level3 import flash_attention
from pallasbench.baselines import jax_baseline


TASK_REGISTRY: list[dict] = [
    # Level 1: Single operators
    {
        "name": "L1/relu",
        "level": 1,
        "category": "activation",
        "pallas_fn": relu.pallas_kernel,
        "baseline_fn": jax_baseline.jax_relu,
        "input_shapes": relu.input_shapes,
    },
    {
        "name": "L1/gelu",
        "level": 1,
        "category": "activation",
        "pallas_fn": gelu.pallas_kernel,
        "baseline_fn": jax_baseline.jax_gelu,
        "input_shapes": gelu.input_shapes,
    },
    {
        "name": "L1/softmax",
        "level": 1,
        "category": "softmax",
        "pallas_fn": softmax.pallas_kernel,
        "baseline_fn": jax_baseline.jax_softmax,
        "input_shapes": softmax.input_shapes,
    },
    {
        "name": "L1/layernorm",
        "level": 1,
        "category": "normalization",
        "pallas_fn": layernorm.pallas_kernel,
        "baseline_fn": jax_baseline.jax_layernorm,
        "input_shapes": layernorm.input_shapes,
    },
    {
        "name": "L1/rmsnorm",
        "level": 1,
        "category": "normalization",
        "pallas_fn": rmsnorm.pallas_kernel,
        "baseline_fn": jax_baseline.jax_rmsnorm,
        "input_shapes": rmsnorm.input_shapes,
    },
    {
        "name": "L1/matmul",
        "level": 1,
        "category": "matmul",
        "pallas_fn": matmul.pallas_kernel,
        "baseline_fn": jax_baseline.jax_matmul,
        "input_shapes": matmul.input_shapes,
    },
    {
        "name": "L1/reduce_sum",
        "level": 1,
        "category": "reduce",
        "pallas_fn": reduce_sum.pallas_kernel,
        "baseline_fn": jax_baseline.jax_reduce_sum,
        "input_shapes": reduce_sum.input_shapes,
    },
    # Level 2: Fusion patterns
    {
        "name": "L2/matmul_relu",
        "level": 2,
        "category": "matmul_activation",
        "pallas_fn": matmul_relu.pallas_kernel,
        "baseline_fn": jax_baseline.jax_matmul_relu,
        "input_shapes": matmul_relu.input_shapes,
    },
    {
        "name": "L2/rmsnorm_residual",
        "level": 2,
        "category": "norm_residual",
        "pallas_fn": rmsnorm_residual.pallas_kernel,
        "baseline_fn": jax_baseline.jax_rmsnorm_residual,
        "input_shapes": rmsnorm_residual.input_shapes,
    },
    {
        "name": "L2/swiglu",
        "level": 2,
        "category": "mlp_fusion",
        "pallas_fn": swiglu.pallas_kernel,
        "baseline_fn": jax_baseline.jax_swiglu,
        "input_shapes": swiglu.input_shapes,
    },
    # Level 3: Architecture components
    {
        "name": "L3/flash_attention",
        "level": 3,
        "category": "attention",
        "pallas_fn": flash_attention.pallas_kernel,
        "baseline_fn": jax_baseline.jax_flash_attention,
        "input_shapes": flash_attention.input_shapes,
    },
]


def get_tasks(
    levels: list[int] | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    tasks = TASK_REGISTRY
    if levels:
        tasks = [t for t in tasks if t["level"] in levels]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]
    return tasks
