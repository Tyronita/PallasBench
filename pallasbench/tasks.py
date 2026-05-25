"""Task registry: maps task names to kernel/baseline pairs with metadata."""

from __future__ import annotations

from pallasbench.kernels.level1 import (
    relu, gelu, silu, sigmoid, tanh_act,
    softmax, log_softmax,
    layernorm, rmsnorm,
    matmul, batched_matmul, outer_product,
    reduce_sum, reduce_max, reduce_mean,
    exp_op, log_op, add_op, multiply_op, rsqrt_op, clamp,
    cross_entropy, mse_loss, cosine_sim,
    embedding_lookup, one_hot,
)
from pallasbench.kernels.level2 import (
    matmul_relu, matmul_gelu, matmul_silu,
    rmsnorm_residual, layernorm_residual,
    swiglu, geglu, linear_bias_relu,
    qk_softmax, fused_softmax_cross_entropy, sigmoid_bce,
)
from pallasbench.kernels.level3 import (
    flash_attention, multi_head_attention, gated_mlp, transformer_block,
)
from pallasbench.baselines import jax_baseline


TASK_REGISTRY: list[dict] = [
    # =========================================================================
    # Level 1: Single operators
    # =========================================================================

    # --- Activation ---
    {"name": "L1/relu",     "level": 1, "category": "activation",    "pallas_fn": relu.pallas_kernel,     "baseline_fn": jax_baseline.jax_relu,     "input_shapes": relu.input_shapes},
    {"name": "L1/gelu",     "level": 1, "category": "activation",    "pallas_fn": gelu.pallas_kernel,     "baseline_fn": jax_baseline.jax_gelu,     "input_shapes": gelu.input_shapes},
    {"name": "L1/silu",     "level": 1, "category": "activation",    "pallas_fn": silu.pallas_kernel,     "baseline_fn": jax_baseline.jax_silu,     "input_shapes": silu.input_shapes},
    {"name": "L1/sigmoid",  "level": 1, "category": "activation",    "pallas_fn": sigmoid.pallas_kernel,  "baseline_fn": jax_baseline.jax_sigmoid,  "input_shapes": sigmoid.input_shapes},
    {"name": "L1/tanh",     "level": 1, "category": "activation",    "pallas_fn": tanh_act.pallas_kernel, "baseline_fn": jax_baseline.jax_tanh,     "input_shapes": tanh_act.input_shapes},

    # --- Normalization ---
    {"name": "L1/layernorm", "level": 1, "category": "normalization", "pallas_fn": layernorm.pallas_kernel, "baseline_fn": jax_baseline.jax_layernorm, "input_shapes": layernorm.input_shapes},
    {"name": "L1/rmsnorm",   "level": 1, "category": "normalization", "pallas_fn": rmsnorm.pallas_kernel,   "baseline_fn": jax_baseline.jax_rmsnorm,   "input_shapes": rmsnorm.input_shapes},

    # --- MatMul ---
    {"name": "L1/matmul",          "level": 1, "category": "matmul", "pallas_fn": matmul.pallas_kernel,          "baseline_fn": jax_baseline.jax_matmul,          "input_shapes": matmul.input_shapes},
    {"name": "L1/batched_matmul",  "level": 1, "category": "matmul", "pallas_fn": batched_matmul.pallas_kernel,  "baseline_fn": jax_baseline.jax_batched_matmul,  "input_shapes": batched_matmul.input_shapes},
    {"name": "L1/outer_product",   "level": 1, "category": "matmul", "pallas_fn": outer_product.pallas_kernel,   "baseline_fn": jax_baseline.jax_outer_product,   "input_shapes": outer_product.input_shapes},

    # --- Reduce ---
    {"name": "L1/reduce_sum",  "level": 1, "category": "reduce", "pallas_fn": reduce_sum.pallas_kernel,  "baseline_fn": jax_baseline.jax_reduce_sum,  "input_shapes": reduce_sum.input_shapes},
    {"name": "L1/reduce_max",  "level": 1, "category": "reduce", "pallas_fn": reduce_max.pallas_kernel,  "baseline_fn": jax_baseline.jax_reduce_max,  "input_shapes": reduce_max.input_shapes},
    {"name": "L1/reduce_mean", "level": 1, "category": "reduce", "pallas_fn": reduce_mean.pallas_kernel, "baseline_fn": jax_baseline.jax_reduce_mean, "input_shapes": reduce_mean.input_shapes},

    # --- Softmax ---
    {"name": "L1/softmax",     "level": 1, "category": "softmax", "pallas_fn": softmax.pallas_kernel,     "baseline_fn": jax_baseline.jax_softmax,     "input_shapes": softmax.input_shapes},
    {"name": "L1/log_softmax", "level": 1, "category": "softmax", "pallas_fn": log_softmax.pallas_kernel, "baseline_fn": jax_baseline.jax_log_softmax, "input_shapes": log_softmax.input_shapes},

    # --- Elementwise ---
    {"name": "L1/exp",      "level": 1, "category": "elementwise", "pallas_fn": exp_op.pallas_kernel,      "baseline_fn": jax_baseline.jax_exp,      "input_shapes": exp_op.input_shapes},
    {"name": "L1/log",      "level": 1, "category": "elementwise", "pallas_fn": log_op.pallas_kernel,      "baseline_fn": jax_baseline.jax_log,      "input_shapes": log_op.input_shapes},
    {"name": "L1/add",      "level": 1, "category": "elementwise", "pallas_fn": add_op.pallas_kernel,      "baseline_fn": jax_baseline.jax_add,      "input_shapes": add_op.input_shapes},
    {"name": "L1/multiply", "level": 1, "category": "elementwise", "pallas_fn": multiply_op.pallas_kernel, "baseline_fn": jax_baseline.jax_multiply, "input_shapes": multiply_op.input_shapes},
    {"name": "L1/rsqrt",    "level": 1, "category": "elementwise", "pallas_fn": rsqrt_op.pallas_kernel,    "baseline_fn": jax_baseline.jax_rsqrt,    "input_shapes": rsqrt_op.input_shapes},
    {"name": "L1/clamp",    "level": 1, "category": "elementwise", "pallas_fn": clamp.pallas_kernel,       "baseline_fn": jax_baseline.jax_clamp,    "input_shapes": clamp.input_shapes},

    # --- Loss ---
    {"name": "L1/cross_entropy", "level": 1, "category": "loss", "pallas_fn": cross_entropy.pallas_kernel, "baseline_fn": jax_baseline.jax_cross_entropy, "input_shapes": cross_entropy.input_shapes},
    {"name": "L1/mse_loss",      "level": 1, "category": "loss", "pallas_fn": mse_loss.pallas_kernel,      "baseline_fn": jax_baseline.jax_mse_loss,      "input_shapes": mse_loss.input_shapes},
    {"name": "L1/cosine_sim",    "level": 1, "category": "loss", "pallas_fn": cosine_sim.pallas_kernel,    "baseline_fn": jax_baseline.jax_cosine_sim,    "input_shapes": cosine_sim.input_shapes},

    # --- Index ---
    {"name": "L1/embedding_lookup", "level": 1, "category": "index", "pallas_fn": embedding_lookup.pallas_kernel, "baseline_fn": jax_baseline.jax_embedding_lookup, "input_shapes": embedding_lookup.input_shapes},
    {"name": "L1/one_hot",          "level": 1, "category": "index", "pallas_fn": one_hot.pallas_kernel,          "baseline_fn": jax_baseline.jax_one_hot,          "input_shapes": one_hot.input_shapes},

    # =========================================================================
    # Level 2: Fusion patterns
    # =========================================================================

    {"name": "L2/matmul_relu",               "level": 2, "category": "matmul_activation",   "pallas_fn": matmul_relu.pallas_kernel,               "baseline_fn": jax_baseline.jax_matmul_relu,               "input_shapes": matmul_relu.input_shapes},
    {"name": "L2/matmul_gelu",               "level": 2, "category": "matmul_activation",   "pallas_fn": matmul_gelu.pallas_kernel,               "baseline_fn": jax_baseline.jax_matmul_gelu,               "input_shapes": matmul_gelu.input_shapes},
    {"name": "L2/matmul_silu",               "level": 2, "category": "matmul_activation",   "pallas_fn": matmul_silu.pallas_kernel,               "baseline_fn": jax_baseline.jax_matmul_silu,               "input_shapes": matmul_silu.input_shapes},
    {"name": "L2/rmsnorm_residual",          "level": 2, "category": "norm_residual",       "pallas_fn": rmsnorm_residual.pallas_kernel,          "baseline_fn": jax_baseline.jax_rmsnorm_residual,          "input_shapes": rmsnorm_residual.input_shapes},
    {"name": "L2/layernorm_residual",        "level": 2, "category": "norm_residual",       "pallas_fn": layernorm_residual.pallas_kernel,        "baseline_fn": jax_baseline.jax_layernorm_residual,        "input_shapes": layernorm_residual.input_shapes},
    {"name": "L2/swiglu",                    "level": 2, "category": "mlp_fusion",          "pallas_fn": swiglu.pallas_kernel,                    "baseline_fn": jax_baseline.jax_swiglu,                    "input_shapes": swiglu.input_shapes},
    {"name": "L2/geglu",                     "level": 2, "category": "mlp_fusion",          "pallas_fn": geglu.pallas_kernel,                     "baseline_fn": jax_baseline.jax_geglu,                     "input_shapes": geglu.input_shapes},
    {"name": "L2/linear_bias_relu",          "level": 2, "category": "matmul_activation",   "pallas_fn": linear_bias_relu.pallas_kernel,          "baseline_fn": jax_baseline.jax_linear_bias_relu,          "input_shapes": linear_bias_relu.input_shapes},
    {"name": "L2/qk_softmax",               "level": 2, "category": "attention_component", "pallas_fn": qk_softmax.pallas_kernel,               "baseline_fn": jax_baseline.jax_qk_softmax,               "input_shapes": qk_softmax.input_shapes},
    {"name": "L2/fused_softmax_cross_entropy","level": 2, "category": "loss_fusion",        "pallas_fn": fused_softmax_cross_entropy.pallas_kernel,"baseline_fn": jax_baseline.jax_fused_softmax_cross_entropy,"input_shapes": fused_softmax_cross_entropy.input_shapes},
    {"name": "L2/sigmoid_bce",              "level": 2, "category": "loss_fusion",          "pallas_fn": sigmoid_bce.pallas_kernel,               "baseline_fn": jax_baseline.jax_sigmoid_bce,               "input_shapes": sigmoid_bce.input_shapes},

    # =========================================================================
    # Level 3: Architecture components
    # =========================================================================

    {"name": "L3/flash_attention",      "level": 3, "category": "attention",  "pallas_fn": flash_attention.pallas_kernel,      "baseline_fn": jax_baseline.jax_flash_attention,      "input_shapes": flash_attention.input_shapes},
    {"name": "L3/multi_head_attention", "level": 3, "category": "attention",  "pallas_fn": multi_head_attention.pallas_kernel, "baseline_fn": jax_baseline.jax_multi_head_attention, "input_shapes": multi_head_attention.input_shapes},
    {"name": "L3/gated_mlp",           "level": 3, "category": "mlp",        "pallas_fn": gated_mlp.pallas_kernel,            "baseline_fn": jax_baseline.jax_gated_mlp,            "input_shapes": gated_mlp.input_shapes},
    {"name": "L3/transformer_block",    "level": 3, "category": "full_model", "pallas_fn": transformer_block.pallas_kernel,    "baseline_fn": jax_baseline.jax_transformer_block,    "input_shapes": transformer_block.input_shapes},
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
