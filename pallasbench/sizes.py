"""Parametric benchmark sizes: SMALL, MEDIUM, LARGE configs per task.

Each task can be run at multiple sizes to measure scaling behavior.
Sizes are chosen to span from quick CI smoke tests (SMALL) to
production-scale workloads (LARGE).
"""

from __future__ import annotations


SIZE_CONFIGS: dict[str, dict[str, list[tuple]]] = {
    # =========================================================================
    # Level 1
    # =========================================================================

    # Activation (1-input elementwise)
    "L1/relu":          {"SMALL": [(512, 512)],      "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/gelu":          {"SMALL": [(512, 512)],      "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/silu":          {"SMALL": [(512, 512)],      "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/sigmoid":       {"SMALL": [(512, 512)],      "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/tanh":          {"SMALL": [(512, 512)],      "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},

    # Normalization (1-input row-wise)
    "L1/layernorm":     {"SMALL": [(256, 256)],      "MEDIUM": [(2048, 1024)],    "LARGE": [(8192, 4096)]},
    "L1/rmsnorm":       {"SMALL": [(256, 256)],      "MEDIUM": [(2048, 1024)],    "LARGE": [(8192, 4096)]},

    # MatMul (2-input)
    "L1/matmul":        {"SMALL": [(256, 256), (256, 256)],   "MEDIUM": [(1024, 1024), (1024, 1024)],   "LARGE": [(4096, 4096), (4096, 4096)]},
    "L1/batched_matmul": {"SMALL": [(4, 64, 64), (4, 64, 64)], "MEDIUM": [(8, 256, 256), (8, 256, 256)], "LARGE": [(16, 512, 512), (16, 512, 512)]},
    "L1/outer_product":  {"SMALL": [(256,), (256,)],  "MEDIUM": [(1024,), (1024,)], "LARGE": [(4096,), (4096,)]},

    # Reduce (1-input -> smaller output)
    "L1/reduce_sum":    {"SMALL": [(512, 256)],       "MEDIUM": [(4096, 2048)],    "LARGE": [(16384, 8192)]},
    "L1/reduce_max":    {"SMALL": [(512, 256)],       "MEDIUM": [(4096, 2048)],    "LARGE": [(16384, 8192)]},
    "L1/reduce_mean":   {"SMALL": [(512, 256)],       "MEDIUM": [(4096, 2048)],    "LARGE": [(16384, 8192)]},

    # Softmax (1-input row-wise)
    "L1/softmax":       {"SMALL": [(256, 256)],       "MEDIUM": [(2048, 2048)],    "LARGE": [(8192, 8192)]},
    "L1/log_softmax":   {"SMALL": [(256, 256)],       "MEDIUM": [(2048, 2048)],    "LARGE": [(8192, 8192)]},

    # Elementwise
    "L1/exp":           {"SMALL": [(512, 512)],       "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/log":           {"SMALL": [(512, 512)],       "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/add":           {"SMALL": [(512, 512), (512, 512)],         "MEDIUM": [(4096, 4096), (4096, 4096)],         "LARGE": [(16384, 16384), (16384, 16384)]},
    "L1/multiply":      {"SMALL": [(512, 512), (512, 512)],         "MEDIUM": [(4096, 4096), (4096, 4096)],         "LARGE": [(16384, 16384), (16384, 16384)]},
    "L1/rsqrt":         {"SMALL": [(512, 512)],       "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},
    "L1/clamp":         {"SMALL": [(512, 512)],       "MEDIUM": [(4096, 4096)],    "LARGE": [(16384, 16384)]},

    # Loss
    "L1/cross_entropy": {"SMALL": [(128, 64), (128, 64)],    "MEDIUM": [(1024, 512), (1024, 512)],    "LARGE": [(4096, 2048), (4096, 2048)]},
    "L1/mse_loss":      {"SMALL": [(256, 128), (256, 128)],  "MEDIUM": [(2048, 1024), (2048, 1024)],  "LARGE": [(8192, 4096), (8192, 4096)]},
    "L1/cosine_sim":    {"SMALL": [(256, 128), (256, 128)],  "MEDIUM": [(2048, 1024), (2048, 1024)],  "LARGE": [(8192, 4096), (8192, 4096)]},

    # Index
    "L1/embedding_lookup": {"SMALL": [(8000, 256), (128,)],   "MEDIUM": [(32000, 768), (512,)],   "LARGE": [(128000, 1024), (2048,)]},
    "L1/one_hot":          {"SMALL": [(128,)],                 "MEDIUM": [(512,)],                 "LARGE": [(2048,)]},

    # =========================================================================
    # Level 2
    # =========================================================================

    "L2/matmul_relu":               {"SMALL": [(256, 256), (256, 256)],             "MEDIUM": [(1024, 1024), (1024, 1024)],             "LARGE": [(4096, 4096), (4096, 4096)]},
    "L2/matmul_gelu":               {"SMALL": [(256, 256), (256, 256)],             "MEDIUM": [(1024, 1024), (1024, 1024)],             "LARGE": [(4096, 4096), (4096, 4096)]},
    "L2/matmul_silu":               {"SMALL": [(256, 256), (256, 256)],             "MEDIUM": [(1024, 1024), (1024, 1024)],             "LARGE": [(4096, 4096), (4096, 4096)]},
    "L2/rmsnorm_residual":          {"SMALL": [(256, 256), (256, 256)],             "MEDIUM": [(2048, 1024), (2048, 1024)],             "LARGE": [(8192, 4096), (8192, 4096)]},
    "L2/layernorm_residual":        {"SMALL": [(256, 256), (256, 256)],             "MEDIUM": [(2048, 1024), (2048, 1024)],             "LARGE": [(8192, 4096), (8192, 4096)]},
    "L2/swiglu":                    {"SMALL": [(128, 256), (256, 512), (256, 512)], "MEDIUM": [(512, 1024), (1024, 2048), (1024, 2048)], "LARGE": [(2048, 4096), (4096, 8192), (4096, 8192)]},
    "L2/geglu":                     {"SMALL": [(128, 256), (256, 512), (256, 512)], "MEDIUM": [(512, 1024), (1024, 2048), (1024, 2048)], "LARGE": [(2048, 4096), (4096, 8192), (4096, 8192)]},
    "L2/linear_bias_relu":          {"SMALL": [(256, 256), (256, 512), (512,)],     "MEDIUM": [(1024, 1024), (1024, 2048), (2048,)],     "LARGE": [(4096, 4096), (4096, 8192), (8192,)]},
    "L2/qk_softmax":               {"SMALL": [(64, 32), (64, 32)],                 "MEDIUM": [(256, 64), (256, 64)],                    "LARGE": [(1024, 128), (1024, 128)]},
    "L2/fused_softmax_cross_entropy": {"SMALL": [(128, 64), (128, 64)],             "MEDIUM": [(1024, 512), (1024, 512)],                "LARGE": [(4096, 2048), (4096, 2048)]},
    "L2/sigmoid_bce":               {"SMALL": [(256, 128), (256, 128)],             "MEDIUM": [(2048, 1024), (2048, 1024)],              "LARGE": [(8192, 4096), (8192, 4096)]},

    # =========================================================================
    # Level 3
    # =========================================================================

    "L3/flash_attention":       {"SMALL": [(128, 32), (128, 32), (128, 32)],                  "MEDIUM": [(512, 64), (512, 64), (512, 64)],                   "LARGE": [(2048, 128), (2048, 128), (2048, 128)]},
    "L3/multi_head_attention":  {"SMALL": [(4, 64, 32), (4, 64, 32), (4, 64, 32)],            "MEDIUM": [(8, 256, 64), (8, 256, 64), (8, 256, 64)],          "LARGE": [(16, 512, 128), (16, 512, 128), (16, 512, 128)]},
    "L3/gated_mlp":             {"SMALL": [(64, 128), (128, 256), (128, 256), (256, 128)],     "MEDIUM": [(256, 512), (512, 1024), (512, 1024), (1024, 512)], "LARGE": [(1024, 2048), (2048, 4096), (2048, 4096), (4096, 2048)]},
    "L3/transformer_block":     {"SMALL": [(32, 64), (64, 32), (64, 32), (64, 32), (32, 64), (64, 64)], "MEDIUM": [(128, 256), (256, 64), (256, 64), (256, 64), (64, 256), (256, 256)], "LARGE": [(512, 1024), (1024, 128), (1024, 128), (1024, 128), (128, 1024), (1024, 1024)]},

    # =========================================================================
    # Genomics
    # =========================================================================

    # nucleotide_onehot: 1D integer sequence -> (N, 4) one-hot
    "L1/nucleotide_onehot":     {"SMALL": [(1024,)],                                   "MEDIUM": [(4096,)],                                    "LARGE": [(16384,)]},

    # pwm_scan: (seq_len, 4) one-hot DNA + (4, motif_len) PWM
    "L2/pwm_scan":              {"SMALL": [(256, 4), (4, 8)],                           "MEDIUM": [(1024, 4), (4, 12)],                         "LARGE": [(4096, 4), (4, 16)]},

    # pairwise_distance: (N, D) points -> (N, N) distance matrix
    "L2/pairwise_distance":     {"SMALL": [(64, 16)],                                   "MEDIUM": [(256, 32)],                                  "LARGE": [(1024, 64)]},

    # triangle_update: (N, N, C) pair repr + (N, N) mask
    "L3/triangle_update":       {"SMALL": [(16, 16, 16), (16, 16)],                     "MEDIUM": [(64, 64, 32), (64, 64)],                     "LARGE": [(128, 128, 64), (128, 128)]},
}


def get_size_config(task_name: str, size: str = "MEDIUM") -> list[tuple] | None:
    task_sizes = SIZE_CONFIGS.get(task_name)
    if task_sizes is None:
        return None
    return task_sizes.get(size)


# =============================================================================
# Multi-Size Scaling Suite
# =============================================================================
# Maps size-name -> task-name -> list of shape tuples (matching the kernel's
# arity).  Used by scripts/run_size_scaling.py to run each kernel at multiple
# problem sizes for scaling-curve analysis.
#
# Primary sizes: N256, N1024, N4096, N8192
# Optional:      N128, N16384
# =============================================================================

SCALING_SIZE_NAMES: list[str] = ["N256", "N1024", "N4096", "N8192"]

SCALING_SUITE: dict[str, dict[str, list[tuple[int, ...]]]] = {
    # ------------------------------------------------------------------
    # N128: tiny — launch overhead dominates, stress-test small blocks
    # ------------------------------------------------------------------
    "N128": {
        # Level 1 — activation (single 2D input)
        "L1/relu": [(128, 128)],
        "L1/gelu": [(128, 128)],
        "L1/silu": [(128, 128)],
        "L1/sigmoid": [(128, 128)],
        "L1/tanh": [(128, 128)],
        # Level 1 — normalization (single 2D input, row-wise)
        "L1/layernorm": [(128, 128)],
        "L1/rmsnorm": [(128, 128)],
        # Level 1 — matmul (two 2D inputs)
        "L1/matmul": [(128, 128), (128, 128)],
        "L1/batched_matmul": [(2, 64, 64), (2, 64, 64)],
        "L1/outer_product": [(128,), (128,)],
        # Level 1 — reduce (single 2D input)
        "L1/reduce_sum": [(128, 128)],
        "L1/reduce_max": [(128, 128)],
        "L1/reduce_mean": [(128, 128)],
        # Level 1 — softmax (single 2D input, row-wise)
        "L1/softmax": [(128, 128)],
        "L1/log_softmax": [(128, 128)],
        # Level 1 — elementwise (1 or 2 inputs)
        "L1/exp": [(128, 128)],
        "L1/log": [(128, 128)],
        "L1/add": [(128, 128), (128, 128)],
        "L1/multiply": [(128, 128), (128, 128)],
        "L1/rsqrt": [(128, 128)],
        "L1/clamp": [(128, 128)],
        # Level 1 — loss (2 inputs, first dim is batch)
        "L1/cross_entropy": [(128, 64), (128, 64)],
        "L1/mse_loss": [(128, 64), (128, 64)],
        "L1/cosine_sim": [(128, 64), (128, 64)],
        # Level 1 — index
        "L1/embedding_lookup": [(1024, 64), (32,)],
        "L1/one_hot": [(64,)],
        # Level 1 — genomics
        "L1/nucleotide_onehot": [(256,)],
        # Level 2 — matmul + activation
        "L2/matmul_relu": [(128, 128), (128, 128)],
        "L2/matmul_gelu": [(128, 128), (128, 128)],
        "L2/matmul_silu": [(128, 128), (128, 128)],
        # Level 2 — norm + residual
        "L2/rmsnorm_residual": [(128, 128), (128, 128)],
        "L2/layernorm_residual": [(128, 128), (128, 128)],
        # Level 2 — gated MLP
        "L2/swiglu": [(64, 128), (128, 256), (128, 256)],
        "L2/geglu": [(64, 128), (128, 256), (128, 256)],
        # Level 2 — linear
        "L2/linear_bias_relu": [(128, 128), (128, 256), (256,)],
        # Level 2 — attention components
        "L2/qk_softmax": [(64, 16), (64, 16)],
        "L2/fused_softmax_cross_entropy": [(64, 32), (64, 32)],
        "L2/sigmoid_bce": [(128, 64), (128, 64)],
        # Level 2 — genomics
        "L2/pwm_scan": [(64, 4), (4, 8)],
        "L2/pairwise_distance": [(32, 8)],
        # Level 3 — attention
        "L3/flash_attention": [(32, 8), (32, 8), (32, 8)],
        "L3/multi_head_attention": [(2, 32, 8), (2, 32, 8), (2, 32, 8)],
        # Level 3 — MLP
        "L3/gated_mlp": [(16, 32), (32, 64), (32, 64), (64, 32)],
        # Level 3 — full model
        "L3/transformer_block": [(8, 16), (16, 8), (16, 8), (16, 8), (8, 16), (16, 16)],
        # Level 3 — genomics
        "L3/triangle_update": [(4, 4, 8), (4, 4)],
    },
    # ------------------------------------------------------------------
    # N256: smallest primary size — still dominated by launch overhead
    # ------------------------------------------------------------------
    "N256": {
        "L1/relu": [(256, 256)],
        "L1/gelu": [(256, 256)],
        "L1/silu": [(256, 256)],
        "L1/sigmoid": [(256, 256)],
        "L1/tanh": [(256, 256)],
        "L1/layernorm": [(256, 256)],
        "L1/rmsnorm": [(256, 256)],
        "L1/matmul": [(256, 256), (256, 256)],
        "L1/batched_matmul": [(4, 64, 64), (4, 64, 64)],
        "L1/outer_product": [(256,), (256,)],
        "L1/reduce_sum": [(256, 256)],
        "L1/reduce_max": [(256, 256)],
        "L1/reduce_mean": [(256, 256)],
        "L1/softmax": [(256, 256)],
        "L1/log_softmax": [(256, 256)],
        "L1/exp": [(256, 256)],
        "L1/log": [(256, 256)],
        "L1/add": [(256, 256), (256, 256)],
        "L1/multiply": [(256, 256), (256, 256)],
        "L1/rsqrt": [(256, 256)],
        "L1/clamp": [(256, 256)],
        "L1/cross_entropy": [(128, 64), (128, 64)],
        "L1/mse_loss": [(256, 128), (256, 128)],
        "L1/cosine_sim": [(256, 128), (256, 128)],
        "L1/embedding_lookup": [(4096, 128), (64,)],
        "L1/one_hot": [(128,)],
        "L1/nucleotide_onehot": [(512,)],
        "L2/matmul_relu": [(256, 256), (256, 256)],
        "L2/matmul_gelu": [(256, 256), (256, 256)],
        "L2/matmul_silu": [(256, 256), (256, 256)],
        "L2/rmsnorm_residual": [(256, 256), (256, 256)],
        "L2/layernorm_residual": [(256, 256), (256, 256)],
        "L2/swiglu": [(128, 256), (256, 512), (256, 512)],
        "L2/geglu": [(128, 256), (256, 512), (256, 512)],
        "L2/linear_bias_relu": [(256, 256), (256, 512), (512,)],
        "L2/qk_softmax": [(64, 32), (64, 32)],
        "L2/fused_softmax_cross_entropy": [(128, 64), (128, 64)],
        "L2/sigmoid_bce": [(256, 128), (256, 128)],
        "L2/pwm_scan": [(128, 4), (4, 8)],
        "L2/pairwise_distance": [(64, 16)],
        "L3/flash_attention": [(64, 16), (64, 16), (64, 16)],
        "L3/multi_head_attention": [(4, 64, 16), (4, 64, 16), (4, 64, 16)],
        "L3/gated_mlp": [(32, 64), (64, 128), (64, 128), (128, 64)],
        "L3/transformer_block": [(16, 32), (32, 16), (32, 16), (32, 16), (16, 32), (32, 32)],
        "L3/triangle_update": [(8, 8, 16), (8, 8)],
    },
    # ------------------------------------------------------------------
    # N1024: moderate — typical Kaggle / Colab scale
    # ------------------------------------------------------------------
    "N1024": {
        "L1/relu": [(1024, 1024)],
        "L1/gelu": [(1024, 1024)],
        "L1/silu": [(1024, 1024)],
        "L1/sigmoid": [(1024, 1024)],
        "L1/tanh": [(1024, 1024)],
        "L1/layernorm": [(1024, 1024)],
        "L1/rmsnorm": [(1024, 1024)],
        "L1/matmul": [(1024, 1024), (1024, 1024)],
        "L1/batched_matmul": [(4, 256, 256), (4, 256, 256)],
        "L1/outer_product": [(1024,), (1024,)],
        "L1/reduce_sum": [(1024, 1024)],
        "L1/reduce_max": [(1024, 1024)],
        "L1/reduce_mean": [(1024, 1024)],
        "L1/softmax": [(1024, 1024)],
        "L1/log_softmax": [(1024, 1024)],
        "L1/exp": [(1024, 1024)],
        "L1/log": [(1024, 1024)],
        "L1/add": [(1024, 1024), (1024, 1024)],
        "L1/multiply": [(1024, 1024), (1024, 1024)],
        "L1/rsqrt": [(1024, 1024)],
        "L1/clamp": [(1024, 1024)],
        "L1/cross_entropy": [(512, 256), (512, 256)],
        "L1/mse_loss": [(1024, 512), (1024, 512)],
        "L1/cosine_sim": [(1024, 512), (1024, 512)],
        "L1/embedding_lookup": [(16000, 512), (256,)],
        "L1/one_hot": [(256,)],
        "L1/nucleotide_onehot": [(1024,)],
        "L2/matmul_relu": [(1024, 1024), (1024, 1024)],
        "L2/matmul_gelu": [(1024, 1024), (1024, 1024)],
        "L2/matmul_silu": [(1024, 1024), (1024, 1024)],
        "L2/rmsnorm_residual": [(1024, 1024), (1024, 1024)],
        "L2/layernorm_residual": [(1024, 1024), (1024, 1024)],
        "L2/swiglu": [(256, 512), (512, 1024), (512, 1024)],
        "L2/geglu": [(256, 512), (512, 1024), (512, 1024)],
        "L2/linear_bias_relu": [(512, 512), (512, 1024), (1024,)],
        "L2/qk_softmax": [(128, 64), (128, 64)],
        "L2/fused_softmax_cross_entropy": [(512, 256), (512, 256)],
        "L2/sigmoid_bce": [(1024, 512), (1024, 512)],
        "L2/pwm_scan": [(256, 4), (4, 8)],
        "L2/pairwise_distance": [(128, 32)],
        "L3/flash_attention": [(128, 32), (128, 32), (128, 32)],
        "L3/multi_head_attention": [(4, 128, 32), (4, 128, 32), (4, 128, 32)],
        "L3/gated_mlp": [(64, 128), (128, 256), (128, 256), (256, 128)],
        "L3/transformer_block": [(32, 64), (64, 32), (64, 32), (64, 32), (32, 64), (64, 64)],
        "L3/triangle_update": [(16, 16, 32), (16, 16)],
    },
    # ------------------------------------------------------------------
    # N4096: large — standard production scale
    # ------------------------------------------------------------------
    "N4096": {
        "L1/relu": [(4096, 4096)],
        "L1/gelu": [(4096, 4096)],
        "L1/silu": [(4096, 4096)],
        "L1/sigmoid": [(4096, 4096)],
        "L1/tanh": [(4096, 4096)],
        "L1/layernorm": [(4096, 4096)],
        "L1/rmsnorm": [(4096, 4096)],
        "L1/matmul": [(4096, 4096), (4096, 4096)],
        "L1/batched_matmul": [(8, 512, 512), (8, 512, 512)],
        "L1/outer_product": [(4096,), (4096,)],
        "L1/reduce_sum": [(4096, 4096)],
        "L1/reduce_max": [(4096, 4096)],
        "L1/reduce_mean": [(4096, 4096)],
        "L1/softmax": [(4096, 4096)],
        "L1/log_softmax": [(4096, 4096)],
        "L1/exp": [(4096, 4096)],
        "L1/log": [(4096, 4096)],
        "L1/add": [(4096, 4096), (4096, 4096)],
        "L1/multiply": [(4096, 4096), (4096, 4096)],
        "L1/rsqrt": [(4096, 4096)],
        "L1/clamp": [(4096, 4096)],
        "L1/cross_entropy": [(2048, 1024), (2048, 1024)],
        "L1/mse_loss": [(4096, 2048), (4096, 2048)],
        "L1/cosine_sim": [(4096, 2048), (4096, 2048)],
        "L1/embedding_lookup": [(32000, 768), (512,)],
        "L1/one_hot": [(512,)],
        "L1/nucleotide_onehot": [(4096,)],
        "L2/matmul_relu": [(4096, 4096), (4096, 4096)],
        "L2/matmul_gelu": [(4096, 4096), (4096, 4096)],
        "L2/matmul_silu": [(4096, 4096), (4096, 4096)],
        "L2/rmsnorm_residual": [(4096, 4096), (4096, 4096)],
        "L2/layernorm_residual": [(4096, 4096), (4096, 4096)],
        "L2/swiglu": [(512, 1024), (1024, 2048), (1024, 2048)],
        "L2/geglu": [(512, 1024), (1024, 2048), (1024, 2048)],
        "L2/linear_bias_relu": [(1024, 1024), (1024, 2048), (2048,)],
        "L2/qk_softmax": [(256, 128), (256, 128)],
        "L2/fused_softmax_cross_entropy": [(2048, 1024), (2048, 1024)],
        "L2/sigmoid_bce": [(4096, 2048), (4096, 2048)],
        "L2/pwm_scan": [(1024, 4), (4, 12)],
        "L2/pairwise_distance": [(256, 64)],
        "L3/flash_attention": [(512, 64), (512, 64), (512, 64)],
        "L3/multi_head_attention": [(8, 256, 64), (8, 256, 64), (8, 256, 64)],
        "L3/gated_mlp": [(256, 512), (512, 1024), (512, 1024), (1024, 512)],
        "L3/transformer_block": [(64, 128), (128, 64), (128, 64), (128, 64), (64, 128), (128, 128)],
        "L3/triangle_update": [(32, 32, 64), (32, 32)],
    },
    # ------------------------------------------------------------------
    # N8192: very large — approaches HBM capacity for FP32 multi-input
    # ------------------------------------------------------------------
    "N8192": {
        "L1/relu": [(8192, 8192)],
        "L1/gelu": [(8192, 8192)],
        "L1/silu": [(8192, 8192)],
        "L1/sigmoid": [(8192, 8192)],
        "L1/tanh": [(8192, 8192)],
        "L1/layernorm": [(8192, 8192)],
        "L1/rmsnorm": [(8192, 8192)],
        "L1/matmul": [(8192, 8192), (8192, 8192)],
        "L1/batched_matmul": [(16, 512, 512), (16, 512, 512)],
        "L1/outer_product": [(8192,), (8192,)],
        "L1/reduce_sum": [(8192, 8192)],
        "L1/reduce_max": [(8192, 8192)],
        "L1/reduce_mean": [(8192, 8192)],
        "L1/softmax": [(8192, 8192)],
        "L1/log_softmax": [(8192, 8192)],
        "L1/exp": [(8192, 8192)],
        "L1/log": [(8192, 8192)],
        "L1/add": [(8192, 8192), (8192, 8192)],
        "L1/multiply": [(8192, 8192), (8192, 8192)],
        "L1/rsqrt": [(8192, 8192)],
        "L1/clamp": [(8192, 8192)],
        "L1/cross_entropy": [(4096, 2048), (4096, 2048)],
        "L1/mse_loss": [(8192, 4096), (8192, 4096)],
        "L1/cosine_sim": [(8192, 4096), (8192, 4096)],
        "L1/embedding_lookup": [(64000, 1024), (1024,)],
        "L1/one_hot": [(1024,)],
        "L1/nucleotide_onehot": [(8192,)],
        "L2/matmul_relu": [(8192, 8192), (8192, 8192)],
        "L2/matmul_gelu": [(8192, 8192), (8192, 8192)],
        "L2/matmul_silu": [(8192, 8192), (8192, 8192)],
        "L2/rmsnorm_residual": [(8192, 8192), (8192, 8192)],
        "L2/layernorm_residual": [(8192, 8192), (8192, 8192)],
        "L2/swiglu": [(1024, 2048), (2048, 4096), (2048, 4096)],
        "L2/geglu": [(1024, 2048), (2048, 4096), (2048, 4096)],
        "L2/linear_bias_relu": [(2048, 2048), (2048, 4096), (4096,)],
        "L2/qk_softmax": [(512, 256), (512, 256)],
        "L2/fused_softmax_cross_entropy": [(4096, 2048), (4096, 2048)],
        "L2/sigmoid_bce": [(8192, 4096), (8192, 4096)],
        "L2/pwm_scan": [(2048, 4), (4, 16)],
        "L2/pairwise_distance": [(512, 128)],
        "L3/flash_attention": [(1024, 128), (1024, 128), (1024, 128)],
        "L3/multi_head_attention": [(16, 512, 128), (16, 512, 128), (16, 512, 128)],
        "L3/gated_mlp": [(512, 1024), (1024, 2048), (1024, 2048), (2048, 1024)],
        "L3/transformer_block": [(128, 256), (256, 128), (256, 128), (256, 128), (128, 256), (256, 256)],
        "L3/triangle_update": [(64, 64, 128), (64, 64)],
    },
    # ------------------------------------------------------------------
    # N16384: extreme — may OOM on A100 80GB for multi-input FP32 kernels
    # ------------------------------------------------------------------
    "N16384": {
        "L1/relu": [(16384, 16384)],
        "L1/gelu": [(16384, 16384)],
        "L1/silu": [(16384, 16384)],
        "L1/sigmoid": [(16384, 16384)],
        "L1/tanh": [(16384, 16384)],
        "L1/layernorm": [(16384, 4096)],
        "L1/rmsnorm": [(16384, 4096)],
        "L1/matmul": [(16384, 16384), (16384, 16384)],
        "L1/batched_matmul": [(16, 1024, 1024), (16, 1024, 1024)],
        "L1/outer_product": [(16384,), (16384,)],
        "L1/reduce_sum": [(16384, 16384)],
        "L1/reduce_max": [(16384, 16384)],
        "L1/reduce_mean": [(16384, 16384)],
        "L1/softmax": [(16384, 16384)],
        "L1/log_softmax": [(16384, 16384)],
        "L1/exp": [(16384, 16384)],
        "L1/log": [(16384, 16384)],
        "L1/add": [(16384, 16384), (16384, 16384)],
        "L1/multiply": [(16384, 16384), (16384, 16384)],
        "L1/rsqrt": [(16384, 16384)],
        "L1/clamp": [(16384, 16384)],
        "L1/cross_entropy": [(8192, 4096), (8192, 4096)],
        "L1/mse_loss": [(16384, 8192), (16384, 8192)],
        "L1/cosine_sim": [(16384, 8192), (16384, 8192)],
        "L1/embedding_lookup": [(128000, 1024), (2048,)],
        "L1/one_hot": [(2048,)],
        "L1/nucleotide_onehot": [(16384,)],
        "L2/matmul_relu": [(16384, 16384), (16384, 16384)],
        "L2/matmul_gelu": [(16384, 16384), (16384, 16384)],
        "L2/matmul_silu": [(16384, 16384), (16384, 16384)],
        "L2/rmsnorm_residual": [(16384, 4096), (16384, 4096)],
        "L2/layernorm_residual": [(16384, 4096), (16384, 4096)],
        "L2/swiglu": [(2048, 4096), (4096, 8192), (4096, 8192)],
        "L2/geglu": [(2048, 4096), (4096, 8192), (4096, 8192)],
        "L2/linear_bias_relu": [(4096, 4096), (4096, 8192), (8192,)],
        "L2/qk_softmax": [(1024, 512), (1024, 512)],
        "L2/fused_softmax_cross_entropy": [(8192, 4096), (8192, 4096)],
        "L2/sigmoid_bce": [(16384, 8192), (16384, 8192)],
        "L2/pwm_scan": [(4096, 4), (4, 16)],
        "L2/pairwise_distance": [(1024, 256)],
        "L3/flash_attention": [(2048, 256), (2048, 256), (2048, 256)],
        "L3/multi_head_attention": [(32, 1024, 256), (32, 1024, 256), (32, 1024, 256)],
        "L3/gated_mlp": [(1024, 2048), (2048, 4096), (2048, 4096), (4096, 2048)],
        "L3/transformer_block": [(256, 512), (512, 256), (512, 256), (512, 256), (256, 512), (512, 512)],
        "L3/triangle_update": [(128, 128, 256), (128, 128)],
    },
}
