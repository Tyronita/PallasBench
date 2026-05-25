"""Level 2: Fusion pattern Pallas kernels."""

from pallasbench.kernels.level2 import (
    matmul_relu,
    matmul_gelu,
    matmul_silu,
    rmsnorm_residual,
    layernorm_residual,
    swiglu,
    geglu,
    linear_bias_relu,
    qk_softmax,
    fused_softmax_cross_entropy,
    sigmoid_bce,
    pwm_scan,
    pairwise_distance,
)
