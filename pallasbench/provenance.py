"""Provenance tracking: maps every PallasBench task to its official source."""

PROVENANCE: dict[str, dict] = {
    # =========================================================================
    # Level 1: Single Operators
    # =========================================================================

    # --- Activation ---
    "L1/relu": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Pallas quickstart tutorial elementwise kernel pattern",
        "domain": "JAX Core",
    },
    "L1/gelu": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Transcendental function (tanh) inside Pallas kernel",
        "domain": "JAX Core",
    },
    "L1/silu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "SiLU gate path from tokamax gated_linear_unit kernel",
        "domain": "OpenXLA",
    },
    "L1/sigmoid": {
        "source": "jax-ml/jax",
        "reference": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.sigmoid.html",
        "description": "Logistic sigmoid, gating primitive",
        "domain": "JAX Core",
    },
    "L1/tanh": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Core transcendental used in GELU approximation",
        "domain": "JAX Core",
    },

    # --- Normalization ---
    "L1/layernorm": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "tokamax.layer_norm production kernel",
        "domain": "OpenXLA",
    },
    "L1/rmsnorm": {
        "source": "linhkid/pallas-forge",
        "reference": "https://github.com/linhkid/pallas-forge",
        "description": "pallas-forge RMSNorm kernel (3.44x over XLA)",
        "domain": "Community",
    },

    # --- MatMul ---
    "L1/matmul": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/tpu/matmul.html",
        "description": "Official Pallas TPU matmul tutorial with BlockSpec tiling",
        "domain": "JAX Core",
    },
    "L1/batched_matmul": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/tpu/matmul.html",
        "description": "Batched GEMM for multi-head attention",
        "domain": "JAX Core",
    },
    "L1/outer_product": {
        "source": "google-deepmind/alphafold3",
        "reference": "https://github.com/google-deepmind/alphafold3",
        "description": "Rank-1 update pattern used in Evoformer outer product mean",
        "domain": "Scientific AI",
    },

    # --- Reduce ---
    "L1/reduce_sum": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Row-wise reduction pattern from Pallas quickstart",
        "domain": "JAX Core",
    },
    "L1/reduce_max": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/issues/34620",
        "description": "Max reduction, tested against argmax TPU issue #34620",
        "domain": "JAX Core",
    },
    "L1/reduce_mean": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Mean reduction, building block for normalization",
        "domain": "JAX Core",
    },

    # --- Softmax ---
    "L1/softmax": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Numerically stable softmax with max subtraction",
        "domain": "JAX Core",
    },
    "L1/log_softmax": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "Log-softmax component of tokamax cross-entropy kernel",
        "domain": "OpenXLA",
    },

    # --- Elementwise ---
    "L1/exp": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Transcendental exp, core of softmax and loss functions",
        "domain": "JAX Core",
    },
    "L1/log": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Transcendental log, used in cross-entropy",
        "domain": "JAX Core",
    },
    "L1/add": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Binary elementwise addition for residual connections",
        "domain": "JAX Core",
    },
    "L1/multiply": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Binary elementwise multiply for gating and scaling",
        "domain": "JAX Core",
    },
    "L1/rsqrt": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Reciprocal sqrt critical in normalization layers",
        "domain": "JAX Core",
    },
    "L1/clamp": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/quickstart.html",
        "description": "Clip/clamp for gradient clipping patterns",
        "domain": "JAX Core",
    },

    # --- Loss ---
    "L1/cross_entropy": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "Derived from tokamax linear_softmax_cross_entropy_loss",
        "domain": "OpenXLA",
    },
    "L1/mse_loss": {
        "source": "jax-ml/jax",
        "reference": "https://jax.readthedocs.io/en/latest/jax.numpy.html",
        "description": "Standard regression loss",
        "domain": "JAX Core",
    },
    "L1/cosine_sim": {
        "source": "jax-ml/jax",
        "reference": "https://jax.readthedocs.io/en/latest/jax.numpy.html",
        "description": "Cosine similarity for embedding retrieval",
        "domain": "JAX Core",
    },

    # --- Index ---
    "L1/embedding_lookup": {
        "source": "jax-ml/jax",
        "reference": "https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html",
        "description": "Gather-style embedding, SparseCore optimized on TPU",
        "domain": "JAX Core",
    },
    "L1/one_hot": {
        "source": "jax-ml/jax",
        "reference": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.one_hot.html",
        "description": "One-hot encoding for label preparation",
        "domain": "JAX Core",
    },

    # =========================================================================
    # Level 2: Fusion Patterns
    # =========================================================================

    "L2/matmul_relu": {
        "source": "keras-team/keras-io",
        "reference": "https://keras.io/guides/define_custom_kernel/",
        "description": "Keras FusedDense tutorial: matmul + activation in one kernel",
        "domain": "Keras",
    },
    "L2/matmul_gelu": {
        "source": "keras-team/keras-io",
        "reference": "https://keras.io/guides/define_custom_kernel/",
        "description": "FusedDense variant with GELU activation",
        "domain": "Keras",
    },
    "L2/matmul_silu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "SiLU gate matmul from tokamax gated_linear_unit",
        "domain": "OpenXLA",
    },
    "L2/rmsnorm_residual": {
        "source": "linhkid/pallas-forge",
        "reference": "https://github.com/linhkid/pallas-forge",
        "description": "pallas-forge fused RMSNorm+residual (3.44x over XLA)",
        "domain": "Community",
    },
    "L2/layernorm_residual": {
        "source": "AI-Hypercomputer/maxtext",
        "reference": "https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html",
        "description": "Pre-norm residual pattern from MaxText attention blocks",
        "domain": "Google AI",
    },
    "L2/swiglu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "tokamax.gated_linear_unit — SwiGLU variant",
        "domain": "OpenXLA",
    },
    "L2/geglu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "tokamax.gated_linear_unit — GeGLU variant (PaLM/Gemma)",
        "domain": "OpenXLA",
    },
    "L2/linear_bias_relu": {
        "source": "keras-team/keras-io",
        "reference": "https://keras.io/guides/define_custom_kernel/",
        "description": "Keras FusedDense with bias: matmul+bias+relu",
        "domain": "Keras",
    },
    "L2/qk_softmax": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py",
        "description": "QK^T+softmax component from JAX flash_attention.py",
        "domain": "JAX Core",
    },
    "L2/fused_softmax_cross_entropy": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "tokamax.linear_softmax_cross_entropy_loss — memory efficient",
        "domain": "OpenXLA",
    },
    "L2/sigmoid_bce": {
        "source": "jax-ml/jax",
        "reference": "https://jax.readthedocs.io/en/latest/jax.numpy.html",
        "description": "Numerically stable fused sigmoid + binary cross-entropy",
        "domain": "JAX Core",
    },

    # =========================================================================
    # Level 3: Architecture Components
    # =========================================================================

    "L3/flash_attention": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py",
        "description": "Official JAX Pallas flash attention TPU kernel",
        "domain": "JAX Core",
    },
    "L3/multi_head_attention": {
        "source": "AI-Hypercomputer/maxtext",
        "reference": "https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html",
        "description": "MaxText splash attention training kernel pattern",
        "domain": "Google AI",
    },
    "L3/gated_mlp": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax",
        "description": "Full gated MLP block using tokamax gated_linear_unit",
        "domain": "OpenXLA",
    },
    "L3/transformer_block": {
        "source": "AI-Hypercomputer/maxtext",
        "reference": "https://github.com/AI-Hypercomputer/maxtext",
        "description": "Simplified pre-norm transformer block from MaxText arch",
        "domain": "Google AI",
    },
}


def get_provenance(task_name: str) -> dict | None:
    return PROVENANCE.get(task_name)


def provenance_table() -> str:
    lines = [
        f"| {'Task':<35} | {'Source':<30} | {'Domain':<15} |",
        f"|{'-'*37}|{'-'*32}|{'-'*17}|",
    ]
    for name, info in sorted(PROVENANCE.items()):
        lines.append(
            f"| {name:<35} | {info['source']:<30} | {info['domain']:<15} |"
        )
    return "\n".join(lines)
