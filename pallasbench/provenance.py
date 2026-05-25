"""Provenance tracking: maps every PallasBench task to its official source."""

PROVENANCE: dict[str, dict] = {
    # =========================================================================
    # Level 1: Single Operators
    # =========================================================================

    # --- Activation ---
    "L1/relu": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Pallas quickstart tutorial elementwise kernel pattern",
        "domain": "JAX Core",
    },
    "L1/gelu": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Transcendental function (tanh) inside Pallas kernel",
        "domain": "JAX Core",
    },
    "L1/silu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py",
        "description": "SiLU gate path from tokamax gated_linear_unit kernel",
        "domain": "OpenXLA",
    },
    "L1/sigmoid": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py",
        "description": "Logistic sigmoid, gating primitive",
        "domain": "JAX Core",
    },
    "L1/tanh": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Core transcendental used in GELU approximation",
        "domain": "JAX Core",
    },

    # --- Normalization ---
    "L1/layernorm": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/normalization/api.py",
        "description": "tokamax.layer_norm production kernel",
        "domain": "OpenXLA",
    },
    "L1/rmsnorm": {
        "source": "linhkid/pallas-forge",
        "reference": "https://github.com/linhkid/pallas-forge/blob/main/pallas_forge/kernels/rmsnorm.py",
        "description": "pallas-forge RMSNorm kernel (3.44x over XLA)",
        "domain": "Community",
        "change": "use jax.lax.rsqrt to match the JAX 0.10 API where jnp.rsqrt was removed",
    },

    # --- MatMul ---
    "L1/matmul": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/matmul.ipynb",
        "description": "Official Pallas TPU matmul tutorial with BlockSpec tiling",
        "domain": "JAX Core",
    },
    "L1/batched_matmul": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/matmul.ipynb",
        "description": "Batched GEMM for multi-head attention",
        "domain": "JAX Core",
    },
    "L1/outer_product": {
        "source": "google-deepmind/alphafold3",
        "reference": "https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/network/evoformer.py",
        "description": "Rank-1 update pattern used in Evoformer outer product mean",
        "domain": "Scientific AI",
    },

    # --- Reduce ---
    "L1/reduce_sum": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
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
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Mean reduction, building block for normalization",
        "domain": "JAX Core",
    },

    # --- Softmax ---
    "L1/softmax": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Numerically stable softmax with max subtraction",
        "domain": "JAX Core",
    },
    "L1/log_softmax": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py",
        "description": "Log-softmax component of tokamax cross-entropy kernel",
        "domain": "OpenXLA",
    },

    # --- Elementwise ---
    "L1/exp": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Transcendental exp, core of softmax and loss functions",
        "domain": "JAX Core",
    },
    "L1/log": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Transcendental log, used in cross-entropy",
        "domain": "JAX Core",
    },
    "L1/add": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Binary elementwise addition for residual connections",
        "domain": "JAX Core",
    },
    "L1/multiply": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Binary elementwise multiply for gating and scaling",
        "domain": "JAX Core",
    },
    "L1/rsqrt": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py",
        "description": "Reciprocal sqrt critical in normalization layers",
        "domain": "JAX Core",
        "change": "call jax.lax.rsqrt instead of deprecated jnp.rsqrt so CPU interpret mode stays valid",
    },
    "L1/clamp": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb",
        "description": "Clip/clamp for gradient clipping patterns",
        "domain": "JAX Core",
    },

    # --- Loss ---
    "L1/cross_entropy": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py",
        "description": "Derived from tokamax linear_softmax_cross_entropy_loss",
        "domain": "OpenXLA",
    },
    "L1/mse_loss": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py",
        "description": "Standard regression loss via jnp.mean / jnp.square",
        "domain": "JAX Core",
    },
    "L1/cosine_sim": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py",
        "description": "Cosine similarity for embedding retrieval",
        "domain": "JAX Core",
    },

    # --- Index ---
    "L1/embedding_lookup": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/sparsecore.ipynb",
        "description": "Gather-style embedding, SparseCore optimized on TPU",
        "domain": "JAX Core",
    },
    "L1/one_hot": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py",
        "description": "One-hot encoding for label preparation",
        "domain": "JAX Core",
    },
    "L1/nucleotide_onehot": {
        "source": "google-deepmind/deepmind-research",
        "reference": "https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer.py",
        "description": "DNA nucleotide one-hot encoding (A/C/G/T -> 4-channel) from Enformer input pipeline",
        "domain": "Genomics",
    },

    # =========================================================================
    # Level 2: Fusion Patterns
    # =========================================================================

    "L2/matmul_relu": {
        "source": "keras-team/keras-io",
        "reference": "https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py",
        "description": "Keras FusedDense tutorial: matmul + activation in one kernel",
        "domain": "Keras",
    },
    "L2/matmul_gelu": {
        "source": "keras-team/keras-io",
        "reference": "https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py",
        "description": "FusedDense variant with GELU activation",
        "domain": "Keras",
    },
    "L2/matmul_silu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py",
        "description": "SiLU gate matmul from tokamax gated_linear_unit",
        "domain": "OpenXLA",
    },
    "L2/rmsnorm_residual": {
        "source": "linhkid/pallas-forge",
        "reference": "https://github.com/linhkid/pallas-forge/blob/main/pallas_forge/kernels/rmsnorm.py",
        "description": "pallas-forge fused RMSNorm+residual (3.44x over XLA)",
        "domain": "Community",
        "change": "reuse the shared jax.lax.rsqrt fallback so fused RMSNorm works on CPU interpret mode",
    },
    "L2/layernorm_residual": {
        "source": "AI-Hypercomputer/maxtext",
        "reference": "https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/normalizations.py",
        "description": "Pre-norm residual pattern from MaxText attention blocks",
        "domain": "Google AI",
    },
    "L2/swiglu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py",
        "description": "tokamax.gated_linear_unit — SwiGLU variant",
        "domain": "OpenXLA",
    },
    "L2/geglu": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py",
        "description": "tokamax.gated_linear_unit — GeGLU variant (PaLM/Gemma)",
        "domain": "OpenXLA",
    },
    "L2/linear_bias_relu": {
        "source": "keras-team/keras-io",
        "reference": "https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py",
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
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py",
        "description": "tokamax.linear_softmax_cross_entropy_loss — memory efficient",
        "domain": "OpenXLA",
    },
    "L2/sigmoid_bce": {
        "source": "jax-ml/jax",
        "reference": "https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py",
        "description": "Numerically stable fused sigmoid + binary cross-entropy",
        "domain": "JAX Core",
    },
    "L2/pwm_scan": {
        "source": "google-deepmind/deepmind-research",
        "reference": "https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer.py",
        "description": "Position Weight Matrix motif scanning — Enformer conv tower / PWMScan (Bioinformatics 2018)",
        "domain": "Genomics",
    },
    "L2/pairwise_distance": {
        "source": "google-deepmind/alphafold3",
        "reference": "https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/jax/geometry/vector.py",
        "description": "Pairwise Euclidean distance matrix for structural biology distance maps",
        "domain": "Genomics",
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
        "reference": "https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/kernels/attention/splash_attention_kernel.py",
        "description": "MaxText splash attention training kernel pattern",
        "domain": "Google AI",
    },
    "L3/gated_mlp": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py",
        "description": "Full gated MLP block using tokamax gated_linear_unit",
        "domain": "OpenXLA",
    },
    "L3/transformer_block": {
        "source": "AI-Hypercomputer/maxtext",
        "reference": "https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/decoders.py",
        "description": "Simplified pre-norm transformer block from MaxText arch",
        "domain": "Google AI",
        "change": "swap jnp.rsqrt for jax.lax.rsqrt so CPU interpret mode keeps matching the baseline",
    },
    "L3/triangle_update": {
        "source": "openxla/tokamax",
        "reference": "https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/triangle_multiplication/api.py",
        "description": "Triangle multiplication update used in AlphaFold-style Pairformer",
        "domain": "Genomics",
    },
}


def get_provenance(task_name: str) -> dict | None:
    return PROVENANCE.get(task_name)


def describe_task(task_name: str, base_doc: str) -> str:
    """Attach provenance metadata (and change notes) to kernel docs."""

    info = PROVENANCE.get(task_name)
    if not info:
        return base_doc

    lines = [base_doc.rstrip()]
    lines.append("")
    lines.append(f"Provenance: {info['source']} — {info['reference']}")
    change_note = info.get("change")
    if change_note:
        lines.append(f"Change: {change_note}")
    return "\n".join(lines)


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
