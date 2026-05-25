# PallasBench: Benchmarking Custom Kernel Generation in JAX Pallas

> **Can LLMs (and humans) write efficient Pallas kernels? A provenance-traced benchmark suite for JAX's hardware-agnostic kernel DSL, with CI/CD evaluation and parametric sizing.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.30%2B-green.svg)](https://github.com/jax-ml/jax)
[![CI](https://github.com/Tyronita/PallasBench/actions/workflows/cpu-correctness.yml/badge.svg)](https://github.com/Tyronita/PallasBench/actions)

---

## Motivation

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Stanford, ICML 2025) established the gold standard for evaluating LLM-generated CUDA/Triton GPU kernels. Follow-up work---[KernelBench-v2](https://github.com/Lossfunk/KernelBench-v2), [KernelBench-X](https://github.com/BonnieW05/KernelBenchX), [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench)---extended this to more DSLs and hardware.

**PallasBench** is the first benchmark focused specifically on [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), JAX's built-in kernel language targeting both TPU (via Mosaic) and GPU (via Mosaic GPU / Triton). Key properties:

- **Hardware-agnostic**: one kernel definition targets TPU *and* GPU
- **Python-native**: uses JAX tracing, `jax.numpy`, standard Python
- **Composable**: Pallas kernels work with `jax.jit`, `jax.vmap`, `jax.grad`
- **Provenance-traced**: every task links to an official source (JAX, Tokamax, MaxText, Keras, etc.)
- **CI/CD integrated**: automated correctness testing, performance regression detection, and scoring
- **Parametric**: each task runs at SMALL/MEDIUM/LARGE sizes for scaling analysis

---

## Landscape: Kernel Benchmarks at a Glance

| Benchmark | Tasks | DSLs | Hardware | Metric | Year |
|-----------|-------|------|----------|--------|------|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 250 | CUDA, Triton, CUTLASS, TK | NVIDIA GPU (L40S) | `fast_p` | 2025 |
| [KernelBench-v2](https://github.com/Lossfunk/KernelBench-v2) | 250+ | CUDA, Triton | NVIDIA GPU | `fast_p` | 2025 |
| [KernelBench-X](https://github.com/BonnieW05/KernelBenchX) | 176 | Triton | NVIDIA GPU (6 cards) | Call/Exe/Perf | 2025 |
| [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) | 285 | CUDA, Triton, AscendC, TileLang, Pallas, SYCL | GPU, NPU, TPU | Compile@k, Pass@k | 2025 |
| **PallasBench** (this work) | **42** | **Pallas (JAX)** | **CPU + TPU + GPU** | **`fast_p`** | **2026** |

---

## Task Provenance

Every PallasBench task traces to an official or well-documented source. This table shows the full provenance:

| Task | Source | Domain |
|------|--------|--------|
| **Level 1: Single Operators (28 tasks)** | | |
| `L1/relu` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — Pallas quickstart | JAX Core |
| `L1/gelu` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — transcendental kernel | JAX Core |
| `L1/silu` | [openxla/tokamax · gated_linear_unit/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py) — SiLU gate | OpenXLA |
| `L1/sigmoid` | [jax-ml/jax · nn/functions.py](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py) | JAX Core |
| `L1/tanh` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — GELU building block | JAX Core |
| `L1/layernorm` | [openxla/tokamax · normalization/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/normalization/api.py) — layer_norm | OpenXLA |
| `L1/rmsnorm` | [pallas-forge · kernels/rmsnorm.py](https://github.com/linhkid/pallas-forge/blob/main/pallas_forge/kernels/rmsnorm.py) — 3.44x over XLA | Community |
| `L1/matmul` | [jax-ml/jax · tpu/matmul.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/matmul.ipynb) — TPU matmul tutorial | JAX Core |
| `L1/batched_matmul` | [jax-ml/jax · tpu/matmul.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/matmul.ipynb) — batch dims for MHA | JAX Core |
| `L1/outer_product` | [alphafold3 · evoformer.py](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/network/evoformer.py) — outer product mean | Scientific AI |
| `L1/reduce_sum` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — reduction pattern | JAX Core |
| `L1/reduce_max` | [jax-ml/jax · issue #34620](https://github.com/jax-ml/jax/issues/34620) — tested against TPU issue | JAX Core |
| `L1/reduce_mean` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — normalization prereq | JAX Core |
| `L1/softmax` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — numerically stable | JAX Core |
| `L1/log_softmax` | [openxla/tokamax · cross_entropy_loss/reference.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py) — log-softmax | OpenXLA |
| `L1/exp` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — transcendental | JAX Core |
| `L1/log` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — loss functions | JAX Core |
| `L1/add` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — residual pattern | JAX Core |
| `L1/multiply` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — gating/scaling | JAX Core |
| `L1/rsqrt` | [jax-ml/jax · lax/lax.py](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py) — normalization core | JAX Core |
| `L1/clamp` | [jax-ml/jax · quickstart.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/quickstart.ipynb) — gradient clipping | JAX Core |
| `L1/cross_entropy` | [openxla/tokamax · cross_entropy_loss/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py) — cross-entropy | OpenXLA |
| `L1/mse_loss` | [jax-ml/jax · numpy/lax_numpy.py](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py) — regression loss | JAX Core |
| `L1/cosine_sim` | [jax-ml/jax · nn/functions.py](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py) — embedding similarity | JAX Core |
| `L1/embedding_lookup` | [jax-ml/jax · tpu/sparsecore.ipynb](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/sparsecore.ipynb) — SparseCore gather | JAX Core |
| `L1/one_hot` | [jax-ml/jax · nn/functions.py](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py) — label prep | JAX Core |
| `L1/nucleotide_onehot` | [deepmind-research · enformer/enformer.py](https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer.py) — DNA encoding | Genomics |
| **Level 2: Fusion Patterns (13 tasks)** | | |
| `L2/matmul_relu` | [keras-io · guides/define_custom_kernel.py](https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py) — FusedDense | Keras |
| `L2/matmul_gelu` | [keras-io · guides/define_custom_kernel.py](https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py) — FusedDense+GELU | Keras |
| `L2/matmul_silu` | [openxla/tokamax · gated_linear_unit/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py) — gate matmul | OpenXLA |
| `L2/rmsnorm_residual` | [pallas-forge · kernels/rmsnorm.py](https://github.com/linhkid/pallas-forge/blob/main/pallas_forge/kernels/rmsnorm.py) — 3.44x speedup | Community |
| `L2/layernorm_residual` | [maxtext · layers/normalizations.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/normalizations.py) — attention blocks | Google AI |
| `L2/swiglu` | [openxla/tokamax · gated_linear_unit/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py) — SwiGLU | OpenXLA |
| `L2/geglu` | [openxla/tokamax · gated_linear_unit/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py) — PaLM/Gemma variant | OpenXLA |
| `L2/linear_bias_relu` | [keras-io · guides/define_custom_kernel.py](https://github.com/keras-team/keras-io/blob/master/guides/define_custom_kernel.py) — bias fusion | Keras |
| `L2/qk_softmax` | [jax-ml/jax · flash_attention.py](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) — QK^T+softmax | JAX Core |
| `L2/fused_softmax_cross_entropy` | [openxla/tokamax · cross_entropy_loss/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py) — memory efficient | OpenXLA |
| `L2/sigmoid_bce` | [jax-ml/jax · nn/functions.py](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py) — numerically stable BCE | JAX Core |
| `L2/pwm_scan` | [deepmind-research · enformer/enformer.py](https://github.com/google-deepmind/deepmind-research/blob/master/enformer/enformer.py) — PWM motif scanning | Genomics |
| `L2/pairwise_distance` | [alphafold3 · geometry/vector.py](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/jax/geometry/vector.py) — structural distance maps | Genomics |
| **Level 3: Architecture Components (5 tasks)** | | |
| `L3/flash_attention` | [jax-ml/jax · flash_attention.py](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) — official TPU kernel | JAX Core |
| `L3/multi_head_attention` | [maxtext · splash_attention_kernel.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/kernels/attention/splash_attention_kernel.py) — splash attention | Google AI |
| `L3/gated_mlp` | [openxla/tokamax · gated_linear_unit/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/gated_linear_unit/api.py) — full block | OpenXLA |
| `L3/transformer_block` | [maxtext · layers/decoders.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/decoders.py) — pre-norm block | Google AI |
| `L3/triangle_update` | [openxla/tokamax · triangle_multiplication/api.py](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/triangle_multiplication/api.py) — Pairformer | Genomics |

### Source Domains

| Domain | Source Projects | Tasks |
|--------|----------------|-------|
| **JAX Core** | jax-ml/jax (official Pallas docs, tutorials, ops) | 22 |
| **OpenXLA** | openxla/tokamax (production kernel library) | 9 |
| **Keras** | keras-team/keras-io (FusedDense tutorial) | 3 |
| **Google AI** | AI-Hypercomputer/maxtext (training framework) | 3 |
| **Community** | pallas-forge (auto-tuned kernels) | 2 |
| **Scientific AI** | google-deepmind/alphafold3 (Evoformer patterns) | 1 |
| **Genomics** | google-deepmind/alphafold3, deepmind-research/Enformer | 4 |

---

## Parametric Benchmark Sizes

Every task supports three sizes to measure scaling behavior:

| Size | Purpose | Typical Dimensions | CI Use |
|------|---------|-------------------|--------|
| **SMALL** | Quick CI smoke test | 128-512 elements per dim | `cpu-correctness.yml` |
| **MEDIUM** | Standard benchmark (default) | 1024-4096 per dim | `gpu-benchmark.yml` |
| **LARGE** | Production-scale stress test | 4096-16384+ per dim | `tpu-benchmark.yml` |

```bash
# Run with specific size
python scripts/run_benchmark.py --levels 1 2 3 --size SMALL
python scripts/run_benchmark.py --levels 1 2 3 --size LARGE
```

---

## CI/CD Architecture

PallasBench includes four GitHub Actions workflows for automated evaluation:

### 1. CPU Correctness (`cpu-correctness.yml`) — **Mandatory**
Runs on every push/PR to `main`/`master`. Tests all kernels using `interpret=True` (CPU emulation of Pallas) with SMALL sizes.

- **Runners**: `ubuntu-latest`, `macos-latest` (multi-OS matrix)
- **Python**: 3.11, 3.12 (multi-version matrix)
- **No hardware required**: uses JAX CPU backend with Pallas interpret mode

### 2. GPU Performance Benchmark (`gpu-benchmark.yml`) — **Mandatory + On-demand**
Runs automatically on push to `main`/`master` when kernel code changes, plus manual `workflow_dispatch`.

- **Runner**: self-hosted GPU (Ampere+ recommended for Pallas GPU, CC 8.0+)
- **Scorer**: `benchmark-action/github-action-benchmark` with configurable alert threshold
- **Alert**: PR comments on regression, workflow failure if threshold exceeded

### 3. TPU Performance Benchmark (`tpu-benchmark.yml`) — **Mandatory + On-demand**
Runs automatically on push to `main`/`master` when kernel code changes, plus manual `workflow_dispatch`.

- **Runner**: self-hosted TPU (v4-8, v5e-1, v5e-4, v6-1)
- **Provisioning**: via [terraform-google-github-actions-runners](https://github.com/terraform-google-modules/terraform-google-github-actions-runners)
- **Scorer**: same `benchmark-action` integration

### 4. CI Scorer (`ci-scorer.yml`) — **Mandatory**
Runs on every push/PR that touches kernel code. Posts a score report comment:

```
## PallasBench CI Score Report
| Metric              | Value |
|---------------------|-------|
| Total Tasks         | 42    |
| Correct             | 40    |
| fast_0 (correctness)| 94.7% |
| fast_1 (faster)     | 78.9% |
```

### Hardware Notes

| Platform | interpret=True | Native | Notes |
|----------|---------------|--------|-------|
| **CPU** | All kernels | N/A | For correctness testing only |
| **GPU (Ampere+)** | All kernels | All kernels | Pallas GPU requires CC 8.0+ |
| **GPU (T4/Turing)** | All kernels | Limited | CC 7.5, some Pallas GPU ops unsupported |
| **TPU v5e** | Via TPU interpret | Native | $1.20/chip-hr, MXU 128x128 |
| **TPU v6 Trillium** | Via TPU interpret | Native | $2.70/chip-hr, MXU 256x256 |

---

## Evaluation Metrics

We adopt KernelBench's `fast_p` metric:

```
fast_p = (# tasks where kernel is correct AND speedup > p) / (# total tasks)
```

| Metric | Meaning |
|--------|---------|
| `fast_0` | Fraction of correct kernels (any speed) |
| `fast_1` | Fraction that are correct *and* faster than JAX baseline |
| `fast_2` | Fraction that are correct *and* 2x+ faster |
| `fast_5` | Fraction that are correct *and* 5x+ faster |

**Correctness** is checked by running both the reference and the Pallas kernel on `N=5` randomly generated inputs and verifying:
```
|output_pallas - output_ref| < atol + rtol * |output_ref|
```

**Speedup** is measured as:
```
speedup = median_time(jax_baseline) / median_time(pallas_kernel)
```

---

## Installation

```bash
git clone https://github.com/Tyronita/PallasBench.git
cd PallasBench
pip install -e .

# For TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# For GPU (CUDA 12)
pip install jax[cuda12]
```

---

## Quick Start

```bash
# All levels, default MEDIUM size
python scripts/run_benchmark.py --levels 1 2 3

# CPU correctness only (no hardware needed)
python scripts/run_benchmark.py --levels 1 2 3 --size SMALL --interpret --correctness-only

# GPU benchmark with specific size
python scripts/run_benchmark.py --levels 1 2 3 --size LARGE

# Filter by category
python scripts/run_benchmark.py --levels 1 --category activation

# Analyze results
python scripts/analyze_results.py --results-dir results/
```

---

## Repository Structure

```
PallasBench/
├── README.md
├── LICENSE
├── pyproject.toml
├── .github/workflows/
│   ├── cpu-correctness.yml        # Multi-OS correctness via interpret=True
│   ├── gpu-benchmark.yml          # On-demand GPU performance + scoring
│   ├── tpu-benchmark.yml          # On-demand TPU performance + scoring
│   └── ci-scorer.yml              # PR score report comments
├── pallasbench/
│   ├── __init__.py
│   ├── benchmark.py               # Core evaluation harness
│   ├── metrics.py                 # fast_p computation
│   ├── utils.py                   # Timing, correctness, RNG
│   ├── provenance.py              # Task provenance registry
│   ├── sizes.py                   # Parametric SMALL/MEDIUM/LARGE configs
│   ├── tasks.py                   # Task registry (42 tasks)
│   ├── kernels/
│   │   ├── level1/                # 28 single-operator tasks
│   │   ├── level2/                # 13 fusion tasks
│   │   └── level3/                # 5 architecture tasks (incl. transformer block, triangle update)
│   └── baselines/
│       └── jax_baseline.py        # Pure JAX reference impls (42 functions)
├── paper/
│   └── PAPER.md
├── results/
└── scripts/
    ├── run_benchmark.py           # CLI with --size, --interpret, --correctness-only
    └── analyze_results.py         # Results analysis & tables
```

---

## Why Pallas is Hard for LLMs

MultiKernelBench's analysis revealed systematic failure modes unique to Pallas:

1. **API Hallucination (22.4% of failures)**: LLMs invent non-existent `pallas_call` keyword arguments
2. **Rank Constraints (5%)**: Pallas TPU lowering requires blocks of rank >= 1
3. **Memory Model Confusion**: LLMs conflate GPU semantics (HBM/SRAM) with TPU semantics (VMEM/SMEM)
4. **Limited Training Data**: Pallas code is orders of magnitude rarer than CUDA/Triton

---

## Contributing

We welcome contributions:

- **New tasks**: Add a kernel + baseline pair with provenance
- **New hardware results**: Run the suite on different TPU/GPU generations
- **LLM evaluations**: Use PallasBench to evaluate frontier models
- **Optimized kernels**: Submit faster Pallas implementations

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

```bibtex
@software{pallasbench2026,
  title   = {PallasBench: Benchmarking Custom Kernel Generation in JAX Pallas},
  author  = {Tyronita},
  year    = {2026},
  url     = {https://github.com/Tyronita/PallasBench},
  note    = {Inspired by KernelBench (Stanford, ICML 2025)}
}
```

---

## Acknowledgements

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — benchmark design and `fast_p` metric
- [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) — cross-platform Pallas evaluation insights
- [JAX/Pallas team](https://github.com/jax-ml/jax) — kernel DSL and documentation
- [openxla/tokamax](https://github.com/openxla/tokamax) — production kernel library
- [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) — training framework Pallas kernels
- [keras-team/keras-io](https://keras.io/guides/define_custom_kernel/) — FusedDense custom kernel tutorial
- [pallas-forge](https://github.com/linhkid/pallas-forge) — auto-tuning patterns and reference kernels
- [ejkernel](https://github.com/erfanzar/ejkernel) — production kernel patterns
- [sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax) — ragged paged attention
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — TPU paged attention integration

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
