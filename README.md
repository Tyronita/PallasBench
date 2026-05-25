# PallasBench: Benchmarking Custom Kernel Generation in JAX Pallas

> **Can LLMs (and humans) write efficient Pallas kernels? A benchmark suite for JAX's hardware-agnostic kernel DSL, inspired by KernelBench.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.6%2B-green.svg)](https://github.com/jax-ml/jax)

---

## Motivation

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Stanford, ICML 2025) established the gold standard for evaluating LLM-generated CUDA/Triton GPU kernels with 250 tasks across 4 difficulty levels. Follow-up work---[KernelBench-v2](https://github.com/Lossfunk/KernelBench-v2), [KernelBench-X](https://github.com/BonnieW05/KernelBenchX) (176 tasks, 15 categories), [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) (285 tasks, CUDA/Triton/AscendC/Pallas/SYCL), and [kernelbench.com](https://github.com/Infatoshi/kernelbench.com)---extended this to more DSLs, hardware, and difficulty.

**However, no benchmark focuses specifically on [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)**, JAX's built-in kernel language that targets both TPU (via Mosaic) and GPU (via Mosaic GPU / Triton). Pallas occupies a unique niche:

- **Hardware-agnostic**: one kernel definition can target TPU *and* GPU
- **Python-native**: uses JAX tracing, `jax.numpy`, and standard Python---no C++/CUDA
- **Composable**: Pallas kernels work with `jax.jit`, `jax.vmap`, `jax.grad`
- **Underrepresented in LLM training data**: MultiKernelBench found Pallas Pass@1 peaks at just 8.4% (Claude Sonnet 4) vs. 45%+ for CUDA

PallasBench fills this gap with **150 benchmark tasks** across 3 levels, a **reference JAX baseline** for every task, and a **speedup-aware evaluation harness** (`fast_p`) adapted from KernelBench.

---

## Landscape: Kernel Benchmarks at a Glance

| Benchmark | Tasks | DSLs | Hardware | Metric | Year |
|-----------|-------|------|----------|--------|------|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 250 | CUDA, Triton, CUTLASS, TK | NVIDIA GPU (L40S) | `fast_p` | 2025 |
| [KernelBench-v2](https://github.com/Lossfunk/KernelBench-v2) | 250+ | CUDA, Triton | NVIDIA GPU | `fast_p` | 2025 |
| [KernelBench-X](https://github.com/BonnieW05/KernelBenchX) | 176 | Triton | NVIDIA GPU (6 cards) | Call/Exe/Perf | 2025 |
| [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) | 285 | CUDA, Triton, AscendC, TileLang, Pallas, SYCL | GPU, NPU, TPU | Compilation@k, Pass@k, SpeedUp | 2025 |
| [kernelbench.com](https://github.com/Infatoshi/kernelbench.com) | 7-58 | CUDA | NVIDIA GPU (3090, H100, B200) | Speedup | 2026 |
| **PallasBench** (this work) | **150** | **Pallas (JAX)** | **TPU + GPU** | **`fast_p`** | **2026** |

---

## Benchmark Design

### Task Levels

Inspired by KernelBench's hierarchical structure, PallasBench defines three levels:

#### Level 1: Single Operators (80 tasks)
Foundational building blocks. Each task provides a JAX reference implementation (`jax.numpy` / `jax.lax`) and asks for a Pallas kernel replacement.

| Category | Example Tasks | Count |
|----------|--------------|-------|
| **Activation** | ReLU, GELU, SiLU, Swish, Mish | 10 |
| **MatMul** | GEMM, batched matmul, outer product | 10 |
| **Reduce** | sum, max, mean, argmax, prod | 8 |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm, GroupNorm | 8 |
| **Elementwise** | add, multiply, exp, log, rsqrt, clamp | 10 |
| **Softmax** | row softmax, online softmax, log-softmax | 6 |
| **Convolution** | conv1d, conv2d, depthwise conv | 6 |
| **Pooling** | max pool, avg pool, adaptive pool | 6 |
| **Loss** | cross-entropy, MSE, cosine similarity | 6 |
| **Index/Scatter** | gather, scatter_add, one_hot, embedding | 10 |

#### Level 2: Fusion Patterns (50 tasks)
Fused kernels where a single Pallas `pallas_call` should outperform chained JAX ops.

| Category | Example Tasks | Count |
|----------|--------------|-------|
| **MatMul + Activation** | matmul+relu, matmul+gelu, matmul+silu | 8 |
| **Norm + Residual** | RMSNorm+residual, LayerNorm+dropout | 8 |
| **Attention Components** | QK^T+mask+softmax, softmax+dropout+V | 8 |
| **MLP Fusions** | SwiGLU, GeGLU, linear+bias+act | 8 |
| **Conv Fusions** | conv+bias+relu, conv+bn+relu | 6 |
| **Loss Fusions** | log_softmax+nll, sigmoid+bce | 6 |
| **Optimizer Steps** | adam_update, sgd_momentum | 6 |

#### Level 3: Full Architectures (20 tasks)
End-to-end model components or full forward passes.

| Category | Example Tasks | Count |
|----------|--------------|-------|
| **Transformer Blocks** | multi-head attention, flash attention | 5 |
| **MLP Blocks** | SwiGLU MLP, gated MLP | 4 |
| **Full Models** | MiniGPT block, Mamba block, MobileNet block | 5 |
| **Specialized** | paged attention, ring attention, RoPE | 6 |

### Evaluation Metrics

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
with `atol=1e-3, rtol=1e-3` (configurable).

**Speedup** is measured as:
```
speedup = median_time(jax_baseline) / median_time(pallas_kernel)
```
using 100 timed iterations with 10 warmup runs, after `jax.block_until_ready()`.

---

## Installation

```bash
# Clone
git clone https://github.com/Tyronita/PallasBench.git
cd PallasBench

# Install (requires Python 3.10+, JAX 0.6+)
pip install -e .

# For TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# For GPU (CUDA 12)
pip install jax[cuda12]
```

---

## Quick Start

### Run a single benchmark task

```python
from pallasbench import benchmark, kernels, baselines

# Get a Level 1 task
task = kernels.level1.softmax
baseline = baselines.jax_softmax

# Run evaluation
result = benchmark.evaluate_kernel(
    pallas_fn=task.pallas_kernel,
    baseline_fn=baseline,
    input_shapes=[(1024, 1024)],
    dtype="float32",
    n_correctness=5,
    n_warmup=10,
    n_trials=100,
)

print(f"Correct: {result.correct}")
print(f"Speedup: {result.speedup:.2f}x")
print(f"fast_1:  {result.speedup > 1.0 and result.correct}")
```

### Run the full benchmark suite

```bash
# All levels, default settings
python scripts/run_benchmark.py --levels 1 2 3

# Single level with custom threshold
python scripts/run_benchmark.py --levels 1 --fast-p 2.0

# Specific category
python scripts/run_benchmark.py --levels 1 --category softmax
```

### Analyze results

```bash
python scripts/analyze_results.py --results-dir results/
```

---

## Repository Structure

```
PallasBench/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Package metadata
├── pallasbench/
│   ├── __init__.py
│   ├── benchmark.py                   # Core evaluation harness
│   ├── metrics.py                     # fast_p computation
│   ├── utils.py                       # Timing, correctness, RNG
│   ├── kernels/
│   │   ├── level1/                    # 80 single-operator tasks
│   │   │   ├── matmul.py             # Tiled matmul with BlockSpec
│   │   │   ├── softmax.py            # Row-wise online softmax
│   │   │   ├── layernorm.py          # Fused layer normalization
│   │   │   ├── rmsnorm.py            # Root mean square norm
│   │   │   ├── relu.py               # Elementwise ReLU
│   │   │   ├── gelu.py               # Gaussian error linear unit
│   │   │   ├── reduce_sum.py         # Parallel reduction
│   │   │   └── embedding_lookup.py   # Gather-based embedding
│   │   ├── level2/                    # 50 fusion tasks
│   │   │   ├── matmul_relu.py        # Fused GEMM + ReLU
│   │   │   ├── rmsnorm_residual.py   # RMSNorm + residual add
│   │   │   └── swiglu.py             # Fused SwiGLU activation
│   │   └── level3/                    # 20 architecture tasks
│   │       └── flash_attention.py     # Tiled flash attention
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── jax_baseline.py           # Pure JAX reference impls
│   └── tasks.py                       # Task registry & metadata
├── paper/
│   └── PAPER.md                       # Full technical paper
├── results/                           # Benchmark output (JSON)
└── scripts/
    ├── run_benchmark.py               # CLI benchmark runner
    └── analyze_results.py             # Results analysis & tables
```

---

## Pallas Kernel Anatomy

Every PallasBench task follows this pattern:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def my_kernel(x_ref, o_ref):
    """The kernel body operates on Ref types (mutable SRAM buffers)."""
    x = x_ref[...]          # Read from HBM -> SRAM (GPU) or SRAM -> regs (TPU)
    o_ref[...] = f(x)       # Write result back

def my_op(x: jax.Array) -> jax.Array:
    """The host-side wrapper using pallas_call."""
    return pl.pallas_call(
        my_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_blocks,),
        in_specs=[pl.BlockSpec((block_size,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block_size,), lambda i: (i,)),
    )(x)
```

Key Pallas concepts exercised by PallasBench:

| Concept | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|
| `pallas_call` + `grid` | All tasks | All tasks | All tasks |
| `BlockSpec` tiling | MatMul, Conv | All fusions | All |
| `program_id` indexing | Reduce, Softmax | Attention | Flash Attn |
| Accumulator patterns | Reduce | Norm+Residual | Transformer |
| `run_scoped` (GPU) | - | - | Flash Attn |
| Multi-dim grids | MatMul, Conv | MLP fusions | Full models |
| `jax.lax.fori_loop` | Softmax | Attention | Ring Attn |

---

## Relationship to Existing Work

### KernelBench (Stanford)
PallasBench directly adopts KernelBench's `fast_p` metric, hierarchical level structure, and evaluation methodology. The key difference: KernelBench targets **PyTorch -> CUDA/Triton** transpilation on NVIDIA GPUs, while PallasBench targets **JAX -> Pallas** on both TPU and GPU.

### MultiKernelBench
MultiKernelBench (285 tasks) includes Pallas as one of six backends but reports only 8.4% Pass@1 for the best model. PallasBench goes deeper: more Pallas-specific tasks, reference implementations in idiomatic Pallas, and analysis of *why* LLMs struggle with Pallas (API hallucination, rank constraints, BlockSpec errors).

### pallas-forge
[pallas-forge](https://github.com/linhkid/pallas-forge) provides 3 auto-tuned production kernels (MatMul, RMSNorm+residual, SwiGLU). PallasBench includes these as Level 2 reference points and extends to 150 tasks.

### ejkernel / EasyDeL
[ejkernel](https://github.com/erfanzar/ejkernel) provides production Pallas+Triton kernels for LLM serving. PallasBench draws on these patterns for Level 3 architecture tasks.

---

## Why Pallas is Hard for LLMs

MultiKernelBench's analysis revealed systematic failure modes unique to Pallas:

1. **API Hallucination (22.4% of failures)**: LLMs invent non-existent `pallas_call` keyword arguments
2. **Rank Constraints (5%)**: Pallas TPU lowering requires blocks of rank >= 1; LLMs produce scalar BlockSpecs
3. **Runtime vs Compile Errors**: Python's dynamic typing shifts errors from compile-time to runtime, making debugging harder
4. **Memory Model Confusion**: LLMs conflate GPU memory semantics (HBM/SRAM/registers) with TPU semantics (VMEM/SMEM/registers)
5. **Limited Training Data**: Pallas code is orders of magnitude rarer than CUDA/Triton in public corpora

PallasBench's level structure explicitly targets these failure modes, progressing from simple patterns (Level 1) that establish API fluency to complex compositions (Level 3) that require deep hardware understanding.

---

## Contributing

We welcome contributions:

- **New tasks**: Add a kernel + baseline pair following the existing pattern
- **New hardware results**: Run the suite on different TPU/GPU generations
- **LLM evaluations**: Use PallasBench to evaluate frontier models on Pallas generation
- **Optimized kernels**: Submit faster Pallas implementations as community solutions

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

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) by Stanford Scaling Intelligence Lab for the benchmark design and `fast_p` metric
- [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) for cross-platform Pallas evaluation insights
- [JAX/Pallas team](https://github.com/jax-ml/jax) at Google for the kernel DSL and documentation
- [pallas-forge](https://github.com/linhkid/pallas-forge) for auto-tuning patterns and reference kernels
- [ejkernel](https://github.com/erfanzar/ejkernel) for production kernel patterns

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
