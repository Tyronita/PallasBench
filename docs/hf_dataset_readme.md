---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - gpu-kernels
  - pallas
  - jax
  - benchmark
  - a100
  - triton
  - robust-evaluation
pretty_name: PallasBench Robust GPU Kernel Benchmark
size_categories:
  - n<1K
language:
  - en
---

# PallasBench: Robust Pallas GPU Kernel Benchmark Dataset

## Dataset Description

PallasBench is a benchmark dataset of 45 JAX Pallas GPU kernels with robust evaluation results, compilation artifacts, and GPU performance metrics. It adapts the robust evaluation methodology from SakanaAI's "Towards Robust Agentic CUDA Kernel Benchmarking" to the Pallas/JAX compilation pipeline, providing the first GPU-focused benchmark for Pallas kernels.

Each kernel has been fixed for GPU compatibility (resolving Triton's 1M element limit via block size clamping) and evaluated with five robustness filters to distinguish genuinely correct implementations from degenerate ones.

### Links

- **Source Repository**: [https://github.com/Tyronita/PallasBench](https://github.com/Tyronita/PallasBench)
- **Technical Report**: "Towards Robust Pallas Kernel Benchmarking, Verification, and Optimization on GPU"

## Intended Use

This dataset is intended for:

- **LLM-driven kernel optimization**: Training and evaluating LLMs that generate or optimize Pallas GPU kernels
- **Compiler research**: Studying the JAX/Pallas compilation pipeline (Jaxpr --> StableHLO --> Triton MLIR --> PTX)
- **Benchmark development**: Extending GPU kernel benchmarks beyond CUDA to portable DSLs
- **Robustness evaluation research**: Studying and improving robustness filters for kernel correctness verification

## Dataset Structure

### Format

KernelBook-compatible JSONL, one JSON object per kernel.

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `kernel_name` | string | Unique kernel identifier (e.g., `relu`, `attention_forward`) |
| `level` | int | Difficulty level: 1 (single op), 2 (fused pattern), 3 (architecture component) |
| `category` | string | Functional category (see table below) |
| `source_original` | string | Original PallasBench kernel source (TPU-oriented) |
| `source_fixed` | string | GPU-compatible kernel with block size clamping fix |
| `diff` | string | Unified diff between original and fixed source |
| `jaxpr` | string | JAX intermediate representation (Jaxpr DAG) |
| `stablehlo` | string | StableHLO MLIR intermediate representation |
| `result.correctness_passed` | bool | Whether the kernel output matches JAX reference within tolerance |
| `result.robustness_passed` | bool | Whether all five robustness filters passed |
| `result.max_abs_error` | float | Maximum absolute error vs. reference |
| `result.max_rel_error` | float | Maximum relative error vs. reference |
| `result.wall_time_seconds` | float | Total wall-clock time including JIT compilation |
| `result.jit_compile_time_seconds` | float | JIT compilation time (dominated by ptxas) |
| `result.filters.output_range` | object | Output range filter result |
| `result.filters.output_std` | object | Output standard deviation filter result |
| `result.filters.axes_variation` | object | Axes variation filter result |
| `result.filters.input_impact` | object | Input impact filter result |
| `result.filters.source_analysis` | object | Source analysis filter result |
| `result.block_shape` | list[int] | Block dimensions used for tiling |
| `result.grid_shape` | list[int] | Grid dimensions for kernel launch |
| `result.gpu_memory_used_bytes` | int | GPU memory consumed during execution |

### Example Entry

```json
{
  "kernel_name": "relu",
  "level": 1,
  "category": "activation",
  "source_original": "def relu_kernel(x_ref, o_ref):\n    o_ref[...] = jnp.maximum(x_ref[...], 0.0)\n",
  "source_fixed": "def relu_kernel(x_ref, o_ref):\n    o_ref[...] = jnp.maximum(x_ref[...], 0.0)\n",
  "diff": "--- a/relu.py\n+++ b/relu.py\n@@ -5,7 +5,8 @@\n-block_size = n\n+MAX_BLOCK = 65536\n+block_size = min(n, MAX_BLOCK)\n",
  "jaxpr": "{ lambda ; a:f32[4096]. let\n    b:f32[4096] = pallas_call[...] a\n  in (b,) }",
  "stablehlo": "module @relu { ... }",
  "result": {
    "correctness_passed": true,
    "robustness_passed": true,
    "max_abs_error": 0.0,
    "wall_time_seconds": 415.3,
    "jit_compile_time_seconds": 415.0,
    "filters": {
      "output_range": {"passed": true, "value": 6.28},
      "output_std": {"passed": true, "value": 1.42},
      "axes_variation": {"passed": true},
      "input_impact": {"passed": true, "value": 0.31},
      "source_analysis": {"passed": true}
    }
  }
}
```

## Complete Kernel List

### Level 1: Single Operations (27 kernels)

| # | Kernel | Category | Description |
|---|--------|----------|-------------|
| 1 | `relu` | activation | Rectified linear unit |
| 2 | `leaky_relu` | activation | Leaky rectified linear unit |
| 3 | `gelu` | activation | Gaussian error linear unit |
| 4 | `silu` | activation | Sigmoid linear unit (Swish) |
| 5 | `sigmoid` | activation | Logistic sigmoid |
| 6 | `tanh_activation` | activation | Hyperbolic tangent |
| 7 | `softplus` | activation | Smooth ReLU approximation |
| 8 | `elu` | activation | Exponential linear unit |
| 9 | `hardswish` | activation | Hard Swish activation |
| 10 | `mish` | activation | Mish self-regularized activation |
| 11 | `layer_norm` | normalization | Layer normalization |
| 12 | `rms_norm` | normalization | Root mean square normalization |
| 13 | `batch_norm_1d` | normalization | 1D batch normalization |
| 14 | `matmul` | matmul | Matrix multiplication |
| 15 | `row_sum` | reduce | Row-wise summation |
| 16 | `col_sum` | reduce | Column-wise summation |
| 17 | `softmax` | softmax | Softmax normalization |
| 18 | `log_softmax` | softmax | Log-softmax normalization |
| 19 | `vector_add` | elementwise | Element-wise vector addition |
| 20 | `elementwise_mul` | elementwise | Element-wise multiplication |
| 21 | `elementwise_max` | elementwise | Element-wise maximum |
| 22 | `cross_entropy` | loss | Cross-entropy loss |
| 23 | `mse_loss` | loss | Mean squared error loss |
| 24 | `huber_loss` | loss | Huber (smooth L1) loss |
| 25 | `gather` | index | Gather elements by index |
| 26 | `scatter_add` | index | Scatter-add by index |
| 27 | `one_hot` | index | One-hot encoding |

### Level 2: Fused Patterns (13 kernels)

| # | Kernel | Category | Description |
|---|--------|----------|-------------|
| 28 | `fused_relu_matmul` | activation | ReLU followed by matrix multiply |
| 29 | `fused_gelu_bias` | activation | GELU with bias addition |
| 30 | `fused_layer_norm_relu` | normalization | Layer norm fused with ReLU |
| 31 | `fused_matmul_bias` | matmul | Matrix multiply with bias |
| 32 | `fused_matmul_relu` | matmul | Matrix multiply with ReLU |
| 33 | `fused_softmax_cross_entropy` | softmax | Softmax fused with cross-entropy |
| 34 | `fused_residual_norm` | normalization | Residual connection with normalization |
| 35 | `fused_bias_gelu_dropout` | activation | Bias + GELU + dropout fusion |
| 36 | `kmer_count` | genomics | K-mer frequency counting |
| 37 | `reverse_complement` | genomics | DNA reverse complement |
| 38 | `hamming_distance` | genomics | Hamming distance between sequences |
| 39 | `sequence_match` | genomics | Sequence alignment matching |
| 40 | `gc_content` | genomics | GC nucleotide content ratio |

### Level 3: Architecture Components (5 kernels)

| # | Kernel | Category | Description |
|---|--------|----------|-------------|
| 41 | `attention_forward` | attention | Scaled dot-product attention |
| 42 | `multi_head_attention` | attention | Multi-head attention mechanism |
| 43 | `mlp_block` | mlp | Feed-forward MLP block |
| 44 | `transformer_block` | full_model | Full transformer encoder block |
| 45 | `conv1d_genomic` | genomics | 1D convolution for genomic sequences |

## Hardware and Software

### Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA A100 80GB PCIe |
| GPU Memory | 80 GB HBM2e |
| Streaming Multiprocessors | 108 |
| Peak HBM Bandwidth | 2,039 GB/s |
| L2 Cache | 40 MB |
| Compute Capability | 8.0 |
| Instance | Azure Standard_NC24ads_A100_v4 |
| CPU | AMD EPYC 7V13 (24 vCPUs) |
| System RAM | 216 GB |

### Software

| Component | Version |
|-----------|---------|
| JAX | 0.10.1 |
| jaxlib (CUDA) | 0.10.1+cuda12 |
| Triton | 3.7.0 |
| Python | 3.11.9 |
| CUDA Toolkit | 12.6 |

## Provenance

This dataset is derived from the [PallasBench](https://github.com/Tyronita/PallasBench) repository. The original kernels were designed for TPU execution and have been adapted for GPU through a systematic block size clamping fix (35 files, 106 lines changed). The robust evaluation framework is adapted from SakanaAI's [robust-kbench](https://github.com/SakanaAI/robust-kbench) methodology.

### Key Modifications from Original PallasBench

1. **Block size clamping**: All kernels modified to clamp block dimensions to max 65,536 elements, resolving Triton's 1M element limit on GPU
2. **Robustness filters**: Five filters added (output range, output std, axes variation, input impact, source analysis) to detect degenerate kernel implementations
3. **IR artifact capture**: Jaxpr DAGs and StableHLO IR captured for each kernel
4. **Hardware metrics**: GPU memory, throughput, and bandwidth utilization recorded

## Limitations

- **JIT compilation overhead**: First-run timing includes JAX/Triton JIT compilation (~60-100 minutes for full suite), dominated by `ptxas` assembly. Warm-run performance is orders of magnitude faster.
- **Single hardware platform**: Results are specific to NVIDIA A100 80GB PCIe. Performance characteristics may differ on other GPUs or TPUs.
- **Numerical precision**: Some kernels (matmul, attention) require relaxed tolerance (`atol=1e-4`) due to floating-point accumulation differences between Pallas and JAX reference implementations.
- **Axes variation filter**: May produce false positives for inherently sparse outputs (e.g., `one_hot`, `scatter_add`).

## Citation

If you use this dataset, please cite:

```bibtex
@misc{pallasbench_robust_2025,
  title={Towards Robust Pallas Kernel Benchmarking, Verification, and Optimization on GPU},
  author={O'Leary, Evan},
  year={2025},
  howpublished={HuggingFace Dataset},
  url={https://huggingface.co/datasets/eoleary/pallasbench-robust}
}

@misc{pallasbench_2025,
  title={PallasBench: A Benchmark for JAX Pallas Kernels},
  author={Tyronita},
  year={2025},
  howpublished={GitHub},
  url={https://github.com/Tyronita/PallasBench}
}

@misc{lange2025robust,
  title={Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization},
  author={Lange, Robert Tjarko and Tang, Yujin and Ha, David},
  year={2025},
  howpublished={SakanaAI Technical Report}
}
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
