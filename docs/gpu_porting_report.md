# PallasBench on GPU: Motivation, Findings, and Next Steps

## Why JAX

JAX is Google's numerical computing library built on XLA (Accelerated Linear Algebra). Unlike PyTorch, which compiles operator-by-operator, JAX traces entire computation graphs and compiles them as a unit through XLA. This gives JAX three properties that matter for kernel engineering:

- **Functional purity**: every operation is a pure function over immutable arrays, which makes program transformations (autodiff, vectorization, parallelism) composable and predictable.
- **XLA compilation**: `jax.jit` lowers Python to StableHLO IR, then to device-specific code. On GPU this means CUDA/PTX; on TPU it means TPU HLO. The compiler handles fusion, tiling, and memory placement automatically.
- **Hardware portability**: the same JAX program runs on CPU, GPU, and TPU without source changes — the backend handles the translation.

For AI workloads, JAX is the training framework behind Gemini, PaLM, and most of Google DeepMind's research. It dominates TPU workloads and has a growing GPU footprint, especially for large-scale distributed training.

## Why Pallas

Pallas is JAX's embedded DSL for writing custom kernels. It sits between "write CUDA by hand" and "hope XLA fuses things correctly":

```python
from jax.experimental import pallas as pl

def _relu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.maximum(x, 0)

out = pl.pallas_call(
    _relu_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(n // block_size,),
    in_specs=[pl.BlockSpec((block_size, cols), lambda i: (i, 0))],
    out_specs=pl.BlockSpec((block_size, cols), lambda i: (i, 0)),
)(x)
```

You write Python that looks like NumPy, but you explicitly control:
- **Grid**: how many blocks to launch
- **BlockSpec**: what tile each block reads/writes
- **Memory refs**: explicit load/store through `Ref` objects

Pallas then compiles this to:
- **Triton IR** on NVIDIA GPUs (via the Triton compiler)
- **Mosaic** on Google TPUs (via the TPU compiler)

This is powerful because the same Pallas kernel can target both TPU and GPU — the abstraction handles the backend translation. No other framework offers this.

### Why not just write Triton directly?

Triton is NVIDIA-only. Pallas is portable. If you're building kernels for a system that might run on TPU (training) and GPU (inference), Pallas lets you write once. It also integrates natively with JAX's transformation system — you get `jax.grad`, `jax.vmap`, and `jax.jit` composition for free.

### Why not just let XLA auto-fuse?

XLA is good at fusing simple patterns (elementwise chains, reduce-broadcast), but it cannot discover complex fusion patterns like flash attention, fused SwiGLU, or tiled matrix-multiply-accumulate. For these, you need explicit tiling control — that's what Pallas provides.

## What is PallasBench

[PallasBench](https://github.com/Tyronita/PallasBench) is a benchmark suite of 45 Pallas kernels across three difficulty levels:

| Level | Count | Description | Examples |
|-------|-------|-------------|----------|
| L1 | 27 | Single operators | relu, softmax, matmul, layernorm, reduce_sum |
| L2 | 13 | Fused patterns | matmul+gelu, SwiGLU, fused softmax cross-entropy |
| L3 | 5 | Architecture components | flash attention, multi-head attention, transformer block |

Each kernel comes with:
- A **Pallas implementation** (the kernel under test)
- A **JAX baseline** (reference implementation using standard `jnp` ops)
- **Input shapes** and evaluation harness for correctness + timing

### Is it built for a specific use case?

PallasBench was originally designed for **TPU evaluation**. The kernels use Pallas primitives that map naturally to TPU's systolic array architecture. GPU support was added later (the repo's single commit message: *"Fix Pallas GPU lowering by using Triton backend"*), but the kernels were **not tuned for GPU**.

This means:
- Block sizes don't account for GPU SM count (108 on A100), warp size (32), or shared memory (164KB/SM)
- Tiling doesn't target GPU memory hierarchy (L2 cache: 40MB, HBM bandwidth: 2039 GB/s)
- No use of GPU-specific Triton features (tl.dot, async loads, persistent kernels)

This is actually what makes PallasBench interesting for our work: it's a clean set of **unoptimized** kernels that represent what an LLM would generate when asked to write Pallas code without GPU-specific tuning knowledge.

## What We Found: GPU Compilation Issues

When we ran PallasBench on an NVIDIA A100 80GB, every kernel failed with:

```
INVALID_ARGUMENT: Maximum allowed number of elements is 1048576,
but tensor<1024x4096xi32> has more than that
```

The Triton backend enforces a **1M element limit per tensor operation**. PallasBench kernels use `block_size = min(1024, n)` with the full second dimension in the block, giving blocks like `(1024, 4096)` = 4M elements.

### Our Fix

We patched all 35 affected kernel files to clamp block sizes to respect the Triton limit:

```python
# Before (fails on GPU):
block_size = min(1024, n)

# After (GPU-compatible):
cols = 1
for s in x.shape[1:]:
    cols *= s
block_size = min(min(1024, n), max(1, 1048576 // cols))
```

For `(4096, 4096)` inputs, this gives `block_size = 256`, and blocks of `(256, 4096)` = 1,048,576 elements — exactly at the Triton limit.

The full patch modifies 35 files with +106/-12 lines. Every change is mechanical: compute the column product, clamp the row block size. The fix preserves correctness because Pallas's `BlockSpec` handles the tiling — we're just choosing smaller tiles.

## What We're Capturing

For every kernel, we collect:

| Artifact | Description |
|----------|-------------|
| `original.py` | Upstream PallasBench source (pre-fix) |
| `fixed.py` | Our GPU-compatible version |
| `fix.diff` | Unified diff showing exact changes |
| `jaxpr.txt` | JAX's functional IR — the compute DAG with grid/block metadata |
| `stablehlo.txt` | StableHLO IR with embedded Triton MLIR bytecode |
| `result.json` | Correctness, baseline vs kernel timing, speedup, throughput |
| `stdout.log` | Full compilation and runtime output |
| `stderr.log` | XLA/Triton compiler diagnostics |
| GPU snapshots | Memory usage, utilization, temperature, power before/after |

### Metrics per kernel

- **Correctness**: pass/fail across 3 random seeds, max absolute error
- **Baseline time (ms)**: standard JAX `jnp` implementation
- **Kernel time (ms)**: Pallas kernel on GPU via Triton
- **Speedup**: baseline / kernel time
- **Throughput (GB/s)**: bytes read + written / kernel time
- **HW bandwidth utilization (%)**: throughput / A100 peak (2039 GB/s)
- **GPU memory delta (MB)**: HBM allocated by the kernel
- **JIT compilation time**: wall-clock time including Triton compile

## What We Want to Expand To

### 1. Multi-backend comparison

Run the same PallasBench kernels on:
- **GPU via Triton** (current — A100)
- **GPU via Mosaic GPU** (JAX's newer CUDA backend, when available)
- **TPU** (the original target — on Google Cloud TPU v4/v5)
- **CPU** (XLA CPU backend, for baseline)

This gives a cross-platform kernel performance matrix that doesn't exist anywhere else.

### 2. Multi-size scaling

Run each kernel at multiple input sizes to characterize:
- Compute-bound vs memory-bound crossover point
- Tiling efficiency at different scales
- Kernel launch overhead vs computation

Target sizes: `(256,256)`, `(1024,1024)`, `(4096,4096)`, `(8192,8192)`

### 3. NCU profiling integration

Use NVIDIA Nsight Compute to capture per-kernel:
- SM occupancy
- L2 cache hit rate
- Warp execution efficiency
- Arithmetic intensity (FLOP/byte)
- Memory throughput breakdown (L1/L2/HBM)

### 4. Kernel optimization as training data

The PallasBench kernels are deliberately unoptimized. We want to:
1. Use LLMs to generate **optimized versions** of each kernel
2. Evaluate them with the same harness (correctness + speedup)
3. Build a dataset of `(unoptimized kernel, optimized kernel, speedup)` triples
4. Use this as SFT/RL training data for code-generation models

This fits the **KernelBench** paradigm: given a reference implementation, produce a faster kernel. But for Pallas instead of CUDA/Triton.

### 5. Integration with ShinkaEvolve

Our evolutionary optimization framework ([ShinkaEvolve](https://github.com/Tyronita/ShinkaEvolve)) already runs on CVDP (Verilog design) and KernelBench (CUDA). Adding PallasBench as a target means:
- Evolve Pallas kernels with correctness + speedup as the fitness function
- Capture full evolutionary traces (generations, patches, reward signals)
- Build the same KernelBook-format dataset we produce for CVDP and Verilog

### 6. Cross-benchmark dataset unification

We're building a unified kernel benchmark dataset across:

| Benchmark | Language | Tasks | Status |
|-----------|----------|-------|--------|
| KernelBench | CUDA/Triton | 250 | Downloaded |
| PallasBench | Pallas/JAX | 45 | Running eval |
| CVDP | SystemVerilog | 304 | Uploaded to HF |
| Verilog Eval | Verilog | 157 | Uploaded to HF |
| MultiKernelBench | CUDA/Triton/Pallas/AscendC | 285 | Cloned |
| KernelBench-v2 | Triton | ~250 | Cloned |
| KernelBot | CUDA (competition) | varies | Downloaded |

All formatted as KernelBook-compatible JSONL with:
- Task description and reference implementation
- Generated kernel source
- Correctness signal (pass/fail)
- Performance signal (speedup, throughput, utilization)
- Hardware context (GPU model, memory, clocks)
- IR representations (Jaxpr, StableHLO, Triton MLIR where applicable)

## Hardware Context

All GPU results are from:
- **NVIDIA A100 80GB PCIe** (Azure Standard_NC24ads_A100_v4)
- 108 SMs, 6912 CUDA cores, 432 Tensor cores
- 80GB HBM2e, 2039 GB/s bandwidth
- 40MB L2 cache
- 19.5 TFLOPS FP32, 312 TFLOPS FP16 Tensor
- Driver: latest Azure-managed
- JAX 0.10.1, Triton 3.7.0

## Conclusion

PallasBench fills a gap: there are many CUDA and Triton kernel benchmarks, but no systematic Pallas benchmark with GPU results, IR captures, and optimization traces. By fixing the GPU compilation issues, capturing comprehensive metrics, and formatting the results as training data, we're building the dataset needed to train LLMs that can write and optimize Pallas kernels — the only kernel DSL that targets both GPU and TPU from a single source.

---

*Generated 2026-05-30. Dataset and code at [github.com/Tyronita/PallasBench](https://github.com/Tyronita/PallasBench).*
