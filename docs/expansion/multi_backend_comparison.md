# Multi-Backend Comparison: Scoping Document & Implementation Plan

## Goal

Run the same 45 PallasBench kernels on four backends:

| Backend | Target Hardware | Compiler Stack | Status |
|---------|----------------|----------------|--------|
| **GPU via Triton** | NVIDIA A100 80GB (Azure) | JAX -> StableHLO -> Triton MLIR -> PTX -> CUBIN | Working (current) |
| **GPU via Mosaic GPU** | NVIDIA A100 80GB | JAX -> StableHLO -> Mosaic GPU (JAX CUDA) | Experimental |
| **TPU** | Cloud TPU v4/v5 (GCP) | JAX -> StableHLO -> Mosaic (TPU) -> TPU HLO | Original target |
| **CPU** | x86-64 (same VM / GCP) | JAX -> StableHLO -> XLA CPU -> LLVM | Reference baseline |

This produces the first cross-platform kernel performance matrix for Pallas, revealing how Pallas's portable abstraction translates (or fails to translate) across backends.

## Why This Matters

Pallas's key value proposition is **write once, run on GPU or TPU**. No published benchmark validates this claim systematically. A 4-backend matrix would:

- Quantify the **portability tax**: how much performance is lost on each backend compared to a native implementation
- Identify **backend-specific regressions**: kernels that work on TPU but fail/are slow on GPU (and vice versa)
- Provide **ground truth data** for LLM training: models that learn to write Pallas kernels need to understand backend-specific constraints (Triton element limits, TPU systolic array shapes, CPU vectorization)
- Validate the **robustness filter framework** across backends — do filters that catch degenerate kernels on GPU also catch them on TPU/CPU?

## Comparison Dimensions

### 1. Compilation Time

| Phase | GPU (Triton) | GPU (Mosaic) | TPU | CPU |
|-------|-------------|--------------|-----|-----|
| JAX tracing | ~0.3s | ~0.3s | ~0.3s | ~0.3s |
| StableHLO lowering | ~0.1s | ~0.1s | ~0.1s | ~0.1s |
| Backend compile | ~3s (Triton) | ? | ? | ? |
| Device assembly | ~412s (ptxas) | ? | ? | ? |
| **Total first-run** | **~415s** | **?** | **?** | **?** |
| Warm execution | ~0.2ms | ? | ? | ? |

Questions to answer:
- Does Mosaic GPU avoid the `ptxas` bottleneck?
- How does TPU compilation time compare with GPU?
- Does XLA CPU compile faster due to mature LLVM backend?

### 2. Numerical Precision

PallasBench kernels are written in float32 by default. Different backends have different native dtype behavior:

| Backend | Default Compute | Key Concern |
|---------|----------------|-------------|
| GPU (A100) | float32 (native) | Baseline precision |
| GPU (Mosaic) | float32 or bfloat16 | Depends on Mosaic lowering |
| TPU v4/v5 | bfloat16 (native) | float32 ops emulated at lower throughput |
| CPU | float64 (XLA default) | May produce more precise results |

Protocol:
- Run every kernel in float32 on all backends (TPU may emulate float32 via bfloat16 accumulation)
- Run every kernel in bfloat16 where supported
- Record: max abs error, max rel error, `allclose(atol=1e-5, rtol=1e-5)` pass/fail
- Flag kernels where dtype choice changes correctness outcome

### 3. Kernel Performance vs Peak

| Metric | GPU (A100) | TPU v4 | TPU v5 | CPU |
|--------|-----------|--------|--------|-----|
| Peak FLOPS (FP32) | 19.5 TFLOPS | ~275 TFLOPS (bf16) | ~450 TFLOPS (bf16) | ~1 TFLOPS |
| HBM/BW | 2,039 GB/s | 1,200 GB/s | 1,600 GB/s | ~100 GB/s (DDR) |
| Key limitation | SM count, shared mem | systolic array shape | warp-like lanes | cache hierarchy |

Per-kernel metrics:
- **Throughput (GFLOP/s)**: raw compute throughput
- **Bandwidth utilization (%)**: measured BW / peak BW
- **Roofline position**: compute-bound vs memory-bound on each backend
- **Block/grid efficiency**: how well tiling maps to backend's parallel units

### 4. Robustness Filter Portability

The five robustness filters (output range, output std, axes variation, input impact, source analysis) were designed for GPU Triton. Do they transfer?

| Filter | GPU | TPU | CPU | Notes |
|--------|-----|-----|-----|-------|
| Output Range | ✓ | ? | ? | Should be backend-agnostic |
| Output Std | ✓ | ? | ? | Should be backend-agnostic |
| Axes Variation | FP for one_hot | ? | ? | Sparse patterns may differ |
| Input Impact | ✓ | ? | ? | Should be backend-agnostic |
| Source Analysis | ✓ (static) | ✓ (static) | ✓ (static) | Identical source analyzed |

Goal: Validate that the robustness framework produces consistent results across backends. Note any backend-specific false positives/negatives.

## Infrastructure Needed

### GCP TPU Quota

| Resource | Required | Details |
|----------|----------|---------|
| TPU v4-8 | 1 slice | 4 chips, ~8 cores, 32GB HBM |
| TPU v5p-8 | 1 slice (if available) | Next-gen, higher BW |
| GCP project | 1 | With TPU API enabled |
| Quota increase | TPU v4: 1 slice (default may be 0) | Request via Cloud Console |
| Service account | 1 | With `tpu.googleapis.com` access |
| Estimated cost | $15-30/hr (v4-8) | ~2 hours total eval = $30-60 |

Steps to acquire:
1. Enable Cloud TPU API in GCP project
2. Request quota increase for TPU v4 (us-central1)
3. Create TPU VM with `--accelerator-type=v4-8`
4. Install JAX with TPU support: `pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

### Azure / On-Prem GPU

Already have access to Azure Standard_NC24ads_A100_v4. No additional infra needed for GPU Triton.

For Mosaic GPU, may need a nightly JAX build or specific jaxlib that exposes the Mosaic GPU path.

### JAX Mosaic GPU Setup

Mosaic GPU is JAX's experimental CUDA backend alternative to Triton. As of JAX 0.10.x:

```bash
# May require building jaxlib from source or using a specific nightly
pip install --upgrade jax jaxlib
# Mosaic GPU is activated via compiler flags, not a separate backend
# Set environment variable:
#   XLA_FLAGS="--xla_gpu_enable_mosaic_gpu=true"
```

Note: Mosaic GPU availability is limited. If not yet stable, defer this backend and document as "future work."

### CPU Baseline

- Run on the same Azure VM (AMD EPYC 7V13, 24 vCPUs)
- No special setup — `jax.default_backend()` returns "cpu" by default if CUDA is not visible
- Can force CPU via `CUDA_VISIBLE_DEVICES=""` or `JAX_PLATFORMS=cpu`

## Expected Challenges

### 1. Different Default Dtypes

TPU v4/v5 natively compute in bfloat16. float32 operations are either:
- Emulated with bfloat16 accumulation (lower precision)
- Run at reduced throughput

This means a kernel that passes correctness checks on GPU (float32 native) may fail on TPU due to precision loss, or vice versa.

**Mitigation**: Run all kernels in both float32 and bfloat16. Report precision differences explicitly. Use relaxed tolerance (`atol=1e-4`) for cross-backend comparison.

### 2. Backend-Specific Compiler Behaviors

| Backend | Known Behavior |
|---------|----------------|
| GPU/Triton | 1M element limit (fixed), `ptxas` bottleneck, shared memory constraints |
| GPU/Mosaic | May have different tiling constraints, possibly no element limit |
| TPU/Mosaic | No element limit, but systolic array shape constraints (128x128 preferred) |
| CPU/XLA | No tiling constraints, but vector width limits (256/512-bit) |

**Mitigation**: Document each backend's constraints in a companion "backend notes" file. Do not expect identical tiling configurations across backends.

### 3. Hardware Availability

| Backend | Availability | Risk |
|---------|-------------|------|
| GPU (Triton) | Have access | Low |
| GPU (Mosaic) | Software-only | Medium — may require specific JAX build |
| TPU v4 | Need GCP quota | Medium — quota may take days |
| TPU v5p | Limited availability | High — may be restricted |
| CPU | Everywhere | Low |

**Mitigation**: Tiers of completion:
- Tier 1 (guaranteed): GPU Triton + CPU
- Tier 2 (likely): + TPU v4
- Tier 3 (aspirational): + TPU v5p + Mosaic GPU

### 4. Reproducibility Across Runs

GPU and TPU evaluation times vary due to:
- VM preemption (TPU spot)
- Thermal throttling (GPU)
- System load (shared VM)

**Mitigation**: Run each kernel 3 times on each backend. Report median, min, max. Record system state (temperature, power, utilization) per run.

## Implementation Steps

### Phase 1: Infrastructure Setup (Week 1-2)

1. **Provision GCP TPU v4-8 slice**
   - Enable Cloud TPU API
   - Request quota increase (us-central1)
   - Create TPU VM with `queued-resources create`
   - Install JAX/TPU: `pip install jax[tpu] jaxlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
   - Verify device detection: `jax.devices()` shows 8 TPU cores

2. **Prepare all backend environments**
   - GPU Triton (Azure A100): already working
   - CPU (same Azure VM): test with `CUDA_VISIBLE_DEVICES=""`
   - Mosaic GPU: install nightly jaxlib, test with `XLA_FLAGS="--xla_gpu_enable_mosaic_gpu=true"`
   - Create Conda/pip venv per backend to avoid dependency conflicts

3. **Write backend-agnostic evaluation harness**
   - Extend `scripts/run_robust_eval.py` to accept `--backend` parameter
   - Abstract away backend-specific compilation flags
   - Support per-backend dtype configuration
   - Handle backend-specific block size constraints

### Phase 2: Calibration & Validation (Week 2-3)

4. **Calibrate metrics on GPU (Triton) baseline**
   - Run all 45 kernels on A100 (baseline already exists from robust eval)
   - Verify metrics match previous runs
   - Establish reference correctness + performance

5. **Validate CPU baseline**
   - Run all 45 kernels on CPU
   - Verify correctness matches GPU (within numerical tolerance)
   - Record CPU compilation + execution times
   - Note any kernels that fail on CPU (e.g., due to different XLA lowering)

6. **Validate TPU environment**
   - Run a subset of 5-10 kernels on TPU v4-8 (start with L1 activations)
   - Verify correctness against GPU reference
   - Tune block sizes for TPU's systolic array (128x128 blocks)
   - Debug any TPU-specific compilation failures

7. **Validate Mosaic GPU (if available)**
   - Run the same 5-10 kernel subset
   - Compare compilation times against Triton path
   - Document differences in lowering behavior

### Phase 3: Full Evaluation (Week 3-4)

8. **Run full 45-kernel suite on all available backends**
   - Execute `scripts/run_multi_backend.py --backend <backend> --kernels all`
   - Capture per-kernel: correctness, performance, robustness filters, IR artifacts
   - Run each kernel 3x for statistical validity
   - Record hardware telemetry (temperature, power, utilization)

9. **Collect compilation artifacts per backend**
   - Jaxpr DAG (backend-agnostic)
   - StableHLO IR (backend-agnostic)
   - Backend-specific IR:
     - GPU Triton: Triton MLIR, PTX
     - GPU Mosaic: Mosaic GPU IR (if exposed)
     - TPU: TPU HLO (via XLA dump)
     - CPU: LLVM IR (via XLA dump)

10. **Serialize results to unified JSONL format**
    - Extend KernelBook JSONL schema with `backend` field
    - One file per backend, or a single file with backend dimension
    - Include all artifacts + metrics

### Phase 4: Analysis & Reporting (Week 4-5)

11. **Produce comparison matrix**
    - 45 kernels x 4 backends = 180 cells
    - Each cell: correctness (pass/fail), speedup vs JAX baseline, BW utilization %
    - Visualize as heatmap (rows = kernels, columns = backends)

12. **Analyze compilation time breakdown**
    - Stacked bar chart per backend showing trac->StableHLO->device compile->assemble
    - Identify backend-specific bottlenecks
    - Compare against existing GPU Triton data

13. **Analyze robustness filter consistency**
    - For each kernel, does the same filter set produce the same pass/fail across backends?
    - Document any backend-specific filter mismatches
    - Propose backend-aware filter thresholds if needed

14. **Write multi-backend comparison report**
    - `docs/expansion/multi_backend_results.md`
    - Include all comparison matrices, analysis, insights
    - Discuss portability tax and backend-specific findings

## Success Criteria

A completed multi-backend comparison must deliver:

1. **Coverage**: At least 3 of 4 backends evaluated (GPU Triton, CPU, TPU v4 minimum). 4 of 4 (including Mosaic GPU or TPU v5p) is aspirational.

2. **Correctness**: All 45 kernels pass correctness + robustness filters on each evaluated backend (document exceptions with root cause analysis).

3. **Performance data**: Per-kernel speedup vs JAX baseline on every backend. Minimum 3 runs per kernel per backend.

4. **Compilation data**: Per-kernel compilation time broken down by phase. Artifacts captured at each IR level.

5. **Robustness validation**: Cross-backend robustness filter consistency analysis. Documented false positives/negatives per backend.

6. **Reproducibility**: All code and data published. Environment specifications (JAX version, hardware config, OS) recorded per backend. Instructions to reproduce any cell in the matrix.

7. **Report**: Comprehensive analysis document covering all comparison dimensions, with visualizations and actionable insights about Pallas portability.

## Deliverables

| Artifact | Format | Location |
|----------|--------|----------|
| Evaluation harness | Python | `scripts/run_multi_backend.py` |
| Results, Triton GPU | JSONL | `results/triton_gpu/` |
| Results, Mosaic GPU | JSONL | `results/mosaic_gpu/` |
| Results, TPU | JSONL | `results/tpu/` |
| Results, CPU | JSONL | `results/cpu/` |
| Unified comparison matrix | CSV/Parquet | `results/multi_backend_comparison.parquet` |
| Analysis notebook | .ipynb | `notebooks/multi_backend_analysis.ipynb` |
| Final report | .md | `docs/expansion/multi_backend_results.md` |
| HuggingFace dataset | HF Dataset | `eoleary/pallasbench-multi-backend` |

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| TPU quota denied | No TPU data | Medium | Request early; fall back to GPU + CPU only |
| Mosaic GPU not ready | 3 backends instead of 4 | High | Defer; document as future work |
| Kernel fails on TPU | Partial coverage | Medium | Document root cause; try block size tuning |
| Compilation time too high | Budget overrun | Low | Use pre-compiled cache where possible |
| Numerical precision mismatches | Hard to compare | Medium | Run float32 + bfloat16; report both |
| TPU VM preemption | Lost progress | Medium | Use checkpointing in eval harness |
