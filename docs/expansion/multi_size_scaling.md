# Multi-Size Scaling — Expansion Plan

## Goal

Run each PallasBench kernel at multiple input sizes to characterize:

- **Compute-bound vs memory-bound crossover point** — where kernel time shifts from being dominated by data movement to being dominated by arithmetic
- **Tiling efficiency at different scales** — how block-size selection affects occupancy and achieved FLOP/s as total problem size grows
- **Kernel launch overhead vs computation** — the fixed cost of `pallas_call` / Triton launch relative to the actual work done

This directly addresses the question: *"How well does a given Pallas kernel generalize across problem sizes?"* The answer determines whether per-task tuning is needed or a single kernel can serve a range of deployment shapes.

## Target Sizes

Primary suite (every kernel runs at these):

| Name      | Tuple(s)                  | Total elements | Approx memory (FP32) |
|-----------|---------------------------|----------------|----------------------|
| `N256`    | `(256, 256)`              | 65K            | 0.25 MB              |
| `N1024`   | `(1024, 1024)`            | 1M             | 4 MB                 |
| `N4096`   | `(4096, 4096)`            | 16M            | 64 MB                |
| `N8192`   | `(8192, 8192)`            | 67M            | 256 MB               |

Optional extended suite (for kernels that fit in GPU memory):

| Name      | Tuple(s)                  | Total elements | Approx memory (FP32) |
|-----------|---------------------------|----------------|----------------------|
| `N128`    | `(128, 128)`              | 16K            | 64 KB                |
| `N16384`  | `(16384, 16384)`          | 268M           | 1 GB                 |

*Note:* Multi-input kernels (e.g. matmul, swiglu) use all-guanine-pairs — e.g. for `N1024` a 2-input kernel gets `[(1024,1024), (1024,1024)]`. The `SCALING_SUITE` constant in `pallasbench/sizes.py` encodes this mapping per task.

## Metrics Per Size

For every (kernel, size) pair we collect:

| Metric              | Source                                  | Notes                                                        |
|---------------------|-----------------------------------------|--------------------------------------------------------------|
| Kernel time (ms)    | `pallasbench.utils.time_fn` (median)    | After JIT warm-up                                            |
| Baseline time (ms)  | Same harness, JAX `jnp` function        | Same input shapes, same RNG seed                             |
| Speedup             | `baseline_ms / kernel_ms`               | < 1 means kernel slower than stock JAX                       |
| Throughput (GB/s)   | `(bytes_read + bytes_written) / kernel_time_s` | From static shape analysis, not runtime counters       |
| BW utilization (%)  | `throughput / peak_bandwidth`           | A100 peak: 2039 GB/s                                         |
| GPU memory delta    | Before / after allocation (MB)          | Captured via `jax.device_get` or `nvidia-smi` query          |

## Analysis Outputs

1. **Scaling curves** — log-log plots of kernel time vs problem size, one curve per kernel (overlaid with baseline)
2. **Roofline overlay** — achieved throughput vs arithmetic intensity for each (kernel, size) point, plotted against A100 peak bandwidth and peak FP32 FLOP/s ceilings
3. **Crossover identification** — for each kernel, the problem size where speedup crosses 1.0x (i.e. Pallas kernel becomes faster than JAX baseline)
4. **Summary table** — per kernel: best speedup, size at which best speedup occurs, memory-limited vs compute-limited classification at `(4096,4096)`

## Harness Modifications

The existing `evaluate_kernel` in `benchmark.py` already accepts `input_shapes`. The changes needed are:

1. **`run_size_scaling.py` (new script)** — orchestrates the outer loop over sizes
2. **`benchmark.py`** — add a `measure_memory_delta()` helper; no API changes needed
3. **`sizes.py`** — add `SCALING_SUITE` constant that maps each task name to a list of per-size shape tuples
4. **`metrics.py`** — add `throughput_gbps()` and `bandwidth_utilization()` that compute from element counts and measured kernel time

The existing `evaluate_kernel` signature works as-is: it iterates trials internally. The scaling script calls it once per size.

## Scoring

Each (kernel, size, trial) result gets a scoring:

- **Throughput (GB/s)**: `total_bytes = sum(tensor.numel() * dtype.itemsize for in/out tensors) * 2` (read + write), divided by kernel time
- **Bandwidth utilization (%)**: throughput / device peak
- **Speedup quantiles**: median, p10, p90 across trials (existing `time_fn` already returns median over `n_trials`)

Results are stored in a per-run JSON file with structure:

```json
{
  "meta": { "device": "A100", "driver": "CUDA 12.x", "date": "...", "sizes": ["N256", "N1024", ...] },
  "results": {
    "L1/relu": {
      "N256": { "kernel_ms": 0.12, "baseline_ms": 0.15, "speedup": 1.25, "throughput_gbps": 890, "bw_util_pct": 43.7, "memory_delta_mb": 0.5 },
      "N1024": { ... },
      ...
    },
    ...
  }
}
```

## Implementation Steps

1. **Define `SCALING_SUITE`** in `pallasbench/sizes.py` — a dict mapping every TASK_REGISTRY name to a list of size tuples for each scaling size name (`N128`, `N256`, `N1024`, `N4096`, `N8192`, `N16384`). For single-input kernels the tuple is e.g. `(256,256)`; for multi-input kernels it's the full list of per-arg shape tuples.

2. **Add throughput helpers** to `pallasbench/metrics.py` — `compute_throughput(total_bytes, kernel_time_ms)` and `bandwidth_utilization(throughput_gbps, peak_gbps=2039)`.

3. **Add memory measurement helper** to `pallasbench/utils.py` or `pallasbench/benchmark.py` — use `jax.device_get` / `nvidia_smi` queries to capture HBM delta before and after kernel launch.

4. **Create `scripts/run_size_scaling.py`** — CLI script that:
   - Accepts `--kernels` (list of task names or "all"), `--sizes` (comma-separated size names), `--output-dir`
   - Loops: for each kernel → for each size → calls `evaluate_kernel` with the corresponding shapes from `SCALING_SUITE`
   - Collects per-size metrics (kernel time, baseline time, speedup, throughput, BW util, memory delta)
   - Writes structured JSON output
   - With `--plot`, generates log-log scaling curves via matplotlib

5. **Implement plotting** — `--plot` flag saves per-kernel scaling curves:
   - X-axis: problem size (total elements, log scale)
   - Y-axis: time (ms, log scale), two traces (kernel, baseline)
   - Optional: roofline overlay with matplotlib patches

6. **Add memory delta capture** — query `jax.devices()[0].memory_stats()` before and after kernel execution (available on JAX GPU backend) to get peak/current HBM usage.

7. **Validate on one kernel** — run `scripts/run_size_scaling.py --kernels L1/relu --sizes N256,N1024,N4096` to confirm scaling behaves as expected.

8. **Batch run** — full suite across primary 4 sizes for all 45 kernels; expected runtime ~1-2 hours depending on compilation overhead.

9. **Produce analysis artifacts** — generate summary tables and scaling-curve PNGs for inclusion in reports.

## Expected Challenges

| Challenge | Mitigation |
|-----------|------------|
| **OOM at (8192,8192) or (16384,16384)** for memory-heavy kernels (flash_attention, transformer_block, triangle_update) | Skip oversized dims per kernel; mark `max_safe_size` in SCALING_SUITE; catch `jax.errors.ResourcExhausted` gracefully |
| **Variable input dimensionality** — not all kernels are 2D | SCALING_SUITE encodes per-task shape lists; 1D kernels get 1D sizes, 3D kernels get 3D sizes, etc. |
| **Triton compilation per size** — JIT cache is per shape, so each new size triggers a recompile for the first run | Accept as measurement cost; report compile time separately from kernel time; use `n_warmup` to amortize |
| **Block efficiency cliff** — at small sizes the Triton 1M-element limit is irrelevant; at large sizes blocks shrink relative to SM count, reducing occupancy | Note the effective block size in results; correlate with measured utilization |
| **Floating-point non-determinism** across sizes due to different accumulation order | Use same random seed per kernel across all sizes; report max-abs error from correctness check at each size |
| **Multi-input kernels (e.g. swiglu, gated_mlp)** have complex shape relationships — can't simply scale all dims uniformly | SCALING_SUITE must hand-curate per-task shape lists; for fused matmul patterns, scale the common hidden dimension while keeping other dims proportionate |

## Success Criteria

1. **45 kernel × 4 size matrix** complete with no crashes (excluding expected OOM skips)
2. **Scaling curves produced** for ≥40 kernels showing clear trend lines
3. **Crossover points identified** — at least 10 kernels show a measurable compute/memory crossover (speedup crossing 1.0x)
4. **Roofline plot generated** with all data points colored by kernel category
5. **Per-kernel best size identified** — the size at which each kernel achieves its maximum speedup
6. **Summary table** ranking kernels by speedup consistency across sizes
7. **Scripts reusable** — `run_size_scaling.py --kernels all --sizes N256,N4096,N16384` works out of the box on any GPU with JAX/Pallas installed
