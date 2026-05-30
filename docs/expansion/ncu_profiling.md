# NCU Profiling Integration

## Goal

Integrate NVIDIA Nsight Compute (NCU) into the PallasBench evaluation pipeline to capture detailed hardware performance counters for every Pallas kernel, going beyond the high-level throughput and bandwidth metrics currently collected via JAX's built-in profiling. This enables per-kernel roofline analysis, warp-level microarchitecture characterization, and memory-hierarchy bottleneck identification.

## Metrics to Capture

| Metric | NCU Counter / Formula | Purpose |
|--------|----------------------|---------|
| SM Occupancy | `sm__occupancy.avg.pct` | Fraction of available warp slots occupied per SM |
| L2 Cache Hit Rate | `lts__t_sectors_hit_rate.pct` | Efficiency of L2 cache usage |
| L1 Cache Hit Rate | `l1tex__t_sectors_hit_rate.pct` | Efficiency of L1/texture cache |
| Warp Execution Efficiency | `sm__warp_issue_stalled_percentage.avg.pct` per stall reason | How often warps issue vs. stall |
| Arithmetic Intensity | `flop_count_sp` / `dram__bytes` | FLOP/byte ratio for roofline |
| Memory Throughput (HBM) | `dram__throughput.avg.pct_of_peak_sustained` | HBM bandwidth utilization |
| Memory Throughput (L1/L2) | `l1tex__throughput.avg.pct_of_peak` / `lts__throughput.avg.pct_of_peak` | Cache bandwidth utilization |
| Stall Reasons | `sm__warp_issue_stalled_*` | Breakdown: wait, scoreboard, not selected, etc. |
| Achieved FLOP/s | `flop_count_sp` / duration | Measured throughput |
| Register Pressure | `launch__registers_per_thread` | Register file utilization |
| Shared Memory Usage | `launch__shared_mem_per_block_dynamic` + static | Shared memory pressure |
| Sector Stores/Loads | `l1tex__t_sector_{op}_pipe_{unit}` | Memory access patterns |

## Roofline Analysis

For each kernel, plot achievable FLOP/s vs. arithmetic intensity on a log-log chart:

- **X-axis**: Arithmetic intensity (FLOP/byte) = `flop_count_sp` / `dram__bytes`
- **Y-axis**: Achieved FLOP/s = `flop_count_sp` / kernel_duration
- **Ceiling lines**: Peak FP32 FLOP/s (19.5 TFLOPS on A100), peak HBM bandwidth (2,039 GB/s), peak L2 bandwidth, peak L1 bandwidth
- **Kernel points**: Each kernel plotted at its measured (AI, FLOP/s) coordinate, color-coded by kernel category

The plot identifies each kernel as:
- **Memory-bound**: Point lies below the bandwidth ceiling (left of ridge point)
- **Compute-bound**: Point lies below the FLOP ceiling (right of ridge point)
- **Optimization headroom**: Distance from ceiling indicates untapped potential

## Integration Approach

### Option A: External NCU Wrapper (Recommended)

Wrap the entire PallasBench evaluation process with `ncu` as an external profiler:

```
ncu --set full --target-processes all --kernel-name regex:".*pallas.*" \
    --csv --log-file kernel_metrics.csv \
    python scripts/run_benchmark.py --levels 1 2 3
```

Pros: Zero code changes to evaluation pipeline, NCU captures all kernel launches.
Cons: Profiles every kernel launch (including JAX baselines), requires filtering to isolate Pallas kernels.

### Option B: Per-Kernel NCU Subprocess

For each kernel individually, spawn `ncu --replay-mode kernel`:

```python
import subprocess, json

def profile_kernel_ncu(kernel_name: str, script: str, device: int = 0) -> dict:
    cmd = [
        "ncu", "--target-processes", "all",
        "--kernel-name", f"*{kernel_name}*",
        "--set", "full",
        "--csv",
        "--log-file", f"{kernel_name}_ncu.csv",
        "--device", str(device),
        "python", script,
    ]
    subprocess.run(cmd, check=True)
    return parse_ncu_csv(f"{kernel_name}_ncu.csv")
```

Pros: Clean per-kernel isolation, targeted counter collection.
Cons: Each kernel incurs NCU driver overhead + JIT recompilation.

### Option C: NVML Bindings (pynvml)

Use `pynvml` to poll GPU metrics synchronously during kernel execution, without NCU.

Pros: Lightweight, no external binary needed.
Cons: Cannot capture SM-level counters (occupancy, warp stalls) or memory-hierarchy breakdown.

We adopt **Option B** as the primary approach, with Option A as a batch alternative and Option C as a fallback for environments where NCU is unavailable.

## Scripting

### Primary interface: `ncu` CLI via subprocess

```python
import subprocess
import csv
import json
from pathlib import Path


def run_ncu_profile(
    kernel_script: str,
    kernel_name: str,
    output_dir: Path,
    device: int = 0,
    metric_set: str = "full",
) -> dict:
    """Run ncu profiling for a single kernel and return parsed metrics."""
    csv_path = output_dir / f"{kernel_name}_ncu_raw.csv"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", f"*{kernel_name}*",
        "--set", metric_set,
        "--csv",
        "--log-file", str(csv_path),
        "--device", str(device),
        "--replay-mode", "kernel",
        "python", kernel_script,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_ncu_csv(csv_path)


def parse_ncu_csv(csv_path: Path) -> dict:
    """Parse NCU CSV output into a structured dict of metrics."""
    metrics = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get("Metric Name", row.get("Metric", ""))
            metric_value = row.get("Value", row.get("Metric Value", ""))
            unit = row.get("Unit", "")
            if metric_name and metric_value:
                try:
                    metrics[metric_name] = {
                        "value": float(metric_value),
                        "unit": unit,
                    }
                except ValueError:
                    metrics[metric_name] = {"value": metric_value, "unit": unit}
    return metrics
```

### Alternative: `ncu --nvtx` range-based capture

If kernels are annotated with NVTX ranges in the JAX pipeline, use:

```
ncu --nvtx --nvtx-include "pallas_kernel_*" --set full ...
```

### Fallback: pynvml polling

```python
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
rates = nvmlDeviceGetUtilizationRates(handle)
# rates.gpu, rates.memory -- coarse-grained only
```

## Data Format

NCU results are merged into the existing JSONL dataset as a new `ncu_profile` field per kernel entry:

```json
{
  "kernel_name": "relu",
  "level": 1,
  "category": "activation",
  "source_original": "...",
  "source_fixed": "...",
  "result": {
    "correctness_passed": true,
    "wall_time_seconds": 415.3,
    "gpu_memory_used_bytes": 4096
  },
  "ncu_profile": {
    "device": "NVIDIA A100 80GB PCIe",
    "driver_version": "560.35.03",
    "ncu_version": "2024.3.1",
    "occupancy": {
      "sm_occupancy_pct": 68.2,
      "achieved_occupancy_pct": 62.1,
      "theoretical_occupancy_pct": 75.0
    },
    "cache_hit_rates": {
      "l1_hit_rate_pct": 81.3,
      "l2_hit_rate_pct": 72.8
    },
    "warp_efficiency": {
      "warp_execution_efficiency_pct": 94.5,
      "stall_wait_pct": 2.1,
      "stall_scoreboard_pct": 1.8,
      "stall_not_selected_pct": 0.9,
      "stall_other_pct": 0.7
    },
    "arithmetic_intensity": {
      "flop_count_sp": 8192.0,
      "dram_bytes_read": 16384.0,
      "dram_bytes_write": 4096.0,
      "arithmetic_intensity_flop_per_byte": 0.4
    },
    "memory_throughput": {
      "hbm_throughput_gbps": 1590.0,
      "hbm_utilization_pct": 78.0,
      "l1_throughput_gbps": 4200.0,
      "l2_throughput_gbps": 3100.0
    },
    "roofline": {
      "ridge_point_flop_per_byte": 9.56,
      "kernel_is_compute_bound": false,
      "achieved_flops_per_second": 12400000000.0,
      "peak_flops_per_second": 19500000000.0,
      "hbm_bandwidth_bytes_per_second": 2039000000000.0
    },
    "resource_usage": {
      "registers_per_thread": 32,
      "shared_mem_per_block_static_bytes": 0,
      "shared_mem_per_block_dynamic_bytes": 4096,
      "max_threads_per_block": 1024,
      "threads_per_warp": 32
    },
    "raw_counters": {
      "sm__inst_executed": 123456,
      "sm__saa_cycles": 98765,
      "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 5000,
      "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 1000,
      "dram__sectors_read": 4096,
      "dram__sectors_write": 1024
    }
  }
}
```

## Implementation Steps

1. **Verify NCU availability**: Check `ncu --version` on the target GPU machine. Install via `apt install cuda-nsight-compute-12-6` or download from NVIDIA. Confirm `ncu --list-sets` shows `full`, `roofline`, `memory`, etc.

2. **Create `ncu_profile.py` CLI script**: Implement the argument parser with `--kernels`, `--output-dir`, `--device`, and `--roofline` flags. Add a `--metric-set` option (`full`, `roofline`, `memory`) and `--kernel-name-filter` for regex targeting.

3. **Implement `run_ncu_profile` function**: Core subprocess wrapper that runs `ncu` with the specified metrics set, captures CSV output, and parses structured results. Handle `--replay-mode kernel` for accurate per-kernel timing.

4. **Implement `parse_ncu_csv` function**: Parse NCU's CSV output format into a hierarchical dict matching the JSONL schema. Handle the column naming variations across NCU versions.

5. **Implement roofline computation**: From `flop_count_sp` and `dram__bytes_{read,write}`, compute arithmetic intensity. Load device peak specs from a lookup table (A100, H100, V100). Generate matplotlib log-log roofline plot with kernel point overlaid on memory and compute ceilings.

6. **Integrate with evaluation pipeline**: Add an `--ncu` flag to `run_benchmark.py` that triggers per-kernel NCU profiling alongside the standard correctness/timing evaluation. Merge NCU results into the existing JSONL output.

7. **Add per-kernel profiling script**: Create `scripts/profile_kernel.py` that imports and runs a single named kernel with NCU profiling, enabling the per-kernel subprocess pattern.

8. **Write integration test**: Run NCU on a single kernel (e.g., `relu`) and verify the output JSONL contains the `ncu_profile` field with `occupancy`, `cache_hit_rates`, `arithmetic_intensity`, and `memory_throughput` subsections.

9. **Batch profiling**: Run NCU across all 45 kernels. Collect a complete performance micro-benchmark dataset. Identify the top-5 most memory-bound and compute-bound kernels.

10. **Document findings**: Update the robust evaluation report with NCU findings. Add a new section analyzing per-kernel microarchitecture characteristics, bottleneck patterns, and optimization opportunities for LLM-driven tuning.

## Expected Challenges

| Challenge | Mitigation |
|-----------|-----------|
| **NCU overhead distorts timing** | Use `--replay-mode kernel` to limit profiling to kernel launches; compare warm execution times with/without NCU; treat profiling runs as separate from timing runs |
| **Kernel launch latency with JIT** | JAX's JIT compilation happens before the kernel is captured; ensure warm-up run occurs before profiling pass; NCU may capture `ptxas` compilation kernels |
| **JIT compilation interaction** | `ncu --target-processes all` will profile the entire Python process; use `--kernel-name` regex to filter only Pallas kernels; exclude JAX runtime kernels |
| **Multiple kernel launches per pallas_call** | Each `pl.pallas_call` may launch multiple Triton kernels; NCU's kernel-name filter may need wildcards; verify capture coverage with `--print-summary=per-kernel` |
| **NCU replay mode with JAX** | `--replay-mode kernel` requires deterministic kernel execution; JAX's functional semantics guarantee this, but async dispatch may cause issues; add `jax.block_until_ready()` |
| **CSV schema varies by NCU version** | Handle multiple CSV column naming conventions in `parse_ncu_csv`; use `--csv` with `--print-summary=per-kernel` consistently |
| **Elevated privileges** | NCU may require `sudo` for GPU performance counter access; document `sudo ncu ...` usage or the `nvidia-persistenced` setup |
| **Large CSV output** | Full metric sets produce hundreds of counters per kernel; structure output hierarchically; provide a `--metrics` subset flag for targeted collection |
| **Non-deterministic occupancy** | Occupancy varies with input size and concurrent workloads; fix clock rates with `nvidia-smi -ac`; run in isolation |

## Success Criteria

1. **All 45 kernels profiled**: NCU successfully captures hardware counter data for every PallasBench kernel without crashes or timeouts.
2. **Structured output**: Each kernel's NCU profile is parsed into the hierarchical JSONL schema with all required subsections.
3. **Roofline plots generated**: Per-kernel roofline plots are produced for all 45 kernels, clearly showing the ridge point and each kernel's position relative to memory/compute ceilings.
4. **Bottleneck classification**: Each kernel is classified as memory-bound, compute-bound, or latency-bound based on roofline analysis and stall reason breakdown.
5. **Reproducible**: Running the profiling pipeline twice on the same hardware produces metrics within 5% relative difference for the key counters (occupancy, L2 hit rate, arithmetic intensity).
6. **Zero false captures**: The `--kernel-name` filter excludes JAX runtime/Triton compiler kernels, with verified per-kernel NCU summary showing exactly the expected number of Pallas kernel launches.
