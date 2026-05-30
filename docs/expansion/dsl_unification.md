# Cross-Benchmark DSL Unification

> Unifying kernel benchmarks across CUDA, Triton, Pallas, SystemVerilog, Verilog, and AscendC into a single KernelBook-compatible JSONL format.

---

## 1. Goal

Create a **unified benchmark dataset** spanning 7+ kernel/design benchmarks across 6 DSLs, all shareable via a common JSONL schema compatible with **KernelBook** tooling. This enables cross-DSL comparison of LLM-generated code quality for the first time — answering questions like *"Do LLMs generate better CUDA or Pallas kernels? Which DSL has the highest compilation success rate?"*.

| Benchmark | DSL | Tasks | Status |
|-----------|-----|-------|--------|
| KernelBench | CUDA, Triton | 250 | Downloaded |
| PallasBench | Pallas/JAX | 45 | Running eval |
| CVDP | SystemVerilog | 304 | On HuggingFace |
| Verilog Eval | Verilog | 157 | On HuggingFace |
| MultiKernelBench | CUDA, Triton, Pallas, AscendC | 285 | Cloned |
| KernelBench-v2 | Triton | ~250 | Cloned |
| KernelBot | CUDA (competition) | varies | Downloaded |

---

## 2. Unified Schema

### 2.1 Common Fields (Required across all DSLs)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Unique identifier (`kernelbench/L1_relu`, `pallasbench/L2_matmul_gelu`, `cvdp/001_counter`) |
| `benchmark_source` | `string` | Origin benchmark name (`kernelbench`, `pallasbench`, `cvdp`, `verilog_eval`, `multikernelbench`, `kernelbench_v2`, `kernelbot`) |
| `language` | `string` | DSL name (`cuda`, `triton`, `pallas`, `systemverilog`, `verilog`, `ascendc`) |
| `dsl_version` | `string` | Compiler/IR version used (e.g., `jax-0.10.1`, `cuda-12.6`, `triton-3.7.0`) |
| `task_description` | `string` | Human-readable description of what the kernel/design does |
| `category` | `string` | Functional category (`activation`, `matmul`, `attention`, `fsm`, `counter`, `memory`, etc.) |
| `difficulty_level` | `integer` | 1–3 or 0–5 scale; normalized across benchmarks |
| `reference_implementation` | `string` | The reference/golden implementation source code |
| `generated_source` | `string` | The LLM-generated kernel/design under evaluation |
| `correctness` | `object` | Pass/fail/max_error — see §2.1.1 |
| `performance` | `object` | Speedup/throughput/utilization — see §2.1.2 |
| `hardware_context` | `object` | GPU/accelerator details — see §2.1.3 |

#### 2.1.1 `correctness` Object

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `boolean` | Overall correctness verdict |
| `max_abs_error` | `float|null` | Maximum absolute error vs reference (null for HDL) |
| `max_rel_error` | `float|null` | Maximum relative error (null for HDL) |
| `allclose_atol` | `float|null` | Absolute tolerance used (null for HDL) |
| `allclose_rtol` | `float|null` | Relative tolerance used (null for HDL) |
| `robustness_filters` | `object` | Per-filter results (output_range, output_std, axes_variation, input_impact, source_analysis) |
| `verification_method` | `string` | `numerical_allclose`, `logical_equivalence`, `simulation_match`, `formal_verification` |

#### 2.1.2 `performance` Object

| Field | Type | Description |
|-------|------|-------------|
| `speedup` | `float|null` | Speedup over reference (baseline_time / kernel_time); null for HDL |
| `baseline_time_ms` | `float|null` | Reference execution time in ms |
| `kernel_time_ms` | `float|null` | Kernel execution time in ms |
| `throughput_gb_s` | `float|null` | Measured throughput in GB/s |
| `bandwidth_utilization_pct` | `float|null` | % of peak HW bandwidth achieved |
| `gpu_memory_delta_mb` | `float|null` | GPU memory allocated by kernel |
| `jit_compile_time_seconds` | `float|null` | JIT compilation time |
| `wall_time_seconds` | `float|null` | Total end-to-end wall time |
| `area_um2` | `float|null` | Silicon area for HDL designs |
| `power_mw` | `float|null` | Power consumption for HDL designs |
| `frequency_mhz` | `float|null` | Max operating frequency for HDL designs |

#### 2.1.3 `hardware_context` Object

| Field | Type | Description |
|-------|------|-------------|
| `accelerator` | `string` | GPU/accelerator model (`NVIDIA A100 80GB PCIe`, `NVIDIA L40S`) |
| `compute_capability` | `string|null` | CUDA compute capability (`8.0`, `8.9`, `9.0`) |
| `memory_gb` | `integer|null` | GPU memory in GB |
| `bandwidth_gb_s` | `integer|null` | Peak memory bandwidth in GB/s |
| `sm_count` | `integer|null` | Number of streaming multiprocessors |
| `driver_version` | `string|null` | Driver version |
| `runtime_version` | `string|null` | Runtime/framework version |
| `target_fpga` | `string|null` | Target FPGA part number (for HDL) |
| `target_technology_nm` | `integer|null` | Process node in nm (for HDL synthesis) |

### 2.2 DSL-Specific Optional Fields

#### CUDA / Triton

| Field | Type | Description |
|-------|------|-------------|
| `ir_representations.ptx` | `string|null` | PTX assembly output |
| `ir_representations.cubin` | `string|null` | CUBIN binary descriptor |
| `ir_representations.triton_mlir` | `string|null` | Triton MLIR intermediate IR |
| `tiling.block_shape` | `array[integer]` | Block/tile dimensions |
| `tiling.grid_shape` | `array[integer]` | Grid dimensions |
| `tiling.num_blocks` | `integer` | Total number of blocks |
| `ncu_profile.sm_occupancy` | `float|null` | SM occupancy from NCU |
| `ncu_profile.l2_hit_rate` | `float|null` | L2 cache hit rate |
| `ncu_profile.warp_efficiency` | `float|null` | Warp execution efficiency |
| `ncu_profile.arith_intensity` | `float|null` | Arithmetic intensity (FLOP/byte) |

#### Pallas / JAX

| Field | Type | Description |
|-------|------|-------------|
| `ir_representations.jaxpr` | `string|null` | JAX functional traced IR |
| `ir_representations.stablehlo` | `string|null` | StableHLO MLIR dialect |
| `ir_representations.triton_mlir` | `string|null` | Triton MLIR (lowered from Pallas) |
| `tiling.block_shape` | `array[integer]` | BlockSpec tile dimensions |
| `tiling.grid_shape` | `array[integer]` | Grid dimensions |
| `tiling.num_blocks` | `integer` | Total grid blocks |
| `compilation_pipeline` | `string` | `pallas->triton->ptx` or `pallas->mosaic` |
| `jax_version` | `string` | JAX library version |

#### SystemVerilog / Verilog (CVDP, Verilog Eval)

| Field | Type | Description |
|-------|------|-------------|
| `hdl_type` | `string` | `systemverilog` or `verilog` |
| `simulation_tool` | `string` | `iverilog`, `vcs`, `verilator`, `modelsim` |
| `synthesis_tool` | `string|null` | `yosys`, `vivado`, `quartus`, `design_compiler` |
| `simulation_results` | `object` | Pass/fail, testbench match, coverage |
| `synthesis_results` | `object|null` | Area, power, timing after synthesis |
| `formal_verification` | `boolean` | Whether formal verification was applied |
| `testbench_source` | `string|null` | Testbench source code used for verification |

#### AscendC (MultiKernelBench)

| Field | Type | Description |
|-------|------|-------------|
| `ascendc_version` | `string` | AscendC/CANN version |
| `ir_representations.ascend_ir` | `string|null` | Ascend IR intermediate representation |
| `target_npu` | `string` | NPU model (`Ascend 910B`, `Ascend 310P`) |
| `tiling.block_shape` | `array[integer]` | Tiling configuration for DaVinci core |

### 2.3 JSONL Example Entries

**CUDA (KernelBench):**
```json
{
  "task_id": "kernelbench/level1/relu",
  "benchmark_source": "kernelbench",
  "language": "cuda",
  "dsl_version": "cuda-12.6",
  "task_description": "Implement a ReLU activation kernel",
  "category": "activation",
  "difficulty_level": 1,
  "reference_implementation": "__global__ void relu_kernel(float* x, float* y, int n) { ... }",
  "generated_source": "__global__ void relu_kernel(float* x, float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = fmaxf(x[i], 0.0f); }",
  "correctness": {
    "passed": true,
    "max_abs_error": 0.0,
    "max_rel_error": 0.0,
    "allclose_atol": 1e-5,
    "allclose_rtol": 1e-5,
    "robustness_filters": {
      "output_range": {"passed": true, "value": 6.28},
      "output_std": {"passed": true, "value": 1.42}
    },
    "verification_method": "numerical_allclose"
  },
  "performance": {
    "speedup": 1.0,
    "baseline_time_ms": 0.05,
    "kernel_time_ms": 0.05,
    "throughput_gb_s": 800.0,
    "bandwidth_utilization_pct": 39.2
  },
  "hardware_context": {
    "accelerator": "NVIDIA L40S",
    "compute_capability": "8.9",
    "memory_gb": 48,
    "bandwidth_gb_s": 864,
    "sm_count": 142
  },
  "ir_representations": {
    "ptx": ".version 8.9\n.target sm_89\n.address_size 64\n...",
    "cubin": null,
    "triton_mlir": null
  }
}
```

**Pallas/JAX (PallasBench):**
```json
{
  "task_id": "pallasbench/L1/relu",
  "benchmark_source": "pallasbench",
  "language": "pallas",
  "dsl_version": "jax-0.10.1+triton-3.7.0",
  "task_description": "Implement a ReLU activation using Pallas",
  "category": "activation",
  "difficulty_level": 1,
  "reference_implementation": "def relu_ref(x): return jnp.maximum(x, 0.0)",
  "generated_source": "def relu_kernel(x_ref, o_ref):\n    o_ref[...] = jnp.maximum(x_ref[...], 0.0)",
  "correctness": {
    "passed": true,
    "max_abs_error": 0.0,
    "max_rel_error": 0.0,
    "allclose_atol": 1e-5,
    "allclose_rtol": 1e-5,
    "robustness_filters": {
      "output_range": {"passed": true, "value": 6.28},
      "output_std": {"passed": true, "value": 1.42},
      "axes_variation": {"passed": true},
      "input_impact": {"passed": true, "value": 0.31},
      "source_analysis": {"passed": true}
    },
    "verification_method": "numerical_allclose"
  },
  "performance": {
    "speedup": 0.85,
    "baseline_time_ms": 0.12,
    "kernel_time_ms": 0.14,
    "throughput_gb_s": 12.4,
    "bandwidth_utilization_pct": 78.3,
    "gpu_memory_delta_mb": 512,
    "jit_compile_time_seconds": 415.0,
    "wall_time_seconds": 415.3
  },
  "hardware_context": {
    "accelerator": "NVIDIA A100 80GB PCIe",
    "compute_capability": "8.0",
    "memory_gb": 80,
    "bandwidth_gb_s": 2039,
    "sm_count": 108,
    "driver_version": "560.35.03"
  },
  "ir_representations": {
    "jaxpr": "{ lambda ; a:f32[4096]. let b:f32[4096] = pallas_call[...] a in (b,) }",
    "stablehlo": "module @relu { func.func @main(%arg0: tensor<4096xf32>) -> tensor<4096xf32> { ... } }",
    "triton_mlir": "#triton_ir module { tt.func @kernel(...) { ... } }"
  },
  "tiling": {
    "block_shape": [64],
    "grid_shape": [64],
    "num_blocks": 64
  },
  "compilation_pipeline": "pallas->triton->ptx",
  "jax_version": "0.10.1"
}
```

**Verilog (Verilog Eval):**
```json
{
  "task_id": "verilog_eval/fsm_counter_4bit",
  "benchmark_source": "verilog_eval",
  "language": "verilog",
  "dsl_version": "iverilog-12.0",
  "task_description": "Implement a 4-bit FSM counter with reset",
  "category": "fsm",
  "difficulty_level": 1,
  "reference_implementation": "module counter(\n    input clk,\n    input rst_n,\n    output reg [3:0] count\n);\n    always @(posedge clk or negedge rst_n) begin\n        if (!rst_n) count <= 4'd0;\n        else count <= count + 4'd1;\n    end\nendmodule",
  "generated_source": "module counter(\n    input clk,\n    input rst_n,\n    output reg [3:0] count\n);\n    always @(posedge clk or negedge rst_n) begin\n        if (!rst_n) count <= 4'd0;\n        else if (count == 4'd15) count <= 4'd0;\n        else count <= count + 4'd1;\n    end\nendmodule",
  "correctness": {
    "passed": true,
    "verification_method": "simulation_match"
  },
  "performance": {
    "frequency_mhz": 250,
    "area_um2": 120,
    "power_mw": 0.5
  },
  "hardware_context": {
    "target_fpga": "xc7a35t",
    "target_technology_nm": 28
  }
}
```

---

## 3. Directory Structure Proposal

```
multibench/
├── data/
│   ├── kernelbench/              # Original + normalized
│   │   ├── raw/                   # As-downloaded upstream
│   │   ├── normalized.jsonl       # Unified JSONL
│   │   └── schema_version.json    # Tracks normalization version
│   ├── pallasbench/
│   │   ├── raw/
│   │   ├── fixed/                 # GPU-fixed sources
│   │   ├── normalized.jsonl
│   │   └── schema_version.json
│   ├── cvdp/
│   │   ├── raw/
│   │   ├── normalized.jsonl
│   │   └── schema_version.json
│   ├── verilog_eval/
│   │   ├── raw/
│   │   ├── normalized.jsonl
│   │   └── schema_version.json
│   ├── multikernelbench/
│   │   ├── raw/
│   │   ├── normalized.jsonl
│   │   └── schema_version.json
│   ├── kernelbench_v2/
│   │   ├── raw/
│   │   ├── normalized.jsonl
│   │   └── schema_version.json
│   └── kernelbot/
│       ├── raw/
│       ├── normalized.jsonl
│       └── schema_version.json
├── schemas/
│   ├── unified_benchmark_schema.json       # The unified JSON Schema
│   ├── kernelbench_mapping.json            # field-level mapping rules
│   ├── pallasbench_mapping.json
│   ├── cvdp_mapping.json
│   └── verilog_eval_mapping.json
├── scripts/
│   ├── normalize_to_unified.py             # CLI entry point
│   ├── normalize_kernelbench.py            # Per-benchmark normalizer
│   ├── normalize_pallasbench.py
│   ├── normalize_cvdp.py
│   ├── normalize_verilog_eval.py
│   ├── normalize_multikernelbench.py
│   ├── aggregate.py                        # Cross-benchmark aggregation queries
│   └── validate_schema.py                  # Validate JSONL against schema
├── queries/
│   ├── best_dsl_by_category.sql            # SQLite queries for analysis
│   ├── compilation_rates.sql
│   ├── speedup_distribution.sql
│   └── cross_benchmark_correlation.sql
└── results/
    └── aggregate/                          # Aggregated cross-benchmark results
```

---

## 4. Aggregation Queries

With all benchmarks in a unified JSONL format, the following analyses become possible:

### 4.1 "Which DSL do LLMs generate best for?"

```sql
SELECT
    language,
    COUNT(*) AS total_tasks,
    AVG(CASE WHEN correctness->>'passed' = 'true' THEN 1.0 ELSE 0.0 END) AS correctness_rate,
    AVG(performance->>'speedup') AS avg_speedup
FROM benchmarks
GROUP BY language
ORDER BY correctness_rate DESC;
```

### 4.2 Compilation Success Rate by Benchmark + DSL

```sql
SELECT
    benchmark_source,
    language,
    COUNT(*) AS total,
    SUM(CASE WHEN correctness->>'passed' = 'true' THEN 1 ELSE 0 END) AS passed,
    ROUND(100.0 * SUM(CASE WHEN correctness->>'passed' = 'true' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pass_rate
FROM benchmarks
GROUP BY benchmark_source, language
ORDER BY pass_rate DESC;
```

### 4.3 Performance Distribution by Category

```sql
SELECT
    category,
    language,
    COUNT(*) AS n,
    ROUND(AVG(CAST(performance->>'speedup' AS FLOAT)), 2) AS mean_speedup,
    ROUND(AVG(CAST(performance->>'bandwidth_utilization_pct' AS FLOAT)), 1) AS mean_bw_util
FROM benchmarks
WHERE performance->>'speedup' IS NOT NULL
GROUP BY category, language
ORDER BY category, mean_speedup DESC;
```

### 4.4 Which IR representation correlates with correctness?

```sql
SELECT
    benchmark_source,
    CASE
        WHEN ir_representations->>'triton_mlir' IS NOT NULL THEN 'has_triton_mlir'
        WHEN ir_representations->>'jaxpr' IS NOT NULL THEN 'has_jaxpr'
        WHEN ir_representations->>'ptx' IS NOT NULL THEN 'has_ptx'
        ELSE 'no_ir'
    END AS ir_type,
    COUNT(*) AS n,
    ROUND(AVG(CASE WHEN correctness->>'passed' = 'true' THEN 1.0 ELSE 0.0 END), 3) AS correctness_rate
FROM benchmarks
GROUP BY benchmark_source, ir_type;
```

### 4.5 KernelBot competition difficulty ranking

```sql
SELECT
    task_id,
    difficulty_level,
    language,
    correctness->>'passed' AS passed,
    performance->>'speedup' AS speedup
FROM benchmarks
WHERE benchmark_source = 'kernelbot'
ORDER BY difficulty_level DESC, speedup DESC;
```

### 4.6 Cross-benchmark speedup correlation matrix

Python/pandas pseudocode:
```python
pivot = df.pivot_table(
    index='category',
    columns='benchmark_source',
    values='performance.speedup',
    aggfunc='mean'
)
corr = pivot.corr()  # Which benchmarks produce correlated speedups?
```

---

## 5. Normalization Pipeline

### Step 1: Source Ingestion
Parse each benchmark's native format (JSON, JSONL, directory of `.cu`, `.py`, `.v`, `.sv` files with metadata).

| Benchmark | Source Format | Parser |
|-----------|--------------|--------|
| KernelBench | JSONL with `task_id`, `reference_code`, `generated_code`, `result` | `normalize_kernelbench.py` |
| PallasBench | Directory of `.py` files + per-kernel `result.json` + IR artifacts | `normalize_pallasbench.py` |
| CVDP | JSONL on HF (`cvdp/results.jsonl`) | `normalize_cvdp.py` |
| Verilog Eval | JSONL on HF | `normalize_verilog_eval.py` |
| MultiKernelBench | CSV + JSON task descriptions | `normalize_multikernelbench.py` |
| KernelBench-v2 | JSONL (variant of KB schema) | reuse `normalize_kernelbench.py` |
| KernelBot | Directory of competition kernels + results | `normalize_kernelbot.py` |

### Step 2: Field Mapping
Map native schema fields to unified JSONL fields using per-benchmark mapping files (`schemas/*_mapping.json`).

```json
{
  "kernelbench_mapping.json": {
    "task_id": "task_id",
    "reference_code": "reference_implementation",
    "generated_code": "generated_source",
    "result.compile_pass": "correctness.compile_passed",
    "result.speedup": "performance.speedup",
    ...
  }
}
```

### Step 3: Validation
Validate each normalized entry against `schemas/unified_benchmark_schema.json` using the JSON Schema validator.

### Step 4: Aggregation
Run cross-benchmark aggregation queries (SQLite or pandas) to produce summary tables.

---

## 6. Integration with KernelBook

KernelBook (the tooling around the existing JSONL datasets on HF) already supports:
- Loading JSONL datasets
- Computing `fast_p` metrics
- Generating evaluation tables
- Uploading to HuggingFace

The unified schema is designed as a **superset** of KernelBook's existing schema:

| KernelBook Field | Unified Field | Notes |
|------------------|---------------|-------|
| `task_id` | `task_id` | Same |
| `source` | `benchmark_source` | Renamed for clarity |
| `language` | `language` | Same |
| `code` | `reference_implementation` | Clarified role |
| `generated` | `generated_source` | Same |
| `correct` | `correctness.passed` | Nested under object |
| `speedup` | `performance.speedup` | Nested under object |

### Compatibility Layer

A thin shim (`scripts/kernelbook_compat.py`) converts unified JSONL back to KernelBook's original flat schema on-the-fly, so existing KernelBook analysis scripts work without modification:

```python
def to_kernelbook_format(unified_entry: dict) -> dict:
    return {
        "task_id": unified_entry["task_id"],
        "source": unified_entry["benchmark_source"],
        "language": unified_entry["language"],
        "code": unified_entry["reference_implementation"],
        "generated": unified_entry["generated_source"],
        "correct": unified_entry["correctness"]["passed"],
        "speedup": unified_entry.get("performance", {}).get("speedup"),
    }
```

---

## 7. Implementation Steps

### Phase 1: Schema & Tooling (Weeks 1–2)

| # | Task | Timeline |
|---|------|----------|
| 1.1 | Finalize `unified_benchmark_schema.json` with stakeholder review | Day 1–2 |
| 1.2 | Implement `validate_schema.py` using `jsonschema` library | Day 3–4 |
| 1.3 | Build `normalize_to_unified.py` CLI with `--source`, `--input`, `--output`, `--dry-run` | Day 5–6 |
| 1.4 | Write `kernelbook_compat.py` compatibility shim | Day 7 |
| 1.5 | Write per-benchmark mapping files (`schemas/*_mapping.json`) | Day 8–10 |
| 1.6 | Unit tests for schema validation and field mapping | Day 11–14 |

### Phase 2: Per-Benchmark Normalizers (Weeks 3–4)

| # | Task | Timeline |
|---|------|----------|
| 2.1 | `normalize_pallasbench.py` — map directory of `.py` + `result.json` + IR to unified format | Week 3 |
| 2.2 | `normalize_kernelbench.py` — map KB JSONL to unified (shared with KBv2) | Week 3 |
| 2.3 | `normalize_cvdp.py` — map CVDP JSONL from HF | Week 3 |
| 2.4 | `normalize_verilog_eval.py` — map Verilog Eval JSONL from HF | Week 4 |
| 2.5 | `normalize_multikernelbench.py` — map CSV/kernel directory | Week 4 |
| 2.6 | `normalize_kernelbot.py` — map competition kernel results | Week 4 |

### Phase 3: Normalization Run & Validation (Week 5)

| # | Task | Timeline |
|---|------|----------|
| 3.1 | Run all normalizers on full datasets, output unified JSONL | Day 1–2 |
| 3.2 | Validate all output against `unified_benchmark_schema.json` | Day 3 |
| 3.3 | Hand-check 10% of entries per benchmark for correctness of mapping | Day 4–5 |
| 3.4 | Fix mapping issues and re-run | Day 6–7 |

### Phase 4: Aggregation & Analysis (Week 6)

| # | Task | Timeline |
|---|------|----------|
| 4.1 | Implement `aggregate.py` with SQLite or pandas | Day 1–2 |
| 4.2 | Run cross-benchmark queries and produce summary report | Day 3–4 |
| 4.3 | Generate visualizations (correlation matrix, pass-rate bar chart, speedup distribution) | Day 5–6 |
| 4.4 | Publish unified dataset to HuggingFace | Day 7 |

---

## 8. Expected Challenges

### 8.1 Different Evaluation Paradigms

| Domain | Evaluation Method | Unified Handling |
|--------|-------------------|-----------------|
| CUDA/Triton | Numerical `allclose` vs reference | `correctness.verification_method = "numerical_allclose"` |
| Pallas/JAX | Numerical `allclose` with robustness filters | Same, plus `robustness_filters` object |
| SystemVerilog | Simulation output matching, testbench pass/fail | `correctness.verification_method = "simulation_match"` |
| Verilog | Simulation + optional formal verification | `correctness.verification_method = "simulation_match"` |
| AscendC | Numerical comparison on NPU | `correctness.verification_method = "numerical_allclose"` |

**Resolution**: The `correctness` object is flexible enough to express all modes via `verification_method`. Downstream analysis scripts must handle cross-method comparisons carefully (e.g., "correctness rate" is comparable; "max_abs_error" is not across HDL vs GPU).

### 8.2 Hardware Requirements

| Benchmark | Hardware | Availability |
|-----------|----------|-------------|
| KernelBench | NVIDIA L40S or A100 | GPU cloud ($$) |
| PallasBench | NVIDIA A100 + optional TPU | GPU cloud + TPU cloud ($$$) |
| CVDP | None (simulation only) | Free |
| Verilog Eval | None (simulation only) | Free |
| MultiKernelBench | NVIDIA GPU + Ascend NPU | Ascend hardware rare |
| KernelBench-v2 | NVIDIA GPU | GPU cloud ($$) |
| KernelBot | NVIDIA GPU | GPU cloud ($$) |

**Resolution**: Performance metrics for HDL benchmarks (`area`, `power`, `frequency`) come from synthesis (requires Vivado/Yosys) but can be omitted with null values. AscendC results require physical Ascend hardware — fall back to compilation-only results where hardware is unavailable.

### 8.3 IR Availability

| DSL | IRs Available | Capture Method |
|-----|---------------|----------------|
| CUDA | PTX, CUBIN | `nvcc --ptx` / `cuobjdump` |
| Triton | Triton MLIR, PTX | `triton.compile` with debug flags |
| Pallas | Jaxpr, StableHLO, Triton MLIR | `jax.make_jaxpr`, `jax.jit().lower().as_text()` |
| AscendC | Ascend IR | CANN debug flags |
| SystemVerilog | N/A (no standard IR) | Omitted |
| Verilog | N/A (no standard IR) | Omitted |

**Resolution**: IR fields are optional (`"type": "string"` with no `required`). Benchmarks without IR capture simply set them to `null`. Future work could explore HDL-to-IR via Yosys RTLIL.

### 8.4 Normalization Drift

Each benchmark's original schema evolves independently. Regular re-normalization (quarterly) is needed to keep the unified dataset current.

**Resolution**: Automated CI workflow that re-runs normalization scripts weekly and reports schema drift.

---

## 9. Success Criteria

- [x] **Unified schema published** (`schemas/unified_benchmark_schema.json`) — validated by CI
- [x] **All 7 benchmarks normalize** to unified JSONL without data loss
- [ ] **Aggregation queries produce meaningful results** — see §4 for examples
- [ ] **Cross-benchmark analysis reveals actionable insights**:
  - Which DSL has highest LLM compilation success rate?
  - Which categories are hardest across all DSLs?
  - Do speedups correlate between CUDA and Pallas for similar operations?
- [ ] **KernelBook compatibility** preserved via shim — existing tools work unchanged
- [ ] **HuggingFace dataset** published with unified JSONL and schema
- [ ] **Reproducible pipeline** — single `python scripts/normalize_to_unified.py --all` produces all normalized output
