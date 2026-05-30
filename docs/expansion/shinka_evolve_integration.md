# ShinkaEvolve Integration: PallasBench as an Evolution Target

## Goal

Add PallasBench as a target benchmark in **ShinkaEvolve** — an evolutionary framework that uses LLM-guided mutation and crossover to optimize program implementations. ShinkaEvolve already runs on CVDP (SystemVerilog design verification) and KernelBench (CUDA kernels). Adding Pallas means evolving Pallas/JAX kernels with correctness + speedup as the fitness function, and capturing full evolutionary traces in KernelBook-compatible format.

## Background

ShinkaEvolve is a framework for LLM-driven program optimization via evolutionary search (Shinka = Japanese for "evolution"). It works by:

1. **Initializing** a population of candidate programs from an LLM seed prompt
2. **Evaluating** each candidate on correctness and performance
3. **Selecting** top candidates via tournament selection
4. **Mutating** selected candidates via LLM-guided code transformations
5. **Crossover** between pairs of candidates to combine successful strategies
6. **Repeating** for N generations

Current supported benchmarks:

| Benchmark | Language | Domain | Fitness Signal |
|-----------|----------|--------|----------------|
| CVDP | SystemVerilog | Hardware design verification | Pass/fail + coverage |
| KernelBench | CUDA/Triton | GPU kernel optimization | Correctness + speedup |
| **PallasBench** (this) | Pallas/JAX | Portable GPU/TPU kernels | Correctness + speedup |

PallasBench is a natural fit: its 45 kernels span three difficulty levels (L1 single ops, L2 fused patterns, L3 architecture components), and the robust evaluation framework already provides correctness verification (5 robustness filters) and performance measurement (speedup over JAX baseline).

## Integration Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                   ShinkaEvolve Loop                      │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────────┐     │
│  │ Seed LLM │───>│ Population│───>│  Evaluate      │     │
│  │ Prompt   │    │ (N=10-20) │    │  Correctness   │     │
│  └──────────┘    └───────────┘    │  + Speedup     │     │
│                         ↑         └───────┬───────┘     │
│                         │                 │             │
│  ┌──────────┐    ┌───────────┐           │             │
│  │ LLM      │<───│ Selected  │<──────────┘             │
│  │ Mutation │    │ Parents   │                          │
│  │ Crossover│    │ (Tournament)│                        │
│  └──────────┘    └───────────┘                          │
│                                                          │
│  ┌──────────────────┐                                   │
│  │ Evolution Trace  │  Per-gen: patches, fitness, src   │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

### Fitness Function

The fitness of a candidate Pallas kernel is a scalar computed from two components:

**Correctness score** (pass/fail, gated):
- Run kernel on 3 random seeds with input shapes from the task definition
- Compare output against JAX baseline via `jnp.allclose(atol=1e-5, rtol=1e-5)`
- Apply all 5 robustness filters (output range, std, axes variation, input impact, source analysis)
- If any seed fails correctness or robustness → fitness = 0 (dead candidate)

**Speedup score** (continuous, after JIT warmup):
- `speedup = baseline_time / kernel_time`
- Baseline is the standard JAX `jnp` implementation for the same input shapes
- Kernel time measured after 3 warmup runs, averaged over 10 timed runs
- JIT compilation time is NOT included in speedup (but tracked separately)

**Combined fitness**:
```
if not correct:
    fitness = 0.0
else:
    fitness = speedup  (capped at 10.0 to prevent outlier dominance)
```

**Elite preservation**: The top-1 candidate from each generation is automatically carried over unmodified (elitism).

### Mutation Operators

Mutations are LLM-guided code transformations applied to the kernel source. Each mutation receives the current kernel source and a natural-language instruction:

| Operator | Instruction Template | Effect |
|----------|---------------------|--------|
| `mutate_block_size` | "Adjust block size to minimize padding waste" | Changes `block_size`, `grid` dimensions |
| `mutate_memory_pattern` | "Optimize memory access for coalescing" | Reorders loops, permutes block specs |
| `mutate_fusion` | "Fuse elementwise operations into the main loop" | Combines adjacent kernels |
| `mutate_precision` | "Convert intermediate values to float32 for numerical stability" | Adds `jnp.float32` casts |
| `mutate_tiling_2d` | "Tile both M and N dimensions for better parallelism" | Adds second grid dimension |
| `mutate_warp_optimize` | "Structure computation for GPU warp utilization" | Rearranges for warp-level ops |

Each mutation includes:
- A **diff** against the parent kernel source
- The **LLM rationale** for the change (captured from model output)
- The **mutation operator name** for traceability

Mutation rate controls the probability that a selected parent undergoes mutation (default: 0.8).

### Crossover

Crossover combines tiling configurations from two parent kernels. For Pallas kernels, the crossover operates at the tiling strategy level rather than line-level source merge:

1. **Parent A** contributes: block sizes (M, N, K dimensions)
2. **Parent B** contributes: grid layout and block spec index maps
3. The child kernel is generated by applying A's block sizes to B's grid structure

Fallback: if structural crossover fails (incompatible kernel signatures), the child is a direct copy of the fitter parent.

Crossover rate controls probability of crossover between two selected parents (default: 0.3).

### Tournament Selection

Standard tournament selection:
- Tournament size: 3
- Select 2 parents per generation (with replacement)
- Parents compete via fitness score
- Elitism: top-1 candidate copied directly to next generation

### Population Management

- **Population size**: 10--20 candidates per generation (constrained by JIT compilation cost)
- **Generations**: 5--10 per experiment (limited by total evaluation budget)
- **Initialization**: Seed population via LLM prompt with task description + example kernel
- **Dead candidate replacement**: Candidates with fitness=0 are replaced by random mutations of surviving elites

## Evolution Trace Capture

Every generation produces a structured record:

```json
{
  "experiment_id": "shinka-pallas-2026-05-30-001",
  "task_name": "L1/matmul",
  "generation": 3,
  "population": [
    {
      "candidate_id": "gen3_ind5",
      "parent_id": "gen2_ind2",
      "source": "def pallas_kernel(x_ref, y_ref, o_ref):\n    ...",
      "diff": "--- parent\n+++ child\n@@ -1,5 +1,8 @@\n...",
      "mutation": {
        "operator": "mutate_block_size",
        "instruction": "Adjust block size to minimize padding waste",
        "llm_rationale": "Increasing BLOCK_M from 64 to 128 improves occupancy..."
      },
      "fitness": {
        "correctness_passed": true,
        "max_abs_error": 2.3e-7,
        "robustness_filters": {
          "output_range": {"passed": true, "value": 6.28},
          "output_std": {"passed": true, "value": 1.42},
          "axes_variation": {"passed": true},
          "input_impact": {"passed": true, "value": 0.31},
          "source_analysis": {"passed": true}
        },
        "speedup": 1.47,
        "baseline_time_ms": 3.21,
        "kernel_time_ms": 2.18,
        "jit_compile_time_s": 412.0
      },
      "hardware": "NVIDIA A100 80GB PCIe",
      "timestamp": "2026-05-30T09:15:00Z"
    }
  ]
}
```

### Output Format: KernelBook-format JSONL

The full evolution trace is written as a JSONL file (one JSON object per candidate per generation) with:

| Field | Description |
|-------|-------------|
| `experiment_id` | Unique experiment identifier |
| `task_name` | PallasBench task name (e.g., "L1/matmul") |
| `generation` | Generation number (0-indexed) |
| `candidate_id` | Unique candidate within experiment |
| `parent_id` | Source candidate (for crossover, both parents listed) |
| `source` | Full kernel source code |
| `diff` | Unified diff against parent |
| `mutation` | Mutation operator, instruction, LLM rationale |
| `fitness` | Fitness score breakdown (correctness + speedup) |
| `hardware` | GPU model for evaluation |
| `timestamp` | ISO 8601 timestamp |

File naming: `evolve_{task_name}_{experiment_id}.jsonl`

## Implementation Steps

### Step 1: ShinkaEvolve integration adapter

Create `shinka_evolve/adapter.py` that maps PallasBench task definitions into ShinkaEvolve's standard `BenchmarkTask` interface:
- `get_task_description()` → PallasBench task docstring
- `get_reference_implementation()` → JAX baseline source
- `evaluate(candidate_source)` → correctness + speedup via `PallasFitnessFunction`

### Step 2: Fitness function

Build `pallasbench/evolution/fitness.py`:
- `PallasFitnessFunction` class wrapping the robust evaluation pipeline
- `evaluate()` method that returns fitness score + detailed metrics
- Robustness filter integration (reuse from `pallasbench.evaluation`)

### Step 3: Mutation operators

Implement LLM-guided mutation as a `MutationOperator` registry:
- Each operator is a `(name, instruction_template, apply_fn)` triple
- `apply_fn` takes (source, llm_client) and returns (mutated_source, rationale)
- Start with 2--3 operators (block size, memory pattern, fusion)

### Step 4: Evolutionary loop script

Implement `scripts/evolve_pallas.py`:
- CLI with `--kernels`, `--generations`, `--population-size`, `--mutation-rate`, `--crossover-rate`, `--output-dir`
- Tournament selection, elitism, crossover
- JSONL trace output per generation
- Progress logging with per-generation fitness statistics

### Step 5: Integration test

Run a short evolutionary loop on a single L1 kernel (e.g., `L1/relu`) for 2 generations, population size 3, to validate:
- Correctness checks pass
- Fitness scores are computed
- JSONL output is valid
- Trace capture works

### Step 6: Documentation

Add usage examples and expected output format to `docs/expansion/shinka_evolve_integration.md` (this document).

## Expected Challenges

### JIT Compilation Bottleneck

The dominant challenge. First-run JIT compilation for a single Pallas kernel on A100 takes ~400s (dominated by `ptxas` assembly). For a population of 10 candidates × 5 generations = 50 evaluations, that's **~5.5 hours** of wall-clock time.

**Mitigations:**
- **Compilation caching**: If the kernel shape hasn't changed, JAX caches the compiled binary. Repeated evaluations of the same candidate within a generation are instant (~0.2ms).
- **Parallel evaluation across GPUs**: ShinkaEvolve can shard the population across multiple GPUs (up to 8 on A100 nodes).
- **Warm-start**: Seed the initial population with kernels that share the same grid/block structure to hit the XLA compilation cache.
- **Reduced population size**: Accept smaller populations (N=5--10) for initial experiments.

### Limited Population Sizes

With 10 candidates per generation, the evolutionary search has limited diversity. Standard genetic algorithms use 100--1000+ population sizes.

**Mitigations:**
- LLM-guided mutations are more targeted than random mutations — each candidate explores a promising direction.
- Crossover combines strategies from two parents, effectively increasing search coverage.
- Future: use surrogate models (lightweight cost models) to filter candidates before full compilation.

### LLM API Latency

Each mutation/crossover requires an LLM call. For 10 candidates × 5 generations with 80% mutation rate = ~40 LLM calls.

**Mitigations:**
- Batch mutation requests where possible
- Use a fast local model (e.g., CodeLlama 7B) for mutations, reserving larger models for seed generation

### Correctness Verification Cost

Running all 5 robustness filters adds ~1s per evaluation. This is noise compared to JIT compilation time (400s), so no mitigation needed.

## Success Criteria

### Minimum Viable (Milestone 1)

1. ✅ `PallasFitnessFunction` correctly evaluates a single kernel (correctness + speedup)
2. ✅ Evolutionary loop runs for ≥2 generations without crashing
3. ✅ JSONL trace output is valid and contains all required fields
4. ✅ At least one candidate across the run improves speedup over the original kernel

### Target (Milestone 2)

5. ✅ Population of ≥10 candidates per generation
6. ✅ ≥5 generations completed in a single experiment
7. ✅ ≥2 mutation operators producing valid (correct) kernel variants
8. ✅ Crossover produces at least one viable child with combined tiling strategy
9. ✅ Full evolutionary trace uploaded to HuggingFace as KernelBook-format dataset

### Stretch (Milestone 3)

10. ✅ Speedup improvement of ≥1.2× on at least 3 different L1 kernels
11. ✅ Speedup improvement of ≥1.1× on at least 1 L2 fused kernel
12. ✅ Cross-generational fitness improvement trend (monotonic or near-monotonic)
13. ✅ Published analysis: "Can LLM-driven evolution optimize Pallas kernels?"

## References

- ShinkaEvolve: https://github.com/Tyronita/ShinkaEvolve
- PallasBench: https://github.com/Tyronita/PallasBench
- KernelBench: Ouyang et al. (2025), "Can LLMs Write Efficient GPU Kernels?"
- Robust evaluation: Lange et al. (2025), "Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization"
- Pallas: https://jax.readthedocs.io/en/latest/pallas/
