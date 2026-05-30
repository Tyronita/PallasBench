# Expansion Plans

This directory contains design documents for future PallasBench expansion directions.

## Plans

| Document | Area | Priority | Status |
|----------|------|----------|--------|
| [dsl_unification.md](dsl_unification.md) | Cross-benchmark DSL unification | High | Planning |
| *(coming soon)* Multi-backend comparison | GPU + TPU + CPU kernel perf matrix | Medium | Proposed |
| *(coming soon)* NCU profiling | Nsight Compute integration for roofline analysis | Medium | Proposed |
| *(coming soon)* ShinkaEvolve integration | Evolutionary LLM kernel optimization | High | Proposed |

## Unified Dataset

The DSL unification plan defines a common JSONL schema shared across 7 benchmarks and 6 DSLs:

| Benchmark | DSL | Tasks | Schema |
|-----------|-----|-------|--------|
| KernelBench | CUDA, Triton | 250 | `schemas/unified_benchmark_schema.json` |
| PallasBench | Pallas/JAX | 45 | Same |
| CVDP | SystemVerilog | 304 | Same |
| Verilog Eval | Verilog | 157 | Same |
| MultiKernelBench | CUDA, Triton, Pallas, AscendC | 285 | Same |
| KernelBench-v2 | Triton | ~250 | Same |
| KernelBot | CUDA | varies | Same |

See [dsl_unification.md](dsl_unification.md) for the full plan, [schemas/unified_benchmark_schema.json](../../schemas/unified_benchmark_schema.json) for the JSON Schema, and [scripts/normalize_to_unified.py](../../scripts/normalize_to_unified.py) for the CLI tool.
