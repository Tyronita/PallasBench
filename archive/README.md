# PallasBench Archive

An archive dataset generator for PallasBench, mirroring the [SakanaAI/AI-CUDA-Engineer-Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) format but targeting **JAX/Pallas** on T4 GPU (sm_75).

## Architecture

```
archive/
├── schema.py          # 40-col SakanaAI-compatible schema + JAX extensions
├── problems.py        # All 45 problems with JAX refs + seed Pallas kernels
├── evaluator.py       # Correctness/timing harness (interpret=True on T4)
├── ir_tools.py        # Jaxpr IR-DAG, StableHLO, PTX/SASS extraction
├── llm_generator.py   # Azure LLM kernel generation (DeepSeek-V3-2 / Kimi-K2-6)
├── archive_builder.py # Orchestrates gen+eval, checkpointing, parquet output
└── scripts/
    └── generate_archive.py  # CLI
```

## Quick Start

```bash
# Install deps
pip install pyarrow pandas rich click tqdm jax-cuda12-plugin==0.10.1

# Source T4 env (sets CUDA paths)
source activate_t4.sh

# Seeds only (no LLM key needed)
cd archive && python scripts/generate_archive.py run --n-variants 0

# Full run with DeepSeek-V3-2
export AZURE_OPENAI_API_KEY=<your-key>
python scripts/generate_archive.py run --n-variants 20 --model DeepSeek-V3-2
```

## Schema Extensions vs SakanaAI

| Column | Type | Notes |
|--------|------|-------|
| `Jaxpr_IR` | string | Shape-annotated Jaxpr compute graph |
| `StableHLO_IR` | string | StableHLO MLIR text |
| `PTX_Code` | string | PTX assembly (sm_80+ only) |
| `SASS_Code` | string | SASS disassembly |
| `JAX_Native_Runtime` | float | JAX eager timing |
| `JAX_XLA_Compiled_Runtime` | float | jit-compiled timing |
| `Target_Hardware` | string | GPU model + compute capability |
| `NCU_Profile` | string | Nsight Compute JSON (when available) |

## T4 Note

JAX 0.10.1 Pallas backends (Mosaic GPU / Triton) require sm_80+. T4 is sm_75.
The archive uses `interpret=True` for correctness on T4; JAX reference timings are measured on GPU.
For native Pallas perf numbers, run on A100/RTX3090+.

## HuggingFace Dataset

Generated archives are published to [EvanOLeary/pallasbench-t4-archive](https://huggingface.co/datasets/EvanOLeary/pallasbench-t4-archive).
