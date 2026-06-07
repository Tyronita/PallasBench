# PallasBench Archive Pipeline

LLM-driven generation of a SakanaAI AI-CUDA-Engineer-style archive for **JAX/Pallas**
kernels, organized in KernelBench-style levels (L1–L4) and generated with the
**generate → evaluate → feedback → retry** loop from the AI CUDA Engineer and
KernelBench papers.

## What this is

For every benchmark problem we:
1. Take a **seed** Pallas kernel (generation 0) + a JAX reference,
2. Ask an LLM (Azure **DeepSeek-V3-2**, fallback gpt-4o-mini) for **20 diverse
   variants** using rotating optimization-strategy prompts,
3. Evaluate each variant for **correctness** and **speedup vs JAX**, capturing the
   Jaxpr IR-DAG and StableHLO,
4. Feed compile/correctness errors back and retry (≤2),
5. Keep **all** attempts (including failures) as training signal,
6. Emit a HuggingFace parquet dataset with per-row device attribution.

## Levels (104 problems, KernelBench-parity structure)

| Level | What | Count |
|-------|------|-------|
| **L1** | Single ops (activations, norms, reductions, elementwise, loss, index, genomics) | 46 |
| **L2** | Fused patterns (matmul+act, norm+residual, gating, attention scores, RoPE, recurrence) | 23 |
| **L3** | Architecture components (FFN, MoE gate, cross-attention, GLU, flash attention) | 10 |
| **L4** | **Full model forward passes** (LLaMA / BERT / GPT-2 / ViT / Whisper / Mamba / Griffin / ResNet / MobileNetV2 / Conformer / T5 / DeepSeek-MoE / U-Net / GCN / AlphaFold triangle + finance & physics) | 25 |

Each problem carries real **GitHub provenance** (`GitHub_URL`) to the JAX/Pallas
source it was derived from (jax-ml/jax, MaxText, Flax, RecurrentGemma, AlphaFold3,
EasyDeL, sglang-jax, jax-md, Brax, HuggingFace Transformers, …).

## Target device — IMPORTANT

All runs here were produced on an **NVIDIA Tesla T4 (sm_75, 16 GB)**.

> JAX 0.10.1 Pallas backends (Mosaic GPU / Triton) require **sm_80+** (Ampere).
> On the T4 (sm_75) Pallas executes via `interpret=True` for **correctness**, while
> JAX reference timings are measured on the real GPU. Speedup numbers are therefore
> a conservative relative signal, **not** absolute native-Pallas GPU performance.
> Every dataset row is tagged with `Target_Hardware` / `Compute_Capability` so T4
> and A100 records never get conflated. For native Pallas timings, re-run on sm_80+.

## Datasets (HuggingFace)

- [`EvanOLeary/pallasbench-archive`](https://huggingface.co/datasets/EvanOLeary/pallasbench-archive) — full 104-problem archive, splits `level_1..level_4` (T4).
- [`EvanOLeary/pallasbench-unified`](https://huggingface.co/datasets/EvanOLeary/pallasbench-unified) — deduplicated T4 + A100 merge, device-tagged.
- [`EvanOLeary/pallasbench-robust-gpu-a100`](https://huggingface.co/datasets/EvanOLeary/pallasbench-robust-gpu-a100) — original 45-problem A100 seed set.

## Layout

```
pallas_bench/
  schema.py            # 40-col SakanaAI-compatible schema + JAX IR columns
  problems.py          # original 45 problems (L1/L2/L3)
  extended_problems.py # 59 new problems (L1–L4) with GitHub provenance
  evaluator.py         # correctness + timing; T4 interpret-mode aware
  ir_tools.py          # Jaxpr IR-DAG, StableHLO, PTX/SASS, block/grid, NCU
  llm_generator.py     # feedback-loop generator (DeepSeek-V3-2 primary)
  archive_builder.py   # orchestration, checkpointing, parquet assembly
scripts/
  run_archive.py       # original 45-problem runner
  run_extended.py      # 100-problem L1–L4 runner
  generate_archive.py  # CLI (run / check / list-problems / push)
activate_t4.sh         # sources the CUDA/ptxas/libdevice env for the T4 box
```

## Run

```bash
source activate_t4.sh
export AZURE_OPENAI_API_KEY=...   # Azure DeepSeek-V3-2 endpoint
python scripts/run_extended.py --levels 1 --levels 2 --levels 3 --levels 4 \
    --n 20 --model DeepSeek-V3-2 \
    --output results/extended_archive \
    --push-hf EvanOLeary/pallasbench-archive
```

## Metrics

KernelBench `fast_p`: fraction of variants that are correct **and** have speedup > p.
We report `fast_0` (correct), `fast_1` (>1×), `fast_2` (>2×), `fast_5` (>5×) per row.

## Credits

- Benchmark base: [Tyronita/PallasBench](https://github.com/Tyronita/PallasBench)
- Method: SakanaAI *AI CUDA Engineer* + Stanford *KernelBench*
- Generation: Azure-hosted DeepSeek-V3-2
