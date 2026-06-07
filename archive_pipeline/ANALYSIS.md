# PallasBench Archive — Results Analysis & Integrity Audit

Device: **NVIDIA Tesla T4 (sm_75)** · JAX 0.10.1 · Pallas via `interpret=True`
(correctness), JAX references timed on GPU. 104 problems, 217 correct LLM variants.

## TL;DR

- **No reward hacking.** 216 / 217 correct kernels use a real `pallas_call`; the one
  exception is a documented JAX-only fallback (`gated_mlp`).
- **The headline "speedups" are a baseline artifact.** They are measured against the
  **unjitted** JAX reference (`Pallas_Speedup_Native`). Against the fair **jitted /
  XLA-compiled** baseline (`Pallas_Speedup_Compiled`) only **10%** of correct kernels
  beat JAX, and every top-10 entry collapses to **0.57×–1.05×**.

## Top 10 by raw speedup — and what they really are

| # | Op | L | vs **native** (unjitted) | vs **compiled** (fair) | Verdict |
|---|----|---|--------------------------|------------------------|---------|
| 1 | linear_recurrence | 4 | **112.8×** | 1.79× | baseline artifact |
| 2 | griffin_recurrent_block | 4 | **60.2×** | 1.04× | baseline artifact |
| 3 | fused_softmax_cross_entropy | 2 | **14.9×** | 1.05× | baseline artifact |
| 4 | gelu | 1 | **10.8×** | 0.64× | artifact (slower fair) |
| 5 | geglu | 2 | **10.7×** | 0.91× | baseline artifact |
| 6 | sigmoid_bce | 2 | **9.3×** | 0.57× | artifact (slower fair) |
| 7 | fused_gelu_bias | 2 | **8.4×** | 0.62× | artifact (slower fair) |
| 8 | gc_content | 1 | **7.4×** | 0.59× | artifact (slower fair) |
| 9 | molecular_distance_matrix | 4 | **7.2×** | 0.94× | baseline artifact |
| 10 | sigmoid_bce | 2 | **7.2×** | 0.56× | artifact (slower fair) |

**Why the gap:** `Pallas_Speedup_Native` divides by the *eager* JAX reference. For
scan-heavy ops (`linear_recurrence`, `griffin`) eager `vmap(lax.scan(...))` dispatches
every timestep separately (~100 ms), so a jitted fused Pallas loop looks 60–112×
faster. That is real *vs eager Python*, but the honest comparison is vs `jax.jit`
(XLA-compiled), where the same kernels are roughly tied or slower.

The two scan kernels were inspected line-by-line: both implement a genuine in-kernel
recurrence (`for t in range(T): h = a_t*h + x_t`) inside `pallas_call`, output
`max_diff = 0.0`. They are correct and legitimately fused — the inflation is purely
the denominator.

## Reward-hacking checks (all passed)

| Check | Result |
|-------|--------|
| Correct LLM variants | 217 |
| Skipped `pallas_call` (pure JAX masquerading as a kernel) | **1** (`gated_mlp`, declared fallback) |
| Output is degenerate / input passthrough | none found (max_diff distribution sane) |
| Correctness via weak tolerance | atol/rtol = 1e-2; fp32 ops mostly 0.0, fp16 attention ~1e-3 |
| Beats **compiled** JAX (`sc>1`) | 22 / 217 = **10%** |
| Beats **native** JAX (`sn>1`) | 99 / 217 = **45%** |

The 45% → 10% drop between native and compiled baselines **is** the anomaly, and it is
a benchmark-design issue (which baseline you divide by), not model cheating.

## Caveat: L3/L4 "whole model in Pallas" is partly ill-posed

The survey shows prior-art depth collapsing to 0–1 real-Pallas seeds at L3/L4, and LLM
correctness collapsing with it (0–1%). This is not only difficulty — it reflects how
the JAX/XLA stack is *designed to be used*:

- **Pallas is a kernel language, not a model language.** It produces one fused array op
  with hand-specified tiling/grid/on-chip memory, then hands control back to the XLA
  graph. It has no mechanism for the control flow, parameter management, or sharding a
  whole model needs.
- **XLA already optimizes the model level.** You write the model in `jnp`/`lax` and XLA
  fuses/schedules the whole graph. You only drop to Pallas at the *leaves* where XLA
  measurably underperforms.
- **Those leaves are few and known:** memory-bound patterns — flash attention (avoid
  materializing the N×N scores), fused norms, and sequential scans (Mamba/Griffin
  recurrence). That is essentially the entire public Pallas catalog. The rest of a
  model is compute-bound matmuls that cuBLAS/XLA already run near peak, where a Pallas
  rewrite only loses to the vendor library.
- **Economics:** Pallas kernels are hard to write, hardware-specific (good backends are
  sm_80+ only), and API-churny. Nobody hand-writes ~50 kernels for a ResNet to match
  what XLA gives for free.

So the realistic L4 target is **"a JAX model with a Pallas attention/scan kernel swapped
in"** — exactly what MaxText, RecurrentGemma, and AlphaFold actually do — not a whole
forward pass rewritten in Pallas. Our L4 problems use a JAX reference as the seed and
ask the LLM to Pallas-ify; the near-zero pass rate is the expected signal that the *hot
kernel*, not the whole model, is the correct unit of optimization. Treat L4
`vs_compiled` numbers as "did the LLM find the fusible kernel inside the model," not
"did it beat XLA on the full model."

## Recommendation

Report `fast_p` against the **compiled** baseline (the KernelBench convention —
torch.compile ≈ jax.jit). Keep `Pallas_Speedup_Native` only as a secondary "vs eager"
column. On sm_80+ hardware, re-time with native Pallas (no interpret mode) for
absolute numbers. For L4, score the *embedded hot kernel* (attention/scan/norm), not
the entire forward pass.

## Future work — multi-DSL tracks and the L4 cliff

The L1→L4 ladder (single op → fused pattern → architecture component → whole model) is
universal — it's just increasing graph size and decreasing data locality, which is
hardware physics, not framework-specific. What differs across kernel ecosystems is
*where the prior-art cliff sits*, and that is dictated by whether a graph compiler sits
above the kernel language. A future multi-DSL PallasBench (MultiKernelBench-style)
should set the L4 scoring rule per track accordingly.

| DSL / stack | Layer it occupies | L1/L2 prior art | L4 prior art | Same cliff as Pallas? |
|-------------|-------------------|-----------------|--------------|-----------------------|
| **Pallas** (JAX) | leaf kernels under XLA | medium | ~none | — (reference) |
| **CUDA** | raw, no compiler above | deep | exists (vLLM, FasterTransformer, TensorRT) | no — whole models hand-written |
| **CUTLASS** | GEMM/epilogue templates | deep (GEMM, fused epilogues) | ~none | steeper — it's a GEMM lib, not a model tool |
| **Mojo / MAX** | spans kernel **and** graph by design | thin (young) | thin (young) | no cliff by design; scarcity is ecosystem age |
| **MLX** (Apple) | leaf kernels under MLX graph | thin–medium | ~none | **yes — same cause** |
| **Metal** (raw, Apple) | raw, no compiler above | deep | exists (llama.cpp Metal, MLX-LM) | no — like CUDA |
| **MPS / MPSGraph** | tuned primitives + graph | deep (primitives) | ~none | yes — primitive lib |

Two cross-vendor signals worth recording:

1. **The optimized-kernel set converges.** Independently, JAX/Pallas and Apple/MLX
   ship the *same* hand-tuned leaves: attention, RMSNorm, LayerNorm, RoPE/scan.
   MLX's built-in `mlx.core.fast.*` (`scaled_dot_product_attention`, `rms_norm`,
   `layer_norm`, `rope`) ≈ the entire public Pallas catalog. The unit of
   hand-optimization is the fusible **memory-bound leaf**, regardless of vendor — so
   L1/L2 of any track should be seeded mostly from those.

2. **The "score the embedded hot kernel, not the full forward pass" rule is specific to
   compiler-fronted DSLs** (Pallas, MLX, CUTLASS). For raw-language tracks (CUDA,
   Metal) L4 becomes a *legitimate* whole-model task with real prior art, and should be
   scored against a real full-model baseline, not the hot-kernel proxy.

Apple-specific caveat for a future MLX track: inference would run on Apple-Silicon
**unified memory**, which shrinks the classic flash-attention win ("don't materialize
the N×N scores to VRAM") because there is no separate VRAM to spill to. The
memory-bound calculus — and therefore which kernels actually show speedups — shifts,
so an MLX track needs its own baseline calibration rather than reusing T4/A100 numbers.
