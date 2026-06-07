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

## Recommendation

Report `fast_p` against the **compiled** baseline (the KernelBench convention —
torch.compile ≈ jax.jit). Keep `Pallas_Speedup_Native` only as a secondary "vs eager"
column. On sm_80+ hardware, re-time with native Pallas (no interpret mode) for
absolute numbers.
