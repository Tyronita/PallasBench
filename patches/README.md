# GPU Compilation Fix: Triton 1M Element Limit

## The Problem

Triton enforces a **1,048,576 (2^20) element limit** per program/block. Pallas kernels that use
`BlockSpec` shapes whose element count (product of all dimensions) exceeds this limit fail to
compile on GPU backends (CUDA/Metal).

For example, a kernel with input shape `(4096, 4096)` and `block_size = min(1024, 4096)` creates
a `BlockSpec((1024, 4096), ...)` — that's **4,194,304 elements**, 4× over the limit.

## The Fix: Block Size Clamping

Each kernel's block size is clamped to respect the 1M element budget:

| Pattern | Before | After |
|---------|--------|-------|
| 1D elementwise | `block_size = min(1024, n)` | `block_size = min(n, MAX_BLOCK)` with `MAX_BLOCK = 65536` |
| Row-wise 2D | `block_rows = min(128, n_rows)` | `block_rows = min(n_rows, MAX_BLOCK)` |
| Matmul 2D | `bm = min(512, m)` | `BLOCK_M = min(m, 128)` |
| Matmul 2D | `bn = min(512, n)` | `BLOCK_N = min(n, 128)` |
| Gated MLP | `bm = min(256, m)` | `BLOCK_M = min(m, 128)` |
| Attention | `block_q = min(128, seq_len)` | `BLOCK_Q = min(seq_len, 128)` |

## Scope

**35 files** were modified across all three levels with **+106 / −12** lines changed.

| Level | Modified | Description |
|-------|----------|-------------|
| level1 | 22 | Elementwise, matmul, reduction, normalization, loss, softmax kernels |
| level2 | 10 | Fused matmul-activation, norm-residual, gated MLP, fused loss kernels |
| level3 | 3 | Flash attention, gated MLP, attention component kernels |

## Core Fix Pattern

```python
# Before
block_size = min(1024, n)

# After
MAX_BLOCK = 65536
block_size = min(n, MAX_BLOCK)
```

All changes are **idempotent** — running the fix script multiple times is safe.

## Scripts

- `apply_gpu_fix.py` — Programmatically applies the block-size clamping fix to all kernel files.
- `generate_patch_files.py` — Generates `.diff` patch files for each modified kernel.
- `gpu_compatibility_check.py` — Scans kernel files for potential GPU issues and reports findings.
