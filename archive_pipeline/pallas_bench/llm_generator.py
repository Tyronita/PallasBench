"""
LLM-driven Pallas kernel generation with feedback loop.

Architecture (from AI CUDA Engineer + KernelBench papers):
  1. Generate diverse variant with strategy hint
  2. Evaluate: correctness + JAX reference timing
  3. If error → feed error message back, retry (up to 2 retries)
  4. If correct but slow → feed speedup back, try next strategy
  5. Store ALL attempts (including failures) — critical for training data

Models: DeepSeek-V3-2 (primary), gpt-4o-mini (fast fallback/retries)
"""
from __future__ import annotations

import asyncio
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Azure LLM client (from Evolve project)
# ---------------------------------------------------------------------------
_EVOLVE_PATH = Path("/home/OLeary/Evolve")
if str(_EVOLVE_PATH) not in sys.path:
    sys.path.insert(0, str(_EVOLVE_PATH))

try:
    from metashinka.jax_evolution.azure_llm import AzureLLM, AZURE_DEPLOYMENTS
    _HAS_AZURE = True
except ImportError:
    _HAS_AZURE = False
    AzureLLM = None
    AZURE_DEPLOYMENTS = {}

PRIMARY_MODEL   = "DeepSeek-V3-2"
FALLBACK_MODEL  = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# System prompt  (paper insight: keep simple, no heavy few-shot examples)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert in JAX/Pallas GPU kernel programming.
You write JAX Pallas kernels using `jax.experimental.pallas as pl`.

## Critical Pallas API rules

Pallas uses BLOCK-LEVEL programming, NOT thread-level. Each kernel invocation processes
one entire block — you NEVER use pl.thread_id() or pl.program_id() directly.

The ONLY correct pattern:
```python
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def my_kernel(x_ref, o_ref):          # x_ref has shape = block_shape
    o_ref[...] = jnp.maximum(x_ref[...], 0.0)   # operates on the whole block

def my_fn(x):
    n = x.shape[0]
    block = 1024
    return pl.pallas_call(
        my_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),                        # number of blocks
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],  # i = block index
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
```

Key rules:
- NEVER use pl.thread_id(), pl.program_id(), or lane-level ops
- x_ref[...] gives the whole block as a jnp array — do jnp operations on it
- out_ref[...] = result  writes the block back
- index_map lambda: `lambda i: (i,)` means block i covers rows [i*block:(i+1)*block]
- For 2D: `lambda i, j: (i, j)` with grid=(M//BM, N//BN)
- Block size product must be ≤ 1,048,576
- No fori_loop with dynamic bounds — use static integers for loop ranges
- Never import torch or use Hopper-only ops (wgmma, cp.async.bulk)

Output ONLY a ```python ... ``` block containing imports + kernel + wrapper.
""").strip()

# ---------------------------------------------------------------------------
# 20 diverse strategy prompts
# ---------------------------------------------------------------------------

STRATEGIES = [
    "Use a different tiling strategy — try to maximize memory coalescing.",
    "Reduce memory bandwidth by using shared memory (SMEM) scratchpad.",
    "Try wider block widths for better vectorized loads (e.g. block=2048 for 1D).",
    "Focus on arithmetic intensity — fuse as many ops as possible inside the kernel body.",
    "Use float32 accumulation with bfloat16 inputs for a mix of precision and throughput.",
    "Try a 2D tiling strategy even for a 1D problem to explore spatial reuse.",
    "Use online normalization (Welford or log-sum-exp) to reduce memory passes.",
    "Fuse the backward-compatible elementwise ops directly into the kernel.",
    "Unroll the inner loop manually using explicit slice indexing instead of fori_loop.",
    "Use smaller block sizes (e.g. 64 or 128) to maximize occupancy on Turing/Ampere.",
    "Try block sizes that are powers of 2 closest to sqrt(total_elements).",
    "Exploit data symmetry or sparsity if it exists in the operation.",
    "Use half-precision (float16) internally and upcast only for accumulation.",
    "Batch multiple rows into each block to amortize kernel launch overhead.",
    "Use a grid that is a multiple of SM count (2560 SMs for T4) for full occupancy.",
    "Try to minimize synchronization barriers inside the kernel body.",
    "Use separable computation — split complex ops into simpler sequential sub-kernels.",
    "Apply loop strength reduction: replace expensive ops (div, sqrt) with cheaper approximations.",
    "Use a column-major or transposed memory layout if the access pattern benefits.",
    "Combine the operation with a common preceding/following op (e.g. add residual inline).",
]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_initial_prompt(problem, strategy_idx: int, variant_idx: int) -> str:
    strategy = STRATEGIES[strategy_idx % len(STRATEGIES)]
    return textwrap.dedent(f"""
Generate a JAX Pallas kernel for this operation.

## Specification
Function name: `{problem.name}`
Inputs: {problem.input_shapes}  dtypes: {problem.input_dtypes}
Reference implementation:
```python
{problem.jax_functional.strip()}
```

## Seed kernel (generation 0 baseline)
```python
{problem.seed_pallas.strip()}
```

## Your task
Write variant #{variant_idx}. Strategy: {strategy}

Output a complete ```python``` block with the kernel and wrapper named `{problem.name}`.
""").strip()


def _build_error_feedback_prompt(problem, prev_code: str, error: str, attempt: int) -> str:
    return textwrap.dedent(f"""
Your previous kernel (attempt {attempt}) failed with:
```
{error[:600]}
```

Previous code:
```python
{prev_code[:1200]}
```

Fix the error while keeping the optimization intent. Common fixes:
- If "fori_loop boolean conversion": use static Python integers, not traced JAX values, for loop bounds
- If "block size": reduce block dimensions so product ≤ 1,048,576
- If "shape mismatch": ensure out_ref[...] shape matches the BlockSpec block shape
- If "index_map": use (lambda i: (i,)) — block index, not element index

Output a fixed ```python``` block with function named `{problem.name}`.
""").strip()


def _build_speedup_feedback_prompt(problem, prev_code: str, speedup: float, jax_ms: float) -> str:
    return textwrap.dedent(f"""
Your kernel is correct (speedup={speedup:.2f}x, JAX baseline={jax_ms:.3f}ms) but we want faster.
Try one of:
  1. Larger block size to reduce grid launches
  2. Process multiple rows per block (bm=32 instead of 16)
  3. Avoid recomputing values — precompute constants outside the kernel
  4. Fuse a follow-up elementwise op into the kernel body

Previous correct code:
```python
{prev_code[:1200]}
```

Output an optimized ```python``` block with function named `{problem.name}`.
""").strip()

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code(text: str, fn_name: str) -> str:
    """Extract first ```python block; fallback to raw text."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "def " in code:
            return code
    # Fallback: return whole response if it looks like code
    if "def " in text and "import" in text:
        return text.strip()
    return ""

# ---------------------------------------------------------------------------
# Single LLM call
# ---------------------------------------------------------------------------

async def _llm_call(prompt: str, model: str = PRIMARY_MODEL,
                    temperature: float = 0.7, max_tokens: int = 3000) -> str:
    """Call LLM, return raw text content."""
    if not _HAS_AZURE:
        return ""
    try:
        llm = AzureLLM(deployment=model)
        resp = await llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        await llm.close()
        content = resp.get("content", "")
        if resp.get("finish_reason") == "error":
            return ""
        return content
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Main: generate one variant with feedback loop
# ---------------------------------------------------------------------------

async def generate_variant_with_feedback(
    problem,
    variant_idx: int,
    evaluator_fn,                     # callable(problem, code, fn_name) -> row dict
    model: str = PRIMARY_MODEL,
    temperature: float = 0.7,
    max_retries: int = 2,
) -> list[dict]:
    """
    Generate one variant using the feedback loop.
    Returns a list of row dicts (may contain multiple attempts).
    """
    rows: list[dict] = []
    strategy_idx = variant_idx - 1   # 0-based strategy selection

    # -- Initial generation
    prompt = _build_initial_prompt(problem, strategy_idx, variant_idx)
    raw = await _llm_call(prompt, model=model, temperature=temperature)
    code = _extract_code(raw, problem.name)

    if not code:
        # Fallback model
        raw = await _llm_call(prompt, model=FALLBACK_MODEL, temperature=0.5)
        code = _extract_code(raw, problem.name)

    if not code:
        code = problem.seed_pallas   # worst-case: duplicate seed

    # -- Evaluate + feedback loop
    current_code = code
    for attempt in range(max_retries + 1):
        row = evaluator_fn(
            problem,
            kernel_source=current_code,
            kernel_fn_name=problem.name,
            generation_idx=variant_idx,
            llm_model=f"{model}_attempt{attempt}",
            capture_ir=(attempt == 0),
        )
        rows.append(row)

        correct = row.get("Correct", False)
        error   = row.get("Error", "")
        speedup = row.get("Pallas_Speedup_Native", float("nan"))
        jax_ms  = row.get("JAX_Native_Runtime", float("nan"))

        if correct:
            # Mark final row with clean model name
            row["LLM_Model"] = model
            row["Generation_Idx"] = variant_idx
            break

        if error and attempt < max_retries:
            # Feed error back
            fb_prompt = _build_error_feedback_prompt(
                problem, current_code, error, attempt + 1
            )
            raw2 = await _llm_call(
                fb_prompt, model=FALLBACK_MODEL, temperature=0.3
            )
            new_code = _extract_code(raw2, problem.name)
            if new_code and new_code != current_code:
                current_code = new_code

    # Final row: update Generation_Idx and LLM_Model cleanly
    if rows:
        rows[-1]["Generation_Idx"] = variant_idx
        rows[-1]["LLM_Model"] = model

    return rows


# ---------------------------------------------------------------------------
# Generate all N variants for one problem
# ---------------------------------------------------------------------------

async def generate_all_variants(
    problem,
    evaluator_fn,
    n_variants: int = 20,
    model: str = PRIMARY_MODEL,
    temperature: float = 0.7,
    max_concurrent: int = 5,
    progress_cb=None,
) -> list[dict]:
    """
    Generate n_variants for a problem. Returns all row dicts (seed + generations).
    The seed (gen=0) is evaluated separately and prepended.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(idx: int) -> list[dict]:
        async with sem:
            t0 = time.perf_counter()
            rows = await generate_variant_with_feedback(
                problem, idx, evaluator_fn, model=model,
                temperature=temperature + (idx % 3) * 0.05,
            )
            if progress_cb:
                elapsed = time.perf_counter() - t0
                ok = sum(1 for r in rows if r.get("Correct"))
                progress_cb(problem.name, idx, ok, len(rows), elapsed)
            return rows

    tasks = [_bounded(i) for i in range(1, n_variants + 1)]
    nested = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows: list[dict] = []
    for item in nested:
        if isinstance(item, list):
            all_rows.extend(item)
        # exceptions are silently dropped (seed is always present)

    return all_rows


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def ping_azure(model: str = PRIMARY_MODEL) -> bool:
    if not _HAS_AZURE:
        return False
    try:
        llm = AzureLLM(deployment=model)
        r = await llm.generate(system="Reply: OK", prompt="OK?",
                               temperature=0.0, max_tokens=4)
        await llm.close()
        return "OK" in r.get("content", "")
    except Exception:
        return False

# Module-level exports
GENERATION_MODELS = list(AZURE_DEPLOYMENTS.keys()) if AZURE_DEPLOYMENTS else [PRIMARY_MODEL, FALLBACK_MODEL]
check_azure_connectivity = ping_azure


async def generate_variants_for_problem(
    problem,
    n_variants: int = 20,
    model: str = PRIMARY_MODEL,
    temperature: float = 0.7,
) -> list[str]:
    """Adapter: generate n_variants kernel code strings for a problem.
    Returns list of code strings (index 0 = seed placeholder, 1..n = LLM variants).
    """
    codes = [problem.seed_pallas]  # index 0 = seed
    sem = asyncio.Semaphore(3)

    async def _one(idx: int) -> str:
        async with sem:
            prompt = _build_initial_prompt(problem, strategy_idx=idx, variant_idx=idx)
            code = await _llm_call(prompt, model=model, temperature=temperature + (idx % 3) * 0.05)
            return _extract_code(code, problem.name) if code else ""

    results = await asyncio.gather(*[_one(i) for i in range(1, n_variants + 1)])
    codes.extend(results)
    return codes
