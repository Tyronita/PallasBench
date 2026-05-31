"""
Archive builder — orchestrates LLM generation + evaluation and assembles
the final SakanaAI-format parquet splits (level_1, level_2, level_3).

Supports resume: completed rows are checkpointed to JSONL so a crash
doesn't lose work.  Final output is a HuggingFace-compatible dataset.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from tqdm import tqdm

from .evaluator import evaluate_kernel, evaluate_problem_seed, robustness_filters
from .llm_generator import (
    generate_all_variants,
    ping_azure as check_azure_connectivity,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
)
GENERATION_MODELS = [PRIMARY_MODEL, FALLBACK_MODEL]
from .problems import ALL_PROBLEMS, LEVEL_PROBLEMS, Problem
from .schema import PA_SCHEMA, SPLITS, empty_row

console = Console()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(output_dir: Path, problem_name: str) -> Path:
    return output_dir / "checkpoints" / f"{problem_name}.jsonl"


def _load_checkpoint(ckpt_file: Path) -> list[dict]:
    if not ckpt_file.exists():
        return []
    rows = []
    with open(ckpt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _append_checkpoint(ckpt_file: Path, row: dict):
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    # Sanitize NaN/Inf for JSON
    safe = {}
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            safe[k] = None
        else:
            safe[k] = v
    with open(ckpt_file, "a") as f:
        f.write(json.dumps(safe) + "\n")


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------

async def process_problem(
    problem: Problem,
    output_dir: Path,
    n_variants: int = 20,
    model: str = "DeepSeek-V3-2",
    temperature: float = 0.7,
    capture_ir: bool = True,
    capture_ptx: bool = False,
    resume: bool = True,
) -> list[dict]:
    """
    Generate n_variants LLM kernels for a problem, evaluate each, and
    return the list of evaluation row dicts.  Uses checkpointing.
    """
    ckpt = _ckpt_path(output_dir, problem.name)

    existing_rows = []
    existing_gen_indices = set()
    if resume:
        existing_rows = _load_checkpoint(ckpt)
        existing_gen_indices = {r.get("Generation_Idx", -1) for r in existing_rows}

    rows: list[dict] = list(existing_rows)

    # ── Seed kernel (generation_idx=0) ──────────────────────────────────────
    if 0 not in existing_gen_indices:
        seed_row = evaluate_problem_seed(problem, capture_ir=capture_ir, capture_ptx=capture_ptx)
        seed_row["Robustness"] = json.dumps(robustness_filters(seed_row))
        rows.append(seed_row)
        _append_checkpoint(ckpt, seed_row)

    # ── LLM-generated variants ───────────────────────────────────────────────
    needed_variants = [
        i for i in range(1, n_variants + 1)
        if i not in existing_gen_indices
    ]

    if needed_variants:
        variants = await generate_variants_for_problem(
            problem,
            n_variants=len(needed_variants),
            model=model,
            temperature=temperature,
        )
        # variants[0] is seed from generator, skip it; rest are 1..n
        generated_code = variants[1:] if len(variants) > 1 else []

        for i, code in zip(needed_variants, generated_code):
            row = evaluate_kernel(
                problem,
                kernel_source=code,
                kernel_fn_name=problem.name,
                generation_idx=i,
                llm_model=model,
                capture_ir=capture_ir,
                capture_ptx=capture_ptx,
            )
            row["Robustness"] = json.dumps(robustness_filters(row))
            rows.append(row)
            _append_checkpoint(ckpt, row)

    return rows


# ---------------------------------------------------------------------------
# Full archive build
# ---------------------------------------------------------------------------

async def build_archive(
    output_dir: str | Path = "results/archive",
    levels: list[int] | None = None,
    n_variants: int = 20,
    model: str = "DeepSeek-V3-2",
    temperature: float = 0.7,
    capture_ir: bool = True,
    capture_ptx: bool = False,
    resume: bool = True,
    max_concurrent: int = 2,
) -> Path:
    """
    Build the full archive parquet dataset.

    Args:
        output_dir:      Where to write parquet splits + checkpoints.
        levels:          Which levels to run (default: all).
        n_variants:      Number of LLM variants per problem (excl. seed).
        model:           Azure LLM deployment name.
        temperature:     Sampling temperature.
        capture_ir:      Whether to capture Jaxpr + StableHLO.
        capture_ptx:     Whether to capture PTX/SASS (requires JAX-CUDA).
        resume:          Resume from checkpoint.
        max_concurrent:  Max concurrent problem evaluations.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_levels = levels or [1, 2, 3]
    problems = [p for p in ALL_PROBLEMS if p.level in active_levels]

    console.rule("[bold blue]PallasBench Archive Builder")
    console.print(f"  Problems : {len(problems)} across levels {active_levels}")
    console.print(f"  Variants : {n_variants} per problem ({n_variants*len(problems)+len(problems)} total rows)")
    console.print(f"  Model    : {model}")
    console.print(f"  Output   : {output_dir.resolve()}")

    # Check Azure connectivity
    console.print("\n[dim]Checking Azure LLM connectivity...[/dim]")
    azure_ok = await check_azure_connectivity(model)
    if azure_ok:
        console.print(f"  [green]✓ Azure {model} reachable[/green]")
    else:
        console.print(
            f"  [yellow]⚠ Azure unreachable — will use seed kernels only[/yellow]\n"
            f"    Set AZURE_OPENAI_API_KEY to enable LLM generation."
        )
        n_variants = 0  # seeds only

    # ── Run all problems ─────────────────────────────────────────────────────
    all_rows: list[dict] = []
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded_process(p: Problem):
        async with sem:
            console.print(f"  [cyan]→ {p.name}[/cyan] (L{p.level}, {p.category})")
            t0 = time.perf_counter()
            rows = await process_problem(
                p, output_dir, n_variants=n_variants, model=model,
                temperature=temperature, capture_ir=capture_ir,
                capture_ptx=capture_ptx, resume=resume,
            )
            elapsed = time.perf_counter() - t0
            passed  = sum(1 for r in rows if r.get("Correct"))
            console.print(
                f"  [green]✓ {p.name}[/green]: "
                f"{passed}/{len(rows)} pass, {elapsed:.1f}s"
            )
            return rows

    tasks = [bounded_process(p) for p in problems]
    for coro in asyncio.as_completed(tasks):
        rows = await coro
        all_rows.extend(rows)

    # ── Assemble DataFrame + write parquet splits ────────────────────────────
    console.print(f"\n[bold]Writing {len(all_rows)} rows to parquet...[/bold]")

    for split_name, level_ids in SPLITS.items():
        split_rows = [r for r in all_rows if r.get("Level_ID") in level_ids]
        if not split_rows:
            continue

        df = _rows_to_dataframe(split_rows)
        out_file = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_file, index=False)
        console.print(f"  [green]✓ {split_name}.parquet[/green]: {len(df)} rows")

    # ── Write summary ────────────────────────────────────────────────────────
    _write_summary(all_rows, output_dir)
    console.print(f"\n[bold green]Archive complete → {output_dir.resolve()}[/bold green]")
    return output_dir


def _rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert row dicts to a cleaned DataFrame with correct column order."""
    from .schema import COLUMN_NAMES
    df = pd.DataFrame(rows)
    # Add any missing columns
    for col in COLUMN_NAMES:
        if col not in df.columns:
            df[col] = None
    # Reorder to match schema, drop extra cols
    extra = [c for c in df.columns if c not in COLUMN_NAMES]
    df = df[COLUMN_NAMES + extra]
    # Coerce types
    for col in ["Correct", "fast_0", "fast_1", "fast_2", "fast_5"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    for col in ["Level_ID", "Task_ID", "Generation_Idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["Pallas_Runtime", "JAX_Native_Runtime", "JAX_XLA_Compiled_Runtime",
                "Pallas_Speedup_Native", "Pallas_Speedup_Compiled", "Max_Diff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _write_summary(rows: list[dict], output_dir: Path):
    """Write a markdown + JSON summary of archive results."""
    total    = len(rows)
    correct  = sum(1 for r in rows if r.get("Correct"))
    fast_1   = sum(1 for r in rows if r.get("fast_1"))
    fast_2   = sum(1 for r in rows if r.get("fast_2"))
    fast_5   = sum(1 for r in rows if r.get("fast_5"))

    by_level: dict[int, dict] = {}
    for lvl in [1, 2, 3]:
        lvl_rows = [r for r in rows if r.get("Level_ID") == lvl]
        by_level[lvl] = {
            "total":   len(lvl_rows),
            "correct": sum(1 for r in lvl_rows if r.get("Correct")),
            "fast_1":  sum(1 for r in lvl_rows if r.get("fast_1")),
            "fast_2":  sum(1 for r in lvl_rows if r.get("fast_2")),
        }

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "target_hardware": "NVIDIA Tesla T4 (sm_75)",
        "total_rows": total,
        "fast_0": correct,
        "fast_1": fast_1,
        "fast_2": fast_2,
        "fast_5": fast_5,
        "fast_0_pct": round(100 * correct / total, 1) if total else 0,
        "by_level": by_level,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = f"""# PallasBench Archive — T4 Results

Generated: {summary['generated_at']}
Hardware: {summary['target_hardware']}

## Aggregate

| Metric | Count | % |
|--------|-------|---|
| Total rows | {total} | 100% |
| fast_0 (correct) | {correct} | {summary['fast_0_pct']}% |
| fast_1 (>1x) | {fast_1} | {round(100*fast_1/total,1) if total else 0}% |
| fast_2 (>2x) | {fast_2} | {round(100*fast_2/total,1) if total else 0}% |
| fast_5 (>5x) | {fast_5} | {round(100*fast_5/total,1) if total else 0}% |

## By Level

| Level | Total | Correct | fast_1 | fast_2 |
|-------|-------|---------|--------|--------|
{''.join(f"| L{l} | {d['total']} | {d['correct']} | {d['fast_1']} | {d['fast_2']} |" + chr(10) for l, d in by_level.items())}
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(md)
