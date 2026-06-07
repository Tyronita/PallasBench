#!/usr/bin/env python3
"""
Production archive generation runner.

Usage:
  python scripts/run_archive.py [--n 20] [--model DeepSeek-V3-2] [--levels 1 2 3]

Generates LLM variants for all 45 PallasBench problems, evaluates with
feedback loop, checkpoints to JSONL, and writes final parquet splits.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/OLeary/Evolve")

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from pallas_bench.evaluator import evaluate_problem_seed, evaluate_kernel
from pallas_bench.llm_generator import (
    generate_all_variants, ping_azure, PRIMARY_MODEL
)
from pallas_bench.problems import ALL_PROBLEMS, LEVEL_PROBLEMS
from pallas_bench.schema import SPLITS, COLUMN_NAMES
from pallas_bench.archive_builder import _rows_to_dataframe, _write_summary

console = Console()

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt(out_dir: Path, name: str) -> Path:
    p = out_dir / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.jsonl"

def _load_ckpt(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def _save_row(path: Path, row: dict):
    safe = {k: (None if isinstance(v, float) and (
        v != v or abs(v) == float("inf")) else v)
        for k, v in row.items()}
    with open(path, "a") as f:
        f.write(json.dumps(safe) + "\n")

# ---------------------------------------------------------------------------
# Evaluator wrapper (sync, called from async generator)
# ---------------------------------------------------------------------------

def _eval_fn(problem, kernel_source, kernel_fn_name,
             generation_idx, llm_model, capture_ir=False):
    return evaluate_kernel(
        problem,
        kernel_source=kernel_source,
        kernel_fn_name=kernel_fn_name,
        generation_idx=generation_idx,
        llm_model=llm_model,
        capture_ir=capture_ir,
        capture_ptx=False,
    )

# ---------------------------------------------------------------------------
# Per-problem processor
# ---------------------------------------------------------------------------

async def process_problem(
    p,
    out_dir: Path,
    n_variants: int,
    model: str,
    resume: bool,
    progress: Progress,
    task_id,
) -> list[dict]:
    ckpt = _ckpt(out_dir, p.name)
    existing = _load_ckpt(ckpt)
    done_gens = {r.get("Generation_Idx", -1) for r in existing}
    rows = list(existing)

    # Always evaluate seed if not done
    if 0 not in done_gens:
        seed_row = evaluate_problem_seed(p, capture_ir=True)
        rows.append(seed_row)
        _save_row(ckpt, seed_row)

    needed = [i for i in range(1, n_variants + 1) if i not in done_gens]
    if not needed:
        progress.advance(task_id, n_variants)
        return rows

    def _prog_cb(name, idx, ok, total, elapsed):
        progress.advance(task_id, 1)
        status = "✓" if ok else "✗"
        progress.update(task_id, description=f"[cyan]{name}[/] gen {idx} {status}")

    gen_rows = await generate_all_variants(
        p,
        evaluator_fn=_eval_fn,
        n_variants=len(needed),
        model=model,
        max_concurrent=4,
        progress_cb=_prog_cb,
    )

    for row in gen_rows:
        rows.append(row)
        _save_row(ckpt, row)

    progress.advance(task_id, max(0, len(needed) - len(gen_rows)))
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--n",      "-n", default=20,   show_default=True, help="LLM variants per problem")
@click.option("--model",  "-m", default=PRIMARY_MODEL, show_default=True)
@click.option("--levels", "-l", multiple=True, type=int, default=[1, 2, 3])
@click.option("--output", "-o", default="results/archive_llm", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--push-hf", default=None, help="HF dataset repo id to push to after generation")
@click.option("--max-concurrent-problems", default=2, show_default=True)
def main(n, model, levels, output, resume, push_hf, max_concurrent_problems):
    """Generate PallasBench LLM archive with feedback loop."""

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]PallasBench LLM Archive Generator")

    # Verify Azure
    console.print(f"  Checking {model}...", end="")
    ok = asyncio.run(ping_azure(model))
    if ok:
        console.print(f" [green]✓[/]")
    else:
        console.print(f" [red]✗ Azure unreachable — check AZURE_OPENAI_API_KEY[/]")
        sys.exit(1)

    # Set env for GPU
    _setup_gpu_env()

    problems = [p for p in ALL_PROBLEMS if p.level in levels]
    total_gens = len(problems) * n

    console.print(f"  Problems : {len(problems)}  ×  {n} variants  =  {total_gens} generations")
    console.print(f"  Model    : {model}")
    console.print(f"  Output   : {out_dir.resolve()}")
    console.print(f"  Resume   : {resume}\n")

    # Run
    all_rows: list[dict] = []
    sem = asyncio.Semaphore(max_concurrent_problems)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[bold]Total", total=total_gens)

        async def run_all():
            tasks = []
            for p in problems:
                tid = progress.add_task(f"[dim]{p.name}", total=n)
                tasks.append((p, tid))

            async def bounded(p, tid):
                async with sem:
                    rows = await process_problem(
                        p, out_dir, n, model, resume, progress, tid
                    )
                    progress.advance(overall, n)
                    return rows

            results = await asyncio.gather(*[bounded(p, t) for p, t in tasks],
                                           return_exceptions=True)
            return results

        nested = asyncio.run(run_all())
        for item in nested:
            if isinstance(item, list):
                all_rows.extend(item)

    # Write parquet splits
    console.print(f"\n[bold]Writing {len(all_rows)} rows...[/]")
    for split, level_ids in SPLITS.items():
        rows = [r for r in all_rows if r.get("Level_ID") in level_ids]
        if not rows:
            continue
        df = _rows_to_dataframe(rows)
        out_file = out_dir / f"{split}.parquet"
        df.to_parquet(out_file, index=False)
        correct = int(df["Correct"].sum()) if "Correct" in df.columns else 0
        console.print(f"  [green]✓[/] {split}.parquet: {len(df)} rows  ({correct} correct)")

    _write_summary(all_rows, out_dir)
    console.print(f"\n[bold green]Archive complete → {out_dir.resolve()}[/]")

    # HF push
    if push_hf:
        _push_hf(out_dir, push_hf)

    # Print fast_p summary
    _print_fast_p(all_rows)


def _print_fast_p(rows):
    if not rows:
        return
    total = len(rows)
    f0 = sum(1 for r in rows if r.get("fast_0"))
    f1 = sum(1 for r in rows if r.get("fast_1"))
    f2 = sum(1 for r in rows if r.get("fast_2"))
    f5 = sum(1 for r in rows if r.get("fast_5"))
    console.print(f"\n[bold]fast_p summary[/] ({total} total rows)")
    console.print(f"  fast_0 (correct)     : {f0}/{total} = {100*f0//total}%")
    console.print(f"  fast_1 (>1x speedup) : {f1}/{total} = {100*f1//total}%")
    console.print(f"  fast_2 (>2x speedup) : {f2}/{total} = {100*f2//total}%")
    console.print(f"  fast_5 (>5x speedup) : {f5}/{total} = {100*f5//total}%")


def _push_hf(out_dir: Path, repo_id: str):
    try:
        from datasets import Dataset, DatasetDict
        console.print(f"\n[bold]Pushing to HuggingFace: {repo_id}[/]")
        splits_data = {}
        for split in ["level_1", "level_2", "level_3"]:
            pf = out_dir / f"{split}.parquet"
            if pf.exists():
                df = pd.read_parquet(pf)
                splits_data[split] = Dataset.from_pandas(df, preserve_index=False)
                console.print(f"  [cyan]{split}[/]: {len(df)} rows")
        if splits_data:
            DatasetDict(splits_data).push_to_hub(repo_id, private=False)
            console.print(f"  [green]✓ https://huggingface.co/datasets/{repo_id}[/]")
    except Exception as e:
        console.print(f"  [red]HF push failed: {e}[/]")


def _setup_gpu_env():
    nvidia = "/home/OLeary/.local/lib/python3.12/site-packages/nvidia"
    libs = ":".join(str(p) for p in Path(nvidia).glob("*/lib") if p.is_dir())
    if libs:
        os.environ["LD_LIBRARY_PATH"] = libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ptxas = Path(nvidia) / "cuda_nvcc/bin"
    if ptxas.exists():
        os.environ["PATH"] = str(ptxas) + ":" + os.environ.get("PATH", "")
    libdev = Path(nvidia) / "cuda_nvcc/nvvm/libdevice"
    if libdev.exists():
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={libdev}"
    os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU", "1")


if __name__ == "__main__":
    main()
