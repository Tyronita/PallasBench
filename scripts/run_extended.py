#!/usr/bin/env python3
"""Run archive generation for extended problems (100 new, L1-L4)."""
from __future__ import annotations
import asyncio, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/OLeary/Evolve")

import click
import pandas as pd
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

from pallas_bench.evaluator import evaluate_problem_seed, evaluate_kernel
from pallas_bench.llm_generator import generate_all_variants, ping_azure, PRIMARY_MODEL
from pallas_bench.extended_problems import EXT, EXT_BY_LEVEL
from pallas_bench.archive_builder import _rows_to_dataframe, _write_summary

console = Console()

def _setup():
    nv = "/home/OLeary/.local/lib/python3.12/site-packages/nvidia"
    libs = ":".join(str(p) for p in Path(nv).glob("*/lib") if p.is_dir())
    if libs: os.environ["LD_LIBRARY_PATH"] = libs + ":" + os.environ.get("LD_LIBRARY_PATH","")
    ptxas = Path(nv)/"cuda_nvcc/bin"
    if ptxas.exists(): os.environ["PATH"] = str(ptxas)+":"+os.environ.get("PATH","")
    ld = Path(nv)/"cuda_nvcc/nvvm/libdevice"
    if ld.exists(): os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={ld}"
    os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU","1")

def _ckpt(out: Path, name: str) -> Path:
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)
    return out/"checkpoints"/f"{name}.jsonl"

def _load(p: Path) -> list[dict]:
    if not p.exists(): return []
    rows=[]
    for l in p.read_text().splitlines():
        try: rows.append(json.loads(l))
        except: pass
    return rows

def _save(p: Path, row: dict):
    import math
    safe={k:(None if isinstance(v,float) and (math.isnan(v) or math.isinf(v)) else v) for k,v in row.items()}
    with open(p,"a") as f: f.write(json.dumps(safe)+"\n")

def _eval_fn(problem, kernel_source, kernel_fn_name, generation_idx, llm_model, capture_ir=False):
    return evaluate_kernel(problem, kernel_source=kernel_source, kernel_fn_name=kernel_fn_name,
                           generation_idx=generation_idx, llm_model=llm_model, capture_ir=capture_ir)

async def process_one(p, out: Path, n: int, model: str, progress, tid):
    ck = _ckpt(out, p.name)
    existing = _load(ck)
    done = {r.get("Generation_Idx",0) for r in existing}
    rows = list(existing)

    if 0 not in done:
        seed = evaluate_problem_seed(p, capture_ir=True)
        seed["Target_Hardware"] = "NVIDIA Tesla T4 (sm_75, 16GB GDDR6)"
        seed["Compute_Capability"] = "7.5"
        seed["Pallas_Backend"] = "interpret_mode"
        rows.append(seed); _save(ck, seed)

    needed = [i for i in range(1, n+1) if i not in done]
    if not needed:
        progress.advance(tid, n); return rows

    def cb(name, idx, ok, total, elapsed):
        progress.advance(tid, 1)

    gen = await generate_all_variants(p, evaluator_fn=_eval_fn, n_variants=len(needed),
                                      model=model, max_concurrent=4, progress_cb=cb)
    for row in gen:
        row["Target_Hardware"] = "NVIDIA Tesla T4 (sm_75, 16GB GDDR6)"
        row["Compute_Capability"] = "7.5"
        row["Pallas_Backend"] = "interpret_mode"
        rows.append(row); _save(ck, row)

    progress.advance(tid, max(0, len(needed)-len(gen)))
    return rows

@click.command()
@click.option("--levels","-l", multiple=True, type=int, default=[1,2,3,4])
@click.option("--n","-n", default=20)
@click.option("--model","-m", default=PRIMARY_MODEL)
@click.option("--output","-o", default="results/extended_archive")
@click.option("--push-hf", default=None)
def main(levels, n, model, output, push_hf):
    _setup()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Extended Archive Generator")
    ok = asyncio.run(ping_azure(model))
    if not ok: console.print("[red]Azure unreachable[/]"); return

    problems = [p for p in EXT if p.level in levels]
    console.print(f"  {len(problems)} problems × {n} variants | model={model} | levels={list(levels)}")

    sem = asyncio.Semaphore(2)
    all_rows = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
        overall = prog.add_task("[bold]Total", total=len(problems)*n)

        async def run():
            tasks = []
            for p in problems:
                tid = prog.add_task(f"[dim]{p.name}", total=n)
                tasks.append((p, tid))

            async def bounded(p, tid):
                async with sem:
                    r = await process_one(p, out, n, model, prog, tid)
                    prog.advance(overall, n)
                    return r

            results = await asyncio.gather(*[bounded(p,t) for p,t in tasks], return_exceptions=True)
            return results

        nested = asyncio.run(run())
        for item in nested:
            if isinstance(item, list): all_rows.extend(item)

    # Write parquet by level
    console.print(f"\n[bold]Writing {len(all_rows)} rows[/]")
    splits = {}
    for lvl in [1,2,3,4]:
        sr = [r for r in all_rows if r.get("Level_ID")==lvl]
        if not sr: continue
        df = _rows_to_dataframe(sr)
        fname = f"level_{lvl}.parquet"
        df.to_parquet(out/fname, index=False)
        splits[f"level_{lvl}"] = Dataset.from_pandas(df, preserve_index=False)
        correct = int(df["Correct"].sum())
        console.print(f"  [green]✓[/] {fname}: {len(df)} rows | {correct} correct")

    _write_summary(all_rows, out)

    if push_hf and splits:
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("Set HF_TOKEN to push to HuggingFace")
        DatasetDict(splits).push_to_hub(push_hf, token=hf_token,
            commit_message=f"Extended: {len(problems)} problems L{min(levels)}-L{max(levels)} × {n} variants")
        console.print(f"[green]Pushed → https://huggingface.co/datasets/{push_hf}[/]")

    # fast_p summary
    f0=sum(1 for r in all_rows if r.get("fast_0"))
    f1=sum(1 for r in all_rows if r.get("fast_1"))
    console.print(f"\nfast_0={f0}/{len(all_rows)} fast_1={f1}/{len(all_rows)}")

if __name__ == "__main__": main()
