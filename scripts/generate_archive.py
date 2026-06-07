#!/usr/bin/env python3
"""
Main CLI — generate the PallasBench Archive dataset.

Usage examples:

  # Evaluate seed kernels only (no LLM, no API key needed)
  python scripts/generate_archive.py --n-variants 0 --output results/seeds_only

  # Full run: 20 variants per problem using DeepSeek-V3-2
  python scripts/generate_archive.py --n-variants 20 --model DeepSeek-V3-2

  # Only Level 1 problems, 10 variants, resume from checkpoint
  python scripts/generate_archive.py --levels 1 --n-variants 10 --resume

  # Push to HuggingFace after generation
  python scripts/generate_archive.py --n-variants 20 --push-hf EvanOLeary/pallasbench-t4-archive
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make pallas_bench importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

from pallas_bench.archive_builder import build_archive
from pallas_bench.archive_builder import GENERATION_MODELS
from pallas_bench.llm_generator import check_azure_connectivity
from pallas_bench.problems import ALL_PROBLEMS, LEVEL_PROBLEMS
from pallas_bench.schema import COLUMN_NAMES

console = Console()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """PallasBench Archive — JAX/Pallas T4 kernel dataset generator."""
    pass


@cli.command()
@click.option("--levels", "-l", multiple=True, type=int, default=[1, 2, 3],
              show_default=True, help="Levels to include (1, 2, 3)")
@click.option("--n-variants", "-n", default=20, show_default=True,
              help="LLM-generated variants per problem (0 = seeds only)")
@click.option("--model", "-m", default="DeepSeek-V3-2", show_default=True,
              type=click.Choice(GENERATION_MODELS + ["DeepSeek-V3-2", "Kimi-K2-6"]),
              help="Azure LLM deployment")
@click.option("--temperature", "-t", default=0.7, show_default=True,
              help="Sampling temperature")
@click.option("--output", "-o", default="results/archive", show_default=True,
              help="Output directory for parquet splits")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Resume from checkpoint files")
@click.option("--capture-ir/--no-ir", default=True, show_default=True,
              help="Capture Jaxpr + StableHLO IR")
@click.option("--capture-ptx/--no-ptx", default=False, show_default=True,
              help="Capture PTX/SASS (requires JAX-CUDA)")
@click.option("--push-hf", default=None,
              help="HuggingFace dataset repo ID to push to (e.g. EvanOLeary/pallasbench-t4-archive)")
@click.option("--max-concurrent", default=2, show_default=True,
              help="Max concurrent problem evaluations")
def run(levels, n_variants, model, temperature, output, resume,
        capture_ir, capture_ptx, push_hf, max_concurrent):
    """Generate the full PallasBench Archive on T4."""

    _print_env_check()

    out_path = asyncio.run(build_archive(
        output_dir=output,
        levels=list(levels),
        n_variants=n_variants,
        model=model,
        temperature=temperature,
        capture_ir=capture_ir,
        capture_ptx=capture_ptx,
        resume=resume,
        max_concurrent=max_concurrent,
    ))

    if push_hf:
        _push_to_hf(out_path, push_hf)


@cli.command()
def check():
    """Check environment: JAX version, GPU, Azure connectivity."""
    _print_env_check(verbose=True)
    console.print()
    console.print("[bold]Azure LLM connectivity:[/bold]")
    for model in GENERATION_MODELS:
        ok = asyncio.run(check_azure_connectivity(model))
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {status} {model}")


@cli.command()
def list_problems():
    """List all 45 PallasBench problems."""
    t = Table(title="PallasBench Problems (45 total)")
    t.add_column("ID",  style="cyan",  no_wrap=True)
    t.add_column("Name",             style="bold")
    t.add_column("Level", justify="center")
    t.add_column("Category")
    for p in ALL_PROBLEMS:
        t.add_row(str(p.task_id), p.name, f"L{p.level}", p.category)
    console.print(t)


@cli.command()
@click.argument("output_dir")
@click.argument("repo_id")
def push(output_dir, repo_id):
    """Push existing archive parquet files to HuggingFace."""
    _push_to_hf(Path(output_dir), repo_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_env_check(verbose: bool = False):
    console.rule("[bold blue]Environment Check")
    try:
        import jax
        devices = jax.devices()
        console.print(f"  [green]✓ JAX {jax.__version__}[/green]  devices={devices}")
    except Exception as e:
        console.print(f"  [red]✗ JAX unavailable: {e}[/red]")

    import subprocess
    try:
        ncu = subprocess.run(["ncu", "--version"], capture_output=True, text=True, timeout=5)
        if ncu.returncode == 0:
            ver = ncu.stdout.strip().split("\n")[0]
            console.print(f"  [green]✓ ncu: {ver}[/green]")
        else:
            console.print("  [yellow]⚠ ncu not found — NCU_Profile will be empty[/yellow]")
    except Exception:
        console.print("  [yellow]⚠ ncu not found — NCU_Profile will be empty[/yellow]")

    key_set = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    if key_set:
        console.print("  [green]✓ AZURE_OPENAI_API_KEY is set[/green]")
    else:
        console.print(
            "  [yellow]⚠ AZURE_OPENAI_API_KEY not set — seeds only mode[/yellow]\n"
            "    Run:  export AZURE_OPENAI_API_KEY=<your-key>"
        )

    if verbose:
        endpoint = os.environ.get("AZURE_API_ENDPOINT", "https://openai-shinka.cognitiveservices.azure.com/")
        console.print(f"  Azure endpoint: {endpoint}")
        console.print(f"  Problems: {len(ALL_PROBLEMS)} total")


def _push_to_hf(output_dir: Path, repo_id: str):
    """Push parquet splits to HuggingFace Hub as a dataset."""
    try:
        from datasets import load_dataset, DatasetDict, Dataset
        import pandas as pd

        console.print(f"\n[bold]Pushing to HuggingFace: {repo_id}[/bold]")
        splits_data = {}
        for split in ["level_1", "level_2", "level_3"]:
            pq_file = output_dir / f"{split}.parquet"
            if pq_file.exists():
                df = pd.read_parquet(pq_file)
                splits_data[split] = Dataset.from_pandas(df)
                console.print(f"  [cyan]{split}[/cyan]: {len(df)} rows")

        if not splits_data:
            console.print("  [red]No parquet files found.[/red]")
            return

        dd = DatasetDict(splits_data)
        dd.push_to_hub(repo_id, private=False)
        console.print(f"  [green]✓ Pushed → https://huggingface.co/datasets/{repo_id}[/green]")
    except ImportError:
        console.print("  [red]datasets library not installed. Run: pip install datasets[/red]")
    except Exception as e:
        console.print(f"  [red]Push failed: {e}[/red]")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    cli(standalone_mode=True, default_map={"run": {}})
