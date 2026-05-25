#!/usr/bin/env python3
"""Run PallasBench benchmark suite.

Usage:
    python scripts/run_benchmark.py --levels 1 2 3
    python scripts/run_benchmark.py --levels 1 --category softmax
    python scripts/run_benchmark.py --levels 1 2 3 --size SMALL --interpret --correctness-only
    python scripts/run_benchmark.py --levels 1 2 3 --size LARGE --pytest-benchmark-json results/bench.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
from jax.experimental import pallas as pl

from pallasbench.benchmark import evaluate_kernel
from pallasbench.metrics import results_summary
from pallasbench.tasks import get_tasks
from pallasbench.sizes import get_size_config


def main():
    parser = argparse.ArgumentParser(description="Run PallasBench")
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[1, 2, 3],
        help="Benchmark levels to run (1, 2, 3)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by task category",
    )
    parser.add_argument(
        "--size", type=str, default="MEDIUM",
        choices=["SMALL", "MEDIUM", "LARGE"],
        help="Parametric benchmark size",
    )
    parser.add_argument(
        "--interpret", action="store_true",
        help="Run with interpret=True for CPU testing",
    )
    parser.add_argument(
        "--correctness-only", action="store_true",
        help="Only check correctness, skip timing",
    )
    parser.add_argument(
        "--fast-p", type=float, default=1.0,
        help="Speedup threshold for fast_p metric",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of timing trials per task",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--pytest-benchmark-json", type=str, default=None,
        help="Output path for benchmark-action compatible JSON",
    )
    args = parser.parse_args()

    print(f"PallasBench v0.2.0")
    print(f"Platform: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Levels: {args.levels}")
    print(f"Size: {args.size}")
    if args.interpret:
        print("Mode: interpret=True (CPU emulation)")
        if not getattr(pl.pallas_call, "_pallasbench_interpret_default", False):
            original_pallas_call = pl.pallas_call

            @wraps(original_pallas_call)
            def interpret_default(*call_args, **call_kwargs):
                call_kwargs.setdefault("interpret", True)
                return original_pallas_call(*call_args, **call_kwargs)

            interpret_default._pallasbench_interpret_default = True
            pl.pallas_call = interpret_default
    if args.correctness_only:
        print("Mode: correctness-only (no timing)")
    print()

    categories = [args.category] if args.category else None
    tasks = get_tasks(levels=args.levels, categories=categories)

    if not tasks:
        print("No tasks matched the specified filters.")
        return

    # Override input shapes with parametric sizes
    for task in tasks:
        size_shapes = get_size_config(task["name"], args.size)
        if size_shapes is not None:
            task["input_shapes"] = size_shapes

    n_trials = 1 if args.correctness_only else args.n_trials
    n_warmup = 0 if args.correctness_only else args.n_warmup

    print(f"Running {len(tasks)} tasks...\n")
    results = []
    for task in tasks:
        result = evaluate_kernel(
            pallas_fn=task["pallas_fn"],
            baseline_fn=task["baseline_fn"],
            input_shapes=task["input_shapes"],
            task_name=task.get("name", "unnamed"),
            dtype=task.get("dtype", "float32"),
            n_correctness=5,
            n_warmup=n_warmup,
            n_trials=n_trials,
            input_dtypes=task.get("input_dtypes"),
            input_ranges=task.get("input_ranges"),
        )
        results.append(result)
        status = "PASS" if result.correct else "FAIL"
        if args.correctness_only:
            print(f"  [{status}] {result.task_name}")
        else:
            print(
                f"  [{status}] {result.task_name}: "
                f"speedup={result.speedup:.2f}x "
                f"(baseline={result.baseline_time_ms:.3f}ms, "
                f"kernel={result.kernel_time_ms:.3f}ms)"
            )

    summary = results_summary(results)
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total tasks:    {summary['total_tasks']}")
    print(f"Correct:        {summary['correct']}")
    print(f"fast_0:         {summary['fast_0']:.1%}")
    print(f"fast_1:         {summary['fast_1']:.1%}")
    print(f"fast_2:         {summary['fast_2']:.1%}")
    print(f"fast_5:         {summary['fast_5']:.1%}")
    print(f"Mean speedup:   {summary['mean_speedup']:.2f}x")
    print(f"Median speedup: {summary['median_speedup']:.2f}x")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = jax.default_backend()
    output_path = os.path.join(
        args.output_dir, f"pallasbench_{backend}_{timestamp}.json"
    )

    output_data = {
        "meta": {
            "timestamp": timestamp,
            "backend": backend,
            "devices": [str(d) for d in jax.devices()],
            "levels": args.levels,
            "size": args.size,
            "interpret": args.interpret,
            "n_trials": n_trials,
        },
        "summary": summary,
        "tasks": [
            {
                "name": r.task_name,
                "correct": r.correct,
                "baseline_ms": r.baseline_time_ms,
                "kernel_ms": r.kernel_time_ms,
                "speedup": r.speedup,
                "errors": r.errors,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate benchmark-action compatible JSON if requested
    if args.pytest_benchmark_json:
        bench_entries = []
        for r in results:
            bench_entries.append({
                "name": r.task_name,
                "unit": "ms",
                "value": r.kernel_time_ms,
                "extra": (
                    f"baseline={r.baseline_time_ms:.3f}ms, "
                    f"speedup={r.speedup:.2f}x, "
                    f"correct={r.correct}"
                ),
            })
        with open(args.pytest_benchmark_json, "w") as f:
            json.dump(bench_entries, f, indent=2)
        print(f"Benchmark JSON saved to: {args.pytest_benchmark_json}")


if __name__ == "__main__":
    main()
