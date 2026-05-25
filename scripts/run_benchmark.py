#!/usr/bin/env python3
"""Run PallasBench benchmark suite.

Usage:
    python scripts/run_benchmark.py --levels 1 2 3
    python scripts/run_benchmark.py --levels 1 --category softmax
    python scripts/run_benchmark.py --levels 1 --fast-p 2.0
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax

from pallasbench.benchmark import evaluate_suite
from pallasbench.metrics import results_summary
from pallasbench.tasks import get_tasks


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
    args = parser.parse_args()

    print(f"PallasBench v0.1.0")
    print(f"Platform: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Levels: {args.levels}")
    print()

    categories = [args.category] if args.category else None
    tasks = get_tasks(levels=args.levels, categories=categories)

    if not tasks:
        print("No tasks matched the specified filters.")
        return

    print(f"Running {len(tasks)} tasks...\n")
    results = evaluate_suite(
        tasks, n_trials=args.n_trials, n_warmup=args.n_warmup
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
            "n_trials": args.n_trials,
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


if __name__ == "__main__":
    main()
