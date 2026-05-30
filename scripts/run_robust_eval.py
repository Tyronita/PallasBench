#!/usr/bin/env python3
"""Run the robust evaluation suite for Pallas kernels.

Usage:
    python scripts/run_robust_eval.py --kernels all
    python scripts/run_robust_eval.py --kernels L1/relu L1/matmul
    python scripts/run_robust_eval.py --kernels all --output-dir results/robust
    python scripts/run_robust_eval.py --kernels all --device gpu:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

import jax

from pallasbench.robust_eval.evaluator import evaluate_kernel
from pallasbench.tasks import TASK_REGISTRY


def _get_kernel_module(task: dict):
    fn = task["pallas_fn"]
    mod = getattr(fn, "__module__", None)
    if mod:
        import importlib
        try:
            return importlib.import_module(mod)
        except Exception:
            pass
    return fn


def _resolve_tasks(kernel_names: list[str]) -> list[dict]:
    if "all" in kernel_names:
        return TASK_REGISTRY
    matched = []
    for name in kernel_names:
        found = False
        for task in TASK_REGISTRY:
            if task["name"] == name:
                matched.append(task)
                found = True
                break
        if not found:
            print(f"Warning: kernel '{name}' not found in registry", file=sys.stderr)
    return matched


def serialize_result(result) -> dict:
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Robust evaluation for Pallas GPU kernels"
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["all"],
        help="Kernel names to evaluate (or 'all' for full suite)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/robust_eval",
        help="Directory for output JSONL results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (e.g. 'gpu:0')",
    )
    args = parser.parse_args()

    if args.device:
        try:
            jax.config.update("jax_default_device", jax.devices(args.device)[0])
        except Exception as e:
            print(f"Warning: could not set device '{args.device}': {e}", file=sys.stderr)

    print(f"PallasBench Robust Evaluation")
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    tasks = _resolve_tasks(args.kernels)
    if not tasks:
        print("No tasks matched. Available kernels:")
        for t in TASK_REGISTRY:
            print(f"  {t['name']}")
        sys.exit(1)

    print(f"Evaluating {len(tasks)} kernel(s)...")
    print()

    results = []
    for i, task in enumerate(tasks):
        name = task.get("name", "unnamed")
        print(f"[{i + 1}/{len(tasks)}] {name} ... ", end="", flush=True)
        try:
            result = evaluate_kernel(
                kernel_module=_get_kernel_module(task),
                kernel_fn=task["pallas_fn"],
                baseline_fn=task["baseline_fn"],
                input_shapes=task["input_shapes"],
                task_name=name,
                level=task.get("level", 1),
                category=task.get("category", "unknown"),
                dtype=task.get("dtype", "float32"),
                input_dtypes=task.get("input_dtypes"),
                input_ranges=task.get("input_ranges"),
            )
            results.append(result)
            ck = "PASS" if result.correctness_passed else "FAIL"
            rb = "PASS" if result.robustness_passed else "FAIL"
            print(f"correctness={ck} robustness={rb} "
                  f"wall={result.wall_time_seconds:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"robust_eval_{timestamp}.jsonl"
    )

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(serialize_result(r)) + "\n")

    total = len(results)
    correct = sum(1 for r in results if r.correctness_passed)
    robust = sum(1 for r in results if r.robustness_passed)
    print()
    print(f"{'=' * 60}")
    print(f"Robust Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Total kernels:     {total}")
    print(f"Correctness pass:  {correct}/{total}")
    print(f"Robustness pass:   {robust}/{total}")
    print(f"Results saved to:  {output_path}")


if __name__ == "__main__":
    main()
