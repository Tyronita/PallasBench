"""Core benchmark harness for PallasBench."""

from __future__ import annotations

from typing import Callable, Sequence

from pallasbench.utils import (
    BenchmarkResult,
    check_correctness,
    generate_inputs,
    time_fn,
)


def evaluate_kernel(
    pallas_fn: Callable,
    baseline_fn: Callable,
    input_shapes: Sequence[tuple[int, ...]],
    task_name: str = "unnamed",
    dtype: str = "float32",
    n_correctness: int = 5,
    n_warmup: int = 10,
    n_trials: int = 100,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    input_dtypes: Sequence[str] | None = None,
    input_ranges: Sequence[tuple[float, float] | None] | None = None,
) -> BenchmarkResult:
    correct, errors = check_correctness(
        pallas_fn=pallas_fn,
        baseline_fn=baseline_fn,
        input_shapes=input_shapes,
        dtype=dtype,
        n_checks=n_correctness,
        atol=atol,
        rtol=rtol,
        input_dtypes=input_dtypes,
        input_ranges=input_ranges,
    )

    inputs = generate_inputs(
        input_shapes,
        dtype=dtype,
        seed=42,
        dtypes=input_dtypes,
        ranges=input_ranges,
    )
    baseline_time = time_fn(baseline_fn, inputs, n_warmup=n_warmup, n_trials=n_trials)
    kernel_time = time_fn(pallas_fn, inputs, n_warmup=n_warmup, n_trials=n_trials)

    speedup = baseline_time / kernel_time if kernel_time > 0 else 0.0

    return BenchmarkResult(
        task_name=task_name,
        correct=correct,
        baseline_time_ms=baseline_time,
        kernel_time_ms=kernel_time,
        speedup=speedup,
        errors=errors,
    )


def evaluate_suite(
    tasks: list[dict],
    n_correctness: int = 5,
    n_warmup: int = 10,
    n_trials: int = 100,
) -> list[BenchmarkResult]:
    results = []
    for task in tasks:
        result = evaluate_kernel(
            pallas_fn=task["pallas_fn"],
            baseline_fn=task["baseline_fn"],
            input_shapes=task["input_shapes"],
            task_name=task.get("name", "unnamed"),
            dtype=task.get("dtype", "float32"),
            n_correctness=n_correctness,
            n_warmup=n_warmup,
            n_trials=n_trials,
            input_dtypes=task.get("input_dtypes"),
            input_ranges=task.get("input_ranges"),
        )
        results.append(result)
        status = "PASS" if result.correct else "FAIL"
        print(
            f"  [{status}] {result.task_name}: "
            f"speedup={result.speedup:.2f}x "
            f"(baseline={result.baseline_time_ms:.3f}ms, "
            f"kernel={result.kernel_time_ms:.3f}ms)"
        )
    return results
