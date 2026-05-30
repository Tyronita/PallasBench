"""Evaluation metrics for PallasBench, adapted from KernelBench's fast_p."""

from __future__ import annotations

from pallasbench.utils import BenchmarkResult


def fast_p(results: list[BenchmarkResult], p: float = 1.0) -> float:
    """Fraction of tasks that are both correct and achieve speedup > p.

    fast_0: correctness only (p=0)
    fast_1: correct AND faster than baseline (p=1)
    fast_2: correct AND 2x+ faster (p=2)
    """
    if not results:
        return 0.0
    passing = sum(1 for r in results if r.correct and r.speedup > p)
    return passing / len(results)


def correctness_rate(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def mean_speedup(results: list[BenchmarkResult], correct_only: bool = True) -> float:
    filtered = [r for r in results if r.correct] if correct_only else results
    if not filtered:
        return 0.0
    return sum(r.speedup for r in filtered) / len(filtered)


def median_speedup(results: list[BenchmarkResult], correct_only: bool = True) -> float:
    import numpy as np

    filtered = [r for r in results if r.correct] if correct_only else results
    if not filtered:
        return 0.0
    return float(np.median([r.speedup for r in filtered]))


def compute_throughput(total_bytes: float, kernel_time_ms: float) -> float:
    """Compute memory throughput in GB/s.

    Args:
        total_bytes: Total bytes read + written by the kernel.
        kernel_time_ms: Median kernel execution time in milliseconds.

    Returns:
        Throughput in GB/s, or 0.0 if kernel_time_ms is 0.
    """
    if kernel_time_ms <= 0.0:
        return 0.0
    return (total_bytes / 1e9) / (kernel_time_ms / 1e3)


def bandwidth_utilization(throughput_gbps: float, peak_gbps: float = 2039.0) -> float:
    """Fraction of peak memory bandwidth achieved.

    Args:
        throughput_gbps: Measured throughput in GB/s.
        peak_gbps: Device peak memory bandwidth (default A100 80GB: 2039 GB/s).

    Returns:
        Utilization as a percentage (0-100).
    """
    if peak_gbps <= 0.0:
        return 0.0
    return (throughput_gbps / peak_gbps) * 100.0


def results_summary(results: list[BenchmarkResult]) -> dict:
    return {
        "total_tasks": len(results),
        "correct": sum(1 for r in results if r.correct),
        "fast_0": fast_p(results, p=0.0),
        "fast_1": fast_p(results, p=1.0),
        "fast_2": fast_p(results, p=2.0),
        "fast_5": fast_p(results, p=5.0),
        "mean_speedup": mean_speedup(results),
        "median_speedup": median_speedup(results),
    }
