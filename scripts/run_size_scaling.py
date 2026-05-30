#!/usr/bin/env python3
"""Multi-size scaling harness for PallasBench.

Run each selected kernel at multiple input sizes to characterize
compute-bound vs memory-bound crossover, tiling efficiency, and
kernel launch overhead.

Usage:
    python scripts/run_size_scaling.py --kernels L1/relu,L1/matmul
    python scripts/run_size_scaling.py --kernels all --sizes N256,N1024,N4096 --output-dir results/scaling
    python scripts/run_size_scaling.py --kernels all --sizes N256,N1024,N4096,N8192 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax

from pallasbench.benchmark import evaluate_kernel
from pallasbench.metrics import compute_throughput, bandwidth_utilization
from pallasbench.sizes import SCALING_SUITE, SCALING_SIZE_NAMES
from pallasbench.tasks import get_tasks


def measure_memory_delta() -> float:
    """Return current HBM usage delta in MB, or 0.0 if unavailable."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats.get("peak_bytes_in_use", 0) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-size scaling for PallasBench")
    parser.add_argument(
        "--kernels",
        type=str,
        default="all",
        help="Comma-separated kernel names (e.g. 'L1/relu,L1/matmul') or 'all'",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(SCALING_SIZE_NAMES),
        help=f"Comma-separated size names (default: all primary sizes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate scaling-curve plots (requires matplotlib)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of timing trials per (kernel, size)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=2039.0,
        help="Peak memory bandwidth in GB/s (default: A100 2039)",
    )
    args = parser.parse_args()

    size_names = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for sn in size_names:
        if sn not in SCALING_SUITE:
            sys.exit(f"Unknown size name '{sn}'. Available: {list(SCALING_SUITE.keys())}")

    if args.kernels == "all":
        tasks = get_tasks()
    else:
        kernel_names = [k.strip() for k in args.kernels.split(",") if k.strip()]
        all_tasks = get_tasks()
        tasks = [t for t in all_tasks if t["name"] in kernel_names]
        missing = set(kernel_names) - {t["name"] for t in tasks}
        if missing:
            sys.exit(f"Unknown kernels: {missing}")

    print(f"PallasBench Multi-Size Scaling")
    print(f"Device: {jax.devices()[0]}")
    print(f"Kernels: {len(tasks)}")
    print(f"Sizes: {size_names}")
    print(f"Trials: {args.n_trials}, Warmup: {args.n_warmup}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: dict[str, dict] = {}
    for task in tasks:
        task_name: str = task["name"]
        print(f"{task_name}:")
        per_size: dict = {}

        for size_name in size_names:
            shape_list = SCALING_SUITE[size_name].get(task_name)
            if shape_list is None:
                print(f"  SKIP {size_name}: no shape mapping")
                continue

            mem_before = measure_memory_delta()
            result = evaluate_kernel(
                pallas_fn=task["pallas_fn"],
                baseline_fn=task["baseline_fn"],
                input_shapes=shape_list,
                task_name=f"{task_name}@{size_name}",
                dtype=task.get("dtype", "float32"),
                n_correctness=3,
                n_warmup=args.n_warmup,
                n_trials=args.n_trials,
                input_dtypes=task.get("input_dtypes"),
                input_ranges=task.get("input_ranges"),
            )
            mem_after = measure_memory_delta()

            # Compute total bytes (approximate: sum of all input+output elements * dtype size)
            approx_bytes = 0
            jnp_dtype = jax.numpy.dtype(task.get("dtype", "float32"))
            itemsize = jnp_dtype.itemsize
            for shape in shape_list:
                import numpy as np
                approx_bytes += int(np.prod(shape)) * itemsize

            kernel_ms = result.kernel_time_ms
            throughput = compute_throughput(approx_bytes * 2, kernel_ms) if kernel_ms > 0 else 0.0
            bw_util = bandwidth_utilization(throughput, args.peak_bandwidth)

            per_size[size_name] = {
                "kernel_ms": kernel_ms,
                "baseline_ms": result.baseline_time_ms,
                "speedup": result.speedup,
                "correct": result.correct,
                "throughput_gbps": throughput,
                "bw_util_pct": bw_util,
                "memory_delta_mb": max(0.0, mem_after - mem_before),
                "total_bytes": approx_bytes * 2,
            }

            status = "PASS" if result.correct else "FAIL"
            print(
                f"  [{status}] {size_name}: kernel={kernel_ms:.3f}ms "
                f"baseline={result.baseline_time_ms:.3f}ms "
                f"speedup={result.speedup:.2f}x "
                f"throughput={throughput:.1f}GB/s "
                f"BW={bw_util:.1f}%"
            )

        all_results[task_name] = per_size

    output = {
        "meta": {
            "timestamp": timestamp,
            "device": str(jax.devices()[0]),
            "sizes": size_names,
            "peak_bandwidth_gbps": args.peak_bandwidth,
            "n_trials": args.n_trials,
            "n_warmup": args.n_warmup,
        },
        "results": all_results,
    }

    output_path = os.path.join(args.output_dir, f"scaling_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    if args.plot:
        try:
            _generate_plots(all_results, size_names, args.output_dir, timestamp)
        except ImportError:
            print("Warning: matplotlib not installed; skipping plots.")


def _generate_plots(
    results: dict[str, dict],
    size_names: list[str],
    output_dir: str,
    timestamp: str,
) -> None:
    """Generate per-kernel log-log scaling curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kernel_name, per_size in results.items():
        valid = [(s, per_size[s]) for s in size_names if s in per_size and per_size[s]["kernel_ms"] > 0]
        if len(valid) < 2:
            continue

        sizes, data = zip(*valid)
        x_labels = sizes
        x_pos = range(len(sizes))
        kernel_times = [d["kernel_ms"] for d in data]
        baseline_times = [d["baseline_ms"] for d in data]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_pos, kernel_times, "o-", label="Kernel (Pallas)")
        ax.plot(x_pos, baseline_times, "s--", label="Baseline (JAX)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Problem size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Scaling: {kernel_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"scaling_{kernel_name.replace('/', '_')}_{timestamp}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
