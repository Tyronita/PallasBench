#!/usr/bin/env python3
"""Profile PallasBench kernels with NVIDIA Nsight Compute (NCU).

Captures detailed hardware performance counters — SM occupancy, cache hit rates,
warp execution efficiency, arithmetic intensity, and memory throughput breakdown —
for each Pallas kernel. Optionally generates per-kernel roofline plots.

Usage:
    python scripts/ncu_profile.py --kernels relu softmax --output-dir results/ncu
    python scripts/ncu_profile.py --kernels all --device 0 --metric-set full
    python scripts/ncu_profile.py --kernels all --roofline --output-dir results/roofline
    python scripts/ncu_profile.py --kernels "attention_*" --metric-set roofline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_ncu_profile(
    kernel_script: str,
    kernel_name: str,
    output_dir: Path,
    device: int = 0,
    metric_set: str = "full",
    kernel_name_filter: str = "",
) -> dict:
    """Run ncu profiling for a single kernel and return parsed metrics.

    Spawns `ncu` as a subprocess targeting the given kernel script.
    Filters captured kernels by the provided regex pattern to isolate
    Pallas kernel launches from JAX runtime launches.

    Args:
        kernel_script: Path to Python script that runs the kernel.
        kernel_name: Logical name for the kernel (used in output paths).
        output_dir: Directory to write NCU CSV output and parsed JSON.
        device: CUDA device index to profile.
        metric_set: NCU metric set — 'full', 'roofline', 'memory', or custom.
        kernel_name_filter: Regex filter for --kernel-name (defaults to kernel_name).

    Returns:
        Parsed NCU metrics as a nested dict.
    """
    filter_pattern = kernel_name_filter or f"*{kernel_name}*"
    csv_path = output_dir / f"{kernel_name}_ncu_raw.csv"

    cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", filter_pattern,
        "--set", metric_set,
        "--csv",
        "--log-file", str(csv_path),
        "--device", str(device),
        "--replay-mode", "kernel",
        "python", kernel_script,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    result.check_returncode()

    metrics = parse_ncu_csv(csv_path)
    with open(output_dir / f"{kernel_name}_ncu_profile.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_ncu_csv(csv_path: Path) -> dict:
    """Parse NCU CSV output into a structured metrics dict."""
    import csv

    metrics: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get("Metric Name", row.get("Metric", ""))
            metric_value = row.get("Value", row.get("Metric Value", ""))
            unit = row.get("Unit", "")
            if metric_name and metric_value:
                try:
                    metrics[metric_name] = {
                        "value": float(metric_value),
                        "unit": unit,
                    }
                except ValueError:
                    metrics[metric_name] = {"value": metric_value, "unit": unit}
    return metrics


def compute_arithmetic_intensity(
    flop_count_sp: float,
    dram_bytes_read: float,
    dram_bytes_write: float,
) -> float:
    """Compute FLOP/byte arithmetic intensity from NCU counters."""
    total_dram_bytes = dram_bytes_read + dram_bytes_write
    if total_dram_bytes == 0:
        return 0.0
    return flop_count_sp / total_dram_bytes


def compute_ridge_point(
    peak_flops_per_second: float,
    peak_hbm_bandwidth_bytes_per_second: float,
) -> float:
    """Compute the roofline ridge point: minimum AI to achieve peak FLOP/s."""
    if peak_hbm_bandwidth_bytes_per_second == 0:
        return float("inf")
    return peak_flops_per_second / peak_hbm_bandwidth_bytes_per_second


DEVICE_PEAK_SPECS = {
    "A100": {
        "peak_flops_fp32": 19.5e12,
        "peak_hbm_bandwidth": 2039e9,
        "peak_l2_bandwidth": 4000e9,
        "peak_l1_bandwidth": 12000e9,
    },
    "H100": {
        "peak_flops_fp32": 60e12,
        "peak_hbm_bandwidth": 3350e9,
        "peak_l2_bandwidth": 8000e9,
        "peak_l1_bandwidth": 20000e9,
    },
    "V100": {
        "peak_flops_fp32": 15.7e12,
        "peak_hbm_bandwidth": 900e9,
        "peak_l2_bandwidth": 2000e9,
        "peak_l1_bandwidth": 6000e9,
    },
}


def generate_roofline_plot(
    metrics: dict,
    kernel_name: str,
    output_dir: Path,
    device_spec: str = "A100",
):
    """Generate a roofline plot for a single kernel.

    Plots achieved FLOP/s vs. arithmetic intensity on a log-log chart
    with memory and compute ceilings for the target device.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is required for roofline plots. Install with: pip install matplotlib")
        return

    flops = metrics.get("flop_count_sp", {}).get("value", 0)
    dram_read = metrics.get("dram__bytes_read.sum", {}).get("value", 0)
    dram_write = metrics.get("dram__bytes_write.sum", {}).get("value", 0)

    if flops is None or dram_read is None:
        print(f"SKIP {kernel_name}: missing flop_count_sp or dram__bytes_read")
        return

    ai = compute_arithmetic_intensity(flops, dram_read, dram_write)
    peak = DEVICE_PEAK_SPECS.get(device_spec, DEVICE_PEAK_SPECS["A100"])

    ai_range = np.logspace(-2, 4, 500)
    hbm_ceiling = peak["peak_hbm_bandwidth"] * ai_range
    flops_ceiling = np.full_like(ai_range, peak["peak_flops_fp32"])

    plt.figure(figsize=(8, 6))
    plt.loglog(ai_range, hbm_ceiling, "b--", label=f"HBM ({peak['peak_hbm_bandwidth']/1e9:.0f} GB/s)")
    plt.loglog(ai_range, flops_ceiling, "r--", label=f"FP32 ({peak['peak_flops_fp32']/1e12:.1f} TFLOPS)")
    plt.loglog(ai, min(flops, peak["peak_flops_fp32"]), "ko", markersize=8)
    plt.annotate(
        kernel_name,
        (ai, min(flops, peak["peak_flops_fp32"])),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
    ridge_x = compute_ridge_point(peak["peak_flops_fp32"], peak["peak_hbm_bandwidth"])
    plt.axvline(ridge_x, color="gray", linestyle=":", alpha=0.5)
    plt.annotate(
        f"Ridge ({ridge_x:.1f})",
        (ridge_x, peak["peak_flops_fp32"]),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )

    plt.xlabel("Arithmetic Intensity (FLOP/byte)")
    plt.ylabel("Achieved FLOP/s")
    plt.title(f"Roofline: {kernel_name} ({device_spec})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plot_path = output_dir / f"{kernel_name}_roofline.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Roofline plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile PallasBench kernels with NVIDIA Nsight Compute (NCU)"
    )
    parser.add_argument(
        "--kernels", nargs="+", required=True,
        help="Kernel name(s) to profile, or 'all' for all kernels",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ncu_profiles",
        help="Output directory for NCU CSV and parsed JSON results",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index to profile (default: 0)",
    )
    parser.add_argument(
        "--roofline", action="store_true",
        help="Generate per-kernel roofline plots (requires matplotlib)",
    )
    parser.add_argument(
        "--metric-set", type=str, default="full",
        choices=["full", "roofline", "memory"],
        help="NCU metric set to collect (default: full)",
    )
    parser.add_argument(
        "--kernel-name-filter", type=str, default="",
        help="Regex pattern for ncu --kernel-name (defaults to *kernel_name*)",
    )
    parser.add_argument(
        "--kernel-script", type=str, default="",
        help="Path to Python script that executes the kernel",
    )
    parser.add_argument(
        "--device-spec", type=str, default="A100",
        choices=list(DEVICE_PEAK_SPECS.keys()),
        help="Device model for roofline peak specs (default: A100)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ncu_available = subprocess.run(
        ["ncu", "--version"], capture_output=True, text=True
    ).returncode == 0
    if not ncu_available:
        print("ERROR: ncu not found on PATH. Install NVIDIA Nsight Compute.")
        sys.exit(1)
    print(f"NCU version: {subprocess.run(['ncu', '--version'], capture_output=True, text=True).stdout.strip()}")

    kernel_list = args.kernels
    if "all" in kernel_list:
        from pallasbench.tasks import get_tasks
        tasks = get_tasks(levels=[1, 2, 3])
        kernel_list = [t["name"] for t in tasks]
        print(f"Profiling all {len(kernel_list)} kernels...")

    kernel_script = args.kernel_script or os.path.join(
        os.path.dirname(__file__), "profile_kernel.py"
    )

    for kernel_name in kernel_list:
        print(f"\nProfiling: {kernel_name}")
        metrics = run_ncu_profile(
            kernel_script=kernel_script,
            kernel_name=kernel_name,
            output_dir=output_dir,
            device=args.device,
            metric_set=args.metric_set,
            kernel_name_filter=args.kernel_name_filter or f"*{kernel_name}*",
        )
        print(f"  Captured {len(metrics)} NCU metrics")

        if args.roofline:
            generate_roofline_plot(metrics, kernel_name, output_dir, args.device_spec)

    print(f"\nDone. Results in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
