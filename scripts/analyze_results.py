#!/usr/bin/env python3
"""Analyze PallasBench results from JSON output files.

Usage:
    python scripts/analyze_results.py --results-dir results/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    results = []
    for f in sorted(Path(results_dir).glob("pallasbench_*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def print_comparison_table(all_results: list[dict]):
    print(f"\n{'Backend':<12} {'Tasks':>6} {'Correct':>8} {'fast_0':>7} "
          f"{'fast_1':>7} {'fast_2':>7} {'Median':>8}")
    print("-" * 62)

    for run in all_results:
        meta = run["meta"]
        s = run["summary"]
        print(
            f"{meta['backend']:<12} "
            f"{s['total_tasks']:>6} "
            f"{s['correct']:>8} "
            f"{s['fast_0']:>7.1%} "
            f"{s['fast_1']:>7.1%} "
            f"{s['fast_2']:>7.1%} "
            f"{s['median_speedup']:>7.2f}x"
        )


def print_task_details(run: dict):
    print(f"\n{'Task':<25} {'Status':>7} {'Baseline':>10} {'Kernel':>10} {'Speedup':>8}")
    print("-" * 65)

    for task in run["tasks"]:
        status = "PASS" if task["correct"] else "FAIL"
        print(
            f"{task['name']:<25} "
            f"{status:>7} "
            f"{task['baseline_ms']:>9.3f}ms "
            f"{task['kernel_ms']:>9.3f}ms "
            f"{task['speedup']:>7.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze PallasBench results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--detail", action="store_true", help="Show per-task details")
    args = parser.parse_args()

    all_results = load_results(args.results_dir)
    if not all_results:
        print(f"No result files found in {args.results_dir}/")
        return

    print(f"Found {len(all_results)} result file(s)")
    print_comparison_table(all_results)

    if args.detail:
        for run in all_results:
            print(f"\n\n=== {run['meta']['backend']} "
                  f"({run['meta']['timestamp']}) ===")
            print_task_details(run)


if __name__ == "__main__":
    main()
