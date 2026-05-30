#!/usr/bin/env python3
"""Build SFT/RL training datasets from kernel evaluation results.

Converts the output of the robust evaluation pipeline (applied to LLM-generated
candidate kernels) into structured training formats:

  - SFT JSONL:  (instruction, output) — for supervised fine-tuning
  - RL JSONL:   (prompt, chosen, rejected) — for reward model / RL training
  - KernelBook: unified cross-benchmark format with candidates array

Usage:
    python scripts/build_training_dataset.py \\
        --eval-results results/robust_eval/*.jsonl \\
        --output-dir data/datasets

    python scripts/build_training_dataset.py \\
        --eval-results results/robust_eval/eval_20260530.jsonl \\
        --format sft rl kernelbook
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


def load_results(paths: list[str]) -> list[dict]:
    """Load evaluation results from JSONL files.

    Each line is a serialized KernelEvalResult dict.
    """
    results = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def _build_instruction(kernel_name: str, source_original: str) -> str:
    """Build an SFT instruction string from the original kernel."""
    return (
        f"Optimize this Pallas kernel for A100 GPU:\n\n"
        f"```python\n{source_original}\n```"
    )


def _build_output(source_optimized: str) -> str:
    return f"```python\n{source_optimized}\n```"


def build_sft_dataset(
    results: list[dict],
    candidate_key: str = "best_candidate_id",
) -> list[dict]:
    """Build SFT-format dataset: (instruction, output) pairs.

    Uses the best candidate per kernel (or all correctness-passing candidates
    if --include-all-candidates is set).
    """
    dataset = []
    for r in results:
        skipped = 0
        for c in r.get("candidates", []):
            if not c.get("eval", {}).get("correctness_passed", False):
                skipped += 1
                continue
            instruction = _build_instruction(
                r["kernel_name"], r.get("source_original", "")
            )
            output = _build_output(c.get("source_optimized", ""))
            dataset.append({
                "instruction": instruction,
                "output": output,
                "kernel_name": r["kernel_name"],
                "candidate_id": c["candidate_id"],
                "speedup": c.get("eval", {}).get("speedup"),
            })
    return dataset


def build_rl_dataset(
    results: list[dict],
) -> list[dict]:
    """Build RL-format dataset: (prompt, chosen, rejected) triples.

    For each kernel, pairs the best candidate (chosen) with the worst
    correctness-passing candidate (rejected) to create a preference pair.
    """
    dataset = []
    for r in results:
        passing = [
            c for c in r.get("candidates", [])
            if c.get("eval", {}).get("correctness_passed", False)
        ]
        if len(passing) < 2:
            continue  # need at least 2 passing candidates for a pair

        passing.sort(key=lambda c: c.get("eval", {}).get("speedup", 0.0), reverse=True)
        best = passing[0]
        worst = passing[-1]

        prompt = _build_instruction(
            r["kernel_name"], r.get("source_original", "")
        )
        dataset.append({
            "prompt": prompt,
            "chosen": _build_output(best.get("source_optimized", "")),
            "rejected": _build_output(worst.get("source_optimized", "")),
            "kernel_name": r["kernel_name"],
            "chosen_speedup": best.get("eval", {}).get("speedup"),
            "rejected_speedup": worst.get("eval", {}).get("speedup"),
            "chosen_candidate_id": best["candidate_id"],
            "rejected_candidate_id": worst["candidate_id"],
        })
    return dataset


def build_kernelbook_dataset(results: list[dict]) -> list[dict]:
    """Build KernelBook-compatible JSONL with full candidates array."""
    dataset = []
    for r in results:
        entry = {
            "kernel_name": r["kernel_name"],
            "level": r.get("level"),
            "category": r.get("category"),
            "source_original": r.get("source_original"),
            "source_fixed": r.get("source_fixed"),
            "reference_source": r.get("reference_source"),
            "hardware": r.get("hardware", "A100-80GB-PCIe"),
            "candidates": [],
            "best_by_model": {},
            "num_candidates": 0,
            "num_correct": 0,
            "num_robust": 0,
            "best_speedup": 0.0,
            "median_speedup": 0.0,
            "best_candidate_id": None,
        }

        candidates = r.get("candidates", [])
        entry["num_candidates"] = len(candidates)
        entry["candidates"] = candidates

        correct = [c for c in candidates if c.get("eval", {}).get("correctness_passed")]
        entry["num_correct"] = len(correct)
        robust = [c for c in candidates if c.get("eval", {}).get("robustness_passed")]
        entry["num_robust"] = len(robust)

        if correct:
            speeds = [c["eval"]["speedup"] for c in correct if c["eval"].get("speedup")]
            if speeds:
                speeds.sort()
                entry["best_speedup"] = max(speeds)
                mid = len(speeds) // 2
                entry["median_speedup"] = speeds[mid] if len(speeds) % 2 else (speeds[mid - 1] + speeds[mid]) / 2
                best = max(correct, key=lambda c: c.get("eval", {}).get("speedup", 0.0))
                entry["best_candidate_id"] = best["candidate_id"]

        # Group best by model
        by_model = defaultdict(list)
        for c in candidates:
            model = c.get("model", "unknown")
            if c.get("eval", {}).get("correctness_passed") and c["eval"].get("speedup"):
                by_model[model].append((c["eval"]["speedup"], c["candidate_id"]))
        for model, entries in by_model.items():
            entries.sort(reverse=True, key=lambda x: x[0])
            entry["best_by_model"][model] = {
                "candidate_id": entries[0][1],
                "speedup": entries[0][0],
            }

        dataset.append(entry)
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT/RL training datasets from kernel evaluation results"
    )
    parser.add_argument(
        "--eval-results",
        nargs="+",
        required=True,
        help="Paths to JSONL files with evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Output directory for training datasets",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["sft", "rl", "kernelbook"],
        choices=["sft", "rl", "kernelbook"],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--include-all-candidates",
        action="store_true",
        default=False,
        help="Include all correctness-passing candidates (not just best) in SFT",
    )
    args = parser.parse_args()

    results = load_results(args.eval_results)
    if not results:
        print("No evaluation results loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} kernel evaluation results")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if "sft" in args.format:
        sft_data = build_sft_dataset(results)
        sft_path = os.path.join(args.output_dir, f"sft_dataset_{timestamp}.jsonl")
        with open(sft_path, "w") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")
        print(f"SFT dataset: {len(sft_data)} examples → {sft_path}")

    if "rl" in args.format:
        rl_data = build_rl_dataset(results)
        rl_path = os.path.join(args.output_dir, f"rl_dataset_{timestamp}.jsonl")
        with open(rl_path, "w") as f:
            for item in rl_data:
                f.write(json.dumps(item) + "\n")
        print(f"RL dataset:   {len(rl_data)} examples → {rl_path}")

    if "kernelbook" in args.format:
        kb_data = build_kernelbook_dataset(results)
        kb_path = os.path.join(args.output_dir, f"kernelbook_{timestamp}.jsonl")
        with open(kb_path, "w") as f:
            for item in kb_data:
                f.write(json.dumps(item) + "\n")
        print(f"KernelBook:   {len(kb_data)} entries → {kb_path}")


if __name__ == "__main__":
    main()
