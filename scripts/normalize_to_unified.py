#!/usr/bin/env python3
"""
Cross-Benchmark DSL Unification: Normalize any supported benchmark to a common
KernelBook-compatible JSONL format.

Usage:
    python scripts/normalize_to_unified.py \\
        --source {kernelbench,pallasbench,cvdp,verilog_eval,multikernelbench,kernelbench_v2,kernelbot} \\
        --input <path> --output <path> [--dry-run]

Supported benchmarks and DSLs:
  - kernelbench      CUDA, Triton      (250 tasks)
  - pallasbench      Pallas/JAX        (45 tasks)
  - cvdp             SystemVerilog     (304 tasks)
  - verilog_eval     Verilog           (157 tasks)
  - multikernelbench CUDA,Triton,Pallas,AscendC (285 tasks)
  - kernelbench_v2   Triton            (~250 tasks)
  - kernelbot        CUDA              (competition tasks)

Schema:
    schemas/unified_benchmark_schema.json

Example:
    python normalize_to_unified.py --source pallasbench \\
        --input ../data/pallasbench/ --output ../data/pallasbench/normalized.jsonl

    python normalize_to_unified.py --source kernelbench \\
        --input ../data/kernelbench/results.jsonl --output ../data/kernelbench/normalized.jsonl \\
        --dry-run

Output format:
    JSONL (one JSON object per line) conforming to unified_benchmark_schema.json.
"""

import argparse
import json
import sys
from pathlib import Path


# Mapping of benchmark sources to their native parsers.
# Each parser function accepts an input path and yields unified dicts.
PARSERS = {}


def normalize_kernelbench(input_path: Path, dry_run: bool):
    """Normalize KernelBench JSONL to unified format.

    Maps native fields: task_id, reference_code -> reference_implementation,
    generated_code -> generated_source, result.compile_pass -> correctness.passed.
    """
    raise NotImplementedError


def normalize_pallasbench(input_path: Path, dry_run: bool):
    """Normalize PallasBench directory to unified format.

    Reads directory of .py kernel files + per-kernel result.json + IR artifacts
    (jaxpr.txt, stablehlo.txt, triton_mlir.txt) + fix.diff.
    """
    raise NotImplementedError


def normalize_cvdp(input_path: Path, dry_run: bool):
    """Normalize CVDP (SystemVerilog) JSONL from HuggingFace to unified format."""
    raise NotImplementedError


def normalize_verilog_eval(input_path: Path, dry_run: bool):
    """Normalize Verilog Eval JSONL from HuggingFace to unified format."""
    raise NotImplementedError


def normalize_multikernelbench(input_path: Path, dry_run: bool):
    """Normalize MultiKernelBench CSV/kernel directory to unified format.

    MultiKernelBench contains entries for CUDA, Triton, Pallas, and AscendC.
    The language field is inferred from the entry metadata.
    """
    raise NotImplementedError


def normalize_kernelbench_v2(input_path: Path, dry_run: bool):
    """Normalize KernelBench-v2 JSONL to unified format.

    Shares the same schema structure as KernelBench; reuses normalize_kernelbench
    logic with version-specific field adjustments.
    """
    raise NotImplementedError


def normalize_kernelbot(input_path: Path, dry_run: bool):
    """Normalize KernelBot competition results to unified format.

    KernelBot tasks are CUDA competition kernels with correctness and speedup results.
    """
    raise NotImplementedError


# Register parsers
PARSERS["kernelbench"] = normalize_kernelbench
PARSERS["pallasbench"] = normalize_pallasbench
PARSERS["cvdp"] = normalize_cvdp
PARSERS["verilog_eval"] = normalize_verilog_eval
PARSERS["multikernelbench"] = normalize_multikernelbench
PARSERS["kernelbench_v2"] = normalize_kernelbench_v2
PARSERS["kernelbot"] = normalize_kernelbot


def validate_entry(entry: dict) -> list[str]:
    """Validate a single unified entry against required fields.

    Returns a list of missing/invalid field names. Empty list means valid.

    For full schema validation, use:
        python -c "import json, jsonschema; jsonschema.validate(data, schema)"
    """
    errors = []
    required = ["task_id", "benchmark_source", "language", "task_description",
                "category", "difficulty_level", "reference_implementation",
                "generated_source", "correctness", "hardware_context"]
    for field in required:
        if field not in entry:
            errors.append(f"missing required field: {field}")
    return errors


def normalize(source: str, input_path: Path, output_path: Path, dry_run: bool) -> int:
    """Run the normalizer for *source* and write unified JSONL to *output_path*.

    Returns the number of entries written (or would be written in dry-run mode).
    """
    parser = PARSERS.get(source)
    if parser is None:
        available = ", ".join(sorted(PARSERS))
        print(f"Error: unknown source '{source}'. Available: {available}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for entry in parser(input_path, dry_run):
        errors = validate_entry(entry)
        if errors:
            print(f"Validation errors for entry {entry.get('task_id', '?'):s}:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            continue
        entries.append(entry)

    if dry_run:
        print(f"[dry-run] Would write {len(entries)} entries to {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} entries to {output_path}")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize benchmarks to unified KernelBook-compatible JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(PARSERS.keys()),
        help="Benchmark source to normalize",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input path (file or directory, depends on benchmark)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate input without writing output",
    )
    args = parser.parse_args()

    # Validate input path exists
    if not args.input.exists():
        print(f"Error: input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    count = normalize(args.source, args.input, args.output, args.dry_run)
    print(f"Done. Processed {count} entries.")


if __name__ == "__main__":
    main()
