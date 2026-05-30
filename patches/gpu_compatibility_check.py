"""GPU compatibility checker for Pallas kernels.

Reads each kernel file under pallasbench/kernels/ and checks for potential
issues that could cause compilation failures or runtime errors on GPU
backends (CUDA / Metal).
"""

import re
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parent.parent / "pallasbench" / "kernels"

# Maximum elements allowed in a single Triton program block (~1M)
MAX_BLOCK_ELEMENTS = 1_048_576

# TPU-specific operations that may not have GPU implementations
TPU_ONLY_OPS = [
    "tpu_custom_call",
    "tpu_allocate",
    "tpu_annotate",
    "kv_quant",
]


def _check_block_element_count(file_path: Path, content: str) -> list[str]:
    """Check that block sizes in BlockSpecs don't exceed the 1M element limit.

    Returns a list of warning messages.
    """
    warnings = []
    lines = content.splitlines()

    for i, line in enumerate(lines, start=1):
        # Look for BlockSpec((...), ...) patterns
        m = re.search(r"BlockSpec\(\(([^)]+)\)", line)
        if m:
            dims_str = m.group(1)
            # Try to extract integer dimensions from the BlockSpec
            dims = []
            for part in dims_str.split(","):
                part = part.strip()
                try:
                    dims.append(int(part))
                except ValueError:
                    # Could be a variable name - skip variable-based checks
                    # but flag if it contains non-trivial expressions
                    pass
            if dims:
                prod = 1
                for d in dims:
                    prod *= d
                if prod > MAX_BLOCK_ELEMENTS:
                    warnings.append(
                        f"{file_path.name}:{i}  BlockSpec dimensions {dims} "
                        f"- {prod} elements exceeds {MAX_BLOCK_ELEMENTS} limit"
                    )
    return warnings


def _check_tpu_only_ops(file_path: Path, content: str) -> list[str]:
    """Flag any TPU-specific operations that lack GPU support."""
    warnings = []
    for op in TPU_ONLY_OPS:
        for i, line in enumerate(content.splitlines(), start=1):
            if op in line:
                warnings.append(
                    f"{file_path.name}:{i}  TPU-only operation '{op}' may not work on GPU"
                )
    return warnings


def _check_dtype_issues(file_path: Path, content: str) -> list[str]:
    """Flag potential dtype issues for GPU compilation."""
    warnings = []
    # bfloat16 may need special handling on non-TPU backends
    for i, line in enumerate(content.splitlines(), start=1):
        if "bfloat16" in line and "jnp" in line:
            warnings.append(
                f"{file_path.name}:{i}  bfloat16 usage - verify GPU support"
            )
    return warnings


def _check_large_full_block(file_path: Path, content: str) -> list[str]:
    """Flag kernels that use full-shaped BlockSpecs (entire tensor in one block)."""
    warnings = []
    for i, line in enumerate(content.splitlines(), start=1):
        # Look for BlockSpec with just `.shape` (e.g. BlockSpec(table.shape, ...))
        if re.search(r"BlockSpec\(\w+\.shape", line):
            warnings.append(
                f"{file_path.name}:{i}  Full-shaped BlockSpec - may exceed GPU memory "
                "if input is large"
            )
    return warnings


def check_file(file_path: Path) -> list[str]:
    """Run all checks on *file_path* and return a list of warnings."""
    content = file_path.read_text(encoding="utf-8")
    results: list[str] = []
    results.extend(_check_block_element_count(file_path, content))
    results.extend(_check_tpu_only_ops(file_path, content))
    results.extend(_check_dtype_issues(file_path, content))
    results.extend(_check_large_full_block(file_path, content))
    return results


def main() -> None:
    if not KERNELS_DIR.is_dir():
        print(f"Error: kernels directory not found at {KERNELS_DIR}", file=sys.stderr)
        sys.exit(1)

    all_warnings: dict[str, list[str]] = {}
    total_files = 0
    total_warnings = 0

    for py_file in sorted(KERNELS_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        total_files += 1
        warnings = check_file(py_file)
        if warnings:
            rel = py_file.relative_to(KERNELS_DIR.parent)
            all_warnings[str(rel)] = warnings
            total_warnings += len(warnings)

    # Print report
    if not all_warnings:
        print("No GPU compatibility issues found.")
        print(f"Checked {total_files} file(s).")
        return

    print(f"GPU Compatibility Report - {total_warnings} issue(s) in {len(all_warnings)} file(s)\n")
    print(f"{'='*60}\n")

    for rel_path, warnings in sorted(all_warnings.items()):
        print(f"  {rel_path}")
        for w in warnings:
            print(f"    [!] {w}")
        print()

    print(f"Checked {total_files} file(s).")
    sys.exit(1 if total_warnings > 0 else 0)


if __name__ == "__main__":
    main()
