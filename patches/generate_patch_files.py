"""Generate .diff patch files for each modified Pallas kernel.

Scans all kernel files under pallasbench/kernels/, applies the block-size
clamping fix in memory, computes the diff, and writes a ``.diff`` file per
kernel into the ``patches/`` directory.

If a kernel already has the fix applied (i.e. it contains ``MAX_BLOCK`` or
``BLOCK_M`` etc.), no diff is generated for it.
"""

import difflib
import re
import sys
from pathlib import Path

PATCHES_DIR = Path(__file__).resolve().parent
KERNELS_DIR = PATCHES_DIR.parent / "pallasbench" / "kernels"

# Reuse the fix logic from apply_gpu_fix
sys.path.insert(0, str(PATCHES_DIR))
import apply_gpu_fix  # noqa: E402


def _needs_fix(content: str) -> bool:
    """Return True when *content* still uses the un-clamped pattern."""
    patterns = [
        r"block_size = min\(\d+, n\)",
        r"block_rows = min\(\d+, n_rows\)",
        r"bm = min\(\d+, m\)",
        r"bn = min\(\d+, n\)",
        r"block_q = min\(\d+, seq_len\)",
    ]
    for pat in patterns:
        if re.search(pat, content, re.MULTILINE):
            return True
    return False


def generate_diff(original: str, fixed: str, rel_path: str) -> str:
    """Return a unified-diff string between *original* and *fixed*."""
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        fixed_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        lineterm="\n",
    )
    return "".join(diff)


def main() -> None:
    if not KERNELS_DIR.is_dir():
        print(f"Error: {KERNELS_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    generated = 0
    skipped = 0

    for py_file in sorted(KERNELS_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        rel = py_file.relative_to(PATCHES_DIR.parent)
        content = py_file.read_text(encoding="utf-8")

        if not _needs_fix(content):
            skipped += 1
            continue

        fixed, _ = apply_gpu_fix.apply_fix(content)
        if fixed == content:
            skipped += 1
            continue

        diff_text = generate_diff(content, fixed, str(rel).replace("\\", "/"))

        # Derive output filename: replace separators with underscores
        out_name = str(rel).replace("\\", "_").replace("/", "_") + ".diff"
        out_path = PATCHES_DIR / out_name
        out_path.write_text(diff_text, encoding="utf-8")
        print(f"[GEN] {out_name}")
        generated += 1

    print(f"\nDone: {generated} patch file(s) generated, {skipped} file(s) skipped (no changes needed).")


if __name__ == "__main__":
    main()
