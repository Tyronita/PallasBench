"""Apply block-size clamping fix to all Pallas kernels.

Triton enforces a ~1M element limit per program.  This script walks every
kernel file under pallasbench/kernels/ and clamps block-size variables so
the product of all dimensions in a BlockSpec never exceeds 1M elements.

Idempotent — safe to run multiple times.
"""

import argparse
import re
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parent.parent / "pallasbench" / "kernels"

# ---------------------------------------------------------------------------
# Replacement patterns
# ---------------------------------------------------------------------------

def _replace_1d_block_size(text: str) -> str:
    """Replace ``block_size = min(<int>, n)`` with a MAX_BLOCK-clamped form.

    The new pattern is::

        MAX_BLOCK = 65536
        block_size = min(n, MAX_BLOCK)
    """
    # Match:  block_size = min(ANY_NUMBER, n)
    text, n1 = re.subn(
        r"^(    )block_size = min\(\d+, n\)$",
        r"\1MAX_BLOCK = 65536\n\1block_size = min(n, MAX_BLOCK)",
        text,
        flags=re.MULTILINE,
    )
    # Also handle cases like min(1024, n) where a comment or space differs
    text, n2 = re.subn(
        r"^(    )block_size = min\((\d+),\s*n\)$",
        r"\1MAX_BLOCK = 65536\n\1block_size = min(n, MAX_BLOCK)",
        text,
        flags=re.MULTILINE,
    )
    return text, n1 + n2


def _replace_block_rows(text: str) -> str:
    """Replace ``block_rows = min(<int>, n_rows)`` with a MAX_BLOCK-clamped form."""
    text, n = re.subn(
        r"^(    )block_rows = min\(\d+, n_rows\)$",
        r"\1MAX_BLOCK = 65536\n\1block_rows = min(n_rows, MAX_BLOCK)",
        text,
        flags=re.MULTILINE,
    )
    return text, n


def _replace_2d_block_vars(text: str) -> str:
    """Replace 2D block-size variables with clamped uppercase versions.

    Handles matmul-style (bm, bn) and gated-MLP-style (bm only) patterns.
    """
    changes = 0

    # bm = min(INT, m)  →  BLOCK_M = min(m, 128)
    text, n = re.subn(
        r"^(    )bm = min\(\d+, m\)$",
        r"\1BLOCK_M = min(m, 128)",
        text,
        flags=re.MULTILINE,
    )
    changes += n

    # bn = min(INT, n)  →  BLOCK_N = min(n, 128)
    text, n = re.subn(
        r"^(    )bn = min\(\d+, n\)$",
        r"\1BLOCK_N = min(n, 128)",
        text,
        flags=re.MULTILINE,
    )
    changes += n

    # block_q = min(INT, seq_len)  →  BLOCK_Q = min(seq_len, 128)
    text, n = re.subn(
        r"^(    )block_q = min\(\d+, seq_len\)$",
        r"\1BLOCK_Q = min(seq_len, 128)",
        text,
        flags=re.MULTILINE,
    )
    changes += n

    # Also handle bm used in swiglu/geglu/gated_mlp where only bm exists
    return text, changes


def _update_usages(text: str) -> str:
    """Update all references to renamed variables within the same file.

    After renaming bm → BLOCK_M, bn → BLOCK_N, block_q → BLOCK_Q, this
    function updates every usage of those names so the file remains
    internally consistent.
    """
    # Replace bm  → BLOCK_M   (but only after the definition is changed)
    # We use word boundaries so we don't match substrings.
    text = re.sub(r"\bbm\b", "BLOCK_M", text)
    text = re.sub(r"\bbn\b", "BLOCK_N", text)
    text = re.sub(r"\bblock_q\b", "BLOCK_Q", text)
    return text


def apply_fix(content: str) -> tuple[str, int]:
    """Apply all block-size clamping fixes to *content*.

    Returns (new_content, change_count).
    """
    original = content
    total = 0

    content, n = _replace_1d_block_size(content)
    total += n

    content, n = _replace_block_rows(content)
    total += n

    content, n = _replace_2d_block_vars(content)
    total += n

    # Update usages of renamed variables
    if n > 0:
        content = _update_usages(content)

    # Guard against already-applied files — if nothing changed, return original
    if content == original:
        return original, 0

    return content, total


def process_file(path: Path, dry_run: bool = False) -> int:
    """Read, fix, and optionally write *path*.

    Returns the number of changes made.
    """
    content = path.read_text(encoding="utf-8")
    new_content, changes = apply_fix(content)

    if changes == 0:
        return 0

    if dry_run:
        print(f"[DRY-RUN] {path.relative_to(KERNELS_DIR.parent)}  ({changes} change(s))")
        return changes

    # Idempotency check — skip if already applied (MAX_BLOCK already present)
    if "MAX_BLOCK" in content:
        return 0

    path.write_text(new_content, encoding="utf-8")
    print(f"[APPLY]  {path.relative_to(KERNELS_DIR.parent)}  ({changes} change(s))")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply block-size clamping fix to Pallas kernels."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=True,
        dest="apply_",
        help="Apply fixes (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    if not KERNELS_DIR.is_dir():
        print(f"Error: kernels directory not found at {KERNELS_DIR}", file=sys.stderr)
        sys.exit(1)

    total_changes = 0
    total_files = 0

    for py_file in sorted(KERNELS_DIR.rglob("*.py")):
        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue

        changes = process_file(py_file, dry_run=args.dry_run)
        if changes > 0:
            total_files += 1
            total_changes += changes

    if args.dry_run:
        print(f"\nDry-run complete: {total_files} file(s) would be changed ({total_changes} change(s)).")
    else:
        print(f"\nDone: {total_files} file(s) updated ({total_changes} change(s)).")


if __name__ == "__main__":
    main()
