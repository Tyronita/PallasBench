"""
Archive column schema — SakanaAI AI-CUDA-Engineer-Archive format
adapted for JAX/Pallas on T4.

SakanaAI columns kept verbatim where semantically equivalent so the
two archives can be joined/compared directly.  JAX-only columns are
clearly marked.  The three parquet splits are level_1 / level_2 / level_3
matching SakanaAI's naming.
"""
from __future__ import annotations

import pyarrow as pa

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Every column: (name, pyarrow_type, description, sakana_equivalent_or_new)
COLUMNS: list[tuple[str, pa.DataType, str, str]] = [
    # ── Identity ────────────────────────────────────────────────────────────
    ("Op_Name",             pa.string(),  "Kernel name, e.g. 'relu'",                   "Op_Name"),
    ("Level_ID",            pa.int64(),   "Difficulty level 1–3",                       "Level_ID"),
    ("Task_ID",             pa.int64(),   "Sequential task ID 1–45",                    "Task_ID"),
    ("Kernel_Name",         pa.string(),  "Unique name for this variant",               "Kernel_Name"),
    ("Category",            pa.string(),  "activation / normalization / matmul / …",    "NEW"),
    ("Generation_Idx",      pa.int64(),   "0 = seed kernel, 1+ = LLM-generated",        "NEW"),
    ("LLM_Model",           pa.string(),  "Model that generated this variant",          "NEW"),

    # ── Timing (all in milliseconds) ────────────────────────────────────────
    ("Pallas_Runtime",               pa.float64(), "Pallas kernel wall-time (ms)",             "CUDA_Runtime"),
    ("JAX_Native_Runtime",           pa.float64(), "Pure JAX (no JIT) baseline (ms)",          "PyTorch_Native_Runtime"),
    ("JAX_XLA_Compiled_Runtime",     pa.float64(), "JAX + jit() compiled baseline (ms)",       "PyTorch_Compile_Runtime"),
    ("Pallas_Speedup_Native",        pa.float64(), "Pallas / JAX_Native ratio",                "CUDA_Speedup_Native"),
    ("Pallas_Speedup_Compiled",      pa.float64(), "Pallas / JAX_XLA_Compiled ratio",          "CUDA_Speedup_Compile"),

    # ── Code ────────────────────────────────────────────────────────────────
    ("Pallas_Code",           pa.string(), "Pallas kernel source (LLM-generated or seed)", "CUDA_Code"),
    ("JAX_Code_Module",       pa.string(), "JAX reference as a class with __call__",       "PyTorch_Code_Module"),
    ("JAX_Code_Functional",   pa.string(), "JAX reference as a plain function",            "PyTorch_Code_Functional"),
    ("Pallas_Code_Original",  pa.string(), "Original TPU-oriented source before GPU fix",  "NEW"),
    ("Diff",                  pa.string(), "Git-style diff between original and fixed",     "NEW"),

    # ── Correctness ─────────────────────────────────────────────────────────
    ("Correct",    pa.bool_(),   "Output matches JAX reference within tolerance",   "Correct"),
    ("Max_Diff",   pa.float64(), "max|pallas_out - jax_out|",                       "Max_Diff"),
    ("Error",      pa.string(),  "Exception / compile error text, or ''",           "Error"),

    # ── KernelBench fast_p metrics ───────────────────────────────────────────
    ("fast_0",  pa.bool_(), "Correct (any speed)",                                "NEW"),
    ("fast_1",  pa.bool_(), "Correct AND speedup > 1.0",                          "NEW"),
    ("fast_2",  pa.bool_(), "Correct AND speedup > 2.0",                          "NEW"),
    ("fast_5",  pa.bool_(), "Correct AND speedup > 5.0",                          "NEW"),

    # ── JAX IR tools ────────────────────────────────────────────────────────
    ("Jaxpr_IR",       pa.string(), "jax.make_jaxpr() text with shapes",           "replaces Clang_Tidy"),
    ("StableHLO_IR",   pa.string(), "StableHLO MLIR text from jax.export()",       "NEW"),
    ("PTX_Code",       pa.string(), "T4 PTX assembly (via MOSAIC_GPU_DUMP_PTX)",   "NEW"),
    ("SASS_Code",      pa.string(), "T4 SASS assembly (via MOSAIC_GPU_DUMP_SASS)",  "NEW"),

    # ── Pallas kernel geometry ───────────────────────────────────────────────
    ("Block_Shape",  pa.string(), "JSON list: Pallas block_shape dims",             "NEW"),
    ("Grid_Shape",   pa.string(), "JSON list: Pallas grid dims",                    "NEW"),

    # ── Profiling ────────────────────────────────────────────────────────────
    ("NCU_Profile",   pa.string(), "JSON from ncu --metrics on T4",                "NCU_Profile"),
    ("JAX_Profile",   pa.string(), "jax.profiler.StepTraceAnnotation JSON",         "Torch_Profile"),

    # ── Input metadata ───────────────────────────────────────────────────────
    ("Input_Shapes",   pa.string(), "JSON list of input tensor shapes",             "NEW"),
    ("Input_Dtypes",   pa.string(), "JSON list of input dtypes",                    "NEW"),

    # ── Hardware / environment ───────────────────────────────────────────────
    ("Target_Hardware",      pa.string(), "GPU model name",           "NEW"),
    ("Compute_Capability",   pa.string(), "e.g. '7.5' for T4",        "NEW"),
    ("JAX_Version",          pa.string(), "jax.__version__",          "NEW"),
    ("Triton_Version",       pa.string(), "triton.__version__ if available", "NEW"),

    # ── Source provenance ────────────────────────────────────────────────────
    ("GitHub_URL",     pa.string(), "Link to original kernel in PallasBench repo",  "NEW"),
    ("GitHub_Raw_URL", pa.string(), "Raw GitHub URL to kernel file",                "NEW"),
]

# Flat name list and pyarrow schema object
COLUMN_NAMES: list[str] = [c[0] for c in COLUMNS]
PA_SCHEMA = pa.schema([(c[0], c[1]) for c in COLUMNS])

# Names that are direct equivalents of SakanaAI columns
SAKANA_EQUIVALENTS: dict[str, str] = {
    c[0]: c[3] for c in COLUMNS if c[3] not in ("NEW", "replaces Clang_Tidy")
}

# Parquet split → level IDs (mirrors SakanaAI naming)
SPLITS: dict[str, list[int]] = {
    "level_1": [1],
    "level_2": [2],
    "level_3": [3],
}


def empty_row() -> dict:
    """Return a blank row with every column at its zero/null value."""
    defaults: dict[str, object] = {}
    for name, dtype, *_ in COLUMNS:
        if dtype == pa.bool_():
            defaults[name] = False
        elif dtype == pa.int64():
            defaults[name] = 0
        elif dtype == pa.float64():
            defaults[name] = float("nan")
        else:
            defaults[name] = ""
    return defaults
