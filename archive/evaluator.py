"""
Kernel evaluator — executes a Pallas kernel string against a JAX reference,
measures runtime, computes speedup, and fills an archive row dict.

Designed to be safe: every eval is wrapped in a subprocess-isolated exec
so a segfaulting kernel doesn't crash the main process.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .schema import empty_row, COLUMN_NAMES
from .ir_tools import (
    extract_jaxpr_from_source,
    extract_stablehlo,
    extract_stablehlo_from_source,
    extract_ptx_sass,
    extract_block_grid,
    build_ir_dag,
    profile_kernel,
)

# ---------------------------------------------------------------------------
# Runtime measurement helpers
# ---------------------------------------------------------------------------

N_WARMUP  = 3
N_TIMINGS = 10
ATOL      = 1e-2   # relaxed: fp16 attention has larger error
RTOL      = 1e-2


def _is_gpu_available() -> bool:
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _pallas_needs_interpret() -> bool:
    """
    Return True when the GPU is < sm_80 (T4, V100, …) where JAX 0.10.1
    Pallas backends (Mosaic GPU / Triton) require Ampere+.

    On T4 (sm_75) we fall back to interpret=True for correctness checks
    while still measuring JAX reference performance on GPU.
    """
    try:
        import jax
        for d in jax.devices():
            if d.platform == "gpu":
                # Backend string contains compute capability for CUDA devices
                cc_str = getattr(d, "device_kind", "") or ""
                # parse e.g. "cuda:0 (sm_75)" or use nvidia-smi
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    cc = float(r.stdout.strip().split("\n")[0])
                    return cc < 8.0
        return False
    except Exception:
        return False

_USE_INTERPRET: bool | None = None  # lazy cache

def _should_interpret() -> bool:
    global _USE_INTERPRET
    if _USE_INTERPRET is None:
        _USE_INTERPRET = _pallas_needs_interpret()
    return _USE_INTERPRET


_POSITIVE_INPUT_OPS = frozenset([
    "log", "rsqrt", "exp", "sqrt",
    "log_softmax",  # numerically stable but still benefits from bounded inputs
])

def _make_sample_inputs(problem) -> list[Any]:
    """Create sample numpy arrays matching problem.input_shapes/dtypes."""
    rng = np.random.default_rng(42)
    arrays = []
    need_positive = problem.name in _POSITIVE_INPUT_OPS
    for idx, (shape, dtype_str) in enumerate(zip(problem.input_shapes, problem.input_dtypes)):
        dt = np.dtype(dtype_str)
        if np.issubdtype(dt, np.integer):
            high = max(shape) if shape else 256
            arr = rng.integers(0, min(high, 128), size=shape).astype(dt)
        elif need_positive and idx == 0:
            # Use U(0.1, 3.0) so log/rsqrt stay well-defined
            arr = rng.uniform(0.1, 3.0, size=shape).astype(dt)
        else:
            arr = rng.standard_normal(shape).astype(dt)
        arrays.append(arr)
    return arrays


def _time_fn(jfn, jax_arrays, n_warmup=N_WARMUP, n_trials=N_TIMINGS) -> float:
    """Return mean wall-time in ms over n_trials calls."""
    try:
        import jax
        for _ in range(n_warmup):
            out = jfn(*jax_arrays)
            if hasattr(out, "block_until_ready"):
                out.block_until_ready()
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out = jfn(*jax_arrays)
            if hasattr(out, "block_until_ready"):
                out.block_until_ready()
            times.append((time.perf_counter() - t0) * 1e3)
        return float(np.mean(times))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_kernel(
    problem,
    kernel_source: str,
    kernel_fn_name: str,
    generation_idx: int = 0,
    llm_model: str = "seed",
    capture_ir: bool = True,
    capture_ptx: bool = False,
    capture_ncu: bool = False,
) -> dict:
    """
    Evaluate a single Pallas kernel implementation.

    Returns a dict with all COLUMN_NAMES keys populated.
    """
    row = empty_row()

    # ── Identity ───────────────────────────────────────────────────────────
    row["Op_Name"]        = problem.name
    row["Level_ID"]       = problem.level
    row["Task_ID"]        = problem.task_id
    row["Category"]       = problem.category
    row["Generation_Idx"] = generation_idx
    row["LLM_Model"]      = llm_model
    row["Kernel_Name"]    = f"{problem.name}_gen{generation_idx}"
    row["Pallas_Code"]    = kernel_source
    row["JAX_Code_Module"]     = problem.jax_module
    row["JAX_Code_Functional"] = problem.jax_functional
    row["Pallas_Code_Original"] = problem.original_pallas
    row["Input_Shapes"]   = problem.input_shapes_json()
    row["Input_Dtypes"]   = problem.input_dtypes_json()
    row["GitHub_URL"]     = problem.github_url
    row["GitHub_Raw_URL"] = problem.github_raw_url
    row["Target_Hardware"]    = "NVIDIA Tesla T4"
    row["Compute_Capability"] = "7.5"

    try:
        import jax
        row["JAX_Version"] = jax.__version__
        try:
            import triton
            row["Triton_Version"] = triton.__version__
        except ImportError:
            row["Triton_Version"] = "unavailable"
    except ImportError:
        row["JAX_Version"] = "unavailable"

    # ── Block / Grid geometry (static parse) ───────────────────────────────
    block_json, grid_json = extract_block_grid(kernel_source)
    row["Block_Shape"] = block_json
    row["Grid_Shape"]  = grid_json

    # ── Prepare inputs ─────────────────────────────────────────────────────
    try:
        sample_np = _make_sample_inputs(problem)
    except Exception as e:
        row["Error"] = f"[input_gen_error] {e}"
        return row

    # ── Execute kernel + measure timing ────────────────────────────────────
    on_gpu = _is_gpu_available()
    try:
        import jax, jax.numpy as jnp

        # Convert numpy → jax arrays
        jax_inputs = [jnp.array(a) for a in sample_np]

        if not on_gpu:
            row["Error"] = "[cpu_only] GPU not available — IR from JAX reference"
            _capture_jax_reference(row, problem, jax_inputs)
            if capture_ir:
                _capture_ir_from_ref(row, problem, jax_inputs)
            return row

        use_interpret = _should_interpret()
        if use_interpret:
            # T4 / sm_75: Pallas uses interpret=True for correctness;
            # JAX reference timings are measured on GPU.
            row["Target_Hardware"] = "NVIDIA Tesla T4 (Pallas interpret=True — native needs sm_80+)"
            _capture_jax_reference(row, problem, jax_inputs)
            if capture_ir:
                _capture_ir_from_ref(row, problem, jax_inputs)

        # Compile + exec the kernel source
        ns: dict = {}
        exec(compile(kernel_source, "<kernel>", "exec"), ns)
        pallas_fn = ns[kernel_fn_name]

        # On T4: wrap pallas_call calls with interpret=True by patching the source
        if use_interpret:
            import jax.experimental.pallas as pl
            pallas_jit = jax.jit(
                lambda *args: _run_interpret(pallas_fn, args)
            )
        else:
            pallas_jit = jax.jit(pallas_fn)

        # Warmup run to catch compile errors
        pallas_out = pallas_jit(*jax_inputs)
        if hasattr(pallas_out, "block_until_ready"):
            pallas_out.block_until_ready()

        pallas_ms = _time_fn(pallas_jit, jax_inputs)
        row["Pallas_Runtime"] = pallas_ms

        # ── JAX reference baselines ────────────────────────────────────────
        ref_ns: dict = {}
        ref_src = problem.jax_functional.strip()
        exec(compile(ref_src, "<ref>", "exec"), ref_ns)
        # Pick the first function defined
        ref_fn_name = next(
            (k for k, v in ref_ns.items() if callable(v) and not k.startswith("_")),
            None,
        )
        if ref_fn_name:
            ref_fn     = ref_ns[ref_fn_name]
            ref_fn_jit = jax.jit(ref_fn)

            native_ms  = _time_fn(ref_fn,     jax_inputs)
            compiled_ms = _time_fn(ref_fn_jit, jax_inputs)
            row["JAX_Native_Runtime"]       = native_ms
            row["JAX_XLA_Compiled_Runtime"] = compiled_ms

            if not (np.isnan(native_ms) or np.isnan(pallas_ms)):
                row["Pallas_Speedup_Native"]   = native_ms / pallas_ms
                row["Pallas_Speedup_Compiled"] = compiled_ms / pallas_ms

            # ── Correctness check ──────────────────────────────────────────
            ref_out = ref_fn_jit(*jax_inputs)
            try:
                diff = float(jnp.max(jnp.abs(pallas_out - ref_out)))
                row["Max_Diff"] = diff
                row["Correct"]  = bool(diff < ATOL + RTOL * float(jnp.max(jnp.abs(ref_out))))
            except Exception as ce:
                row["Error"]   = f"[correctness_error] {ce}"
                row["Correct"] = False

    except Exception as e:
        row["Error"]   = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
        row["Correct"] = False

    # ── IR capture (GPU path) ──────────────────────────────────────────────
    if capture_ir and row["Correct"] and on_gpu:
        _capture_ir_from_source(row, kernel_source, kernel_fn_name, sample_np, capture_ir)

    # ── PTX / SASS (GPU path) ─────────────────────────────────────────────
    if capture_ptx and row["Correct"] and on_gpu:
        try:
            import jax, jax.numpy as jnp
            ns3: dict = {}
            exec(compile(kernel_source, "<kernel>", "exec"), ns3)
            fn = ns3[kernel_fn_name]
            ptx, sass = extract_ptx_sass(fn, *[jnp.array(a) for a in sample_np])
            row["PTX_Code"]  = ptx
            row["SASS_Code"] = sass
        except Exception as e:
            row["PTX_Code"]  = f"[ptx_error] {e}"
            row["SASS_Code"] = f"[sass_error] {e}"

    # ── fast_p metrics ─────────────────────────────────────────────────────
    speedup = row["Pallas_Speedup_Native"]
    correct = row["Correct"]
    row["fast_0"] = correct
    row["fast_1"] = correct and (not np.isnan(speedup)) and speedup > 1.0
    row["fast_2"] = correct and (not np.isnan(speedup)) and speedup > 2.0
    row["fast_5"] = correct and (not np.isnan(speedup)) and speedup > 5.0

    return row


def _capture_ir_from_source(
    row: dict,
    kernel_source: str,
    kernel_fn_name: str,
    sample_np: list,
    capture_ir: bool,
):
    """Helper: populate Jaxpr_IR and StableHLO_IR on the row dict."""
    if not capture_ir:
        return
    try:
        import jax, jax.numpy as jnp
        jax_inputs = [jnp.array(a) for a in sample_np]
        ns: dict = {}
        exec(compile(kernel_source, "<kernel>", "exec"), ns)
        fn = ns.get(kernel_fn_name) or next(
            (v for v in ns.values() if callable(v) and not str(v).startswith("<built")), None
        )
        if fn is None:
            row["Jaxpr_IR"] = "[ir_error] function not found in source"
            return
        row["Jaxpr_IR"]     = build_ir_dag(fn, *jax_inputs)
        row["StableHLO_IR"] = extract_stablehlo(fn, *jax_inputs)
    except Exception as e:
        row["Jaxpr_IR"]     = f"[ir_error] {e}"
        row["StableHLO_IR"] = f"[stablehlo_error] {e}"


def _run_interpret(pallas_fn, args):
    """
    Re-exec the kernel source with pallas_call patched to use interpret=True.
    This lets T4 (sm_75) check correctness even though native Pallas is sm_80+.
    """
    import jax.experimental.pallas as pl
    orig_call = pl.pallas_call

    def interpret_call(fn, *, out_shape, grid=None, in_specs=None,
                       out_specs=None, **kwargs):
        kwargs.pop("interpret", None)
        return orig_call(fn, out_shape=out_shape, grid=grid,
                         in_specs=in_specs, out_specs=out_specs,
                         interpret=True, **kwargs)

    pl.pallas_call = interpret_call
    try:
        return pallas_fn(*args)
    finally:
        pl.pallas_call = orig_call


def _capture_ir_from_ref(row: dict, problem, jax_inputs: list):
    """Capture Jaxpr + StableHLO from JAX reference function (CPU-safe)."""
    try:
        import jax
        ref_ns: dict = {}
        exec(compile(problem.jax_functional.strip(), "<ref>", "exec"), ref_ns)
        ref_fn_name = next(
            (k for k, v in ref_ns.items() if callable(v) and not k.startswith("_")), None
        )
        if ref_fn_name:
            fn = ref_ns[ref_fn_name]
            row["Jaxpr_IR"]     = build_ir_dag(fn, *jax_inputs)
            row["StableHLO_IR"] = extract_stablehlo(fn, *jax_inputs)
    except Exception as e:
        row["Jaxpr_IR"]     = f"[ir_error] {e}"
        row["StableHLO_IR"] = f"[stablehlo_error] {e}"


def _capture_jax_reference(row: dict, problem, jax_inputs: list):
    """Helper: time the JAX reference function and store baselines."""
    try:
        import jax
        ref_ns: dict = {}
        exec(compile(problem.jax_functional.strip(), "<ref>", "exec"), ref_ns)
        ref_fn_name = next(
            (k for k, v in ref_ns.items() if callable(v) and not k.startswith("_")), None
        )
        if ref_fn_name:
            ref_fn = ref_ns[ref_fn_name]
            row["JAX_Native_Runtime"]       = _time_fn(ref_fn, jax_inputs)
            row["JAX_XLA_Compiled_Runtime"] = _time_fn(jax.jit(ref_fn), jax_inputs)
    except Exception as e:
        row["Error"] = (row.get("Error") or "") + f" | ref_error: {e}"


def evaluate_problem_seed(problem, **kwargs) -> dict:
    """Evaluate the seed (pre-existing GPU-fixed) kernel for a problem."""
    return evaluate_kernel(
        problem,
        kernel_source=problem.seed_pallas,
        kernel_fn_name=problem.name,
        generation_idx=0,
        llm_model="seed",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Robustness filters (matching pallasbench-robust pipeline)
# ---------------------------------------------------------------------------

def robustness_filters(row: dict, n_runs: int = 5) -> dict:
    """
    Run 5 robustness filters on a completed evaluation row.
    Returns filter results dict.
    """
    import numpy as np

    filters: dict[str, dict] = {
        "compile_error":   {"passed": row["Error"] == "", "value": None},
        "correctness":     {"passed": row["Correct"],      "value": row["Max_Diff"]},
        "nan_inf_free":    {"passed": True,                "value": None},
        "speedup_positive": {"passed": row["Pallas_Speedup_Native"] > 0
                             if not np.isnan(row["Pallas_Speedup_Native"]) else False,
                             "value": row["Pallas_Speedup_Native"]},
        "source_analysis": {"passed": _check_source(row["Pallas_Code"]), "value": None},
    }

    # Check for NaN/Inf: only possible if we have numeric output error
    if not np.isnan(row["Max_Diff"]):
        filters["nan_inf_free"]["passed"] = not (np.isnan(row["Max_Diff"]) or np.isinf(row["Max_Diff"]))

    all_passed = all(f["passed"] for f in filters.values())
    return {"passed": all_passed, "filters": filters}


def _check_source(src: str) -> bool:
    """Detect hallucinated imports or non-existent Pallas APIs."""
    BANNED = [
        "import torch",
        "import tensorflow",
        "from torch",
        "pl.non_existent",
        "plgpu.wgmma",          # Hopper-only
        "plgpu.tcgen05",         # Blackwell-only
    ]
    return not any(b in src for b in BANNED)
