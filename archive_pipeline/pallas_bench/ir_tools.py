"""
JAX IR extraction tools for the PallasBench archive.

Captures: Jaxpr (with shapes), StableHLO, PTX (T4/sm_75), SASS,
MLIR pass dumps, and NCU profiler data.  All tools degrade gracefully
if JAX-CUDA or ncu is not available.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Jaxpr IR — shape-annotated computation graph
# ---------------------------------------------------------------------------

def extract_jaxpr(fn: Callable, *args, **kwargs) -> str:
    """
    Return the Jaxpr of fn(*args) as a formatted string.

    The output includes all shapes and dtypes so it can serve as a
    shape-annotated IR-DAG for the archive column Jaxpr_IR.
    """
    try:
        import jax
        jaxpr_obj = jax.make_jaxpr(fn)(*args, **kwargs)
        return str(jaxpr_obj)
    except Exception as e:
        return f"[jaxpr_error] {type(e).__name__}: {e}"


def extract_jaxpr_from_source(
    kernel_source: str,
    fn_name: str,
    sample_args: list[Any],
) -> str:
    """
    Exec kernel_source, locate fn_name, and extract its Jaxpr.
    Returns the Jaxpr string or an error tag.
    """
    try:
        ns: dict = {}
        exec(compile(kernel_source, "<kernel>", "exec"), ns)
        fn = ns[fn_name]
        return extract_jaxpr(fn, *sample_args)
    except Exception as e:
        return f"[jaxpr_error] {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# StableHLO — portable MLIR representation
# ---------------------------------------------------------------------------

def extract_stablehlo(fn: Callable, *args, **kwargs) -> str:
    """
    Return the StableHLO text representation using jax.export.
    Falls back to XLA HLO text if jax.export is unavailable.
    """
    try:
        import jax
        import jax.numpy as jnp

        # Build abstract shapes from concrete args
        abstract_args = [
            jax.ShapeDtypeStruct(a.shape, a.dtype)
            for a in args
            if hasattr(a, "shape")
        ]

        # jax.export produces StableHLO in JAX >= 0.4.25
        try:
            exported = jax.export.export(jax.jit(fn))(*abstract_args)
            return exported.mlir_module_serialized.decode("utf-8", errors="replace")
        except AttributeError:
            pass

        # Fallback: use jax.xla_computation for HLO text
        try:
            comp = jax.xla_computation(fn)(*args, **kwargs)
            return comp.as_hlo_text()
        except Exception as e2:
            return f"[hlo_error] {e2}"

    except Exception as e:
        return f"[stablehlo_error] {type(e).__name__}: {e}"


def extract_stablehlo_from_source(
    kernel_source: str,
    fn_name: str,
    sample_args: list[Any],
) -> str:
    try:
        ns: dict = {}
        exec(compile(kernel_source, "<kernel>", "exec"), ns)
        fn = ns[fn_name]
        return extract_stablehlo(fn, *sample_args)
    except Exception as e:
        return f"[stablehlo_error] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# PTX / SASS — via Mosaic GPU environment variable dumps
# ---------------------------------------------------------------------------

_PTX_ENV  = "MOSAIC_GPU_DUMP_PTX"
_SASS_ENV = "MOSAIC_GPU_DUMP_SASS"
_MLIR_ENV = "MOSAIC_GPU_DUMP_MLIR_PASSES"
_PTXAS_ENV = "MOSAIC_GPU_DUMP_PTXAS"


@contextlib.contextmanager
def _capture_mosaic_dumps(dump_dir: str):
    """
    Context manager: set Mosaic dump env vars pointing to dump_dir,
    restore on exit.
    """
    old = {k: os.environ.get(k) for k in (_PTX_ENV, _SASS_ENV, _MLIR_ENV, _PTXAS_ENV)}
    os.environ[_PTX_ENV]   = "1"
    os.environ[_SASS_ENV]  = "1"
    os.environ[_PTXAS_ENV] = "1"
    os.environ["MOSAIC_GPU_DUMP_TO"] = dump_dir
    try:
        yield dump_dir
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop("MOSAIC_GPU_DUMP_TO", None)


def extract_ptx_sass(
    fn: Callable,
    *args,
    want_ptx: bool = True,
    want_sass: bool = True,
) -> tuple[str, str]:
    """
    Run fn(*args) once with Mosaic PTX/SASS dump env vars set.
    Returns (ptx_text, sass_text).  Both may be empty strings if
    JAX-CUDA / Mosaic is not available.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="pb_ptx_") as dump_dir:
            with _capture_mosaic_dumps(dump_dir):
                import jax
                jax.jit(fn)(*args)  # trigger compilation

            ptx_text  = ""
            sass_text = ""
            for f in Path(dump_dir).glob("*.ptx"):
                ptx_text += f.read_text(errors="replace") + "\n"
            for f in Path(dump_dir).glob("*.sass"):
                sass_text += f.read_text(errors="replace") + "\n"
            return ptx_text or "[ptx_unavailable]", sass_text or "[sass_unavailable]"
    except Exception as e:
        return f"[ptx_error] {e}", f"[sass_error] {e}"


# ---------------------------------------------------------------------------
# Block / Grid shape extractor
# ---------------------------------------------------------------------------

def extract_block_grid(kernel_source: str) -> tuple[str, str]:
    """
    Heuristically parse block_shape and grid dimensions from Pallas kernel
    source code.  Returns (block_json, grid_json) as JSON strings.

    Looks for: pl.BlockSpec((...), ...), grid=(...), BM/BN/BK constants.
    """
    block_dims: list = []
    grid_dims:  list = []

    # Match BlockSpec tuple arguments
    for m in re.finditer(r'BlockSpec\(\s*\(([^)]+)\)', kernel_source):
        raw = m.group(1)
        try:
            dims = [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]
            if dims:
                block_dims.append(dims)
        except ValueError:
            pass

    # Match grid=(...) tuple
    m = re.search(r'grid\s*=\s*\(([^)]+)\)', kernel_source)
    if m:
        raw = m.group(1)
        for token in raw.split(","):
            token = token.strip()
            # Resolve simple BM//BN patterns — just capture the variable names
            if re.match(r'^[A-Z0-9_]+$', token):
                # Constant like BM, BN
                vm = re.search(rf'\b{token}\s*=\s*(\d+)', kernel_source)
                if vm:
                    grid_dims.append(int(vm.group(1)))
            elif token.isdigit():
                grid_dims.append(int(token))

    return json.dumps(block_dims or []), json.dumps(grid_dims or [])


# ---------------------------------------------------------------------------
# NCU profiler wrapper (T4 / sm_75)
# ---------------------------------------------------------------------------

_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed",
]

_NCU_CMD_TEMPLATE = (
    "ncu --metrics {metrics} "
    "--csv --quiet "
    "--target-processes all "
    "{binary}"
)


def run_ncu_on_script(
    script_path: str,
    python: str = sys.executable,
    extra_metrics: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """
    Run ncu on a Python script and return the CSV profiling data as JSON.
    Returns an empty-profile JSON string if ncu is not available.
    """
    metrics = _NCU_METRICS + (extra_metrics or [])
    metrics_str = ",".join(metrics)
    binary = f"{python} {script_path}"
    cmd = _NCU_CMD_TEMPLATE.format(metrics=metrics_str, binary=binary)

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.dumps({
                "raw_csv": result.stdout,
                "stderr": result.stderr[:500] if result.stderr else "",
                "metrics_requested": metrics,
            })
        else:
            return json.dumps({
                "error": f"ncu returned {result.returncode}",
                "stderr": result.stderr[:500],
            })
    except FileNotFoundError:
        return json.dumps({"error": "ncu not found on PATH"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"ncu timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def ncu_profile_kernel(
    kernel_source: str,
    fn_name: str,
    sample_args: list[Any],
    timeout: int = 120,
) -> str:
    """
    Write a self-contained profiling script to a temp file and run ncu on it.
    Returns JSON NCU profile string.
    """
    arg_reprs = []
    import numpy as np
    for a in sample_args:
        if hasattr(a, "shape"):
            arg_reprs.append(
                f"jnp.ones({list(a.shape)}, dtype=jnp.{a.dtype})"
                if hasattr(a, "dtype") else f"jnp.ones({list(a.shape)})"
            )
        elif isinstance(a, (int, float)):
            arg_reprs.append(repr(a))
        else:
            arg_reprs.append("None")

    script = textwrap.dedent(f"""
import sys; sys.path.insert(0, "{Path(__file__).parent.parent}")
import jax, jax.numpy as jnp
{kernel_source}
args = [{", ".join(arg_reprs)}]
fn = jax.jit({fn_name})
for _ in range(3):
    fn(*args).block_until_ready()
""")
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="pb_ncu_"
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        return run_ncu_on_script(tmp_path, timeout=timeout)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# JAX profiler (Chrome trace)
# ---------------------------------------------------------------------------

def profile_kernel(
    fn: Callable,
    *args,
    n_warmup: int = 3,
    n_profile: int = 5,
) -> str:
    """
    Run JAX profiler and return a summary dict as JSON.
    Returns timing summary; full Chrome trace is too large for dataset rows.
    """
    try:
        import jax, time
        jfn = jax.jit(fn)

        # warmup
        for _ in range(n_warmup):
            jfn(*args).block_until_ready() if hasattr(jfn(*args), "block_until_ready") else None

        times = []
        for _ in range(n_profile):
            t0 = time.perf_counter()
            out = jfn(*args)
            if hasattr(out, "block_until_ready"):
                out.block_until_ready()
            times.append((time.perf_counter() - t0) * 1e3)

        return json.dumps({
            "mean_ms": sum(times) / len(times),
            "min_ms":  min(times),
            "max_ms":  max(times),
            "n_trials": n_profile,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Shape-annotated IR-DAG builder
# ---------------------------------------------------------------------------

def build_ir_dag(fn: Callable, *args) -> str:
    """
    Build a human-readable IR-DAG showing node → shape/dtype annotations.
    Uses Jaxpr equations as nodes and annotates each with abstract values.

    Returns a multi-line string suitable for the Jaxpr_IR archive column.
    """
    try:
        import jax
        from jax._src import core as jax_core

        closed_jaxpr = jax.make_jaxpr(fn)(*args)
        lines = ["# IR-DAG (Jaxpr) with shape annotations"]
        lines.append(f"# Inputs:")
        for v in closed_jaxpr.jaxpr.invars:
            lines.append(f"#   {v}: {v.aval.shape} {v.aval.dtype}")
        lines.append(f"# Equations:")
        for eq in closed_jaxpr.jaxpr.eqns:
            outs = ", ".join(
                f"{v}:{v.aval.shape}{v.aval.dtype}"
                if hasattr(v, "aval") and hasattr(v.aval, "shape")
                else str(v)
                for v in eq.outvars
            )
            ins = ", ".join(str(v) for v in eq.invars)
            lines.append(f"  [{outs}] = {eq.primitive.name}({ins})")
        lines.append(f"# Outputs:")
        for v in closed_jaxpr.jaxpr.outvars:
            lines.append(f"#   {v}: {v.aval.shape} {v.aval.dtype}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ir_dag_error] {e}"
