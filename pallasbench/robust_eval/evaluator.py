from __future__ import annotations

import importlib.util
import inspect
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from pallasbench.robust_eval.filters import (
    FilterResult,
    apply_all_filters,
)
from pallasbench.robust_eval.ir_capture import capture_all_ir
from pallasbench.robust_eval.tiling import analyze_tiling


@dataclass
class KernelEvalResult:
    kernel_name: str
    level: int
    category: str

    correctness_passed: bool
    max_abs_error: float
    max_rel_error: float
    allclose_atol: float
    allclose_rtol: float

    robustness_filters: dict[str, FilterResult]
    robustness_passed: bool

    wall_time_seconds: float
    jit_compile_time_seconds: float
    kernel_exec_time_seconds: float
    throughput_gflops: float | None
    bandwidth_utilization_pct: float | None
    gpu_memory_used_bytes: int

    block_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    num_blocks: int

    jaxpr_dag: str
    stablehlo_ir: str
    triton_mlir: str | None

    hardware: str
    jax_version: str
    timestamp: str

    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["robustness_filters"] = {
            k: v.to_dict() for k, v in d["robustness_filters"].items()
        }
        d["block_shape"] = list(d["block_shape"])
        d["grid_shape"] = list(d["grid_shape"])
        return d


def _load_kernel_source(kernel_module) -> str:
    try:
        return inspect.getsource(kernel_module)
    except (TypeError, OSError):
        return "<source not available>"


def _get_hardware_info() -> str:
    try:
        devices = jax.devices()
        if devices:
            return str(devices[0])
        return "unknown"
    except Exception:
        return "unknown"


def _get_gpu_memory() -> int:
    try:
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        if stats:
            return stats.get("used_bytes", 0)
        return 0
    except Exception:
        return 0


def _estimate_flops(
    output_shape: tuple[int, ...], kernel_source: str
) -> int:
    total_elems = max(1, _prod(output_shape))
    op_count = kernel_source.count("jnp.") + kernel_source.count("jax.")
    return total_elems * max(op_count, 1)


def _prod(shape: Sequence[int]) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


def _estimate_bandwidth_utilization(
    output: jnp.ndarray, kernel_exec_time_seconds: float
) -> float | None:
    if kernel_exec_time_seconds <= 0:
        return None
    try:
        num_bytes = output.nbytes
        bw = num_bytes / kernel_exec_time_seconds
        peak_bw = 2039e9
        return (bw / peak_bw) * 100.0
    except Exception:
        return None


def _try_get_block_grid(kernel_module) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        block = getattr(kernel_module, "BLOCK_SIZE", None)
        grid = getattr(kernel_module, "GRID_SIZE", None)
        if block is not None:
            block_shape = (block,) if isinstance(block, int) else tuple(block)
        else:
            block_shape = getattr(kernel_module, "block_shape", (1,))
            if callable(block_shape):
                block_shape = (1,)
        if grid is not None:
            grid_shape = (grid,) if isinstance(grid, int) else tuple(grid)
        else:
            grid_shape = getattr(kernel_module, "grid_shape", (1,))
            if callable(grid_shape):
                grid_shape = (1,)
        return tuple(block_shape), tuple(grid_shape)
    except Exception:
        return (1,), (1,)


def evaluate_kernel(
    kernel_module,
    kernel_fn: Callable,
    baseline_fn: Callable,
    input_shapes: Sequence[tuple[int, ...]],
    task_name: str = "unnamed",
    level: int = 1,
    category: str = "unknown",
    dtype: str = "float32",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    input_dtypes: Sequence[str] | None = None,
    input_ranges: Sequence[tuple[float, float] | None] | None = None,
    seed: int = 0,
) -> KernelEvalResult:
    timestamp = datetime.now(timezone.utc).isoformat()
    hardware = _get_hardware_info()
    jax_version = jax.__version__

    input_dtype_list = list(input_dtypes) if input_dtypes is not None else [dtype] * len(input_shapes)
    input_range_list = list(input_ranges) if input_ranges is not None else [None] * len(input_shapes)

    try:
        key = jax.random.PRNGKey(seed)
        inputs = []
        for i, shape in enumerate(input_shapes):
            key, subkey = jax.random.split(key)
            dt = input_dtype_list[i]
            inputs.append(jax.random.normal(subkey, shape, dtype=jnp.dtype(dt)))
    except Exception as e:
        return KernelEvalResult(
            kernel_name=task_name,
            level=level,
            category=category,
            correctness_passed=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            allclose_atol=atol,
            allclose_rtol=rtol,
            robustness_filters={},
            robustness_passed=False,
            wall_time_seconds=0.0,
            jit_compile_time_seconds=0.0,
            kernel_exec_time_seconds=0.0,
            throughput_gflops=None,
            bandwidth_utilization_pct=None,
            gpu_memory_used_bytes=0,
            block_shape=(1,),
            grid_shape=(1,),
            num_blocks=1,
            jaxpr_dag="<input generation failed>",
            stablehlo_ir="<input generation failed>",
            triton_mlir=None,
            hardware=hardware,
            jax_version=jax_version,
            timestamp=timestamp,
            error=str(e),
        )

    source = _load_kernel_source(kernel_module)

    jaxpr_str = ""
    stablehlo_str = ""
    triton_str: str | None = None
    try:
        irs = capture_all_ir(kernel_fn, *inputs)
        jaxpr_str = irs.get("jaxpr", "")
        stablehlo_str = irs.get("stablehlo", "")
        triton_str = irs.get("triton_mlir")
    except Exception:
        pass

    jit_start = time.perf_counter()
    try:
        jitted_fn = jax.jit(kernel_fn)
        output = jitted_fn(*inputs)
        jax.block_until_ready(output)
    except Exception as e:
        return KernelEvalResult(
            kernel_name=task_name,
            level=level,
            category=category,
            correctness_passed=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            allclose_atol=atol,
            allclose_rtol=rtol,
            robustness_filters={},
            robustness_passed=False,
            wall_time_seconds=0.0,
            jit_compile_time_seconds=0.0,
            kernel_exec_time_seconds=0.0,
            throughput_gflops=None,
            bandwidth_utilization_pct=None,
            gpu_memory_used_bytes=0,
            block_shape=(1,),
            grid_shape=(1,),
            num_blocks=1,
            jaxpr_dag=jaxpr_str,
            stablehlo_ir=stablehlo_str,
            triton_mlir=triton_str,
            hardware=hardware,
            jax_version=jax_version,
            timestamp=timestamp,
            error=f"Kernel execution failed: {e}",
        )
    jit_end = time.perf_counter()
    jit_compile_time = jit_end - jit_start

    exec_start = time.perf_counter()
    try:
        output = jitted_fn(*inputs)
        jax.block_until_ready(output)
    except Exception as e:
        return KernelEvalResult(
            kernel_name=task_name,
            level=level,
            category=category,
            correctness_passed=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            allclose_atol=atol,
            allclose_rtol=rtol,
            robustness_filters={},
            robustness_passed=False,
            wall_time_seconds=jit_compile_time,
            jit_compile_time_seconds=jit_compile_time,
            kernel_exec_time_seconds=0.0,
            throughput_gflops=None,
            bandwidth_utilization_pct=None,
            gpu_memory_used_bytes=0,
            block_shape=(1,),
            grid_shape=(1,),
            num_blocks=1,
            jaxpr_dag=jaxpr_str,
            stablehlo_ir=stablehlo_str,
            triton_mlir=triton_str,
            hardware=hardware,
            jax_version=jax_version,
            timestamp=timestamp,
            error=f"Warm execution failed: {e}",
        )
    exec_end = time.perf_counter()
    kernel_exec_time = exec_end - exec_start
    wall_time = jit_compile_time + kernel_exec_time

    gpu_memory = _get_gpu_memory()

    try:
        ref_output = baseline_fn(*inputs)
        jax.block_until_ready(ref_output)
    except Exception as e:
        return KernelEvalResult(
            kernel_name=task_name,
            level=level,
            category=category,
            correctness_passed=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            allclose_atol=atol,
            allclose_rtol=rtol,
            robustness_filters={},
            robustness_passed=False,
            wall_time_seconds=wall_time,
            jit_compile_time_seconds=jit_compile_time,
            kernel_exec_time_seconds=kernel_exec_time,
            throughput_gflops=None,
            bandwidth_utilization_pct=None,
            gpu_memory_used_bytes=gpu_memory,
            block_shape=(1,),
            grid_shape=(1,),
            num_blocks=1,
            jaxpr_dag=jaxpr_str,
            stablehlo_ir=stablehlo_str,
            triton_mlir=triton_str,
            hardware=hardware,
            jax_version=jax_version,
            timestamp=timestamp,
            error=f"Baseline execution failed: {e}",
        )

    abs_err = float(jnp.max(jnp.abs(output - ref_output)))
    rel_err = float(
        jnp.max(
            jnp.abs(output - ref_output)
            / (jnp.abs(ref_output) + 1e-15)
        )
    )
    correctness_passed = bool(
        jnp.allclose(output, ref_output, atol=atol, rtol=rtol)
    )

    num_elements = _prod(output.shape)
    block_shape, grid_shape = _try_get_block_grid(kernel_module)
    num_blocks = _prod(grid_shape)

    try:
        kernel_fn_for_filter = lambda x: jax.jit(kernel_fn)(x)
        filters = apply_all_filters(
            output, kernel_fn_for_filter, inputs[0], ref_output, source
        )
    except Exception:
        filters = {}
    robustness_passed = all(
        f.passed for f in filters.values()
    ) if filters else False

    throughput = None
    if kernel_exec_time > 0 and correctness_passed:
        flops = _estimate_flops(output.shape, source)
        throughput = flops / kernel_exec_time / 1e9

    bw_util = _estimate_bandwidth_utilization(output, kernel_exec_time)

    return KernelEvalResult(
        kernel_name=task_name,
        level=level,
        category=category,
        correctness_passed=correctness_passed,
        max_abs_error=abs_err,
        max_rel_error=rel_err,
        allclose_atol=atol,
        allclose_rtol=rtol,
        robustness_filters=filters,
        robustness_passed=robustness_passed,
        wall_time_seconds=wall_time,
        jit_compile_time_seconds=jit_compile_time,
        kernel_exec_time_seconds=kernel_exec_time,
        throughput_gflops=throughput,
        bandwidth_utilization_pct=bw_util,
        gpu_memory_used_bytes=gpu_memory,
        block_shape=block_shape,
        grid_shape=grid_shape,
        num_blocks=num_blocks,
        jaxpr_dag=jaxpr_str,
        stablehlo_ir=stablehlo_str,
        triton_mlir=triton_str,
        hardware=hardware,
        jax_version=jax_version,
        timestamp=timestamp,
    )
