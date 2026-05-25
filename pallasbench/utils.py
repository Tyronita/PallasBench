"""Timing, correctness checking, and RNG utilities for PallasBench."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class BenchmarkResult:
    task_name: str
    correct: bool
    baseline_time_ms: float
    kernel_time_ms: float
    speedup: float
    errors: list[str]

    @property
    def passes_fast_p(self) -> Callable[[float], bool]:
        return lambda p: self.correct and self.speedup > p


def generate_inputs(
    shapes: Sequence[tuple[int, ...]],
    dtype: str = "float32",
    seed: int = 0,
) -> list[jax.Array]:
    key = jax.random.PRNGKey(seed)
    inputs = []
    for shape in shapes:
        key, subkey = jax.random.split(key)
        inputs.append(jax.random.normal(subkey, shape, dtype=dtype))
    return inputs


def check_correctness(
    pallas_fn: Callable,
    baseline_fn: Callable,
    input_shapes: Sequence[tuple[int, ...]],
    dtype: str = "float32",
    n_checks: int = 5,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[bool, list[str]]:
    errors = []
    for seed in range(n_checks):
        inputs = generate_inputs(input_shapes, dtype=dtype, seed=seed)
        try:
            out_pallas = pallas_fn(*inputs)
            jax.block_until_ready(out_pallas)
        except Exception as e:
            errors.append(f"Pallas execution error (seed={seed}): {e}")
            continue

        out_baseline = baseline_fn(*inputs)
        jax.block_until_ready(out_baseline)

        if out_pallas.shape != out_baseline.shape:
            errors.append(
                f"Shape mismatch (seed={seed}): "
                f"{out_pallas.shape} vs {out_baseline.shape}"
            )
            continue

        if not jnp.allclose(out_pallas, out_baseline, atol=atol, rtol=rtol):
            max_diff = float(jnp.max(jnp.abs(out_pallas - out_baseline)))
            errors.append(
                f"Value mismatch (seed={seed}): max_diff={max_diff:.6f}"
            )

    return len(errors) == 0, errors


def time_fn(
    fn: Callable,
    inputs: list[jax.Array],
    n_warmup: int = 10,
    n_trials: int = 100,
) -> float:
    for _ in range(n_warmup):
        out = fn(*inputs)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        out = fn(*inputs)
        jax.block_until_ready(out)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return float(np.median(times))
