"""Fitness evaluation for Pallas kernel evolution.

Defines :class:`PallasFitnessFunction`, which wraps the robust evaluation
pipeline to compute a scalar fitness score (correctness + speedup) for a
candidate kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FitnessReport:
    """Result of evaluating a single candidate kernel.

    Attributes
    ----------
    candidate_id : str
        Unique identifier within the experiment.
    parent_id : str | None
        Source candidate (``None`` for generation 0).
    correctness_passed : bool
        Whether the kernel passed all correctness checks (allclose + robustness).
    max_abs_error : float
        Maximum absolute error against the JAX baseline.
    robustness_filters : dict[str, dict[str, Any]]
        Per-filter results keyed by filter name.
    speedup : float
        Baseline wall-time / kernel wall-time (0.0 if correctness failed).
    baseline_time_ms : float
        JAX baseline execution time in milliseconds.
    kernel_time_ms : float
        Pallas kernel execution time in milliseconds.
    jit_compile_time_s : float
        First-run JIT compilation time in seconds.
    source : str
        The kernel source code that was evaluated.
    mutation_info : dict[str, Any] | None
        Mutation operator and rationale used to produce this candidate.
    extra : dict[str, Any] = field(default_factory=dict)
        Catch-all for additional metrics.
    """

    candidate_id: str
    parent_id: str | None
    correctness_passed: bool
    max_abs_error: float
    robustness_filters: dict[str, dict[str, Any]]
    speedup: float
    baseline_time_ms: float
    kernel_time_ms: float
    jit_compile_time_s: float
    source: str
    mutation_info: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class PallasFitnessFunction:
    """Fitness function for evolving Pallas kernels.

    Evaluates a candidate kernel for:
      1. Correctness  — output matches JAX baseline (allclose + robustness)
      2. Performance  — speedup over JAX baseline

    Parameters
    ----------
    task_name : str
        PallasBench task name (e.g., ``"L1/relu"``).
    seed : int
        Random seed for input generation.
    atol : float
        Absolute tolerance for output comparison.
    rtol : float
        Relative tolerance for output comparison.
    """

    def __init__(
        self,
        task_name: str,
        *,
        seed: int = 42,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.atol = atol
        self.rtol = rtol
        self._baseline_fn = None
        self._benchmark_input = None

    def evaluate(self, source: str) -> FitnessReport:
        """Evaluate a single candidate kernel source.

        Parameters
        ----------
        source : str
            Full Python source of the Pallas kernel.

        Returns
        -------
        FitnessReport
            Structured evaluation result.
        """
        _ = source
        msg = (
            f"PallasFitnessFunction.evaluate() is a stub. "
            f"Task: {self.task_name}, atol={self.atol}, rtol={self.rtol}. "
            f"To implement: compile source via jax.jit, run on benchmark input, "
            f"compare output to JAX baseline, apply robustness filters, "
            f"measure timing, return FitnessReport."
        )
        raise NotImplementedError(msg)
