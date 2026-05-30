from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class FilterResult:
    name: str
    passed: bool
    value: Any
    threshold: Any
    reason: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self._serialize_value(self.value),
            "threshold": self._serialize_value(self.threshold),
            "reason": self.reason,
        }

    @staticmethod
    def _serialize_value(v: Any) -> Any:
        if isinstance(v, (jnp.ndarray, jnp.generic)):
            return float(v)
        if isinstance(v, tuple):
            return tuple(
                float(x) if isinstance(x, (jnp.ndarray, jnp.generic)) else x
                for x in v
            )
        if isinstance(v, list):
            return [
                float(x) if isinstance(x, (jnp.ndarray, jnp.generic)) else x
                for x in v
            ]
        return v


@dataclass
class OutputRangeFilter:
    min_range: float = 1e-6
    max_range: float = 1e12

    def check(self, output: jnp.ndarray) -> FilterResult:
        out_range = float(jnp.max(output) - jnp.min(output))
        passed = self.min_range <= out_range <= self.max_range
        return FilterResult(
            name="output_range",
            passed=passed,
            value=out_range,
            threshold=(self.min_range, self.max_range),
            reason=(
                f"Output range {out_range:.6e} "
                f"{'within' if passed else 'outside'} "
                f"[{self.min_range:.0e}, {self.max_range:.0e}]"
            ),
        )


@dataclass
class OutputStdFilter:
    min_std: float = 1e-6

    def check(self, output: jnp.ndarray) -> FilterResult:
        std = float(jnp.std(output))
        passed = std >= self.min_std
        return FilterResult(
            name="output_std",
            passed=passed,
            value=std,
            threshold=self.min_std,
            reason=(
                f"Output std {std:.6e} "
                f"{'above' if passed else 'below'} {self.min_std:.0e}"
            ),
        )


@dataclass
class AxesVariationFilter:
    min_axis_std: float = 1e-6

    def check(self, output: jnp.ndarray) -> FilterResult:
        if output.ndim < 2:
            return FilterResult(
                "axes_variation", True, None, None, "Skipped for 1D output"
            )
        failures = []
        for axis in range(output.ndim):
            axis_std = float(jnp.std(output, axis=axis).mean())
            if axis_std < self.min_axis_std:
                failures.append(f"axis {axis}: std={axis_std:.6e}")
        passed = len(failures) == 0
        return FilterResult(
            name="axes_variation",
            passed=passed,
            value=failures if failures else "all axes vary",
            threshold=self.min_axis_std,
            reason=(
                "All axes vary"
                if passed
                else "Constant along: " + ", ".join(failures)
            ),
        )


@dataclass
class InputImpactFilter:
    perturbation_scale: float = 0.1
    min_output_change: float = 1e-6

    def check(
        self,
        kernel_fn,
        reference_input: jnp.ndarray,
        reference_output: jnp.ndarray,
    ) -> FilterResult:
        try:
            key = jax.random.PRNGKey(42)
            perturbed = reference_input + self.perturbation_scale * (
                jax.random.normal(key, reference_input.shape).astype(
                    reference_input.dtype
                )
            )
            perturbed_output = kernel_fn(perturbed)
            diff = float(jnp.max(jnp.abs(perturbed_output - reference_output)))
        except Exception:
            return FilterResult(
                name="input_impact",
                passed=False,
                value=None,
                threshold=self.min_output_change,
                reason="Failed to compute perturbed output",
            )
        passed = diff >= self.min_output_change
        return FilterResult(
            name="input_impact",
            passed=passed,
            value=diff,
            threshold=self.min_output_change,
            reason=(
                f"Max output change {diff:.6e} "
                f"{'above' if passed else 'below'} {self.min_output_change:.0e}"
            ),
        )


@dataclass
class SourceAnalysisFilter:
    degenerate_patterns: list = field(
        default_factory=lambda: [
            r"o_ref\[\.\.\.?\]\s*=\s*0",
            r"o_ref\[\.\.\.?\]\s*=\s*1",
            r"return\s+jnp\.zeros",
            r"jnp\.full\(.*,\s*0\)",
        ]
    )

    def check(self, source: str) -> FilterResult:
        matches = []
        for pattern in self.degenerate_patterns:
            if re.search(pattern, source):
                matches.append(pattern)
        passed = len(matches) == 0
        return FilterResult(
            name="source_analysis",
            passed=passed,
            value=matches if matches else "no degenerate patterns",
            threshold=None,
            reason=(
                "No degenerate patterns"
                if passed
                else f"Found {len(matches)} degenerate pattern(s)"
            ),
        )


def apply_all_filters(
    output: jnp.ndarray,
    kernel_fn,
    input_data: jnp.ndarray,
    reference_output: jnp.ndarray,
    source: str | None = None,
) -> dict[str, FilterResult]:
    results = {}
    results["output_range"] = OutputRangeFilter().check(output)
    results["output_std"] = OutputStdFilter().check(output)
    results["axes_variation"] = AxesVariationFilter().check(output)
    results["input_impact"] = InputImpactFilter().check(
        kernel_fn, input_data, reference_output
    )
    if source is not None:
        results["source_analysis"] = SourceAnalysisFilter().check(source)
    return results
