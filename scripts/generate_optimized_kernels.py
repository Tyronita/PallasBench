#!/usr/bin/env python3
"""Generate optimized Pallas kernel candidates using LLMs.

Generates candidate optimized versions of PallasBench kernels by prompting
LLMs (OpenAI, Anthropic, or open-source via vLLM). Each candidate is saved
as a structured JSON entry for downstream evaluation.

Usage:
    python scripts/generate_optimized_kernels.py \\
        --model gpt-4o \\
        --kernels L1/relu L1/matmul \\
        --num-candidates-per-kernel 5 \\
        --output-dir data/generated

    python scripts/generate_optimized_kernels.py \\
        --model claude-opus-4 \\
        --kernels all \\
        --num-candidates-per-kernel 3 \\
        --temperature 0.8

    python scripts/generate_optimized_kernels.py \\
        --model deepseek-coder-v2 \\
        --kernels L2/swiglu L3/flash_attention \\
        --api-base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _resolve_kernels(kernel_names: list[str]) -> list[dict]:
    """Resolve kernel names from the PallasBench task registry.

    In production, this imports TASK_REGISTRY from pallasbench.tasks
    and filters by name. For now returns a placeholder.
    """
    if "all" in kernel_names:
        return _get_all_kernels()
    return [{"name": name} for name in kernel_names]


def _get_all_kernels() -> list[dict]:
    """Return all kernels from the task registry (placeholder)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pallasbench.tasks import TASK_REGISTRY
        return TASK_REGISTRY
    except ImportError:
        print("Warning: could not import TASK_REGISTRY. Using placeholder.", file=sys.stderr)
        return [
            {"name": f"L1/{n}", "level": 1}
            for n in ["relu", "gelu", "silu", "matmul", "softmax", "layernorm"]
        ]


def _build_prompt(kernel: dict) -> str:
    """Build the optimization prompt for a given kernel.

    In production, this reads the kernel's source, reference source,
    and hardware context to construct a prompt following the template
    patterns in docs/expansion/llm_optimization_pipeline.md.
    """
    return f"Optimize the Pallas kernel '{kernel.get('name', 'unknown')}' for A100 GPU.\n"


def _call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.8,
    api_base: str | None = None,
) -> str:
    """Call the LLM API and return the generated kernel source.

    Supported providers:
      - OpenAI: model names starting with 'gpt-' or 'o'
      - Anthropic: model names starting with 'claude-'
      - Open-source (vLLM): uses --api-base for self-hosted endpoints

    Placeholder implementation — returns a mock kernel.
    """
    # TODO: Implement actual API calls
    # - openai: from openai import OpenAI
    # - anthropic: from anthropic import Anthropic
    # - vllm: requests.post(f"{api_base}/chat/completions", ...)
    return f"# TODO: generated kernel for prompt: {prompt[:60]}...\n"


def _generate_candidates(
    kernel: dict,
    model: str,
    n: int,
    temperature: float,
    api_base: str | None,
) -> list[dict]:
    """Generate n candidate kernels for a given kernel task."""
    prompt = _build_prompt(kernel)
    candidates = []
    for i in range(n):
        raw = _call_llm(model, prompt, temperature, api_base)
        candidate = {
            "candidate_id": f"{kernel['name']}--{model}--t{temperature}--n{i}",
            "kernel_name": kernel["name"],
            "level": kernel.get("level", 1),
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "raw_response": raw,
            "source_optimized": _extract_kernel_source(raw),
        }
        candidates.append(candidate)
    return candidates


def _extract_kernel_source(response: str) -> str:
    """Extract the kernel function from the LLM response.

    Attempts to extract content from ```python ... ``` blocks.
    Falls back to the full response if no code block is found.
    """
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized Pallas kernel candidates using LLMs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (e.g. gpt-4o, claude-opus-4, deepseek-coder-v2)",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["all"],
        help="Kernel names to optimize (or 'all' for full suite)",
    )
    parser.add_argument(
        "--num-candidates-per-kernel",
        type=int,
        default=5,
        help="Number of candidate kernels to generate per kernel",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated",
        help="Output directory for generated candidates JSONL",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for vLLM-compatible API (self-hosted models)",
    )
    args = parser.parse_args()

    kernels = _resolve_kernels(args.kernels)
    print(f"Model: {args.model}")
    print(f"Kernels: {len(kernels)}")
    print(f"Candidates per kernel: {args.num_candidates_per_kernel}")
    print(f"Temperature: {args.temperature}")
    print()

    all_candidates = []
    for i, kernel in enumerate(kernels):
        name = kernel.get("name", "unnamed")
        print(f"[{i + 1}/{len(kernels)}] {name} ... ", end="", flush=True)
        try:
            candidates = _generate_candidates(
                kernel=kernel,
                model=args.model,
                n=args.num_candidates_per_kernel,
                temperature=args.temperature,
                api_base=args.api_base,
            )
            all_candidates.extend(candidates)
            print(f"{len(candidates)} candidates")
        except Exception as e:
            print(f"ERROR: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"candidates_{args.model}_{timestamp}.jsonl"
    )

    with open(output_path, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c) + "\n")

    print(f"\nSaved {len(all_candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
