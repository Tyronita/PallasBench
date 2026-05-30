"""CLI entry-point for evolving Pallas kernels via ShinkaEvolve.

Usage
-----
    python scripts/evolve_pallas.py --kernels L1/relu L1/matmul \\
        --generations 5 --population-size 10 \\
        --mutation-rate 0.8 --crossover-rate 0.3 \\
        --output-dir ./evolution_traces
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _resolve_kernels(kernel_names: list[str]) -> list[str]:
    """Resolve kernel names; can expand wildcards in the future."""
    return kernel_names


def run_evolution(
    kernel_names: list[str],
    *,
    generations: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    output_dir: str | Path,
) -> None:
    """Run the ShinkaEvolve evolutionary loop for the specified kernels.

    Parameters
    ----------
    kernel_names : list of str
        Task names from the TASK_REGISTRY (e.g., ``["L1/relu", "L1/matmul"]``).
    generations : int
        Number of generations to evolve.
    population_size : int
        Number of candidates per generation.
    mutation_rate : float
        Probability of mutating a selected parent (0.0 -- 1.0).
    crossover_rate : float
        Probability of applying crossover between two parents (0.0 -- 1.0).
    output_dir : str or Path
        Directory for JSONL evolution-trace files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_id = (
        f"shinka-pallas-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )

    for kernel_name in kernel_names:
        print(
            f"[{experiment_id}] Skipping kernel '{kernel_name}' "
            f"({generations} gen, pop={population_size}, "
            f"mut={mutation_rate}, xover={crossover_rate}) — "
            f"placeholder. No evolution performed.",
            file=sys.stderr,
        )

    meta = {
        "experiment_id": experiment_id,
        "generations": generations,
        "population_size": population_size,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "kernels": kernel_names,
        "status": "placeholder",
        "message": "Evolution not yet implemented; run_evolution is a stub.",
    }
    manifest = output_path / f"{experiment_id}_manifest.json"
    manifest.write_text(json.dumps(meta, indent=2))
    print(f"Wrote manifest to {manifest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve Pallas kernels via ShinkaEvolve",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        required=True,
        help="Kernel task names (e.g., L1/relu L1/matmul)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations (default: 5)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Candidates per generation (default: 10)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.8,
        help="Mutation probability per parent (default: 0.8)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.3,
        help="Crossover probability per parent pair (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evolution_traces",
        help="Directory for evolution trace output (default: ./evolution_traces)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_evolution(
        kernel_names=_resolve_kernels(args.kernels),
        generations=args.generations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
