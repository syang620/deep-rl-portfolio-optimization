"""CLI entrypoint for validation-only configuration selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.model_selection import (
    NoPassingCandidateError,
    write_candidate_ranking,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank seed-stable configurations using validation gates.",
    )
    parser.add_argument(
        "--seed-stability",
        required=True,
        help="Path to seed_stability.csv.",
    )
    parser.add_argument(
        "--baseline-root",
        default="artifacts/backtests/baselines_validation",
        help="Directory containing required validation baseline metrics.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Phase 3 evaluation and selection config.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for candidate ranking and selection artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        result = write_candidate_ranking(
            seed_stability_path=Path(args.seed_stability),
            baseline_root=Path(args.baseline_root),
            evaluation_config_path=Path(args.config),
            output_dir=Path(args.output_dir),
        )
    except NoPassingCandidateError as exc:
        _print_outputs(exc.outputs)
        parser.exit(1, f"selection failed: {exc}\n")
    _print_outputs(result.outputs)


def _print_outputs(outputs: dict[str, Path]) -> None:
    for output_type, output_path in outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
