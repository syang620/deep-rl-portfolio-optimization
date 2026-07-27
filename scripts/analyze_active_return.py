"""CLI entrypoint for paired PPO active-return bootstrap analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.statistical_validation import (
    run_active_return_bootstrap,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap selected PPO validation returns against a paired "
            "equal-weight benchmark."
        ),
    )
    parser.add_argument(
        "--ppo-nav",
        required=True,
        help="Selected-checkpoint nav_by_regime.parquet.",
    )
    parser.add_argument(
        "--baseline-nav",
        required=True,
        help="Cost-matched equal-weight validation NAV Parquet.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Phase 3 evaluation config.",
    )
    parser.add_argument(
        "--regime-name",
        default="validation_2024",
        help="Validation regime to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for bootstrap evidence artifacts.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve relative paths.",
    )
    args = parser.parse_args(argv)

    result = run_active_return_bootstrap(
        ppo_nav_path=Path(args.ppo_nav),
        baseline_nav_path=Path(args.baseline_nav),
        evaluation_config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        regime_name=args.regime_name,
        root=Path(args.root),
    )
    for output_type, output_path in result.outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
