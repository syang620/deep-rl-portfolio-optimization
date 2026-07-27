"""CLI entrypoint for selected-checkpoint policy behavior diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.diagnostics import (
    run_policy_behavior_diagnostics,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze allocation, concentration, turnover, and return behavior "
            "for all checkpoints in a selected Phase 3 configuration."
        ),
    )
    parser.add_argument(
        "--selected-configuration",
        required=True,
        help="Path to selected_configuration.json.",
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Path to the selection-ready experiment registry.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Phase 3 evaluation config.",
    )
    parser.add_argument(
        "--universe-config",
        default="configs/universe.yaml",
        help="Asset universe config used for exposure groupings.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for policy-behavior diagnostic artifacts.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve relative paths.",
    )
    args = parser.parse_args(argv)

    result = run_policy_behavior_diagnostics(
        selected_configuration_path=Path(args.selected_configuration),
        registry_path=Path(args.registry),
        evaluation_config_path=Path(args.config),
        universe_config_path=Path(args.universe_config),
        output_dir=Path(args.output_dir),
        root=Path(args.root),
    )
    for output_type, output_path in result.outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
