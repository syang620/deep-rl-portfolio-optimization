"""CLI entrypoint for validation-only cost and regime robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.robustness import (
    run_transaction_cost_robustness,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected seed checkpoints across validation transaction "
            "costs and configured regime windows."
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
        help="Phase 3 evaluation config containing the robustness cost grid.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for cost and regime robustness artifacts.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve relative paths.",
    )
    args = parser.parse_args(argv)

    result = run_transaction_cost_robustness(
        selected_configuration_path=Path(args.selected_configuration),
        registry_path=Path(args.registry),
        evaluation_config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        root=Path(args.root),
    )
    for output_type, output_path in result.outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
