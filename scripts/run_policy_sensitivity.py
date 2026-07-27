"""CLI entrypoint for selected-policy counterfactual sensitivity probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.policy_sensitivity import run_policy_sensitivity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay selected PPO actions and probe counterfactual volatility "
            "and global-risk feature values."
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
        "--diagnostics-dir",
        required=True,
        help="Directory containing reconciled policy behavior diagnostics.",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Phase 3 evaluation config.",
    )
    parser.add_argument(
        "--universe-config",
        default="configs/universe.yaml",
        help="Asset universe config.",
    )
    parser.add_argument(
        "--feature-spec",
        default="artifacts/feature_specs/feature_spec_v1.json",
        help="Versioned feature ordering contract.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for sensitivity artifacts.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve relative paths.",
    )
    args = parser.parse_args(argv)

    result = run_policy_sensitivity(
        selected_configuration_path=Path(args.selected_configuration),
        registry_path=Path(args.registry),
        diagnostics_dir=Path(args.diagnostics_dir),
        evaluation_config_path=Path(args.config),
        universe_config_path=Path(args.universe_config),
        feature_spec_path=Path(args.feature_spec),
        output_dir=Path(args.output_dir),
        root=Path(args.root),
    )
    for output_type, output_path in result.outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
