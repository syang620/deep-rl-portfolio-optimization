"""CLI entrypoint for packaging a selected Phase 3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.training.finalize_model import finalize_selected_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Atomically package a validation-selected model candidate.",
    )
    parser.add_argument(
        "--selected-configuration",
        required=True,
        help="Path to selected_configuration.json.",
    )
    parser.add_argument(
        "--registry",
        default="artifacts/experiments/registry.csv",
        help="Path to the experiment registry CSV.",
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--representative-seed", required=True, type=int)
    parser.add_argument(
        "--output-root",
        default="artifacts/final_model",
        help="Parent directory for versioned final packages.",
    )
    args = parser.parse_args(argv)

    result = finalize_selected_model(
        selected_configuration_path=Path(args.selected_configuration),
        registry_path=Path(args.registry),
        model_version=args.model_version,
        representative_seed=args.representative_seed,
        output_root=Path(args.output_root),
    )
    print(f"final_model: {result.output_dir}")
    print(f"configuration_id: {result.configuration_id}")
    print(f"run_id: {result.run_id}")
    print(f"representative_seed: {result.representative_seed}")
    print("final_test_status: not_run")


if __name__ == "__main__":
    main()
