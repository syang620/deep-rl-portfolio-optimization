"""CLI entrypoint for building the Phase 3 experiment registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.training.registry import write_experiment_registry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build CSV, Parquet, and Markdown experiment registries.",
    )
    parser.add_argument(
        "--experiment-root",
        default="artifacts/experiments",
        help="Directory containing experiment artifact bundles.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/experiments/registry",
        help="Output path prefix without extension.",
    )
    args = parser.parse_args(argv)

    outputs = write_experiment_registry(
        experiment_root=Path(args.experiment_root),
        output_prefix=Path(args.output),
    )
    for output_type, output_path in outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
