"""CLI entrypoint for Phase 3 seed-stability aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.model_selection import (
    write_seed_stability_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize validation stability across experiment seeds.",
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Selection-ready experiment registry CSV or Parquet path.",
    )
    parser.add_argument(
        "--matrix-manifest",
        required=True,
        help="Persisted matrix manifest defining the experiment scope.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for seed_stability.csv and seed_stability.md.",
    )
    args = parser.parse_args(argv)

    outputs = write_seed_stability_report(
        registry_path=Path(args.registry),
        matrix_manifest_path=Path(args.matrix_manifest),
        output_dir=Path(args.output_dir),
    )
    for output_type, output_path in outputs.items():
        print(f"{output_type}: {output_path}")


if __name__ == "__main__":
    main()
