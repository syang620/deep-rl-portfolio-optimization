"""CLI entrypoint for Phase 3 experiment-matrix planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.training.experiment_runner import (
    expand_experiment_matrix,
    write_experiment_matrix_plan,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plan a Phase 3 experiment matrix without training.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a Phase 3 experiment matrix config.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve config paths.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Print validated child runs without executing training.",
    )
    modes.add_argument(
        "--write-plan",
        action="store_true",
        help="Write matrix manifest, CSV, and Markdown planning artifacts.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root for --write-plan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing known plan files when using --write-plan.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.write_plan:
        parser.error(
            "execution is not implemented; pass --dry-run or --write-plan"
        )
    if args.dry_run and (args.output_root is not None or args.force):
        parser.error("--output-root and --force require --write-plan")

    if args.write_plan:
        outputs = write_experiment_matrix_plan(
            Path(args.config),
            root=Path(args.root),
            output_root=(
                Path(args.output_root)
                if args.output_root is not None
                else Path("artifacts/experiment_matrices")
            ),
            force=args.force,
        )
        for output_type, output_path in outputs.items():
            print(f"{output_type}: {output_path}")
        return

    plans = expand_experiment_matrix(Path(args.config), root=Path(args.root))
    print("run_id\tseed\ttotal_timesteps\toverrides")
    for plan in plans:
        overrides = json.dumps(
            plan.overrides, sort_keys=True, separators=(",", ":")
        )
        print(f"{plan.run_id}\t{plan.seed}\t{plan.total_timesteps}\t{overrides}")


if __name__ == "__main__":
    main()
