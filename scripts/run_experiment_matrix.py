"""CLI entrypoint for Phase 3 experiment-matrix planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.training.experiment_runner import expand_experiment_matrix


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print validated child runs without executing training.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("execution is not implemented; pass --dry-run")

    plans = expand_experiment_matrix(
        Path(args.config),
        root=Path(args.root),
    )
    print("run_id\tseed\ttotal_timesteps\toverrides")
    for plan in plans:
        overrides = json.dumps(
            plan.overrides,
            sort_keys=True,
            separators=(",", ":"),
        )
        print(
            f"{plan.run_id}\t{plan.seed}\t{plan.total_timesteps}\t{overrides}"
        )


if __name__ == "__main__":
    main()
