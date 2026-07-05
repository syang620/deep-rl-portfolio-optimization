"""CLI entrypoint for validation backtest comparison reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.evaluation.reports import write_validation_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write a Markdown comparison report for validation backtests.",
    )
    parser.add_argument(
        "--baseline-root",
        default="artifacts/backtests/baselines_validation",
        help="Directory containing one metrics.json per baseline strategy.",
    )
    parser.add_argument(
        "--ppo-metrics",
        default="artifacts/backtests/ppo_validation/metrics.json",
        help="Optional PPO metrics.json path.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/backtests/validation_report.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args(argv)

    output_path = write_validation_report(
        baseline_root=Path(args.baseline_root),
        ppo_metrics_path=Path(args.ppo_metrics),
        output_path=Path(args.output),
    )
    print(output_path)


if __name__ == "__main__":
    main()
