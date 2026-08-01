"""Aggregate PR17 fold-reset-aware outer evaluation evidence."""

from __future__ import annotations

import argparse

from portfolio_rl.evaluation.walk_forward_report import (
    aggregate_walk_forward_results,
)
from portfolio_rl.training.walk_forward_runner import (
    load_walk_forward_campaign_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/ppo_walk_forward.yaml",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    config = load_walk_forward_campaign_config(args.config, root=args.root)
    print(f"aggregate: {aggregate_walk_forward_results(config)}")


if __name__ == "__main__":
    main()
