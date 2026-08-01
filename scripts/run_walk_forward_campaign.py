"""Run staged PR17 walk-forward selection and outer evaluation."""

from __future__ import annotations

import argparse

from portfolio_rl.evaluation.walk_forward_report import run_evaluation_stage
from portfolio_rl.training.walk_forward_runner import (
    load_walk_forward_campaign_config,
    run_selection_stage,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/ppo_walk_forward.yaml",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("pilot", "selection", "evaluation"),
    )
    args = parser.parse_args(argv)
    config = load_walk_forward_campaign_config(args.config, root=args.root)
    if args.stage == "pilot":
        selections = run_selection_stage(config, pilot=True)
        outputs = run_evaluation_stage(config, pilot=True)
        print(f"pilot_selection_freeze: {selections[0].freeze_path}")
        print(f"pilot_evaluation: {outputs[0]}")
    elif args.stage == "selection":
        results = run_selection_stage(config, pilot=False)
        for result in results:
            print(f"{result.fold_id}/seed_{result.seed}: {result.freeze_path}")
    else:
        for output in run_evaluation_stage(config, pilot=False):
            print(f"outer_evaluation: {output}")


if __name__ == "__main__":
    main()
