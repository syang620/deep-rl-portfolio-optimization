"""CLI entrypoint for minimal PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.training.train_ppo import run_ppo_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a Stable-Baselines3 PPO portfolio policy.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing Phase 1 data/ and artifacts/ outputs.",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Path to the Phase 1 data config.",
    )
    parser.add_argument(
        "--env-config",
        default="configs/env.yaml",
        help="Path to the Phase 2 environment config.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train_ppo.yaml",
        help="Path to the PPO training config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional override for smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional model output directory override.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional experiment run id when using configured output_dir.",
    )
    args = parser.parse_args(argv)

    model_path = run_ppo_training(
        root=Path(args.root),
        data_config_path=Path(args.data_config),
        env_config_path=Path(args.env_config),
        train_config_path=Path(args.train_config),
        total_timesteps_override=args.total_timesteps,
        output_dir_override=(
            Path(args.output_dir) if args.output_dir is not None else None
        ),
        run_id=args.run_id,
    )
    print(model_path)


if __name__ == "__main__":
    main()
