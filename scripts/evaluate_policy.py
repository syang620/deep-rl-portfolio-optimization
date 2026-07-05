"""CLI entrypoint for deterministic PPO policy backtests."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.config.loader import load_env_config
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import (
    run_weight_policy_backtest,
    write_backtest_artifacts,
)
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Stable-Baselines3 PPO portfolio policy.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing Phase 1 data/ and artifacts/ outputs.",
    )
    parser.add_argument(
        "--env-config",
        default="configs/env.yaml",
        help="Path to the Phase 2 environment config.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved PPO model zip.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to backtest.",
    )
    parser.add_argument(
        "--strategy",
        default="ppo",
        help="Strategy name recorded in output artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/backtests/ppo_validation",
        help="Directory where policy artifacts will be written.",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required when evaluating the final selected model on the test split.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    args = parser.parse_args(argv)

    output_dir = run_policy_evaluation(
        root=Path(args.root),
        env_config_path=Path(args.env_config),
        model_path=Path(args.model_path),
        split=args.split,
        strategy=args.strategy,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        confirm_final_test=args.confirm_final_test,
    )
    print(output_dir)


def run_policy_evaluation(
    *,
    root: str | Path = ".",
    env_config_path: str | Path = "configs/env.yaml",
    model_path: str | Path,
    split: str = "validation",
    strategy: str = "ppo",
    output_dir: str | Path = "artifacts/backtests/ppo_validation",
    max_steps: int | None = None,
    confirm_final_test: bool = False,
) -> Path:
    """Evaluate a saved PPO model and write deterministic backtest artifacts."""
    root_path = Path(root)
    split_name = split.strip()
    if split_name == "test" and not confirm_final_test:
        raise ValueError(
            "test split evaluation requires confirm_final_test=True "
            "or CLI flag --confirm-final-test"
        )
    env_config = load_env_config(_resolve_path(root_path, env_config_path))
    dataset = load_portfolio_dataset(root_path)
    feature_store = PortfolioFeatureStore(dataset, split=split_name)
    resolved_model_path = _resolve_path(root_path, model_path)
    policy = load_sb3_weight_policy(
        resolved_model_path,
        action_temperature=env_config.action_temperature,
    )
    result = run_weight_policy_backtest(
        feature_store=feature_store,
        policy=policy,
        strategy=strategy,
        rebalance_frequency_trading_days=env_config.rebalance_frequency_trading_days,
        transaction_cost_bps=env_config.transaction_cost_bps,
        max_steps=max_steps,
    )
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root_path / output_path
    write_backtest_artifacts(result, output_path)
    _write_evaluation_metadata(
        output_path=output_path,
        model_path=resolved_model_path,
        split=split_name,
        strategy=strategy,
        confirm_final_test=confirm_final_test,
    )
    return output_path


def _write_evaluation_metadata(
    *,
    output_path: Path,
    model_path: Path,
    split: str,
    strategy: str,
    confirm_final_test: bool,
) -> None:
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "split": split,
        "strategy": strategy,
        "confirm_final_test": bool(confirm_final_test),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


if __name__ == "__main__":
    main()
