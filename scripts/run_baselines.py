"""CLI entrypoint for deterministic baseline backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.config.loader import load_env_config
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import (
    run_weight_policy_backtest,
    write_backtest_artifacts,
)
from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    SingleAssetPolicy,
    WeightPolicy,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic portfolio baseline backtests.",
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
        "--split",
        default="validation",
        help="Dataset split to backtest.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/backtests/baselines_validation",
        help="Directory where strategy artifact folders will be written.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    args = parser.parse_args(argv)

    output_dirs = run_baseline_backtests(
        root=Path(args.root),
        env_config_path=Path(args.env_config),
        split=args.split,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
    )
    for output_dir in output_dirs:
        print(output_dir)


def run_baseline_backtests(
    *,
    root: str | Path = ".",
    env_config_path: str | Path = "configs/env.yaml",
    split: str = "validation",
    output_dir: str | Path = "artifacts/backtests/baselines_validation",
    max_steps: int | None = None,
) -> list[Path]:
    """Run all deterministic baseline policies and write artifacts."""
    root_path = Path(root)
    env_config = load_env_config(_resolve_path(root_path, env_config_path))
    dataset = load_portfolio_dataset(root_path)
    feature_store = PortfolioFeatureStore(dataset, split=split)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root_path / output_path

    policies = _baseline_policies(feature_store.asset_order)
    written_dirs: list[Path] = []
    for strategy, policy in policies.items():
        result = run_weight_policy_backtest(
            feature_store=feature_store,
            policy=policy,
            strategy=strategy,
            rebalance_frequency_trading_days=(
                env_config.rebalance_frequency_trading_days
            ),
            transaction_cost_bps=env_config.transaction_cost_bps,
            max_steps=max_steps,
        )
        strategy_dir = output_path / strategy
        write_backtest_artifacts(result, strategy_dir)
        written_dirs.append(strategy_dir)

    return written_dirs


def _baseline_policies(asset_order: list[str]) -> dict[str, WeightPolicy]:
    n_assets = len(asset_order)
    return {
        "equal_weight_weekly": EqualWeightWeeklyPolicy(n_assets=n_assets),
        "buy_and_hold_equal_weight": BuyAndHoldEqualWeightPolicy(n_assets=n_assets),
        "spy_only": SingleAssetPolicy(asset_order=asset_order, ticker="SPY"),
        "shy_only": SingleAssetPolicy(asset_order=asset_order, ticker="SHY"),
        "inverse_volatility": InverseVolatilityPolicy(n_assets=n_assets),
    }


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


if __name__ == "__main__":
    main()
