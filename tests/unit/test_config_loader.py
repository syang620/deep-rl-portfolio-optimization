from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_rl.config.loader import (
    load_data_config,
    load_env_config,
    load_features_config,
    load_universe_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def test_valid_repo_configs_load_successfully() -> None:
    universe = load_universe_config(CONFIG_DIR / "universe.yaml")
    data = load_data_config(CONFIG_DIR / "data.yaml")
    features = load_features_config(CONFIG_DIR / "features.yaml")
    env = load_env_config(CONFIG_DIR / "env.yaml")

    assert universe.tickers[:3] == ["SPY", "QQQ", "IWM"]
    assert data.train_start_date.isoformat() == "2010-01-01"
    assert features.feature_version == "v1"
    assert env.max_episode_steps == 52


def test_duplicate_tickers_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "universe.yaml"
    config_path.write_text(
        """
universe_name: duplicate_test
assets:
  - ticker: SPY
    asset_class: us_large_cap_equity
  - ticker: spy
    asset_class: duplicate_equity
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="duplicate tickers"):
        load_universe_config(config_path)


def test_asset_order_is_preserved() -> None:
    universe = load_universe_config(CONFIG_DIR / "universe.yaml")

    assert universe.tickers == [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "SHY",
        "LQD",
        "HYG",
        "GLD",
        "DBC",
        "VNQ",
        "XLU",
    ]


def test_unknown_fields_fail_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        """
feature_version: v1
return_windows: [1]
volatility_windows: [21]
drawdown_windows: [63]
rsi_windows: [14]
price_z_windows: [20]
correlation_windows: [63]
unexpected_key: true
winsorization:
  enabled: true
  lower_quantile: 0.005
  upper_quantile: 0.995
normalization:
  method: standard_scaler
  fit_split: train
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_features_config(config_path)


def test_overlapping_split_dates_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        """
raw_start_date: "2007-01-01"
model_start_date: "2010-01-01"
train_start_date: "2010-01-01"
train_end_date: "2024-01-01"
validation_start_date: "2024-01-01"
validation_end_date: "2024-12-31"
test_start_date: "2025-01-01"
test_end_date: null
market_data_source: yfinance
macro_data_source: fred
storage:
  duckdb_path: data/duckdb/portfolio.duckdb
  raw_parquet_dir: data/raw
  interim_parquet_dir: data/interim
  processed_parquet_dir: data/processed
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="train_end_date"):
        load_data_config(config_path)


def test_inconsistent_env_steps_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "env.yaml"
    config_path.write_text(
        """
rebalance_frequency_trading_days: 5
episode_length_trading_days: 260
max_episode_steps: 51
action_transform: softmax
action_temperature: 5.0
initial_weights: equal_weight
transaction_cost_bps: 10.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="max_episode_steps"):
        load_env_config(config_path)
