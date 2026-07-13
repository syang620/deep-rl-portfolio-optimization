from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_rl.config.loader import (
    load_data_config,
    load_env_config,
    load_features_config,
    load_phase3_evaluation_config,
    load_phase3_experiment_config,
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
    assert [series.series_id for series in data.macro_series] == [
        "VIXCLS",
        "DTB3",
        "DGS2",
        "DGS10",
        "T10Y2Y",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
    ]
    assert features.feature_version == "v1"
    assert features.market.benchmark_ticker == "SPY"
    assert features.market.credit_proxy_safe_ticker == "IEF"
    assert features.market.credit_proxy_risk_ticker == "HYG"
    assert env.max_episode_steps == 52


def test_valid_phase3_evaluation_config_loads_successfully() -> None:
    config = load_phase3_evaluation_config(CONFIG_DIR / "evaluation.yaml")

    assert config.validation.split == "validation"
    assert config.validation.metric_for_selection == "sharpe_ratio"
    assert config.final_test.split == "test"
    assert config.final_test.require_confirm_final_test is True
    assert config.robustness.transaction_cost_bps == [0.0, 5.0, 10.0, 25.0, 50.0]
    assert config.robustness.regime_windows[0].name == "validation_2024"
    assert config.selection.primary_metric == "sharpe_ratio"
    assert config.selection.tie_breakers == [
        "max_drawdown",
        "average_weekly_turnover",
        "transaction_cost_drag",
    ]


def test_valid_phase3_experiment_configs_load_successfully() -> None:
    config_paths = sorted((CONFIG_DIR / "experiments").glob("*.yaml"))
    loaded = [load_phase3_experiment_config(path) for path in config_paths]

    assert [config.experiment_name for config in loaded] == [
        "ppo_phase3_default",
        "ppo_phase3_seed_sweep",
        "ppo_phase3_temperature_sweep",
    ]
    assert loaded[0].base_env_config == Path("configs/env.yaml")
    assert loaded[0].seeds == [42]
    assert loaded[0].overrides == {}
    assert loaded[1].overrides["env.action_temperature"] == [0.5]
    assert loaded[2].overrides["ppo.ent_coef"] == [0.005, 0.01]


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
market:
  benchmark_ticker: SPY
  credit_proxy_safe_ticker: IEF
  credit_proxy_risk_ticker: HYG
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


def test_phase3_evaluation_unknown_fields_fail_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        """
validation:
  split: validation
  output_root: artifacts/backtests
  include_baselines: true
  include_ppo: true
  metric_for_selection: sharpe_ratio
final_test:
  split: test
  require_confirm_final_test: true
  output_root: artifacts/final_model
robustness:
  transaction_cost_bps: [10.0]
  regime_windows:
    - name: validation_2024
      start_date: "2024-01-01"
      end_date: "2024-12-31"
selection:
  primary_metric: sharpe_ratio
  higher_is_better: true
  tie_breakers: [max_drawdown]
unexpected_key: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_phase3_evaluation_config(config_path)


def test_phase3_experiment_override_values_must_be_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment_name: invalid_phase3_experiment
base_data_config: configs/data.yaml
base_env_config: configs/env.yaml
base_train_config: configs/train_ppo.yaml
run_id_prefix: invalid_phase3
seeds: [42]
overrides:
  env.action_temperature: 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="override values must be lists"):
        load_phase3_experiment_config(config_path)


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
macro_series:
  - series_id: VIXCLS
    description: CBOE Volatility Index
    frequency: daily
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


def test_duplicate_macro_series_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        _data_config_yaml(
            """
macro_series:
  - series_id: VIXCLS
    description: CBOE Volatility Index
    frequency: daily
  - series_id: vixcls
    description: Duplicate VIX
    frequency: daily
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="duplicate series_id"):
        load_data_config(config_path)


def test_monthly_macro_series_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        _data_config_yaml(
            """
macro_series:
  - series_id: CPIAUCSL
    description: Consumer Price Index
    frequency: monthly
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="frequency must be daily"):
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


def _data_config_yaml(macro_series_yaml: str) -> str:
    return f"""
raw_start_date: "2007-01-01"
model_start_date: "2010-01-01"
train_start_date: "2010-01-01"
train_end_date: "2023-12-31"
validation_start_date: "2024-01-01"
validation_end_date: "2024-12-31"
test_start_date: "2025-01-01"
test_end_date: null
market_data_source: yfinance
macro_data_source: fred
{macro_series_yaml.strip()}
storage:
  duckdb_path: data/duckdb/portfolio.duckdb
  raw_parquet_dir: data/raw
  interim_parquet_dir: data/interim
  processed_parquet_dir: data/processed
"""
