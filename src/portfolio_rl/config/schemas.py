"""Typed configuration schemas for the Phase 1 data pipeline."""

from __future__ import annotations

from datetime import date
from math import isfinite
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictConfigModel(BaseModel):
    """Base model that rejects config drift from unknown YAML keys."""

    model_config = ConfigDict(extra="forbid")


class AssetConfig(StrictConfigModel):
    ticker: str
    asset_class: str

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        return ticker


class UniverseConfig(StrictConfigModel):
    universe_name: str
    assets: list[AssetConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_tickers(self) -> UniverseConfig:
        tickers = [asset.ticker for asset in self.assets]
        if len(tickers) != len(set(tickers)):
            raise ValueError("assets must not contain duplicate tickers")
        return self

    @property
    def tickers(self) -> list[str]:
        return [asset.ticker for asset in self.assets]


class StorageConfig(StrictConfigModel):
    duckdb_path: Path
    raw_parquet_dir: Path
    interim_parquet_dir: Path
    processed_parquet_dir: Path


class MacroSeriesConfig(StrictConfigModel):
    series_id: str
    description: str
    frequency: str

    @field_validator("series_id")
    @classmethod
    def normalize_series_id(cls, value: str) -> str:
        series_id = value.strip().upper()
        if not series_id:
            raise ValueError("series_id must not be empty")
        return series_id

    @field_validator("frequency")
    @classmethod
    def require_daily_frequency(cls, value: str) -> str:
        frequency = value.strip().lower()
        if frequency != "daily":
            raise ValueError("macro series frequency must be daily")
        return frequency


class DataConfig(StrictConfigModel):
    raw_start_date: date
    model_start_date: date
    train_start_date: date
    train_end_date: date
    validation_start_date: date
    validation_end_date: date
    test_start_date: date
    test_end_date: date | None
    market_data_source: str
    macro_data_source: str
    macro_series: list[MacroSeriesConfig] = Field(min_length=1)
    storage: StorageConfig

    @model_validator(mode="after")
    def require_chronological_splits(self) -> DataConfig:
        if self.raw_start_date > self.model_start_date:
            raise ValueError("raw_start_date must be on or before model_start_date")
        if self.model_start_date > self.train_start_date:
            raise ValueError("model_start_date must be on or before train_start_date")
        if self.train_start_date > self.train_end_date:
            raise ValueError("train_start_date must be on or before train_end_date")
        if self.train_end_date >= self.validation_start_date:
            raise ValueError("train_end_date must be before validation_start_date")
        if self.validation_start_date > self.validation_end_date:
            raise ValueError(
                "validation_start_date must be on or before validation_end_date"
            )
        if self.validation_end_date >= self.test_start_date:
            raise ValueError("validation_end_date must be before test_start_date")
        if self.test_end_date is not None and self.test_start_date > self.test_end_date:
            raise ValueError("test_start_date must be on or before test_end_date")
        return self

    @model_validator(mode="after")
    def require_unique_macro_series(self) -> DataConfig:
        series_ids = [series.series_id for series in self.macro_series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("macro_series must not contain duplicate series_id values")
        return self


class WinsorizationConfig(StrictConfigModel):
    enabled: bool
    lower_quantile: float = Field(ge=0.0, le=1.0)
    upper_quantile: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_ordered_quantiles(self) -> WinsorizationConfig:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be less than upper_quantile")
        return self


class NormalizationConfig(StrictConfigModel):
    method: str
    fit_split: str

    @field_validator("fit_split")
    @classmethod
    def require_train_fit_split(cls, value: str) -> str:
        if value != "train":
            raise ValueError("normalization fit_split must be train")
        return value


class MarketFeatureConfig(StrictConfigModel):
    benchmark_ticker: str
    credit_proxy_safe_ticker: str
    credit_proxy_risk_ticker: str

    @field_validator(
        "benchmark_ticker",
        "credit_proxy_safe_ticker",
        "credit_proxy_risk_ticker",
    )
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("feature market tickers must not be empty")
        return ticker


class FeaturesConfig(StrictConfigModel):
    feature_version: str
    market: MarketFeatureConfig
    return_windows: list[int] = Field(min_length=1)
    volatility_windows: list[int] = Field(min_length=1)
    drawdown_windows: list[int] = Field(min_length=1)
    rsi_windows: list[int] = Field(min_length=1)
    price_z_windows: list[int] = Field(min_length=1)
    correlation_windows: list[int] = Field(min_length=1)
    winsorization: WinsorizationConfig
    normalization: NormalizationConfig

    @field_validator(
        "return_windows",
        "volatility_windows",
        "drawdown_windows",
        "rsi_windows",
        "price_z_windows",
        "correlation_windows",
    )
    @classmethod
    def require_positive_windows(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("feature windows must be positive integers")
        return values


class EnvConfig(StrictConfigModel):
    rebalance_frequency_trading_days: int = Field(gt=0)
    episode_length_trading_days: int = Field(gt=0)
    max_episode_steps: int = Field(gt=0)
    action_transform: str
    action_temperature: float = Field(gt=0)
    initial_weights: str
    transaction_cost_bps: float = Field(ge=0)
    reward_type: str = "log_growth"
    reward_scale: float = Field(default=100.0, gt=0)
    terminal_bad_gross_penalty: float = -100.0
    record_arrays_in_info: bool = False

    @field_validator("action_transform")
    @classmethod
    def require_softmax_action_transform(cls, value: str) -> str:
        if value != "softmax":
            raise ValueError("action_transform must be softmax")
        return value

    @field_validator("initial_weights")
    @classmethod
    def require_equal_weight_initial_weights(cls, value: str) -> str:
        if value != "equal_weight":
            raise ValueError("initial_weights must be equal_weight")
        return value

    @field_validator("reward_type")
    @classmethod
    def require_log_growth_reward(cls, value: str) -> str:
        if value != "log_growth":
            raise ValueError("reward_type must be log_growth")
        return value

    @field_validator("terminal_bad_gross_penalty")
    @classmethod
    def require_finite_terminal_penalty(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("terminal_bad_gross_penalty must be finite")
        return value

    @model_validator(mode="after")
    def require_consistent_episode_steps(self) -> EnvConfig:
        expected_steps = (
            self.episode_length_trading_days / self.rebalance_frequency_trading_days
        )
        if expected_steps != self.max_episode_steps:
            raise ValueError(
                "max_episode_steps must equal "
                "episode_length_trading_days / rebalance_frequency_trading_days"
            )
        return self


class PPOHyperparametersConfig(StrictConfigModel):
    learning_rate: float = Field(gt=0)
    gamma: float = Field(gt=0, le=1)
    gae_lambda: float = Field(gt=0, le=1)
    n_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    n_epochs: int = Field(gt=0)
    clip_range: float = Field(gt=0, le=1)
    ent_coef: float = Field(ge=0)
    vf_coef: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)

    @model_validator(mode="after")
    def require_rollout_alignment(self) -> PPOHyperparametersConfig:
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size")
        if self.n_steps % 52 != 0:
            raise ValueError("n_steps must be divisible by 52")
        return self


class NetworkConfig(StrictConfigModel):
    pi: list[int] = Field(min_length=1)
    vf: list[int] = Field(min_length=1)

    @field_validator("pi", "vf")
    @classmethod
    def require_positive_layer_sizes(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("network layer sizes must be positive integers")
        return values


class EvaluationConfig(StrictConfigModel):
    eval_freq_timesteps: int = Field(gt=0)
    deterministic: bool
    metric_for_best_model: str

    @field_validator("metric_for_best_model")
    @classmethod
    def require_supported_best_model_metric(cls, value: str) -> str:
        supported_metrics = {"sharpe_ratio", "final_nav", "total_return"}
        if value not in supported_metrics:
            raise ValueError(
                "metric_for_best_model must be one of "
                f"{sorted(supported_metrics)}"
            )
        return value


class CheckpointConfig(StrictConfigModel):
    save_freq_timesteps: int = Field(gt=0)
    output_dir: Path


class WandbConfig(StrictConfigModel):
    enabled: bool
    project: str
    group: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("project", "group")
    @classmethod
    def require_non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class TrainPPOConfig(StrictConfigModel):
    algorithm: str
    policy: str
    total_timesteps: int = Field(gt=0)
    seed: int = Field(ge=0)
    ppo: PPOHyperparametersConfig
    network: NetworkConfig
    evaluation: EvaluationConfig
    checkpoints: CheckpointConfig
    wandb: WandbConfig

    @field_validator("algorithm")
    @classmethod
    def require_ppo_algorithm(cls, value: str) -> str:
        if value != "PPO":
            raise ValueError("algorithm must be PPO")
        return value

    @field_validator("policy")
    @classmethod
    def require_mlp_policy(cls, value: str) -> str:
        if value != "MlpPolicy":
            raise ValueError("policy must be MlpPolicy")
        return value
