"""Typed configuration schemas for the Phase 1 data pipeline."""

from __future__ import annotations

from datetime import date
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


class FeaturesConfig(StrictConfigModel):
    feature_version: str
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
