"""Validated split contracts and isolated views for walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_yaml
from portfolio_rl.data.dataset import PortfolioDataset, load_portfolio_dataset


@dataclass(frozen=True)
class FoldPeriod:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    inner_train: FoldPeriod
    inner_validation: FoldPeriod
    outer_evaluation: FoldPeriod


@dataclass(frozen=True)
class WalkForwardConfig:
    source_paths: dict[str, str]
    source_hashes: dict[str, str]
    output_root: str
    raw_history_start_date: str
    maximum_feature_lookback_trading_days: int
    folds: tuple[WalkForwardFold, ...]


def load_walk_forward_config(path: str | Path) -> WalkForwardConfig:
    """Load and strictly validate the PR 16 configuration."""
    raw = load_yaml(Path(path))
    if int(_required(raw, "schema_version")) != 1:
        raise ValueError("walk-forward schema_version must be 1")
    sources = _mapping(raw, "sources")
    source_paths = {}
    source_hashes = {}
    for name in (
        "asset_features",
        "global_features",
        "feature_spec",
        "features_config",
        "universe_config",
        "data_config",
    ):
        record = _mapping(sources, name)
        source_paths[name] = _text(record, "path")
        source_hashes[name] = _text(record, "sha256")
    folds = tuple(_parse_fold(item) for item in _list(raw, "folds"))
    validate_walk_forward_folds(folds)
    lookback = int(_required(raw, "maximum_feature_lookback_trading_days"))
    if lookback != 252:
        raise ValueError("maximum feature lookback must remain 252 trading days")
    return WalkForwardConfig(
        source_paths=source_paths,
        source_hashes=source_hashes,
        output_root=_text(raw, "output_root"),
        raw_history_start_date=_text(raw, "raw_history_start_date"),
        maximum_feature_lookback_trading_days=lookback,
        folds=folds,
    )


def validate_walk_forward_folds(folds: tuple[WalkForwardFold, ...]) -> None:
    """Require the exact, ordered Phase 3 walk-forward fold contract."""
    expected = [
        ("WF1", "2010-01-01", "2014-12-31", "2015-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
        ("WF2", "2010-01-01", "2016-12-31", "2017-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
        ("WF3", "2010-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
        ("WF4", "2010-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ]
    observed = [
        (
            fold.fold_id,
            fold.inner_train.start.date().isoformat(),
            fold.inner_train.end.date().isoformat(),
            fold.inner_validation.start.date().isoformat(),
            fold.inner_validation.end.date().isoformat(),
            fold.outer_evaluation.start.date().isoformat(),
            fold.outer_evaluation.end.date().isoformat(),
        )
        for fold in folds
    ]
    if observed != expected:
        raise ValueError("walk-forward folds do not match the frozen WF1-WF4 contract")
    for fold in folds:
        if not (
            fold.inner_train.start <= fold.inner_train.end
            < fold.inner_validation.start
            <= fold.inner_validation.end
            < fold.outer_evaluation.start
            <= fold.outer_evaluation.end
        ):
            raise ValueError(f"fold periods overlap or are unordered: {fold.fold_id}")


def assign_fold_splits(frame: pd.DataFrame, fold: WalkForwardFold) -> pd.DataFrame:
    """Slice a pre-normalization panel and assign the three fold labels."""
    if "date" not in frame or "split" not in frame:
        raise ValueError("source frame must contain date and split columns")
    dates = pd.to_datetime(frame["date"])
    labels = pd.Series(index=frame.index, dtype="object")
    periods = (
        ("inner_train", fold.inner_train),
        ("inner_validation", fold.inner_validation),
        ("outer_evaluation", fold.outer_evaluation),
    )
    for label, period in periods:
        labels.loc[(dates >= period.start) & (dates <= period.end)] = label
    selected = labels.notna()
    result = frame.loc[selected].copy()
    result["split"] = labels.loc[selected].to_numpy()
    if set(result["split"]) != {label for label, _ in periods}:
        raise ValueError(f"fold source data does not cover all periods: {fold.fold_id}")
    return result.reset_index(drop=True)


def load_training_selection_dataset(fold_dir: str | Path) -> PortfolioDataset:
    """Load the physical view that excludes outer-evaluation rows."""
    root = Path(fold_dir)
    dataset = load_portfolio_dataset(
        root,
        model_matrix_path="training_selection_matrix_daily.parquet",
        feature_spec_path="feature_spec.json",
    )
    if set(dataset.splits) != {"inner_train", "inner_validation"}:
        raise ValueError("training-selection view contains unexpected splits")
    return dataset


def load_outer_evaluation_dataset(fold_dir: str | Path) -> PortfolioDataset:
    """Load outer evaluation with strictly earlier return context retained."""
    root = Path(fold_dir)
    dataset = load_portfolio_dataset(
        root,
        model_matrix_path="model_matrix_daily.parquet",
        feature_spec_path="feature_spec.json",
    )
    if set(dataset.splits) != {
        "inner_train",
        "inner_validation",
        "outer_evaluation",
    }:
        raise ValueError("outer-evaluation context dataset has unexpected splits")
    return dataset


def _parse_fold(raw: Any) -> WalkForwardFold:
    if not isinstance(raw, dict):
        raise TypeError("each fold must be a mapping")
    return WalkForwardFold(
        fold_id=_text(raw, "fold_id"),
        inner_train=_period(raw, "inner_train"),
        inner_validation=_period(raw, "inner_validation"),
        outer_evaluation=_period(raw, "outer_evaluation"),
    )


def _period(raw: dict[str, Any], key: str) -> FoldPeriod:
    value = _mapping(raw, key)
    return FoldPeriod(
        start=pd.Timestamp(_text(value, "start")).normalize(),
        end=pd.Timestamp(_text(value, "end")).normalize(),
    )


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required walk-forward key: {key}")
    return mapping[key]


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise TypeError(f"walk-forward key must be a mapping: {key}")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"walk-forward key must be a list: {key}")
    return value


def _text(mapping: dict[str, Any], key: str) -> str:
    value = str(_required(mapping, key)).strip()
    if not value:
        raise ValueError(f"walk-forward key must not be empty: {key}")
    return value
