from __future__ import annotations

import pandas as pd
import pytest

from portfolio_rl.config.schemas import DataConfig
from portfolio_rl.data.splits import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    assert_split_boundaries,
    assign_chronological_splits,
)


def test_assign_chronological_splits_uses_configured_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                "2010-01-01",
                "2023-12-31",
                "2024-01-01",
                "2024-12-31",
                "2025-01-01",
                "2026-04-26",
            ],
            "value": range(6),
        }
    )

    result = assign_chronological_splits(frame, _data_config())

    assert result["split"].to_list() == [
        TRAIN_SPLIT,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
        TEST_SPLIT,
    ]
    assert "split" not in frame.columns


def test_assign_chronological_splits_supports_custom_date_column() -> None:
    frame = pd.DataFrame({"as_of_date": ["2010-01-01", "2024-01-01", "2025-01-01"]})

    result = assign_chronological_splits(
        frame,
        _data_config(),
        date_column="as_of_date",
    )

    assert result["split"].to_list() == [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]


def test_assign_chronological_splits_fails_for_dates_before_train() -> None:
    frame = pd.DataFrame({"date": ["2009-12-31", "2010-01-01"]})

    with pytest.raises(ValueError, match="outside configured split ranges"):
        assign_chronological_splits(frame, _data_config())


def test_assign_chronological_splits_fails_for_dates_after_closed_test() -> None:
    config = _data_config(
        {
            "test_end_date": "2025-12-31",
        }
    )
    frame = pd.DataFrame({"date": ["2025-12-31", "2026-01-01"]})

    with pytest.raises(ValueError, match="outside configured split ranges"):
        assign_chronological_splits(frame, config)


def test_assert_split_boundaries_fails_on_mislabeled_rows() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2025-01-01"],
            "split": [TRAIN_SPLIT, TEST_SPLIT],
        }
    )

    with pytest.raises(ValueError, match="train split ends after configured boundary"):
        assert_split_boundaries(frame, _data_config())


def _data_config(overrides: dict[str, object] | None = None) -> DataConfig:
    values: dict[str, object] = {
        "raw_start_date": "2007-01-01",
        "model_start_date": "2010-01-01",
        "train_start_date": "2010-01-01",
        "train_end_date": "2023-12-31",
        "validation_start_date": "2024-01-01",
        "validation_end_date": "2024-12-31",
        "test_start_date": "2025-01-01",
        "test_end_date": None,
        "market_data_source": "yfinance",
        "macro_data_source": "fred",
        "macro_series": [
            {
                "series_id": "VIXCLS",
                "description": "CBOE Volatility Index",
                "frequency": "daily",
            }
        ],
        "storage": {
            "duckdb_path": "data/duckdb/portfolio.duckdb",
            "raw_parquet_dir": "data/raw",
            "interim_parquet_dir": "data/interim",
            "processed_parquet_dir": "data/processed",
        },
    }
    if overrides is not None:
        values.update(overrides)
    return DataConfig.model_validate(values)
