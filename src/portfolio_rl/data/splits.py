"""Chronological split assignment for Phase 1 datasets."""

from __future__ import annotations

from datetime import date

import pandas as pd

from portfolio_rl.config.schemas import DataConfig


SPLIT_COLUMN = "split"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"


def assign_chronological_splits(
    frame: pd.DataFrame,
    data_config: DataConfig,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return a copy with train/validation/test labels assigned by date."""
    if date_column not in frame.columns:
        raise ValueError(f"frame is missing required date column: {date_column}")

    result = frame.copy()
    dates = pd.to_datetime(result[date_column]).dt.date
    result[SPLIT_COLUMN] = pd.NA

    train_mask = dates.between(
        data_config.train_start_date,
        data_config.train_end_date,
        inclusive="both",
    )
    validation_mask = dates.between(
        data_config.validation_start_date,
        data_config.validation_end_date,
        inclusive="both",
    )
    test_mask = dates >= data_config.test_start_date
    if data_config.test_end_date is not None:
        test_mask &= dates <= data_config.test_end_date

    result.loc[train_mask, SPLIT_COLUMN] = TRAIN_SPLIT
    result.loc[validation_mask, SPLIT_COLUMN] = VALIDATION_SPLIT
    result.loc[test_mask, SPLIT_COLUMN] = TEST_SPLIT

    _assert_all_dates_assigned(result, date_column)
    assert_split_boundaries(result, data_config, date_column=date_column)
    return result


def assert_split_boundaries(
    frame: pd.DataFrame,
    data_config: DataConfig,
    date_column: str = "date",
    split_column: str = SPLIT_COLUMN,
) -> None:
    """Assert that configured split labels are chronological and non-overlapping."""
    for column in (date_column, split_column):
        if column not in frame.columns:
            raise ValueError(f"frame is missing required column: {column}")

    split_dates = pd.to_datetime(frame[date_column]).dt.date
    _assert_split_range(
        split_dates,
        frame[split_column],
        TRAIN_SPLIT,
        data_config.train_start_date,
        data_config.train_end_date,
    )
    _assert_split_range(
        split_dates,
        frame[split_column],
        VALIDATION_SPLIT,
        data_config.validation_start_date,
        data_config.validation_end_date,
    )
    _assert_split_range(
        split_dates,
        frame[split_column],
        TEST_SPLIT,
        data_config.test_start_date,
        data_config.test_end_date,
    )

    max_train = _max_date_for_split(split_dates, frame[split_column], TRAIN_SPLIT)
    min_validation = _min_date_for_split(
        split_dates,
        frame[split_column],
        VALIDATION_SPLIT,
    )
    max_validation = _max_date_for_split(
        split_dates,
        frame[split_column],
        VALIDATION_SPLIT,
    )
    min_test = _min_date_for_split(split_dates, frame[split_column], TEST_SPLIT)

    if max_train is not None and min_validation is not None and max_train >= min_validation:
        raise ValueError("train split must end before validation split starts")
    if max_validation is not None and min_test is not None and max_validation >= min_test:
        raise ValueError("validation split must end before test split starts")


def _assert_all_dates_assigned(frame: pd.DataFrame, date_column: str) -> None:
    if frame[SPLIT_COLUMN].isna().any():
        unassigned_dates = (
            pd.to_datetime(frame.loc[frame[SPLIT_COLUMN].isna(), date_column])
            .dt.date.astype(str)
            .drop_duplicates()
            .sort_values()
            .head(5)
            .to_list()
        )
        raise ValueError(f"dates fall outside configured split ranges: {unassigned_dates}")


def _assert_split_range(
    dates: pd.Series,
    splits: pd.Series,
    split_name: str,
    start_date: date,
    end_date: date | None,
) -> None:
    split_dates = dates.loc[splits == split_name]
    if split_dates.empty:
        return

    if split_dates.min() < start_date:
        raise ValueError(f"{split_name} split starts before configured boundary")
    if end_date is not None and split_dates.max() > end_date:
        raise ValueError(f"{split_name} split ends after configured boundary")


def _min_date_for_split(
    dates: pd.Series,
    splits: pd.Series,
    split_name: str,
) -> date | None:
    split_dates = dates.loc[splits == split_name]
    if split_dates.empty:
        return None
    return split_dates.min()


def _max_date_for_split(
    dates: pd.Series,
    splits: pd.Series,
    split_name: str,
) -> date | None:
    split_dates = dates.loc[splits == split_name]
    if split_dates.empty:
        return None
    return split_dates.max()
