from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.data.storage import (
    read_duckdb_table,
    read_parquet,
    write_duckdb_table,
    write_parquet,
)


def test_write_and_read_parquet_round_trips_rows(tmp_path: Path) -> None:
    frame = _storage_frame()
    output_path = tmp_path / "raw" / "prices_daily.parquet"

    result_path = write_parquet(frame, output_path)
    round_trip = read_parquet(result_path)

    assert result_path == output_path
    assert result_path.exists()
    assert len(round_trip) == len(frame)
    assert round_trip.to_dict("records") == frame.to_dict("records")


def test_write_and_read_duckdb_table_round_trips_rows(tmp_path: Path) -> None:
    frame = _storage_frame()
    database_path = tmp_path / "duckdb" / "portfolio.duckdb"

    row_count = write_duckdb_table(frame, database_path, "prices_daily")
    round_trip = read_duckdb_table(database_path, "prices_daily")

    assert row_count == len(frame)
    assert database_path.exists()
    assert len(round_trip) == len(frame)
    assert round_trip.to_dict("records") == frame.to_dict("records")


def test_write_duckdb_table_replaces_existing_table(tmp_path: Path) -> None:
    database_path = tmp_path / "portfolio.duckdb"

    write_duckdb_table(_storage_frame(), database_path, "prices_daily")
    replacement = _storage_frame().iloc[[0]].reset_index(drop=True)
    row_count = write_duckdb_table(replacement, database_path, "prices_daily")

    round_trip = read_duckdb_table(database_path, "prices_daily")
    assert row_count == 1
    assert len(round_trip) == 1
    assert round_trip.to_dict("records") == replacement.to_dict("records")


def test_write_duckdb_table_can_fail_if_table_exists(tmp_path: Path) -> None:
    database_path = tmp_path / "portfolio.duckdb"

    write_duckdb_table(_storage_frame(), database_path, "prices_daily")

    with pytest.raises(Exception, match="already exists"):
        write_duckdb_table(
            _storage_frame(),
            database_path,
            "prices_daily",
            if_exists="fail",
        )


def test_write_duckdb_table_rejects_empty_table_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="table_name must not be empty"):
        write_duckdb_table(_storage_frame(), tmp_path / "portfolio.duckdb", " ")


def _storage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["SPY", "SPY"],
            "adj_close": [470.5, 471.5],
            "volume": [1000, 1100],
        }
    )
