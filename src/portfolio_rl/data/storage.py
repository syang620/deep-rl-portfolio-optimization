"""Storage helpers for Phase 1 pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd


IfExists = Literal["replace", "fail"]


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet and return the resolved output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(Path(path))


def write_duckdb_table(
    frame: pd.DataFrame,
    database_path: str | Path,
    table_name: str,
    *,
    if_exists: IfExists = "replace",
) -> int:
    """Write a DataFrame to a DuckDB table and return the inserted row count."""
    if if_exists not in ("replace", "fail"):
        raise ValueError("if_exists must be 'replace' or 'fail'")

    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    table_identifier = _quote_identifier(table_name)
    create_clause = "CREATE OR REPLACE TABLE" if if_exists == "replace" else "CREATE TABLE"

    with duckdb.connect(str(db_path)) as connection:
        connection.register("input_frame", frame)
        connection.execute(
            f"{create_clause} {table_identifier} AS SELECT * FROM input_frame"
        )

    return len(frame)


def read_duckdb_table(database_path: str | Path, table_name: str) -> pd.DataFrame:
    """Read a DuckDB table into a DataFrame."""
    table_identifier = _quote_identifier(table_name)
    with duckdb.connect(str(Path(database_path)), read_only=True) as connection:
        return connection.execute(f"SELECT * FROM {table_identifier}").fetchdf()


def _quote_identifier(identifier: str) -> str:
    if not identifier.strip():
        raise ValueError("table_name must not be empty")
    return '"' + identifier.replace('"', '""') + '"'
