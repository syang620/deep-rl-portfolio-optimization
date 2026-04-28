"""CLI entrypoint for Phase 1 raw ETL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.config.loader import load_data_config, load_universe_config
from portfolio_rl.data.etl import run_raw_etl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 raw ETL.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    parser.add_argument(
        "--universe",
        required=True,
        help="Path to configs/universe.yaml",
    )
    args = parser.parse_args(argv)

    result = run_raw_etl(
        data_config=load_data_config(Path(args.config)),
        universe_config=load_universe_config(Path(args.universe)),
    )
    print(
        json.dumps(
            {
                "prices_parquet_path": str(result.prices_parquet_path),
                "macro_parquet_path": str(result.macro_parquet_path),
                "duckdb_path": str(result.duckdb_path),
                "prices_row_count": result.prices_row_count,
                "macro_row_count": result.macro_row_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
