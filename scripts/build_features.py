"""CLI entrypoint for Phase 1 feature artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.config.loader import (
    load_data_config,
    load_features_config,
    load_universe_config,
)
from portfolio_rl.features.pipeline import build_feature_artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 feature artifacts.")
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to configs/data.yaml",
    )
    parser.add_argument(
        "--feature-config",
        required=True,
        help="Path to configs/features.yaml",
    )
    parser.add_argument(
        "--universe",
        required=True,
        help="Path to configs/universe.yaml",
    )
    args = parser.parse_args(argv)

    result = build_feature_artifacts(
        data_config=load_data_config(Path(args.data_config)),
        feature_config=load_features_config(Path(args.feature_config)),
        universe_config=load_universe_config(Path(args.universe)),
    )
    print(
        json.dumps(
            {
                "features_parquet_path": str(result.features_parquet_path),
                "global_features_parquet_path": str(
                    result.global_features_parquet_path
                ),
                "normalized_features_parquet_path": str(
                    result.normalized_features_parquet_path
                ),
                "normalized_global_features_parquet_path": str(
                    result.normalized_global_features_parquet_path
                ),
                "model_matrix_parquet_path": str(result.model_matrix_parquet_path),
                "scaler_artifact_path": str(result.scaler_artifact_path),
                "feature_spec_path": str(result.feature_spec_path),
                "data_quality_report_path": str(result.data_quality_report_path),
                "duckdb_path": str(result.duckdb_path),
                "features_row_count": result.features_row_count,
                "global_features_row_count": result.global_features_row_count,
                "normalized_features_row_count": (
                    result.normalized_features_row_count
                ),
                "normalized_global_features_row_count": (
                    result.normalized_global_features_row_count
                ),
                "model_matrix_row_count": result.model_matrix_row_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
