"""Reconcile, but never refit or approve, the frozen Phase 3B scaler."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.phase3b.governance import canonical_json_bytes
from portfolio_rl.phase3b.scaler_reconciliation import reconcile_frozen_scaler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = reconcile_frozen_scaler(
        scaler_path=root / "artifacts/scalers/feature_scaler_v1.pkl",
        feature_spec_path=root / "artifacts/feature_specs/feature_spec_v1.json",
        raw_asset_features_path=root / "data/processed/features_daily.parquet",
        normalized_asset_features_path=root
        / "data/processed/features_normalized_daily.parquet",
        raw_global_features_path=root
        / "data/processed/global_features_daily.parquet",
        normalized_global_features_path=root
        / "data/processed/global_features_normalized_daily.parquet",
        model_matrix_path=root / "data/processed/model_matrix_daily.parquet",
    ).to_payload()
    encoded = canonical_json_bytes(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
        output.write_bytes(encoded)
    print(encoded.decode("utf-8"))


if __name__ == "__main__":
    main()
