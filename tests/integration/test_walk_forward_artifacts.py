from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import (
    load_outer_evaluation_dataset,
    load_training_selection_dataset,
)
from portfolio_rl.features.fold_pipeline import build_walk_forward_artifacts


def test_walk_forward_artifacts_are_isolated_and_reproducible(
    tmp_path: Path,
) -> None:
    config_path = _fixture(tmp_path)
    first = build_walk_forward_artifacts(
        config_path=config_path,
        root=tmp_path,
        output_root="artifacts/first",
    )
    second = build_walk_forward_artifacts(
        config_path=config_path,
        root=tmp_path,
        output_root="artifacts/second",
    )

    first_campaign = _json(first.campaign_manifest)
    second_campaign = _json(second.campaign_manifest)
    assert first_campaign == second_campaign
    assert first_campaign["contains_2024_or_later"] is False
    for fold_id in ("WF1", "WF2", "WF3", "WF4"):
        first_dir = first.fold_directories[fold_id]
        second_dir = second.fold_directories[fold_id]
        first_manifest = _json(first_dir / "fold_manifest.json")
        second_manifest = _json(second_dir / "fold_manifest.json")
        assert first_manifest == second_manifest
        assert first_manifest["normalization"]["fit_split"] == "inner_train"
        assert first_manifest["access_contract"][
            "outer_accessed_during_training_or_selection"
        ] is False
        assert first_manifest["artifact_hashes"] == second_manifest["artifact_hashes"]
        assert (first_dir / "feature_spec.json").read_bytes() == (
            tmp_path / "feature_spec.json"
        ).read_bytes()

        training = load_training_selection_dataset(first_dir)
        outer = load_outer_evaluation_dataset(first_dir)
        assert set(training.splits) == {"inner_train", "inner_validation"}
        assert set(outer.splits) == {"outer_evaluation"}
        assert outer.dates.max() < pd.Timestamp("2024-01-01")
        store = PortfolioFeatureStore(outer, "outer_evaluation")
        with pytest.raises(IndexError, match="split boundary"):
            store.get_forward_log_returns(store.n_rows - 1, 1)

    with pytest.raises(FileExistsError):
        build_walk_forward_artifacts(
            config_path=config_path,
            root=tmp_path,
            output_root="artifacts/first",
        )


def _fixture(root: Path) -> Path:
    dates = pd.date_range("2010-01-04", "2023-12-29", freq="B")
    asset = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "feature_version": "v1",
                    "ret_1d": 0.001 * np.sin(np.arange(len(dates)) + offset),
                    "split": "legacy",
                }
            )
            for offset, ticker in enumerate(("SPY", "SHY"))
        ],
        ignore_index=True,
    ).sort_values(["date", "ticker"], ignore_index=True)
    global_frame = pd.DataFrame(
        {
            "date": dates,
            "feature_version": "v1",
            "global_signal": np.cos(np.arange(len(dates))),
            "split": "legacy",
        }
    )
    asset.to_parquet(root / "asset.parquet", index=False)
    global_frame.to_parquet(root / "global.parquet", index=False)
    feature_spec = {
        "feature_version": "v1",
        "asset_order": ["SPY", "SHY"],
        "per_asset_features": ["ret_1d"],
        "global_features": ["global_signal"],
        "current_weight_features": ["weight_spy", "weight_shy"],
        "observation_dim": 5,
        "created_at": "frozen",
    }
    (root / "feature_spec.json").write_text(
        json.dumps(feature_spec, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in ("universe.yaml", "data.yaml"):
        (root / name).write_text("fixture: true\n", encoding="utf-8")
    source_paths = {
        "asset_features": root / "asset.parquet",
        "global_features": root / "global.parquet",
        "feature_spec": root / "feature_spec.json",
        "features_config": Path("configs/features.yaml").resolve(),
        "universe_config": root / "universe.yaml",
        "data_config": root / "data.yaml",
    }
    config = {
        "schema_version": 1,
        "output_root": "artifacts/walk_forward/data",
        "raw_history_start_date": "2007-01-01",
        "maximum_feature_lookback_trading_days": 252,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "folds": [
            _fold("WF1", "2014", "2015", "2016", "2017"),
            _fold("WF2", "2016", "2017", "2018", "2019"),
            _fold("WF3", "2018", "2019", "2020", "2021"),
            _fold("WF4", "2020", "2021", "2022", "2023"),
        ],
    }
    path = root / "walk_forward.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fold(
    fold_id: str,
    train_end: str,
    validation_year: str,
    outer_start: str,
    outer_end: str,
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "inner_train": {"start": "2010-01-01", "end": f"{train_end}-12-31"},
        "inner_validation": {
            "start": f"{validation_year}-01-01",
            "end": f"{validation_year}-12-31",
        },
        "outer_evaluation": {
            "start": f"{outer_start}-01-01",
            "end": f"{outer_end}-12-31",
        },
    }


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
