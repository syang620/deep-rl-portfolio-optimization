from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import (
    load_outer_evaluation_dataset,
    load_training_selection_dataset,
)
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.features.fold_pipeline import build_walk_forward_artifacts


class CapturingEqualWeightPolicy:
    def __init__(self) -> None:
        self.trailing_returns: list[np.ndarray] = []

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation
        self.trailing_returns.append(
            np.asarray(info["trailing_log_returns"]).copy()
        )
        return np.array([0.5, 0.5])


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
        assert first_campaign["fold_artifact_hashes"][fold_id] == first_manifest[
            "artifact_hashes"
        ]
        for artifact in first_manifest["artifact_hashes"].values():
            assert len(artifact["file_sha256"]) == 64
            assert len(artifact["logical_sha256"]) == 64
        assert first_manifest["inner_train_end"] < first_manifest[
            "inner_validation_start"
        ]
        assert first_manifest["inner_validation_end"] < first_manifest[
            "outer_evaluation_start"
        ]
        assert first_manifest["outer_evaluation_end"] <= "2023-12-31"
        assert first_manifest["matrix_width_breakdown"] == {
            "metadata_columns": 3,
            "observation_columns": 5,
            "return_columns": 2,
            "total_columns": 10,
        }
        assert first_manifest["feature_contract"] == {
            "asset_order": ["SPY", "SHY"],
            "per_asset_feature_order": ["ret_1d"],
            "global_feature_order": ["global_signal"],
            "current_weight_order": ["weight_spy", "weight_shy"],
            "observation_dimension": 5,
            "return_column_order": ["return_spy_1d", "return_shy_1d"],
        }
        assert (first_dir / "feature_spec.json").read_bytes() == (
            tmp_path / "feature_spec.json"
        ).read_bytes()

        training = load_training_selection_dataset(first_dir)
        outer_with_context = load_outer_evaluation_dataset(first_dir)
        assert set(training.splits) == {"inner_train", "inner_validation"}
        assert set(outer_with_context.splits) == {
            "inner_train",
            "inner_validation",
            "outer_evaluation",
        }
        assert outer_with_context.dates.max() < pd.Timestamp("2024-01-01")
        store = PortfolioFeatureStore(outer_with_context, "outer_evaluation")
        expected_context = store.get_pre_window_log_returns(63)
        policy = CapturingEqualWeightPolicy()
        run_weight_policy_backtest(
            feature_store=store,
            policy=policy,
            strategy="capture",
            max_steps=1,
            inverse_vol_lookback_trading_days=63,
        )
        np.testing.assert_array_equal(policy.trailing_returns[0], expected_context)
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
