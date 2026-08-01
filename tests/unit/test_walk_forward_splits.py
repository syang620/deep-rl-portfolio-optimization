from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import load_features_config
from portfolio_rl.data.walk_forward import (
    assign_fold_splits,
    load_walk_forward_config,
)
from portfolio_rl.features.builder import build_features
from portfolio_rl.features.normalization import fit_normalization_artifact


def test_canonical_walk_forward_folds_are_ordered_and_non_overlapping() -> None:
    config = load_walk_forward_config("configs/walk_forward.yaml")

    assert [fold.fold_id for fold in config.folds] == ["WF1", "WF2", "WF3", "WF4"]
    for fold in config.folds:
        assert fold.inner_train.end < fold.inner_validation.start
        assert fold.inner_validation.end < fold.outer_evaluation.start
        assert fold.outer_evaluation.end < pd.Timestamp("2024-01-01")


def test_assign_fold_splits_excludes_rows_outside_fold() -> None:
    fold = load_walk_forward_config("configs/walk_forward.yaml").folds[0]
    dates = pd.to_datetime(
        ["2009-12-31", "2010-01-04", "2015-01-02", "2016-01-04", "2018-01-02"]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "split": "legacy",
            "feature_version": "v1",
            "value": np.arange(len(dates)),
        }
    )

    assigned = assign_fold_splits(frame, fold)

    assert assigned["split"].tolist() == [
        "inner_train",
        "inner_validation",
        "outer_evaluation",
    ]
    assert assigned["date"].dt.year.tolist() == [2010, 2015, 2016]


def test_fold_scaler_is_invariant_to_validation_and_outer_values() -> None:
    config = load_features_config("configs/features.yaml")
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6),
            "ticker": "SPY",
            "split": ["inner_train"] * 3 + ["inner_validation", "outer_evaluation", "outer_evaluation"],
            "feature_version": "v1",
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    perturbed = frame.copy()
    perturbed.loc[perturbed["split"] != "inner_train", "feature"] = 1_000_000.0

    original = fit_normalization_artifact(frame, config, fit_split="inner_train")
    changed = fit_normalization_artifact(perturbed, config, fit_split="inner_train")

    assert original == changed
    assert original.fit_split == "inner_train"


def test_rolling_feature_prefix_is_invariant_to_future_rows() -> None:
    config = load_features_config("configs/features.yaml")
    dates = pd.date_range("2018-01-02", periods=380, freq="B")
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": base
                    + 0.1 * np.arange(len(dates))
                    + np.sin(np.arange(len(dates)) / 9.0 + offset),
                    "volume": 1_000_000 + 100 * np.arange(len(dates)),
                }
            )
            for ticker, base, offset in (("SPY", 100.0, 0.0), ("QQQ", 130.0, 1.0))
        ],
        ignore_index=True,
    )
    prices["close"] = prices["adj_close"]
    macro = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "series_id": series,
                    "value": base + 0.001 * np.arange(len(dates)),
                }
            )
            for series, base in (
                ("VIXCLS", 20.0),
                ("DGS2", 3.0),
                ("DGS10", 4.0),
                ("T10Y2Y", 1.0),
                ("BAMLH0A0HYM2", 3.5),
            )
        ],
        ignore_index=True,
    )
    cutoff = dates[330]
    full = build_features(prices, macro, config, "SPY", ["SPY", "QQQ"])
    prefix = build_features(
        prices[prices["date"] <= cutoff],
        macro[macro["date"] <= cutoff],
        config,
        "SPY",
        ["SPY", "QQQ"],
    )

    expected_asset = full.asset_features[
        full.asset_features["date"] <= cutoff
    ].reset_index(drop=True)
    expected_global = full.global_features[
        full.global_features["date"] <= cutoff
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix.asset_features, expected_asset)
    pd.testing.assert_frame_equal(prefix.global_features, expected_global)
