"""Model matrix assembly for Phase 1 training and environment data."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from portfolio_rl.features.feature_spec import FeatureSpec, flatten_features


IDENTIFIER_COLUMNS = ("date", "split", "feature_version")
ASSET_IDENTIFIER_COLUMNS = ("date", "ticker", "split", "feature_version")
RETURN_FEATURE_COLUMN = "ret_1d"


def build_model_matrix(
    normalized_asset_features: pd.DataFrame,
    normalized_global_features: pd.DataFrame,
    raw_asset_features: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """Build one row per date with flattened observations and one-day returns."""
    asset_features = _prepare_asset_features(normalized_asset_features, feature_spec)
    global_features = _prepare_global_features(normalized_global_features, feature_spec)
    returns = _prepare_return_frame(raw_asset_features, feature_spec)
    equal_weights = {
        ticker: 1.0 / len(feature_spec.asset_order)
        for ticker in feature_spec.asset_order
    }

    rows = []
    obs_columns = _observation_columns(feature_spec)
    return_columns = _return_columns(feature_spec)
    for date, global_row in global_features.iterrows():
        asset_rows = asset_features.loc[asset_features["date"] == date]
        _assert_asset_rows_for_date(asset_rows, global_row, feature_spec, date)
        if date not in returns.index:
            raise ValueError(f"raw returns are missing date: {date.date()}")

        observation = flatten_features(
            asset_rows,
            global_row,
            equal_weights,
            feature_spec,
        )
        return_values = returns.loc[date, feature_spec.asset_order]
        row = {
            "date": date,
            "split": global_row["split"],
            "feature_version": global_row["feature_version"],
        }
        row.update(dict(zip(obs_columns, observation, strict=True)))
        row.update(dict(zip(return_columns, return_values, strict=True)))
        rows.append(row)

    model_matrix = pd.DataFrame(rows)
    _assert_clean_model_matrix(model_matrix, obs_columns + return_columns)
    return model_matrix


def _prepare_asset_features(
    features: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    _require_columns(
        features,
        [*ASSET_IDENTIFIER_COLUMNS, *feature_spec.per_asset_features],
        "normalized asset features",
    )
    result = features.loc[
        :,
        [*ASSET_IDENTIFIER_COLUMNS, *feature_spec.per_asset_features],
    ].copy()
    result["date"] = pd.to_datetime(result["date"])
    result["ticker"] = result["ticker"].str.upper()
    return result.sort_values(["date", "ticker"], ignore_index=True)


def _prepare_global_features(
    features: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    _require_columns(
        features,
        [*IDENTIFIER_COLUMNS, *feature_spec.global_features],
        "normalized global features",
    )
    result = features.loc[
        :,
        [*IDENTIFIER_COLUMNS, *feature_spec.global_features],
    ].copy()
    result["date"] = pd.to_datetime(result["date"])
    if result["date"].duplicated().any():
        raise ValueError("normalized global features must contain one row per date")
    return result.sort_values("date").set_index("date")


def _prepare_return_frame(
    features: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    _require_columns(
        features,
        ["date", "ticker", RETURN_FEATURE_COLUMN],
        "raw asset features",
    )
    result = features.loc[:, ["date", "ticker", RETURN_FEATURE_COLUMN]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result["ticker"] = result["ticker"].str.upper()
    returns = result.pivot(
        index="date",
        columns="ticker",
        values=RETURN_FEATURE_COLUMN,
    ).sort_index()
    missing_tickers = [
        ticker for ticker in feature_spec.asset_order if ticker not in returns.columns
    ]
    if missing_tickers:
        raise ValueError(f"raw returns are missing tickers: {missing_tickers}")
    return returns.loc[:, feature_spec.asset_order]


def _assert_asset_rows_for_date(
    asset_rows: pd.DataFrame,
    global_row: pd.Series,
    feature_spec: FeatureSpec,
    date: pd.Timestamp,
) -> None:
    tickers = set(asset_rows["ticker"])
    if tickers != set(feature_spec.asset_order):
        raise ValueError(f"normalized asset features have incomplete coverage on {date.date()}")
    if set(asset_rows["split"]) != {global_row["split"]}:
        raise ValueError(f"asset/global split mismatch on {date.date()}")
    if set(asset_rows["feature_version"]) != {global_row["feature_version"]}:
        raise ValueError(f"asset/global feature_version mismatch on {date.date()}")


def _assert_clean_model_matrix(
    model_matrix: pd.DataFrame,
    numeric_columns: list[str],
) -> None:
    if model_matrix.empty:
        raise ValueError("model matrix is empty")
    numeric = model_matrix.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not numeric.notna().all().all():
        raise ValueError("model matrix contains NaN values")
    if not numeric.map(isfinite).all().all():
        raise ValueError("model matrix contains infinite values")


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _observation_columns(feature_spec: FeatureSpec) -> list[str]:
    return [f"obs_{index:03d}" for index in range(feature_spec.observation_dim)]


def _return_columns(feature_spec: FeatureSpec) -> list[str]:
    return [f"return_{ticker.lower()}_1d" for ticker in feature_spec.asset_order]
