"""Asset, exposure, disagreement, and static-mix attribution helpers."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


class ConstantTargetPolicy:
    """Rebalance to one frozen target at every weekly decision."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = _validated_target(target)

    def target_weights(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        del observation, info
        return self._target.copy()


class BuyAndHoldTargetPolicy:
    """Trade to one frozen target once and make no subsequent trades."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = _validated_target(target)
        self._first = True

    def reset(self) -> None:
        self._first = True

    def target_weights(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        del observation
        if self._first:
            self._first = False
            return self._target.copy()
        return np.asarray(info["current_weights"], dtype=np.float64).copy()


def build_asset_contributions(
    *,
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    reference_strategy: str,
) -> pd.DataFrame:
    """Add exact daily cost effects and arithmetic active asset contribution."""
    keys = ["fold_id", "strategy", "date"]
    gross = (
        positions.groupby(keys, as_index=False)["gross_return_contribution"]
        .sum()
        .rename(columns={"gross_return_contribution": "gross_daily_return"})
    )
    daily = nav[keys + ["daily_return"]].merge(
        gross, on=keys, validate="one_to_one"
    )
    daily["transaction_cost_return_effect"] = (
        daily["daily_return"] - daily["gross_daily_return"]
    )
    enriched = positions.merge(
        daily[keys + ["daily_return", "gross_daily_return", "transaction_cost_return_effect"]],
        on=keys,
        validate="many_to_one",
    )
    reference = enriched[enriched["strategy"] == reference_strategy][
        ["fold_id", "date", "ticker", "gross_return_contribution"]
    ].rename(
        columns={"gross_return_contribution": "reference_gross_contribution"}
    )
    enriched = enriched.merge(
        reference, on=["fold_id", "date", "ticker"], validate="many_to_one"
    )
    enriched["active_gross_contribution"] = (
        enriched["gross_return_contribution"]
        - enriched["reference_gross_contribution"]
    )
    reconciliation = enriched.groupby(keys)["gross_return_contribution"].sum()
    expected = daily.set_index(keys)["gross_daily_return"]
    if not np.allclose(reconciliation.sort_index(), expected.sort_index(), atol=1e-10):
        raise AssertionError("asset contributions do not reconcile to gross return")
    return enriched


def build_exposure_paths(
    *, positions: pd.DataFrame, exposure_groups: Mapping[str, list[str]]
) -> pd.DataFrame:
    """Aggregate daily pre-return holdings into exhaustive asset groups."""
    ticker_to_group = _ticker_group_map(exposure_groups)
    result = positions.copy()
    result["exposure_group"] = result["ticker"].map(ticker_to_group)
    if result["exposure_group"].isna().any():
        raise ValueError("positions contain tickers absent from exposure_groups")
    grouped = result.groupby(
        ["fold_id", "strategy", "date", "exposure_group"], as_index=False
    )["pre_return_weight"].sum()
    totals = grouped.groupby(["fold_id", "strategy", "date"])[
        "pre_return_weight"
    ].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-6):
        raise AssertionError("asset-group exposures must sum to one")
    return grouped.rename(columns={"pre_return_weight": "exposure"})


def calculate_seed_disagreement(member_targets: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold/strategy member disagreement at each decision."""
    rows = []
    keys = [
        "fold_id",
        "scenario_group",
        "cost_scenario",
        "execution_delay_closes",
        "strategy",
        "date",
        "decision_step",
    ]
    for values, frame in member_targets.groupby(keys, sort=True):
        pivot = frame.pivot(index="member", columns="ticker", values="target_weight")
        matrix = pivot.to_numpy(dtype=np.float64)
        distances = [
            0.5 * float(np.abs(matrix[left] - matrix[right]).sum())
            for left, right in itertools.combinations(range(len(matrix)), 2)
        ]
        dominant = pivot.idxmax(axis=1)
        mode = str(dominant.mode().iloc[0])
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "median_pairwise_target_half_l1": float(np.median(distances)),
                "max_pairwise_target_half_l1": float(np.max(distances)),
                "mean_asset_target_std": float(np.std(matrix, axis=0).mean()),
                "dominant_asset": mode,
                "dominant_asset_agreement": float(np.mean(dominant == mode)),
            }
        )
    return pd.DataFrame(rows)


def largest_active_weeks(
    *,
    nav: pd.DataFrame,
    costs: pd.DataFrame,
    strategies: list[str],
    reference_strategy: str,
    count_each_tail: int,
) -> pd.DataFrame:
    """Return deterministic best and worst five-day active periods."""
    weekly = _weekly_returns(nav)
    reference = weekly[weekly["strategy"] == reference_strategy][
        ["fold_id", "decision_step", "week_end", "weekly_return"]
    ].rename(columns={"weekly_return": "reference_weekly_return"})
    selected = weekly[weekly["strategy"].isin(strategies)].merge(
        reference,
        on=["fold_id", "decision_step", "week_end"],
        validate="many_to_one",
    )
    selected["active_weekly_return"] = (
        selected["weekly_return"] - selected["reference_weekly_return"]
    )
    cost_summary = costs.copy()
    cost_summary["decision_step"] = cost_summary.groupby(
        ["fold_id", "strategy"]
    ).cumcount()
    selected = selected.merge(
        cost_summary[
            [
                "fold_id",
                "strategy",
                "decision_step",
                "turnover",
                "transaction_cost_fraction",
            ]
        ],
        on=["fold_id", "strategy", "decision_step"],
        validate="one_to_one",
    )
    frames = []
    for strategy, frame in selected.groupby("strategy", sort=True):
        ordered = frame.sort_values(
            ["active_weekly_return", "week_end"], kind="mergesort"
        )
        losses = ordered.head(count_each_tail).assign(tail="loss")
        gains = ordered.tail(count_each_tail).sort_values(
            "active_weekly_return", ascending=False, kind="mergesort"
        ).assign(tail="gain")
        frames.extend([losses, gains])
    return pd.concat(frames, ignore_index=True)


def largest_target_change_windows(
    *, targets: pd.DataFrame, strategies: list[str], count: int, radius: int
) -> pd.DataFrame:
    """Select objective policy-defined regime changes and surrounding decisions."""
    rows = []
    for strategy, frame in targets[targets["strategy"].isin(strategies)].groupby(
        "strategy", sort=True
    ):
        pivot = frame.pivot(index="date", columns="ticker", values="target_weight")
        values = pivot.to_numpy(dtype=np.float64)
        changes = np.concatenate(
            ([np.nan], 0.5 * np.abs(np.diff(values, axis=0)).sum(axis=1))
        )
        ranked = np.argsort(np.nan_to_num(changes, nan=-np.inf))[-count:][::-1]
        for rank, center in enumerate(ranked, start=1):
            for step in range(max(0, center - radius), min(len(pivot), center + radius + 1)):
                rows.append(
                    {
                        "strategy": strategy,
                        "event_rank": rank,
                        "event_date": pivot.index[center],
                        "date": pivot.index[step],
                        "relative_decision": step - center,
                        "event_target_half_l1_change": float(changes[center]),
                    }
                )
    return pd.DataFrame(rows)


def _weekly_returns(nav: pd.DataFrame) -> pd.DataFrame:
    frame = nav.sort_values(["fold_id", "strategy", "date"]).copy()
    frame["decision_step"] = frame.groupby(["fold_id", "strategy"]).cumcount() // 5
    rows = []
    for keys, group in frame.groupby(
        ["fold_id", "strategy", "decision_step"], sort=True
    ):
        rows.append(
            {
                "fold_id": keys[0],
                "strategy": keys[1],
                "decision_step": keys[2],
                "week_end": pd.to_datetime(group["date"]).max(),
                "weekly_return": float(np.prod(1.0 + group["daily_return"]) - 1.0),
            }
        )
    return pd.DataFrame(rows)


def _ticker_group_map(groups: Mapping[str, list[str]]) -> dict[str, str]:
    result = {}
    for group, tickers in groups.items():
        for ticker in tickers:
            if ticker in result:
                raise ValueError(f"ticker appears in multiple exposure groups: {ticker}")
            result[ticker] = group
    return result


def _validated_target(target: np.ndarray) -> np.ndarray:
    result = np.asarray(target, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError("static target must be a finite nonnegative vector")
    if not np.isclose(result.sum(), 1.0):
        raise ValueError("static target must sum to one")
    return result.copy()
