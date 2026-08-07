"""Frozen baseline target generation from one point-in-time snapshot."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from portfolio_rl.phase3b.governance import GovernanceError
from portfolio_rl.phase3b.snapshot_chain import LivePortfolioState, PointInTimeSnapshot
from portfolio_rl.policies.baseline_policies import (
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    MomentumPolicy,
    SingleAssetPolicy,
)


def generate_baseline_targets(
    *,
    snapshot: PointInTimeSnapshot,
    live_state: LivePortfolioState,
    baseline_definitions: Mapping[str, object],
) -> dict[str, tuple[float, ...]]:
    """Generate every frozen baseline target from the same snapshot."""
    definitions = baseline_definitions.get("definitions")
    if not isinstance(definitions, dict):
        raise GovernanceError("baseline definitions are unavailable")
    inverse = definitions.get("inverse_volatility")
    momentum = definitions.get("momentum_63d_top3_equal_weight")
    if not isinstance(inverse, dict) or not isinstance(momentum, dict):
        raise GovernanceError("baseline definitions are malformed")
    asset_order = list(snapshot.asset_order)
    n_assets = len(asset_order)
    trailing = np.asarray(snapshot.trailing_log_returns, dtype=np.float64)
    if trailing.shape != (63, n_assets):
        raise GovernanceError(
            "baseline trailing-return matrix must have shape (63, n_assets)"
        )
    info = {
        "asset_order": asset_order,
        "trailing_log_returns": trailing,
        "date": snapshot.decision_date,
    }
    empty_observation = np.empty(0, dtype=np.float32)
    targets = {
        "equal_weight_weekly": EqualWeightWeeklyPolicy(n_assets).target_weights(
            empty_observation, info
        ),
        "buy_and_hold_equal_weight": np.asarray(
            live_state.weights["buy_and_hold_equal_weight"], dtype=np.float64
        ),
        "inverse_volatility": InverseVolatilityPolicy(
            n_assets=n_assets,
            volatility_floor=float(inverse["volatility_floor"]),
        ).target_weights(empty_observation, info),
        "momentum_63d_top3_equal_weight": MomentumPolicy(
            n_assets=n_assets,
            top_k=int(momentum["top_k"]),
        ).target_weights(empty_observation, info),
        "spy_only": SingleAssetPolicy(asset_order, "SPY").target_weights(
            empty_observation, info
        ),
        "shy_only": SingleAssetPolicy(asset_order, "SHY").target_weights(
            empty_observation, info
        ),
    }
    result = {}
    for strategy, target in targets.items():
        values = np.asarray(target, dtype=np.float64)
        _validate(values, n_assets, strategy)
        result[strategy] = tuple(float(value) for value in values)
    return result


def _validate(values: np.ndarray, n_assets: int, strategy: str) -> None:
    if values.shape != (n_assets,):
        raise GovernanceError(f"baseline target shape mismatch: {strategy}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise GovernanceError(f"baseline target is invalid: {strategy}")
    if not np.isclose(values.sum(), 1.0):
        raise GovernanceError(f"baseline target does not sum to one: {strategy}")
