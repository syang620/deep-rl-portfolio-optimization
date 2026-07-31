"""Executable target-weight ensemble policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from portfolio_rl.policies.baseline_policies import WeightPolicy


@dataclass(frozen=True)
class MemberTargetRecord:
    """One member recommendation observed during ensemble execution."""

    decision_step: int
    date: Any
    member: str
    weights: tuple[float, ...]
    live_current_weights: tuple[float, ...]


class MeanWeightEnsemblePolicy:
    """Average validated member target weights at every decision."""

    def __init__(self, *, member_policies: Mapping[str, WeightPolicy]) -> None:
        if not member_policies:
            raise ValueError("member_policies must not be empty")
        if any(not str(name).strip() for name in member_policies):
            raise ValueError("ensemble member names must not be empty")
        self._member_policies = dict(member_policies)
        self._records: list[MemberTargetRecord] = []
        self._decision_step = 0

    @property
    def member_order(self) -> list[str]:
        """Return the deterministic member evaluation order."""
        return list(self._member_policies)

    @property
    def member_target_records(self) -> tuple[MemberTargetRecord, ...]:
        """Return immutable audit records from the latest execution."""
        return tuple(self._records)

    def reset(self) -> None:
        """Reset ensemble and member state before a new backtest."""
        self._records.clear()
        self._decision_step = 0
        for policy in self._member_policies.values():
            if hasattr(policy, "reset"):
                policy.reset()

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        """Return the arithmetic mean of valid member target weights."""
        if "asset_order" not in info:
            raise ValueError("info must include asset_order")
        n_assets = len(info["asset_order"])
        if "current_weights" not in info:
            raise ValueError("info must include current_weights")
        live_current_weights = np.asarray(
            info["current_weights"],
            dtype=np.float64,
        )
        _validate_weights(live_current_weights, n_assets, "live current")
        member_targets = []
        for member, policy in self._member_policies.items():
            target = np.asarray(
                policy.target_weights(observation, info),
                dtype=np.float64,
            )
            _validate_weights(target, n_assets, f"member {member}")
            member_targets.append(target)
            self._records.append(
                MemberTargetRecord(
                    decision_step=self._decision_step,
                    date=info.get("date"),
                    member=member,
                    weights=tuple(float(value) for value in target),
                    live_current_weights=tuple(
                        float(value) for value in live_current_weights
                    ),
                )
            )
        ensemble_target = np.mean(np.stack(member_targets), axis=0)
        _validate_weights(ensemble_target, n_assets, "ensemble")
        self._decision_step += 1
        return ensemble_target


def _validate_weights(weights: np.ndarray, n_assets: int, label: str) -> None:
    if weights.shape != (n_assets,):
        raise ValueError(f"{label} target weights must have shape ({n_assets},)")
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} target weights must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"{label} target weights must be nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"{label} target weights must sum to one")
