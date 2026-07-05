"""Training callbacks for PPO experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.policies.sb3_policy import SB3WeightPolicy


class ValidationCheckpointCallback(BaseCallback):
    """Periodically backtest the in-memory model and save the best checkpoint."""

    def __init__(
        self,
        *,
        validation_store: PortfolioFeatureStore,
        action_temperature: float,
        rebalance_frequency_trading_days: int,
        transaction_cost_bps: float,
        eval_freq_timesteps: int,
        metric_for_best_model: str,
        output_dir: str | Path,
    ) -> None:
        super().__init__(verbose=0)
        if eval_freq_timesteps <= 0:
            raise ValueError("eval_freq_timesteps must be positive")
        self.validation_store = validation_store
        self.action_temperature = float(action_temperature)
        self.rebalance_frequency_trading_days = int(rebalance_frequency_trading_days)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.eval_freq_timesteps = int(eval_freq_timesteps)
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = Path(output_dir)
        self.best_score: float | None = None
        self.best_metrics: dict[str, float | None] | None = None
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq_timesteps:
            self._run_validation()
        return True

    def _on_training_end(self) -> None:
        if self.best_metrics is None:
            self._run_validation()

    def _run_validation(self) -> None:
        result = run_weight_policy_backtest(
            feature_store=self.validation_store,
            policy=SB3WeightPolicy(
                model=self.model,
                action_temperature=self.action_temperature,
            ),
            strategy="ppo",
            rebalance_frequency_trading_days=self.rebalance_frequency_trading_days,
            transaction_cost_bps=self.transaction_cost_bps,
        )
        self._last_eval_timestep = int(self.num_timesteps)
        score = validation_metric_value(
            result.metrics,
            result.nav,
            self.metric_for_best_model,
        )
        if is_metric_improvement(score, self.best_score):
            self.best_score = float(score)
            self.best_metrics = result.metrics
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.output_dir / "best_model.zip")
            (self.output_dir / "best_metrics_validation.json").write_text(
                json.dumps(result.metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def validation_metric_value(
    metrics: dict[str, float | None],
    nav: pd.DataFrame,
    metric_name: str,
) -> float | None:
    """Return the configured validation metric from backtest outputs."""
    if metric_name == "final_nav":
        if nav.empty:
            return None
        return float(nav.sort_values("date")["nav"].iloc[-1])
    return metrics.get(metric_name)


def is_metric_improvement(
    candidate: float | None,
    best: float | None,
) -> bool:
    """Return whether candidate is a finite higher-is-better improvement."""
    if candidate is None or not np.isfinite(candidate):
        return False
    if best is None:
        return True
    return float(candidate) > float(best)
