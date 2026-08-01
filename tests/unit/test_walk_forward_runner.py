from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rl.evaluation import walk_forward_report
from portfolio_rl.training.walk_forward_runner import (
    SelectionResult,
    load_walk_forward_campaign_config,
)


def test_canonical_walk_forward_campaign_is_frozen() -> None:
    config = load_walk_forward_campaign_config(
        "configs/experiments/ppo_walk_forward.yaml"
    )

    assert config.folds == ("WF1", "WF2", "WF3", "WF4")
    assert config.seeds == (7, 42, 101, 202, 999)
    assert config.total_timesteps == 500_000
    assert config.eval_freq_timesteps == 25_000
    assert (config.pilot_fold, config.pilot_seed, config.pilot_timesteps) == (
        "WF1",
        42,
        50_000,
    )
    assert config.alphas == (0.25, 0.5, 0.75, 1.0)


def test_outer_loader_runs_only_after_selection_freeze_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_walk_forward_campaign_config(
        "configs/experiments/ppo_walk_forward.yaml"
    )
    result = SelectionResult(
        fold_id="WF1",
        seed=42,
        run_id="pilot",
        output_dir=tmp_path,
        freeze_path=tmp_path / "selection_freeze.json",
        selected_model_path=tmp_path / "selected_model.zip",
    )
    calls = []

    def verify(output_dir, *, config):
        del output_dir, config
        calls.append("verify_freeze")
        return result

    def load_outer(fold_dir):
        del fold_dir
        calls.append("load_outer")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(walk_forward_report, "verify_selection_freeze", verify)
    monkeypatch.setattr(
        walk_forward_report,
        "load_outer_evaluation_dataset",
        load_outer,
    )

    with pytest.raises(RuntimeError, match="ordering assertion"):
        walk_forward_report.evaluate_frozen_selection(
            config=config,
            fold_id="WF1",
            selection_results=[result],
            output_dir=tmp_path / "outer",
            pilot=True,
        )

    assert calls == ["verify_freeze", "load_outer"]
