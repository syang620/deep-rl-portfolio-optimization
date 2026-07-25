from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from portfolio_rl.training.experiment_runner import (
    execute_experiment_matrix,
    expand_experiment_matrix,
    write_experiment_matrix_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "configs/experiments/ppo_phase3_smoke.yaml"


def test_experiment_runner_executes_bounded_smoke_matrix(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrices"
    experiment_root = tmp_path / "experiments"
    write_experiment_matrix_plan(
        SMOKE_CONFIG,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plan = expand_experiment_matrix(SMOKE_CONFIG, root=REPO_ROOT)[0]

    result = execute_experiment_matrix(
        SMOKE_CONFIG,
        max_runs=1,
        root=REPO_ROOT,
        matrix_output_root=matrix_root,
        experiment_output_root=experiment_root,
    )

    assert result.attempted_count == 1
    assert result.completed_count == 1
    assert result.skipped_count == 0
    assert result.remaining_count == 0
    assert result.results[0].status == "completed"
    assert result.results[0].model_path.is_file()
    run_manifest = json.loads(
        (experiment_root / plan.run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["git_commit"]
    assert run_manifest["seed"] == 42
    assert run_manifest["total_timesteps"] == 1000
    matrix_manifest = json.loads(
        (
            matrix_root
            / "ppo_phase3_smoke"
            / "experiment_matrix_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert matrix_manifest["runs"][0]["status"] == "completed"
