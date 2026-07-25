from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from portfolio_rl.training.experiment_runner import expand_experiment_matrix
from scripts.run_experiment_matrix import main


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def test_default_experiment_expands_to_one_run() -> None:
    plans = expand_experiment_matrix(
        CONFIG_DIR / "ppo_phase3_default.yaml",
        root=REPO_ROOT,
    )

    assert len(plans) == 1
    assert plans[0].seed == 42
    assert plans[0].total_timesteps == 500000
    assert plans[0].overrides == {}
    assert plans[0].run_id.startswith("ppo_phase3_default_")


def test_seed_sweep_expands_to_five_runs() -> None:
    plans = expand_experiment_matrix(
        CONFIG_DIR / "ppo_phase3_seed_sweep.yaml",
        root=REPO_ROOT,
    )

    assert len(plans) == 5
    assert [plan.seed for plan in plans] == [7, 42, 101, 202, 999]


def test_temperature_sweep_expands_to_twenty_four_runs() -> None:
    plans = expand_experiment_matrix(
        CONFIG_DIR / "ppo_phase3_temperature_sweep.yaml",
        root=REPO_ROOT,
    )

    assert len(plans) == 24
    assert plans[0].overrides == {
        "env.action_temperature": 0.25,
        "ppo.ent_coef": 0.005,
    }
    assert plans[-1].overrides == {
        "env.action_temperature": 1.0,
        "ppo.ent_coef": 0.01,
    }


def test_expansion_and_run_ids_are_deterministic(tmp_path: Path) -> None:
    first_config = _write_experiment_config(
        tmp_path / "first.yaml",
        overrides={
            "env.action_temperature": [0.5, 1.0],
            "ppo.ent_coef": [0.0, 0.01],
        },
    )
    reordered_config = _write_experiment_config(
        tmp_path / "reordered.yaml",
        overrides={
            "ppo.ent_coef": [0.0, 0.01],
            "env.action_temperature": [0.5, 1.0],
        },
    )

    first = expand_experiment_matrix(first_config, root=REPO_ROOT)
    repeated = expand_experiment_matrix(first_config, root=REPO_ROOT)
    reordered = expand_experiment_matrix(reordered_config, root=REPO_ROOT)

    assert first == repeated
    assert first == reordered


def test_unknown_override_path_fails_before_execution(tmp_path: Path) -> None:
    config_path = _write_experiment_config(
        tmp_path / "unknown.yaml",
        overrides={"ppo.unknown": [1]},
    )

    with pytest.raises(ValueError, match="unknown override path"):
        expand_experiment_matrix(config_path, root=REPO_ROOT)


def test_schema_invalid_override_fails_before_execution(tmp_path: Path) -> None:
    config_path = _write_experiment_config(
        tmp_path / "invalid.yaml",
        overrides={"env.action_temperature": [-1.0]},
    )

    with pytest.raises(ValidationError, match="action_temperature"):
        expand_experiment_matrix(config_path, root=REPO_ROOT)


def test_dry_run_output_is_stable_and_writes_no_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_experiment_config(
        tmp_path / "dry_run.yaml",
        overrides={"env.action_temperature": [0.5]},
    )
    expected_paths = {config_path}

    argv = [
        "--config",
        str(config_path),
        "--root",
        str(REPO_ROOT),
        "--dry-run",
    ]
    main(argv)
    first_output = capsys.readouterr().out
    main(argv)
    second_output = capsys.readouterr().out

    assert first_output == second_output
    assert first_output.startswith("run_id\tseed\ttotal_timesteps\toverrides\n")
    assert set(tmp_path.iterdir()) == expected_paths


def test_cli_refuses_execution_without_dry_run(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path / "execute.yaml")

    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(config_path), "--root", str(REPO_ROOT)])


def _write_experiment_config(
    path: Path,
    *,
    overrides: dict[str, list[object]] | None = None,
) -> Path:
    payload = {
        "experiment_name": "test_matrix",
        "base_data_config": str(REPO_ROOT / "configs" / "data.yaml"),
        "base_env_config": str(REPO_ROOT / "configs" / "env.yaml"),
        "base_train_config": str(REPO_ROOT / "configs" / "train_ppo.yaml"),
        "run_id_prefix": "test_matrix",
        "seeds": [7, 42],
        "total_timesteps": 1000,
        "overrides": overrides or {},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
