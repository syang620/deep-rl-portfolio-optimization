from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import portfolio_rl.training.experiment_runner as experiment_runner_module
import scripts.run_experiment_matrix as run_experiment_matrix_script
from portfolio_rl.training.experiment_runner import (
    ExperimentMatrixResult,
    ExperimentRunResult,
    execute_experiment_matrix,
    execute_experiment_run,
    expand_experiment_matrix,
    write_experiment_matrix_plan,
)
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


def test_smoke_experiment_expands_to_one_short_run() -> None:
    plans = expand_experiment_matrix(
        CONFIG_DIR / "ppo_phase3_smoke.yaml",
        root=REPO_ROOT,
    )

    assert len(plans) == 1
    assert plans[0].seed == 42
    assert plans[0].total_timesteps == 1000
    assert plans[0].overrides == {}


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


def test_write_plan_exports_manifest_runs_and_summary(tmp_path: Path) -> None:
    outputs = write_experiment_matrix_plan(
        CONFIG_DIR / "ppo_phase3_temperature_sweep.yaml",
        root=REPO_ROOT,
        output_root=tmp_path,
    )

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    with outputs["runs"].open(newline="", encoding="utf-8") as runs_file:
        rows = list(csv.DictReader(runs_file))
    summary = outputs["summary"].read_text(encoding="utf-8")

    assert list(outputs) == ["manifest", "runs", "summary"]
    assert manifest["schema_version"] == 1
    assert manifest["experiment_name"] == "ppo_phase3_temperature_sweep"
    assert manifest["run_count"] == 24
    assert len(manifest["source_config"]["sha256"]) == 64
    assert set(manifest["base_configs"]) == {"data", "env", "train"}
    assert all(
        len(config["sha256"]) == 64
        for config in manifest["base_configs"].values()
    )
    assert len(rows) == 24
    assert list(rows[0]) == [
        "run_id",
        "seed",
        "total_timesteps",
        "status",
        "env.action_temperature",
        "ppo.ent_coef",
    ]
    assert rows[0]["status"] == "planned"
    assert rows[0]["env.action_temperature"] == "0.25"
    assert rows[-1]["ppo.ent_coef"] == "0.01"
    assert "# Experiment Matrix: ppo_phase3_temperature_sweep" in summary
    assert "- Planned runs: 24" in summary


def test_invalid_matrix_writes_nothing(tmp_path: Path) -> None:
    config_path = _write_experiment_config(
        tmp_path / "invalid_write.yaml",
        overrides={"env.action_temperature": [-1.0]},
    )
    output_root = tmp_path / "outputs"

    with pytest.raises(ValidationError, match="action_temperature"):
        write_experiment_matrix_plan(
            config_path,
            root=REPO_ROOT,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_write_plan_refuses_overwrite_unless_forced(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path / "overwrite.yaml")
    output_root = tmp_path / "outputs"
    outputs = write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=output_root,
    )
    unrelated_path = outputs["manifest"].parent / "keep.txt"
    unrelated_path.write_text("keep", encoding="utf-8")
    outputs["manifest"].write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass force=True"):
        write_experiment_matrix_plan(
            config_path,
            root=REPO_ROOT,
            output_root=output_root,
        )

    refreshed = write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=output_root,
        force=True,
    )
    manifest = json.loads(refreshed["manifest"].read_text(encoding="utf-8"))
    assert manifest["run_count"] == 2
    assert unrelated_path.read_text(encoding="utf-8") == "keep"


def test_write_plan_cli_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_experiment_config(tmp_path / "cli_write.yaml")
    output_root = tmp_path / "matrix_outputs"

    main(
        [
            "--config",
            str(config_path),
            "--root",
            str(REPO_ROOT),
            "--write-plan",
            "--output-root",
            str(output_root),
        ]
    )
    output = capsys.readouterr().out

    matrix_dir = output_root / "test_matrix"
    assert "manifest:" in output
    assert set(path.name for path in matrix_dir.iterdir()) == {
        "experiment_matrix_manifest.json",
        "runs.csv",
        "summary.md",
    }


def test_dry_run_rejects_write_only_options(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path / "invalid_cli.yaml")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--config",
                str(config_path),
                "--root",
                str(REPO_ROOT),
                "--dry-run",
                "--force",
            ]
        )


def test_execute_run_resolves_configs_updates_status_and_skips_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(
        tmp_path / "execute_one.yaml",
        overrides={
            "env.action_temperature": [0.75],
            "ppo.ent_coef": [0.0],
        },
    )
    matrix_root = tmp_path / "matrices"
    experiment_root = tmp_path / "experiments"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plan = expand_experiment_matrix(config_path, root=REPO_ROOT)[0]
    calls: list[dict[str, object]] = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        env_config = yaml.safe_load(
            Path(kwargs["env_config_path"]).read_text(encoding="utf-8")
        )
        train_config = yaml.safe_load(
            Path(kwargs["train_config_path"]).read_text(encoding="utf-8")
        )
        assert env_config["action_temperature"] == 0.75
        assert train_config["seed"] == 7
        assert train_config["total_timesteps"] == 1000
        assert train_config["ppo"]["ent_coef"] == 0.0
        return _write_fake_completed_run(kwargs)

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fake_training,
    )

    completed = execute_experiment_run(
        config_path,
        plan.run_id,
        root=REPO_ROOT,
        matrix_output_root=matrix_root,
        experiment_output_root=experiment_root,
    )
    skipped = execute_experiment_run(
        config_path,
        plan.run_id,
        root=REPO_ROOT,
        matrix_output_root=matrix_root,
        experiment_output_root=experiment_root,
    )

    assert completed.status == "completed"
    assert skipped.status == "skipped"
    assert completed.model_path == experiment_root / plan.run_id / "model.zip"
    assert len(calls) == 1
    manifest = _read_matrix_manifest(matrix_root, "test_matrix")
    record = _manifest_record(manifest, plan.run_id)
    assert record["status"] == "completed"
    assert "completed_at" in record
    assert "skipped_at" in record
    with (
        matrix_root / "test_matrix" / "runs.csv"
    ).open(newline="", encoding="utf-8") as runs_file:
        rows = list(csv.DictReader(runs_file))
    assert rows[0]["status"] == "completed"
    summary = (
        matrix_root / "test_matrix" / "summary.md"
    ).read_text(encoding="utf-8")
    assert "| completed |" in summary
    (
        experiment_root / plan.run_id / "env.yaml"
    ).write_text("conflicting: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="incomplete or conflicts"):
        execute_experiment_run(
            config_path,
            plan.run_id,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=experiment_root,
        )
    assert len(calls) == 1


def test_execute_run_rejects_changed_matrix_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(tmp_path / "changed.yaml")
    matrix_root = tmp_path / "matrices"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plan = expand_experiment_matrix(config_path, root=REPO_ROOT)[0]
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    called = False

    def fake_training(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fake_training,
    )

    with pytest.raises(ValueError, match="source_config"):
        execute_experiment_run(
            config_path,
            plan.run_id,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=tmp_path / "experiments",
        )

    assert called is False


def test_execute_run_rejects_partial_experiment_directory(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path / "partial.yaml")
    matrix_root = tmp_path / "matrices"
    experiment_root = tmp_path / "experiments"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plan = expand_experiment_matrix(config_path, root=REPO_ROOT)[0]
    run_dir = experiment_root / plan.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "model.zip").write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="incomplete or conflicts"):
        execute_experiment_run(
            config_path,
            plan.run_id,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=experiment_root,
        )


def test_execute_run_records_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(tmp_path / "failure.yaml")
    matrix_root = tmp_path / "matrices"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plan = expand_experiment_matrix(config_path, root=REPO_ROOT)[0]

    def fail_training(**kwargs):
        raise RuntimeError("smoke training failed")

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fail_training,
    )

    with pytest.raises(RuntimeError, match="smoke training failed"):
        execute_experiment_run(
            config_path,
            plan.run_id,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=tmp_path / "experiments",
        )

    manifest = _read_matrix_manifest(matrix_root, "test_matrix")
    record = _manifest_record(manifest, plan.run_id)
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "smoke training failed"
    assert "failed_at" in record
    with (
        matrix_root / "test_matrix" / "runs.csv"
    ).open(newline="", encoding="utf-8") as runs_file:
        rows = list(csv.DictReader(runs_file))
    assert rows[0]["status"] == "failed"
    summary = (
        matrix_root / "test_matrix" / "summary.md"
    ).read_text(encoding="utf-8")
    assert "| failed |" in summary


def test_execute_run_cli_executes_only_selected_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "experiments" / "selected" / "model.zip"
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_execute(config_path, run_id, **kwargs):
        captured.append((run_id, kwargs))
        return ExperimentRunResult(
            run_id=run_id,
            status="completed",
            model_path=model_path,
        )

    monkeypatch.setattr(
        run_experiment_matrix_script,
        "execute_experiment_run",
        fake_execute,
    )
    run_experiment_matrix_script.main(
        [
            "--config",
            "configs/experiments/ppo_phase3_smoke.yaml",
            "--root",
            str(REPO_ROOT),
            "--execute-run",
            "selected",
            "--output-root",
            str(tmp_path / "matrices"),
            "--experiment-output-root",
            str(tmp_path / "experiments"),
        ]
    )
    output = capsys.readouterr().out

    assert len(captured) == 1
    assert captured[0][0] == "selected"
    assert "status: completed" in output
    assert f"model: {model_path}" in output


def test_execute_matrix_limits_runs_and_resumes_in_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(tmp_path / "bounded.yaml")
    matrix_root = tmp_path / "matrices"
    experiment_root = tmp_path / "experiments"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plans = expand_experiment_matrix(config_path, root=REPO_ROOT)
    calls: list[str] = []

    def fake_training(**kwargs):
        calls.append(str(kwargs["run_id"]))
        return _write_fake_completed_run(kwargs)

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fake_training,
    )

    first = execute_experiment_matrix(
        config_path,
        max_runs=1,
        root=REPO_ROOT,
        matrix_output_root=matrix_root,
        experiment_output_root=experiment_root,
    )
    second = execute_experiment_matrix(
        config_path,
        max_runs=1,
        root=REPO_ROOT,
        matrix_output_root=matrix_root,
        experiment_output_root=experiment_root,
    )

    assert calls == [plans[0].run_id, plans[1].run_id]
    assert first.attempted_count == 1
    assert first.completed_count == 1
    assert first.skipped_count == 0
    assert first.remaining_count == 1
    assert [result.status for result in first.results] == ["completed"]
    assert second.attempted_count == 1
    assert second.completed_count == 2
    assert second.skipped_count == 1
    assert second.remaining_count == 0
    assert [result.status for result in second.results] == [
        "skipped",
        "completed",
    ]
    manifest = _read_matrix_manifest(matrix_root, "test_matrix")
    assert [
        _manifest_record(manifest, plan.run_id)["status"] for plan in plans
    ] == ["completed", "completed"]


def test_execute_matrix_stops_on_failure_and_refuses_failed_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(tmp_path / "fail_fast.yaml")
    matrix_root = tmp_path / "matrices"
    write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    plans = expand_experiment_matrix(config_path, root=REPO_ROOT)
    calls: list[str] = []

    def fail_training(**kwargs):
        calls.append(str(kwargs["run_id"]))
        raise RuntimeError("campaign child failed")

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fail_training,
    )

    with pytest.raises(RuntimeError, match="campaign child failed"):
        execute_experiment_matrix(
            config_path,
            max_runs=2,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=tmp_path / "experiments",
        )
    with pytest.raises(RuntimeError, match="unresolved running or failed"):
        execute_experiment_matrix(
            config_path,
            max_runs=1,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=tmp_path / "experiments",
        )

    assert calls == [plans[0].run_id]
    manifest = _read_matrix_manifest(matrix_root, "test_matrix")
    assert _manifest_record(manifest, plans[0].run_id)["status"] == "failed"
    assert _manifest_record(manifest, plans[1].run_id)["status"] == "planned"


def test_execute_matrix_rejects_non_positive_limit(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path / "invalid_limit.yaml")

    for max_runs in (0, -1, False):
        with pytest.raises(ValueError, match="positive integer"):
            execute_experiment_matrix(
                config_path,
                max_runs=max_runs,
                root=REPO_ROOT,
            )


def test_execute_matrix_refuses_running_run_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment_config(tmp_path / "running.yaml")
    matrix_root = tmp_path / "matrices"
    outputs = write_experiment_matrix_plan(
        config_path,
        root=REPO_ROOT,
        output_root=matrix_root,
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    manifest["runs"][0]["status"] = "running"
    outputs["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    called = False

    def fake_training(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        experiment_runner_module,
        "_run_ppo_training",
        fake_training,
    )

    with pytest.raises(RuntimeError, match="unresolved running or failed"):
        execute_experiment_matrix(
            config_path,
            max_runs=1,
            root=REPO_ROOT,
            matrix_output_root=matrix_root,
            experiment_output_root=tmp_path / "experiments",
        )

    assert called is False


def test_execute_matrix_cli_requires_limit_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "experiments" / "first" / "model.zip"
    captured: list[dict[str, object]] = []

    def fake_execute(config_path, **kwargs):
        captured.append(kwargs)
        return ExperimentMatrixResult(
            results=(
                ExperimentRunResult(
                    run_id="first",
                    status="completed",
                    model_path=model_path,
                ),
            ),
            attempted_count=1,
            completed_count=1,
            skipped_count=0,
            remaining_count=1,
        )

    monkeypatch.setattr(
        run_experiment_matrix_script,
        "execute_experiment_matrix",
        fake_execute,
    )
    argv = [
        "--config",
        "configs/experiments/ppo_phase3_seed_sweep.yaml",
        "--root",
        str(REPO_ROOT),
        "--execute-matrix",
    ]
    with pytest.raises(SystemExit, match="2"):
        run_experiment_matrix_script.main(argv)

    run_experiment_matrix_script.main([*argv, "--max-runs", "1"])
    output = capsys.readouterr().out

    assert captured[0]["max_runs"] == 1
    assert f"first\tcompleted\t{model_path}" in output
    assert "summary: attempted=1 completed=1 skipped=0 remaining=1" in output


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


def _write_fake_completed_run(kwargs: dict[str, object]) -> Path:
    experiment_dir = Path(kwargs["output_dir_override"])
    experiment_dir.mkdir(parents=True)
    data_config_path = Path(kwargs["data_config_path"])
    env_config_path = Path(kwargs["env_config_path"])
    train_config_path = Path(kwargs["train_config_path"])
    feature_spec_path = REPO_ROOT / "artifacts/feature_specs/feature_spec_v1.json"
    copies = {
        data_config_path: experiment_dir / "config.yaml",
        env_config_path: experiment_dir / "env.yaml",
        train_config_path: experiment_dir / "train_ppo.yaml",
        feature_spec_path: experiment_dir / "feature_spec_v1.json",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    model_path = experiment_dir / "model.zip"
    model_path.write_text("model", encoding="utf-8")
    (experiment_dir / "metrics_validation.json").write_text(
        json.dumps({"sharpe_ratio": 1.0}),
        encoding="utf-8",
    )
    train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    manifest = {
        "run_id": kwargs["run_id"],
        "git_commit": _git_commit(),
        "seed": train_config["seed"],
        "total_timesteps": train_config["total_timesteps"],
        "data_config_hash": _sha256(data_config_path),
        "env_config_hash": _sha256(env_config_path),
        "train_config_hash": _sha256(train_config_path),
        "feature_spec_hash": _sha256(feature_spec_path),
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return model_path


def _read_matrix_manifest(
    matrix_root: Path,
    experiment_name: str,
) -> dict[str, object]:
    path = (
        matrix_root
        / experiment_name
        / "experiment_matrix_manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_record(
    manifest: dict[str, object],
    run_id: str,
) -> dict[str, object]:
    runs = manifest["runs"]
    assert isinstance(runs, list)
    return next(record for record in runs if record["run_id"] == run_id)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
