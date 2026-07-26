from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_robustness_checks as robustness_script
from portfolio_rl.config.schemas import RegimeWindowConfig
from portfolio_rl.evaluation import robustness
from portfolio_rl.evaluation.robustness import (
    RobustnessResult,
    aggregate_transaction_cost_results,
    run_transaction_cost_robustness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG = REPO_ROOT / "configs" / "evaluation.yaml"
SEEDS = [7, 42, 101, 202, 999]


def test_transaction_cost_robustness_evaluates_all_selected_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_path, registry_path = _write_inputs(tmp_path)
    _stub_backtests(monkeypatch)

    result = run_transaction_cost_robustness(
        selected_configuration_path=selected_path,
        registry_path=registry_path,
        evaluation_config_path=EVALUATION_CONFIG,
        output_dir=tmp_path / "robustness",
        root=tmp_path,
    )

    assert list(result.outputs) == [
        "results_csv",
        "summary_csv",
        "regime_results_csv",
        "regime_summary_csv",
        "manifest",
        "report",
    ]
    assert len(result.results) == 25
    assert list(result.results["seed"].drop_duplicates()) == SEEDS
    assert list(result.summary["transaction_cost_bps"]) == [
        0.0,
        5.0,
        10.0,
        25.0,
        50.0,
    ]
    baseline = result.summary[result.summary["transaction_cost_bps"] == 10.0].iloc[0]
    assert baseline["total_return_median_delta_vs_baseline"] == pytest.approx(0.0)
    assert len(result.regime_results) == 15
    assert set(result.regime_results["regime_name"]) == {
        "covid_2020",
        "rate_hike_2022",
        "validation_2024",
    }
    assert len(result.regime_summary) == 3
    assert set(
        result.regime_summary.loc[
            result.regime_summary["in_sample"],
            "regime_name",
        ]
    ) == {"covid_2020", "rate_hike_2022"}
    validation = result.regime_summary[
        result.regime_summary["regime_name"] == "validation_2024"
    ].iloc[0]
    assert bool(validation["in_sample"]) is False
    assert bool(validation["full_split_window"]) is True
    assert result.manifest["test_split_used"] is False
    assert result.manifest["evaluation_count"] == 40
    assert result.manifest["transaction_cost_evaluation_count"] == 25
    assert result.manifest["regime_evaluation_count"] == 15
    assert result.manifest["skipped_regime_windows"] == []
    assert all(result.manifest["diagnostics"].values())
    assert len(result.manifest["models"]) == 5
    assert all(
        len(source["sha256"]) == 64 for source in result.manifest["sources"].values()
    )
    report = result.outputs["report"].read_text(encoding="utf-8")
    assert "The test split was not accessed." in report
    assert "in-sample stress replays" in report
    assert "out-of-sample validation" in report
    assert "These diagnostics describe sensitivity" in report


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"test_split_used": True}, "must not use the test split"),
        ({"validation_only": False}, "must be validation-only"),
        (
            {"gate_results": {"gate_seed_coverage": False}},
            "must pass all validation gates",
        ),
    ],
)
def test_transaction_cost_robustness_rejects_invalid_selection(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    selected_path, registry_path = _write_inputs(tmp_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected.update(mutation)
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_transaction_cost_robustness(
            selected_configuration_path=selected_path,
            registry_path=registry_path,
            evaluation_config_path=EVALUATION_CONFIG,
            output_dir=tmp_path / "robustness",
            root=tmp_path,
        )


def test_transaction_cost_robustness_rejects_missing_selected_seed(
    tmp_path: Path,
) -> None:
    selected_path, registry_path = _write_inputs(tmp_path)
    registry = pd.read_csv(registry_path)
    registry[registry["seed"] != 999].to_csv(registry_path, index=False)

    with pytest.raises(ValueError, match="missing selected seeds"):
        run_transaction_cost_robustness(
            selected_configuration_path=selected_path,
            registry_path=registry_path,
            evaluation_config_path=EVALUATION_CONFIG,
            output_dir=tmp_path / "robustness",
            root=tmp_path,
        )


def test_transaction_cost_aggregation_requires_baseline() -> None:
    results = pd.DataFrame(
        [
            _result_row(seed=7, transaction_cost_bps=0.0),
            _result_row(seed=42, transaction_cost_bps=0.0),
        ]
    )

    with pytest.raises(ValueError, match="missing baseline cost"):
        aggregate_transaction_cost_results(
            results,
            baseline_cost_bps=10.0,
        )


def test_regime_windows_reject_test_and_audit_unavailable_data() -> None:
    dataset = SimpleNamespace(
        dates=pd.DatetimeIndex(["2024-01-02", "2025-01-02"]),
        splits=np.array(["validation", "test"]),
    )
    test_window = RegimeWindowConfig(
        name="forbidden_test",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    with pytest.raises(ValueError, match="touches test split"):
        robustness._resolve_regime_windows(
            dataset=dataset,
            windows=[test_window],
            rebalance_frequency_trading_days=5,
        )

    missing_window = RegimeWindowConfig(
        name="unavailable",
        start_date=date(2019, 1, 1),
        end_date=date(2019, 12, 31),
    )
    resolved, skipped = robustness._resolve_regime_windows(
        dataset=dataset,
        windows=[missing_window],
        rebalance_frequency_trading_days=5,
    )

    assert resolved == []
    assert skipped == [
        {
            "name": "unavailable",
            "configured_start_date": "2019-01-01",
            "configured_end_date": "2019-12-31",
            "reason": "no_available_rows",
        }
    ]


def test_robustness_cli_prints_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "robustness_report.md"
    fake = RobustnessResult(
        outputs={"report": output},
        results=pd.DataFrame(),
        summary=pd.DataFrame(),
        regime_results=pd.DataFrame(),
        regime_summary=pd.DataFrame(),
        manifest={},
    )
    monkeypatch.setattr(
        robustness_script,
        "run_transaction_cost_robustness",
        lambda **_kwargs: fake,
    )

    robustness_script.main(
        [
            "--selected-configuration",
            "selected.json",
            "--registry",
            "registry.csv",
            "--output-dir",
            "robustness",
        ]
    )

    assert capsys.readouterr().out == f"report: {output}\n"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    selected = {
        "schema_version": 1,
        "configuration_id": "selected",
        "experiment_name": "seed_sweep",
        "total_timesteps": 500000,
        "overrides": {
            "env.action_temperature": 0.5,
            "ppo.ent_coef": 0.01,
        },
        "planned_seeds": SEEDS,
        "eligible_seeds": SEEDS,
        "selection_checkpoint_counts": {
            "best_checkpoint": 3,
            "final_endpoint": 2,
        },
        "gate_results": {
            "gate_seed_coverage": True,
            "gate_finite_metrics": True,
        },
        "validation_only": True,
        "test_split_used": False,
    }
    selected_path = tmp_path / "selected_configuration.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    rows = []
    for index, seed in enumerate(SEEDS):
        run_id = f"run_{seed}"
        run_dir = tmp_path / "artifacts" / "experiments" / run_id
        run_dir.mkdir(parents=True)
        checkpoint = "best_checkpoint" if index < 3 else "final_endpoint"
        model_name = (
            "best_model.zip" if checkpoint == "best_checkpoint" else "model.zip"
        )
        (run_dir / model_name).write_bytes(f"model-{seed}".encode())
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "env.yaml").write_text(
            """
rebalance_frequency_trading_days: 5
episode_length_trading_days: 260
max_episode_steps: 52
action_transform: softmax
action_temperature: 0.5
initial_weights: equal_weight
transaction_cost_bps: 10.0
reward_type: log_growth
reward_scale: 100.0
terminal_bad_gross_penalty: -100.0
record_arrays_in_info: false
""".lstrip(),
            encoding="utf-8",
        )
        rows.append(
            {
                "run_id": run_id,
                "experiment_name": "seed_sweep",
                "git_commit": "abc123",
                "seed": seed,
                "total_timesteps": 500000,
                "action_temperature": 0.5,
                "ent_coef": 0.01,
                "selection_eligible": True,
                "eligibility_issues": "",
                "selection_checkpoint": checkpoint,
                "selection_model_path": (
                    f"artifacts/experiments/{run_id}/{model_name}"
                ),
                "manifest_path": (f"artifacts/experiments/{run_id}/manifest.json"),
            }
        )
    registry_path = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry_path, index=False)
    return selected_path, registry_path


def _stub_backtests(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.DatetimeIndex(
        [
            *pd.bdate_range("2020-02-03", periods=6),
            *pd.bdate_range("2022-01-03", periods=6),
            *pd.bdate_range("2024-01-02", periods=6),
        ]
    )
    dataset = SimpleNamespace(
        dates=dates,
        splits=np.array(["train"] * 12 + ["validation"] * 6),
    )
    monkeypatch.setattr(
        robustness,
        "load_portfolio_dataset",
        lambda _root: dataset,
    )

    def fake_feature_store(
        source: SimpleNamespace,
        split: str,
        *,
        start_date: object = None,
        end_date: object = None,
    ) -> SimpleNamespace:
        mask = source.splits == split
        if start_date is not None:
            mask &= source.dates >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= source.dates <= pd.Timestamp(end_date)
        selected_dates = source.dates[mask]
        return SimpleNamespace(
            split=split,
            n_rows=len(selected_dates),
            date_at=lambda index: selected_dates[index],
        )

    monkeypatch.setattr(
        robustness,
        "PortfolioFeatureStore",
        fake_feature_store,
    )
    monkeypatch.setattr(
        robustness,
        "load_sb3_weight_policy",
        lambda model_path, action_temperature: {
            "seed": int(Path(model_path).parent.name.removeprefix("run_")),
            "action_temperature": action_temperature,
        },
    )

    def fake_backtest(
        *,
        policy: dict[str, object],
        transaction_cost_bps: float,
        **_kwargs: object,
    ) -> SimpleNamespace:
        seed = int(policy["seed"])
        total_return = 0.20 + seed / 100000.0 - transaction_cost_bps / 1000.0
        return SimpleNamespace(
            metrics={
                "total_return": total_return,
                "cagr": total_return,
                "sharpe_ratio": 1.5 - transaction_cost_bps / 100.0,
                "max_drawdown": -0.05 - transaction_cost_bps / 10000.0,
                "average_weekly_turnover": 0.20,
                "annualized_turnover": 10.4,
                "transaction_cost_drag": transaction_cost_bps / 1000.0,
            }
        )

    monkeypatch.setattr(
        robustness,
        "run_weight_policy_backtest",
        fake_backtest,
    )


def _result_row(
    *,
    seed: int,
    transaction_cost_bps: float,
) -> dict[str, object]:
    return {
        "configuration_id": "selected",
        "experiment_name": "seed_sweep",
        "run_id": f"run_{seed}",
        "seed": seed,
        "selection_checkpoint": "best_checkpoint",
        "selection_model_path": f"run_{seed}/best_model.zip",
        "transaction_cost_bps": transaction_cost_bps,
        "total_return": 0.10,
        "cagr": 0.10,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.05,
        "average_weekly_turnover": 0.20,
        "annualized_turnover": 10.4,
        "transaction_cost_drag": 0.0,
    }
