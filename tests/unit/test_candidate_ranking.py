from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.select_model as select_model_script
from portfolio_rl.config.loader import load_phase3_evaluation_config
from portfolio_rl.evaluation.model_selection import (
    NoPassingCandidateError,
    rank_candidate_configurations,
    write_candidate_ranking,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG = REPO_ROOT / "configs" / "evaluation.yaml"


def test_candidate_ranking_selects_passing_configuration(
    tmp_path: Path,
) -> None:
    stability_path = tmp_path / "seed_stability.csv"
    pd.DataFrame([_candidate("passing")]).to_csv(stability_path, index=False)
    baseline_root = _write_baselines(tmp_path / "baselines")

    result = write_candidate_ranking(
        seed_stability_path=stability_path,
        baseline_root=baseline_root,
        evaluation_config_path=EVALUATION_CONFIG,
        output_dir=tmp_path / "selection",
    )

    assert list(result.outputs) == [
        "ranking_csv",
        "ranking_markdown",
        "report",
        "selected_configuration",
    ]
    selected = json.loads(
        result.outputs["selected_configuration"].read_text(encoding="utf-8")
    )
    assert selected["configuration_id"] == "passing"
    assert selected["metric_source"] == "best_available_checkpoint"
    assert selected["selection_checkpoint_counts"] == {
        "best_checkpoint": 2,
        "final_endpoint": 1,
    }
    assert selected["overrides"] == {"ppo.ent_coef": 0.01}
    assert selected["eligible_seeds"] == [7, 42, 101]
    assert selected["validation_only"] is True
    assert selected["test_split_used"] is False
    assert len(selected["sources"]["seed_stability"]["sha256"]) == 64
    assert set(selected["sources"]["baselines"]) == {
        "equal_weight_weekly",
        "buy_and_hold_equal_weight",
        "spy_only",
        "shy_only",
        "inverse_volatility",
    }
    report = result.outputs["report"].read_text(encoding="utf-8")
    assert "The test split was not accessed." in report
    assert "Selected configuration: `passing`." in report


@pytest.mark.parametrize(
    ("overrides", "failed_gate"),
    [
        ({"eligible_seed_count": 1}, "gate_seed_coverage"),
        (
            {"validation_total_return_median": 0.03},
            "gate_shy_total_return",
        ),
        (
            {
                "validation_sharpe_ratio_median": 1.0,
                "validation_max_drawdown_median": -0.07,
            },
            "gate_equal_weight_materiality",
        ),
        (
            {"validation_average_weekly_turnover_median": 0.51},
            "gate_weekly_turnover",
        ),
        (
            {"validation_transaction_cost_drag_median": 0.026},
            "gate_transaction_cost_drag",
        ),
    ],
)
def test_candidate_ranking_applies_each_hard_gate(
    overrides: dict[str, object],
    failed_gate: str,
) -> None:
    candidate = _candidate("candidate") | overrides

    ranked = rank_candidate_configurations(
        pd.DataFrame([candidate]),
        _baseline_metrics(),
        load_phase3_evaluation_config(EVALUATION_CONFIG).selection,
    )
    row = ranked.iloc[0]

    assert bool(row["passes_all_gates"]) is False
    assert failed_gate in json.loads(row["failed_gates"])


def test_equal_weight_gate_requires_joint_material_underperformance() -> None:
    sharpe_only = _candidate("sharpe_only") | {
        "validation_sharpe_ratio_median": 1.0,
        "validation_max_drawdown_median": -0.05,
    }
    drawdown_only = _candidate("drawdown_only") | {
        "validation_sharpe_ratio_median": 1.15,
        "validation_max_drawdown_median": -0.07,
    }

    ranked = rank_candidate_configurations(
        pd.DataFrame([sharpe_only, drawdown_only]),
        _baseline_metrics(),
        load_phase3_evaluation_config(EVALUATION_CONFIG).selection,
    )

    assert ranked["gate_equal_weight_materiality"].all()
    assert ranked["passes_all_gates"].all()


def test_candidate_ranking_uses_deterministic_tie_breakers() -> None:
    better_drawdown = _candidate("better_drawdown") | {
        "validation_max_drawdown_median": -0.03,
        "validation_average_weekly_turnover_median": 0.20,
    }
    lower_turnover = _candidate("lower_turnover") | {
        "validation_max_drawdown_median": -0.04,
        "validation_average_weekly_turnover_median": 0.10,
    }

    ranked = rank_candidate_configurations(
        pd.DataFrame([lower_turnover, better_drawdown]),
        _baseline_metrics(),
        load_phase3_evaluation_config(EVALUATION_CONFIG).selection,
    )

    assert list(ranked["configuration_id"]) == [
        "better_drawdown",
        "lower_turnover",
    ]
    assert list(ranked["rank"]) == [1, 2]


def test_candidate_ranking_rejects_missing_or_invalid_baselines() -> None:
    stability = pd.DataFrame([_candidate("candidate")])
    config = load_phase3_evaluation_config(EVALUATION_CONFIG).selection
    missing = _baseline_metrics()
    missing.pop("spy_only")

    with pytest.raises(ValueError, match="missing required"):
        rank_candidate_configurations(stability, missing, config)

    invalid = _baseline_metrics()
    invalid["shy_only"]["total_return"] = None
    with pytest.raises(ValueError, match="invalid validation baseline"):
        rank_candidate_configurations(stability, invalid, config)


def test_no_passing_candidate_writes_audit_outputs_and_cli_fails(
    tmp_path: Path,
) -> None:
    stability_path = tmp_path / "seed_stability.csv"
    pd.DataFrame(
        [_candidate("smoke") | {"eligible_seed_count": 1}]
    ).to_csv(stability_path, index=False)
    baseline_root = _write_baselines(tmp_path / "baselines")
    output_dir = tmp_path / "selection"

    with pytest.raises(NoPassingCandidateError) as exc_info:
        write_candidate_ranking(
            seed_stability_path=stability_path,
            baseline_root=baseline_root,
            evaluation_config_path=EVALUATION_CONFIG,
            output_dir=output_dir,
        )

    assert set(exc_info.value.outputs) == {
        "ranking_csv",
        "ranking_markdown",
        "report",
    }
    assert not (output_dir / "selected_configuration.json").exists()
    report = (output_dir / "validation_selection_report.md").read_text(
        encoding="utf-8"
    )
    assert "No configuration passed all validation gates" in report
    ranking = pd.read_csv(output_dir / "candidate_ranking.csv")
    assert "gate_seed_coverage" in json.loads(ranking.loc[0, "failed_gates"])

    with pytest.raises(SystemExit) as cli_exit:
        select_model_script.main(
            [
                "--seed-stability",
                str(stability_path),
                "--baseline-root",
                str(baseline_root),
                "--config",
                str(EVALUATION_CONFIG),
                "--output-dir",
                str(tmp_path / "cli_selection"),
            ]
        )
    assert cli_exit.value.code == 1
    assert not (
        tmp_path / "cli_selection" / "selected_configuration.json"
    ).exists()


def test_candidate_ranking_rejects_malformed_stability() -> None:
    stability = pd.DataFrame([_candidate("candidate")]).drop(
        columns=["ranking_ready"]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        rank_candidate_configurations(
            stability,
            _baseline_metrics(),
            load_phase3_evaluation_config(EVALUATION_CONFIG).selection,
        )


def _candidate(configuration_id: str) -> dict[str, object]:
    return {
        "configuration_id": configuration_id,
        "experiment_name": "test_matrix",
        "metric_source": "best_available_checkpoint",
        "selection_checkpoint_counts": (
            '{"best_checkpoint":2,"final_endpoint":1}'
        ),
        "total_timesteps": 500000,
        "overrides": '{"ppo.ent_coef":0.01}',
        "planned_seed_count": 3,
        "eligible_seed_count": 3,
        "planned_seeds": "[7,42,101]",
        "eligible_seeds": "[7,42,101]",
        "ranking_ready": True,
        "validation_sharpe_ratio_median": 1.30,
        "validation_sharpe_ratio_std": 0.10,
        "validation_total_return_median": 0.12,
        "validation_max_drawdown_median": -0.04,
        "validation_average_weekly_turnover_median": 0.20,
        "validation_transaction_cost_drag_median": 0.01,
    }


def _baseline_metrics() -> dict[str, dict[str, float | None]]:
    return {
        "equal_weight_weekly": _metrics(0.10, 1.20, -0.04, 0.01, 0.001),
        "buy_and_hold_equal_weight": _metrics(
            0.09,
            1.10,
            -0.05,
            0.00,
            0.0001,
        ),
        "spy_only": _metrics(0.20, 1.50, -0.08, 0.04, 0.002),
        "shy_only": _metrics(0.04, 2.00, -0.01, 0.04, 0.002),
        "inverse_volatility": _metrics(
            0.08,
            1.40,
            -0.03,
            0.10,
            0.005,
        ),
    }


def _metrics(
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    turnover: float,
    cost_drag: float,
) -> dict[str, float | None]:
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "average_weekly_turnover": turnover,
        "transaction_cost_drag": cost_drag,
    }


def _write_baselines(root: Path) -> Path:
    for strategy, metrics in _baseline_metrics().items():
        strategy_dir = root / strategy
        strategy_dir.mkdir(parents=True)
        (strategy_dir / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
    return root
