from __future__ import annotations

from pathlib import Path

from scripts.run_baselines import run_baseline_backtests


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STRATEGIES = [
    "equal_weight_weekly",
    "buy_and_hold_equal_weight",
    "spy_only",
    "shy_only",
    "inverse_volatility",
]
EXPECTED_ARTIFACTS = [
    "metrics.json",
    "nav.parquet",
    "costs.parquet",
    "report.md",
]


def test_run_baselines_writes_expected_artifacts(tmp_path: Path) -> None:
    written_dirs = run_baseline_backtests(
        root=REPO_ROOT,
        env_config_path=REPO_ROOT / "configs/env.yaml",
        split="validation",
        output_dir=tmp_path,
        max_steps=1,
    )

    assert [path.name for path in written_dirs] == EXPECTED_STRATEGIES
    for strategy in EXPECTED_STRATEGIES:
        strategy_dir = tmp_path / strategy
        assert strategy_dir.is_dir()
        for artifact_name in EXPECTED_ARTIFACTS:
            assert (strategy_dir / artifact_name).is_file()
