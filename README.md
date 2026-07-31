# Deep RL Portfolio Optimization

Research codebase for dynamic portfolio allocation with reinforcement learning.
The project builds a reproducible pipeline from market data ingestion through
feature engineering, Gymnasium environment simulation, PPO training, and
baseline comparison.

Phase 2 is complete. The current focus is Phase 3: turning the tested RL
research environment into a reproducible experimentation, model-selection, and
robustness workflow. Phase 3 experiment configs and the experiment registry are
implemented, and persisted experiment matrices support bounded sequential
execution. Selection-ready registries, best-available-checkpoint seed
aggregation, baseline-gated configuration ranking, and validation-only
transaction-cost robustness are implemented. The default five-seed campaign is
complete, and named 2020, 2022, and 2024 regime robustness is implemented.
Selected-checkpoint policy behavior diagnostics now cover allocation,
concentration, turnover, monthly-return dependence, drawdowns, and
high-volatility response across all five seeds and all three regime windows.
Counterfactual sensitivity probes distinguish observed high-volatility
association from direct SPY-volatility and coordinated global-risk responses.
The selected candidate can now be packaged atomically with its frozen training
snapshots, validation evidence, provenance hashes, and model card without
accessing the test split. The frozen seed-42 initialization-sensitivity
diagnostic now compares equal-weight, 63-day inverse-volatility, and 100% SHY
endowments using target and realized pre-trade convergence paths. The executable
five-seed mean-target-weight ensemble now records member recommendations,
cross-seed disagreement, seed tracking error, and 2022 historical behavior;
its initialization sensitivity reuses the same framework. Phase 3A candidate
qualification remains open. The 2024 period is consumed
development/selection data, and Phase 3B is blocked until PM/ML reviewers
approve a new independent holdout because a legacy Phase 2 model was previously
evaluated on the repository's 2025+ test designation.

Turnover is defined as `0.5 * sum(abs(target - drifted current weights))`.
The original campaign artifacts remain legacy results. The corrected five-seed
rerun, model selection, robustness analysis, and policy diagnostics are
versioned by `configs/experiments/ppo_phase3_seed_sweep_turnover_v2.yaml`.

## Architecture

```text
configs/                  YAML configuration for data, features, env, and PPO
scripts/                  Thin CLI entrypoints for pipeline and experiments
src/portfolio_rl/data/    ETL, storage, split assignment, feature-store access
src/portfolio_rl/features/ Feature engineering, normalization, model matrix
src/portfolio_rl/env/     Gymnasium portfolio environment and mechanics
src/portfolio_rl/policies/ Baseline policies and SB3 policy adapter
src/portfolio_rl/evaluation/ Backtests, metrics, and validation reports
src/portfolio_rl/training/ PPO training, validation checkpoints, experiment registry
tests/                    Unit and integration coverage
docs/                     Planning documents and technical references
```

For a fuller technical explanation, see
[`docs/technical_overview.md`](docs/technical_overview.md).

## Core Workflow

```bash
python scripts/run_etl.py
python scripts/build_features.py
python scripts/validate_phase1.py
python scripts/smoke_test_env.py
python scripts/run_baselines.py
python scripts/train_ppo.py
python scripts/evaluate_policy.py --model-path artifacts/experiments/<run_id>/model.zip
python scripts/compare_validation.py
python scripts/list_experiments.py
python scripts/run_experiment_matrix.py \
  --config configs/experiments/ppo_phase3_smoke.yaml \
  --write-plan
python scripts/run_experiment_matrix.py \
  --config configs/experiments/ppo_phase3_smoke.yaml \
  --execute-matrix \
  --max-runs 1
python scripts/list_experiments.py
python scripts/summarize_experiment.py \
  --registry artifacts/experiments/registry.csv \
  --matrix-manifest \
  artifacts/experiment_matrices/ppo_phase3_smoke/experiment_matrix_manifest.json \
  --output-dir artifacts/model_selection/ppo_phase3_smoke
python scripts/select_model.py \
  --seed-stability artifacts/model_selection/<experiment>/seed_stability.csv \
  --baseline-root artifacts/backtests/baselines_validation \
  --config configs/evaluation.yaml \
  --output-dir artifacts/model_selection/<experiment>
python scripts/run_robustness_checks.py \
  --selected-configuration \
  artifacts/model_selection/<experiment>/selected_configuration.json \
  --registry artifacts/experiments/registry.csv \
  --config configs/evaluation.yaml \
  --output-dir artifacts/robustness/<experiment>
python scripts/analyze_policy_behavior.py \
  --selected-configuration \
  artifacts/model_selection/<experiment>/selected_configuration.json \
  --registry artifacts/experiments/registry.csv \
  --config configs/evaluation.yaml \
  --universe-config configs/universe.yaml \
  --output-dir artifacts/diagnostics/<experiment>
python scripts/run_policy_sensitivity.py \
  --selected-configuration \
  artifacts/model_selection/<experiment>/selected_configuration.json \
  --registry artifacts/experiments/registry.csv \
  --diagnostics-dir artifacts/diagnostics/<experiment> \
  --config configs/evaluation.yaml \
  --output-dir artifacts/sensitivity/<experiment>
python scripts/analyze_active_return.py \
  --ppo-nav artifacts/diagnostics/<experiment>/nav_by_regime.parquet \
  --baseline-nav \
  artifacts/backtests/baselines_validation_turnover_v2/equal_weight_weekly/nav.parquet \
  --config configs/evaluation.yaml \
  --output-dir artifacts/statistical_validation/<experiment>
python scripts/finalize_model.py \
  --selected-configuration \
  artifacts/model_selection/<experiment>/selected_configuration.json \
  --registry artifacts/experiments/registry.csv \
  --model-version <model_version> \
  --representative-seed 42 \
  --output-root artifacts/final_model
python scripts/freeze_phase3_campaign.py \
  --config configs/research/phase3_candidate_qualification.yaml \
  --output-root artifacts/research_freeze
python scripts/run_initialization_sensitivity.py \
  --config configs/research/phase3_initialization_sensitivity.yaml
python scripts/evaluate_policy_ensemble.py \
  --config configs/research/phase3_ensemble.yaml
```

Generated data and experiment outputs are written under `data/` and
`artifacts/`, which are intentionally git-ignored.

## Local Setup

```bash
conda create -n drl-portfolio-opt python=3.11
conda activate drl-portfolio-opt
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Current Splits

The default data split contract is defined in `configs/data.yaml`:

```text
train:       2010-01-01 through 2023-12-31
development/selection: 2024-01-01 through 2024-12-31
legacy test designation: 2025-01-01 onward
```

PPO trains on random one-year windows from the train split. The 2024 period was
used for deterministic validation, checkpoint selection, diagnostics, and
statistical analysis, so it is not independent evidence. The turnover-v2
campaign has not accessed 2025+, but a different legacy Phase 2 model has.
Phase 3B therefore remains blocked pending approval of a new holdout.

## Documentation

- [`docs/technical_overview.md`](docs/technical_overview.md): project goal,
  architecture, data flow, and experiment design.
- [`docs/planning/phase_1.md`](docs/planning/phase_1.md): Phase 1 data pipeline
  plan.
- [`docs/planning/phase_2.md`](docs/planning/phase_2.md): completed Phase 2 RL
  environment, training, and evaluation blueprint.
- [`docs/planning/phase_3.md`](docs/planning/phase_3.md): active Phase 3
  experimentation, model-selection, and robustness plan.
- [`docs/planning/phase_3_remaining_regenerated.md`](docs/planning/phase_3_remaining_regenerated.md):
  active Phase 3A candidate-qualification and pre-test governance plan.
- [`notebooks/phase3_policy_behavior_review.ipynb`](notebooks/phase3_policy_behavior_review.ipynb):
  allocation, exposure, turnover, volatility, and sensitivity visual review.
