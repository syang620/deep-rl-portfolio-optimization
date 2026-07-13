# Phase 3 — Institutional Experimentation, Model Selection, and Robustness Plan

**Project:** RL Dynamic Portfolio Allocation  
**Phase:** 3 — Experiment management, PPO research campaign, model selection, robustness diagnostics, and final candidate packaging  
**Audience:** junior engineers, coding-agent operators, project reviewers, and future serving/API implementers  
**Repo baseline reviewed:** `syang620/deep-rl-portfolio-optimization`, latest observed commit `40cb3a709e4fc678e367bd543df33b208d56f740` (`Complete Phase 2 training workflow`)  
**Status:** Ready to plan Phase 3 after the current Phase 2 workflow is locally validated with fresh Phase 1 artifacts.

---

## 0. Executive Summary

Phase 2 built the core RL research platform:

- split-aware `PortfolioFeatureStore`,
- Gymnasium `PortfolioEnv`,
- weekly two-clock stepping,
- drift-aware transaction-cost mechanics,
- deterministic baseline policies,
- shared backtest engine,
- PPO training harness,
- validation checkpoint callback,
- W&B-capable training config,
- validation comparison report,
- guarded final test evaluation CLI.

Phase 3 should **not** be “add more PPO code.” The repo already has a first PPO training path. Phase 3 should turn that path into a disciplined institutional research process:

> Run controlled PPO experiment campaigns, compare candidates against baselines, stress-test policies across regimes and frictions, select one final model without test leakage, and package the selected candidate with a model card and reproducibility manifest.

The Phase 3 deliverable is a **selected, documented, reproducible model candidate** that is good enough to hand to the later serving/API phase.

---

## 1. Current Phase 2 Baseline From GitHub

The current repo has advanced materially beyond the original Phase 2 planning document. Phase 3 should assume the following already exists or is in scope to verify:

### 1.1 Repo-level workflow

The README describes a workflow from ETL through PPO training and validation comparison:

```bash
python scripts/run_etl.py
python scripts/build_features.py
python scripts/validate_phase1.py
python scripts/smoke_test_env.py
python scripts/run_baselines.py
python scripts/train_ppo.py
python scripts/evaluate_policy.py --model-path artifacts/experiments/<run_id>/model.zip
python scripts/compare_validation.py
```

It also documents that current outputs live under git-ignored `data/` and `artifacts/`.

### 1.2 Core modules now present

The repository now has package areas for:

```text
src/portfolio_rl/data/        ETL, storage, splits, feature-store access
src/portfolio_rl/features/    feature engineering, normalization, model matrix
src/portfolio_rl/env/         Gymnasium environment and portfolio mechanics
src/portfolio_rl/policies/    deterministic baselines and SB3 policy wrapper
src/portfolio_rl/evaluation/  backtests, metrics, reports
src/portfolio_rl/training/    PPO harness and validation checkpoint callback
```

### 1.3 Environment mechanics

Current `PortfolioEnv` already implements the core Phase 2 mechanics:

- action space `Box[-1, 1]`,
- temperature-scaled softmax action conversion,
- forward-return window fetch,
- EOF-safe truncation,
- turnover vs current drifted weights,
- 10 bps transaction cost path,
- buy-and-hold simulation over the rebalance window,
- scaled log-growth reward,
- two-clock advancement,
- dynamic observation construction using market features plus current weights.

### 1.4 Current environment defaults

Current `configs/env.yaml` uses:

```yaml
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
```

Important: the original design considered `action_temperature = 5.0`, but Phase 2 diagnostics moved the default to `0.5` because sharper softmax allocations caused excessive turnover and cost drag. Phase 3 should treat `0.5` as the current default, not as an accidental deviation.

### 1.5 PPO training path

Current PPO config uses:

```yaml
algorithm: PPO
policy: MlpPolicy
total_timesteps: 500000
seed: 42
n_steps: 2080
batch_size: 260
eval_freq_timesteps: 25000
metric_for_best_model: sharpe_ratio
```

The training harness:

1. loads data/env/train configs,
2. creates train and validation feature stores,
3. trains PPO with `DummyVecEnv`,
4. periodically backtests the in-memory model on validation,
5. saves `best_model.zip`,
6. saves final `model.zip`,
7. writes validation artifacts,
8. copies configs/feature specs/data-quality report,
9. writes a manifest,
10. optionally logs to W&B.

### 1.6 Backtesting and reports

Current backtest outputs include:

```text
nav.parquet
weights_target.parquet
weights_drifted.parquet
trades.parquet
costs.parquet
metrics.json
report.md
```

Current metrics include total return, CAGR, annualized volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, transaction-cost drag, hit rate, best month, and worst month.

Current reports compare strategies and warn when PPO underperforms equal-weight on total return or Sharpe.

### 1.7 Final test is guarded

`evaluate_policy.py` requires `--confirm-final-test` when evaluating the test split. Phase 3 must preserve this guard. Test evaluation should be run only once for the final selected model.

---

## 2. Phase 3 Objective

Phase 3 turns the existing Phase 2 training workflow into an institutional research and model-selection process.

The core objective:

> Produce a selected PPO allocation policy with documented validation performance, robustness diagnostics, cost sensitivity, model behavior analysis, and a reproducible artifact bundle suitable for downstream API serving.

Phase 3 should answer these questions:

1. **Does PPO add value after costs versus simple baselines?**
2. **Is performance stable across seeds, regimes, and frictions?**
3. **Is the policy behavior explainable enough to trust?**
4. **Is the selected model reproducible from code, configs, data version, feature spec, and seed?**
5. **Can we defend why this model was selected without looking at the test split?**

---

## 3. Non-Goals

Phase 3 should not include:

1. **No live trading integration.**
   - No broker API.
   - No order generation.
   - No execution simulator beyond current transaction-cost model.

2. **No FastAPI serving implementation yet.**
   - Phase 3 may prepare model artifacts for serving.
   - The actual API service should remain a later phase.

3. **No tuning on the test split.**
   - Test is reserved for the final selected model.
   - Validation drives model selection.

4. **No uncontrolled algorithm sprawl.**
   - PPO remains the primary Phase 3 algorithm.
   - SAC or other algorithms are optional late-phase extensions only after the PPO experiment protocol is stable.

5. **No changing Phase 1 feature definitions during model-selection runs unless the run is explicitly labeled as a new feature version.**
   - Phase 3 should not silently mutate `feature_spec_v1.json`.

---

## 4. Inherited Contracts From Phase 2

These are still non-negotiable.

### 4.1 Data timing

For a decision at row `i`:

```text
observation = information known at close of row i
reward window = future return rows i+1 through i+5
next decision = row i+5
```

Never use same-row returns in the reward.

### 4.2 Two-clock environment

The environment keeps:

```python
current_step      # weekly agent decision clock
current_data_idx  # daily market-data row clock
```

Each `step()` advances:

```python
current_step += 1
current_data_idx += 5
```

### 4.3 Dynamic current weights

The observation must be:

```python
concat(market_features_t, live_current_drifted_weights_t)
```

not the static equal-weight slice from the Phase 1 model matrix.

### 4.4 Reward scaling

Training reward is scaled for PPO stability:

```python
reward = log(net_gross_return) * reward_scale
```

But reported NAV, CAGR, Sharpe, drawdown, turnover, and cost drag are always unscaled financial metrics.

### 4.5 No observation VecNormalize

Do not use:

```python
VecNormalize(vec_env, norm_obs=True)
```

Market features are already normalized, and portfolio weights have direct financial meaning.

### 4.6 Test split guard

The test split is for final evaluation only. The guard in `evaluate_policy.py` must remain intact.

---

## 5. Phase 3 Deliverables

Phase 3 should produce these code and artifact deliverables.

### 5.1 Experiment configuration system

Add or formalize:

```text
configs/experiments/
├── ppo_baseline.yaml
├── ppo_temperature_sweep.yaml
├── ppo_entropy_sweep.yaml
├── ppo_seed_sweep.yaml
├── ppo_cost_sensitivity.yaml
└── ppo_final_candidate.yaml
```

Each experiment config should define:

```yaml
experiment_name: ppo_temperature_sweep_v1
base_train_config: configs/train_ppo.yaml
base_env_config: configs/env.yaml
seeds: [7, 42, 101, 202, 999]
grid:
  env.action_temperature: [0.25, 0.5, 0.75, 1.0]
  ppo.ent_coef: [0.0, 0.005, 0.01]
selection_metric: sharpe_ratio
secondary_metrics:
  - total_return
  - max_drawdown
  - average_weekly_turnover
  - transaction_cost_drag
```

### 5.2 Experiment runner

Add:

```text
src/portfolio_rl/training/experiment_runner.py
scripts/run_experiment_matrix.py
```

Responsibilities:

1. Load an experiment matrix config.
2. Generate child run configs.
3. Execute each run with a deterministic run ID.
4. Skip or resume completed runs when artifacts already exist.
5. Save one `experiment_matrix_manifest.json`.
6. Optionally log each child run to W&B with consistent group/tags.

### 5.3 Experiment registry

Add:

```text
src/portfolio_rl/training/registry.py
scripts/list_experiments.py
```

Responsibilities:

1. Scan `artifacts/experiments/*/manifest.json`.
2. Load validation metrics and best validation metrics.
3. Join config values such as temperature, seed, entropy coefficient, learning rate.
4. Produce a tabular run inventory.
5. Export:

```text
artifacts/experiments/registry.parquet
artifacts/experiments/registry.csv
artifacts/experiments/registry.md
```

Minimum registry columns:

```text
run_id
created_at
git_commit
feature_version
seed
algorithm
total_timesteps
action_temperature
learning_rate
ent_coef
n_steps
batch_size
metric_for_best_model
validation_total_return
validation_cagr
validation_sharpe_ratio
validation_max_drawdown
validation_average_weekly_turnover
validation_transaction_cost_drag
best_validation_sharpe_ratio
model_path
best_model_path
manifest_path
```

### 5.4 Model-selection report

Add:

```text
src/portfolio_rl/evaluation/model_selection.py
scripts/select_model.py
```

Output:

```text
artifacts/model_selection/
├── candidate_ranking.csv
├── candidate_ranking.md
├── selected_model.json
├── selected_model_card.md
└── validation_selection_report.md
```

The report should include:

1. Candidate ranking.
2. Baseline comparison.
3. Seed stability.
4. Cost sensitivity.
5. Regime breakdown.
6. Allocation behavior diagnostics.
7. Final model selection rationale.
8. Explicit statement that test split was not used for selection.

### 5.5 Robustness and stress testing

Add:

```text
src/portfolio_rl/evaluation/robustness.py
scripts/run_robustness_checks.py
```

Required robustness dimensions:

1. **Transaction cost sensitivity**

```yaml
transaction_cost_bps: [0.0, 5.0, 10.0, 25.0, 50.0]
```

2. **Regime windows**

Examples, subject to data availability:

```text
2011 eurozone / US downgrade stress
2015-2016 commodity / China slowdown
2018 Q4 risk-off
2020 COVID crash and rebound
2022 inflation / rate-hike regime
2024 validation period
```

3. **Seed stability**

Train multiple seeds under the same config:

```yaml
seeds: [7, 42, 101, 202, 999]
```

4. **Turnover and concentration stability**

Track:

```text
average weekly turnover
annualized turnover
transaction-cost drag
max single-asset weight
Herfindahl-Hirschman concentration index
average weight in SHY
average equity-like exposure
number of assets with >5% allocation
```

### 5.6 Policy behavior diagnostics

Add:

```text
src/portfolio_rl/evaluation/diagnostics.py
scripts/analyze_policy_behavior.py
```

Diagnostics should answer:

1. Is the policy just buying SPY?
2. Is the policy just hiding in SHY?
3. Is the policy rotating too much?
4. Is performance coming from one lucky month?
5. Are allocations stable or chaotic?
6. Does the policy respond sensibly to high-volatility regimes?

Suggested artifact outputs:

```text
artifacts/diagnostics/{run_id}/
├── allocation_summary.json
├── allocation_by_regime.parquet
├── concentration_metrics.parquet
├── turnover_distribution.parquet
├── monthly_returns.parquet
├── drawdown_periods.parquet
└── diagnostics_report.md
```

### 5.7 Final candidate packaging

Add:

```text
artifacts/final_model/{model_version}/
├── model.zip
├── selected_model.json
├── selected_model_card.md
├── manifest.json
├── feature_spec_v1.json
├── env.yaml
├── train_ppo.yaml
├── validation_metrics.json
├── validation_report.md
├── robustness_report.md
├── diagnostics_report.md
└── README.md
```

This folder is the handoff artifact for later serving/API work.

---

## 6. Model Selection Rules

Use validation only.

### 6.1 Primary selection metric

Default:

```yaml
selection_metric: sharpe_ratio
```

This matches the current Phase 2 config.

### 6.2 Required baseline comparisons

The selected PPO model must be compared against:

```text
equal_weight_weekly
buy_and_hold_equal_weight
spy_only
shy_only
inverse_volatility
```

### 6.3 Suggested minimum acceptance gates

A PPO model is eligible for selection only if it satisfies all hard gates:

```text
1. validation_sharpe_ratio is finite.
2. validation_total_return is finite.
3. validation_max_drawdown is finite.
4. average_weekly_turnover is finite and not excessive.
5. transaction_cost_drag is finite and explainable.
6. final validation NAV is positive.
7. PPO beats SHY-only on total return unless the validation regime is extreme risk-off.
8. PPO is not materially worse than equal_weight_weekly on both Sharpe and drawdown.
9. No diagnostic indicates static-allocation collapse unless that behavior is explicitly accepted.
```

Suggested soft gates:

```text
1. PPO Sharpe >= equal_weight_weekly Sharpe.
2. PPO max drawdown <= equal_weight_weekly max drawdown.
3. PPO transaction-cost drag is below a configured threshold.
4. PPO has reasonable allocation diversification.
5. PPO performance is not entirely explained by one asset or one month.
```

### 6.4 Tie-breakers

If multiple candidates are close:

1. Prefer lower turnover.
2. Prefer lower drawdown.
3. Prefer better seed stability.
4. Prefer simpler config.
5. Prefer less reliance on one asset.
6. Prefer clearer behavior diagnostics.

---

## 7. Experiment Matrix Design

Do not start with a massive sweep. Use staged experiments.

### Stage 1 — Smoke and reproducibility

Goal: prove repeated runs are reproducible and artifacts are complete.

```yaml
seeds: [42]
total_timesteps: 25000
configs:
  - current_default
```

Acceptance:

```text
- run completes
- validation callback runs
- artifacts written
- registry sees run
- W&B logs if enabled
- report generated
```

### Stage 2 — Seed sweep on current default

Goal: estimate variance under current config.

```yaml
seeds: [7, 42, 101, 202, 999]
total_timesteps: 500000
action_temperature: 0.5
ent_coef: 0.01
```

Acceptance:

```text
- all runs complete
- metrics aggregated
- median and dispersion reported
- no seed produces catastrophic invalid outputs
```

### Stage 3 — Action-temperature sweep

Goal: test allocation sharpness and turnover tradeoff.

```yaml
action_temperature: [0.25, 0.5, 0.75, 1.0]
seeds: [42, 101, 202]
```

Rationale:

- Lower temperature: diversified, lower turnover, potentially lower alpha.
- Higher temperature: higher conviction, higher turnover, higher cost drag.

Do not reintroduce `5.0` as a default unless there is a deliberate diagnostic run. Prior Phase 2 diagnostics already found higher temperatures problematic.

### Stage 4 — Entropy and learning-rate sweep

Goal: tune exploration and learning stability.

```yaml
ent_coef: [0.0, 0.005, 0.01, 0.02]
learning_rate: [0.0001, 0.0003, 0.0005]
seeds: [42, 101, 202]
```

### Stage 5 — Cost sensitivity

Goal: verify selected policy is not viable only under optimistic costs.

Evaluate selected candidates under:

```yaml
transaction_cost_bps: [0.0, 5.0, 10.0, 25.0, 50.0]
```

This can be evaluation-only at first. Retraining under alternate cost assumptions can be a later experiment.

### Stage 6 — Final validation candidate

Goal: train the selected config with final seed policy and full budget.

```yaml
total_timesteps: 500000
seeds: selected seed set or selected seed
```

Output:

```text
artifacts/final_model/{model_version}/
```

### Stage 7 — Final test evaluation

Only after model selection is frozen:

```bash
python scripts/evaluate_policy.py \
  --model-path artifacts/final_model/{model_version}/model.zip \
  --split test \
  --strategy ppo_final \
  --output-dir artifacts/final_model/{model_version}/test_backtest \
  --confirm-final-test
```

No hyperparameter changes after this.

---

## 8. Required New Configs

### 8.1 `configs/evaluation.yaml`

Add:

```yaml
validation:
  split: validation
  output_root: artifacts/backtests
  include_baselines: true
  include_ppo: true
  metric_for_selection: sharpe_ratio

final_test:
  split: test
  require_confirm_final_test: true
  output_root: artifacts/final_model

robustness:
  transaction_cost_bps: [0.0, 5.0, 10.0, 25.0, 50.0]
  regime_windows:
    - name: validation_2024
      start_date: "2024-01-01"
      end_date: "2024-12-31"
    - name: rate_hike_2022
      start_date: "2022-01-01"
      end_date: "2022-12-31"
    - name: covid_2020
      start_date: "2020-02-01"
      end_date: "2020-12-31"

selection:
  primary_metric: sharpe_ratio
  higher_is_better: true
  tie_breakers:
    - max_drawdown
    - average_weekly_turnover
    - transaction_cost_drag
```

### 8.2 `configs/experiments/ppo_phase3_default.yaml`

Add:

```yaml
experiment_name: ppo_phase3_default
base_data_config: configs/data.yaml
base_env_config: configs/env.yaml
base_train_config: configs/train_ppo.yaml
run_id_prefix: ppo_phase3_default
seeds: [42]
overrides: {}
```

### 8.3 `configs/experiments/ppo_phase3_seed_sweep.yaml`

Add:

```yaml
experiment_name: ppo_phase3_seed_sweep
base_data_config: configs/data.yaml
base_env_config: configs/env.yaml
base_train_config: configs/train_ppo.yaml
run_id_prefix: ppo_phase3_seed
total_timesteps: 500000
seeds: [7, 42, 101, 202, 999]
overrides:
  env.action_temperature: [0.5]
  ppo.ent_coef: [0.01]
```

### 8.4 `configs/experiments/ppo_phase3_temperature_sweep.yaml`

Add:

```yaml
experiment_name: ppo_phase3_temperature_sweep
base_data_config: configs/data.yaml
base_env_config: configs/env.yaml
base_train_config: configs/train_ppo.yaml
run_id_prefix: ppo_phase3_temp
total_timesteps: 500000
seeds: [42, 101, 202]
overrides:
  env.action_temperature: [0.25, 0.5, 0.75, 1.0]
  ppo.ent_coef: [0.005, 0.01]
```

---

## 9. Artifact Layout

Phase 3 should standardize this artifact tree:

```text
artifacts/
├── experiments/
│   ├── {run_id}/
│   │   ├── model.zip
│   │   ├── best_model.zip
│   │   ├── manifest.json
│   │   ├── config.yaml
│   │   ├── env.yaml
│   │   ├── train_ppo.yaml
│   │   ├── feature_spec_v1.json
│   │   ├── metrics_validation.json
│   │   ├── best_metrics_validation.json
│   │   ├── validation_nav.parquet
│   │   ├── validation_weights.parquet
│   │   ├── validation_trades.parquet
│   │   └── validation_costs.parquet
│   ├── registry.parquet
│   ├── registry.csv
│   └── registry.md
│
├── experiment_matrices/
│   └── {experiment_name}/
│       ├── experiment_matrix_manifest.json
│       ├── runs.csv
│       └── summary.md
│
├── model_selection/
│   ├── candidate_ranking.csv
│   ├── candidate_ranking.md
│   ├── validation_selection_report.md
│   ├── selected_model.json
│   └── selected_model_card.md
│
├── robustness/
│   └── {run_id}/
│       ├── cost_sensitivity.csv
│       ├── regime_breakdown.csv
│       ├── seed_stability.csv
│       └── robustness_report.md
│
└── final_model/
    └── {model_version}/
        ├── model.zip
        ├── selected_model.json
        ├── selected_model_card.md
        ├── manifest.json
        ├── validation_report.md
        ├── robustness_report.md
        ├── diagnostics_report.md
        └── test_backtest/       # created only after final selection
```

---

## 10. Implementation PR Plan

Use small PRs. Do not combine experiment orchestration, diagnostics, and final test evaluation in one PR.

### PR 1 — Phase 3 docs and evaluation config

Files:

```text
docs/planning/phase_3.md
configs/evaluation.yaml
configs/experiments/ppo_phase3_default.yaml
configs/experiments/ppo_phase3_seed_sweep.yaml
configs/experiments/ppo_phase3_temperature_sweep.yaml
```

Acceptance:

```text
- Phase 3 plan checked into docs/planning/phase_3.md.
- Evaluation config validates through config loader or simple YAML schema.
- Experiment configs are parseable.
- No training behavior changed yet.
```

### PR 2 — Experiment registry

Status: implemented locally in `src/portfolio_rl/training/registry.py`,
`scripts/list_experiments.py`, and `tests/unit/test_registry.py`.

Files:

```text
src/portfolio_rl/training/registry.py
scripts/list_experiments.py
tests/unit/test_registry.py
```

Acceptance:

```text
- Scans artifacts/experiments/*/manifest.json.
- Loads metrics_validation.json and best_metrics_validation.json when present.
- Extracts config values from copied env.yaml and train_ppo.yaml.
- Writes registry.csv, registry.parquet, registry.md.
- Handles missing optional files gracefully.
```

### PR 3 — Experiment matrix runner

Files:

```text
src/portfolio_rl/training/experiment_runner.py
scripts/run_experiment_matrix.py
tests/unit/training/test_experiment_runner.py
```

Acceptance:

```text
- Expands grid configs into concrete runs.
- Produces deterministic run IDs.
- Supports dry-run mode.
- Supports resume/skip when run artifacts already exist.
- Runs a tiny smoke matrix with total_timesteps=1000.
```

### PR 4 — W&B and training telemetry hardening

Files:

```text
src/portfolio_rl/training/train_ppo.py
src/portfolio_rl/training/callbacks.py
configs/train_ppo_wandb.yaml
tests/integration/test_train_ppo_wandb.py
```

Acceptance:

```text
- W&B tracked mode logs validation metrics at configured eval timesteps.
- W&B logs artifact bundle.
- Local dry-run mode still works with wandb.enabled=false.
- If wandb.enabled=true and package is missing, error message is clear.
- Optional: add SB3 training metrics callback or Monitor wrapper outputs.
```

### PR 5 — Model-selection report

Files:

```text
src/portfolio_rl/evaluation/model_selection.py
scripts/select_model.py
tests/unit/evaluation/test_model_selection.py
```

Acceptance:

```text
- Reads experiment registry.
- Applies hard gates and soft ranking.
- Produces candidate_ranking.csv and candidate_ranking.md.
- Writes selected_model.json.
- Writes validation_selection_report.md.
- Explicitly records that test split was not used.
```

### PR 6 — Robustness checks

Files:

```text
src/portfolio_rl/evaluation/robustness.py
scripts/run_robustness_checks.py
tests/unit/evaluation/test_robustness.py
```

Acceptance:

```text
- Runs transaction-cost sensitivity on selected model.
- Runs named regime-window backtests when dates are available.
- Produces robustness_report.md.
- Does not evaluate test unless explicitly requested by final model process.
```

### PR 7 — Policy behavior diagnostics

Files:

```text
src/portfolio_rl/evaluation/diagnostics.py
scripts/analyze_policy_behavior.py
tests/unit/evaluation/test_diagnostics.py
```

Acceptance:

```text
- Computes allocation concentration metrics.
- Computes average asset weights.
- Computes turnover distribution.
- Flags SPY-only collapse, SHY-only collapse, or excessive concentration.
- Writes diagnostics_report.md.
```

### PR 8 — Final candidate packager

Files:

```text
src/portfolio_rl/training/finalize_model.py
scripts/finalize_model.py
tests/integration/test_finalize_model.py
```

Acceptance:

```text
- Copies selected best_model.zip or model.zip into artifacts/final_model/{model_version}/model.zip.
- Copies configs, feature spec, selection report, robustness report, diagnostics report.
- Writes final manifest.json.
- Writes selected_model_card.md.
- Does not run test evaluation.
```

### PR 9 — Final test evaluation command wrapper

Files:

```text
scripts/run_final_test.py
tests/unit/scripts/test_run_final_test.py
```

Acceptance:

```text
- Requires explicit --confirm-final-test.
- Requires selected_model.json to exist.
- Runs test evaluation exactly through existing guarded evaluation path.
- Writes test_backtest artifacts under final_model/{model_version}/test_backtest.
- Writes a test_report.md.
- Refuses to overwrite existing final test artifacts unless --force is passed.
```

---

## 11. Required Tests

### 11.1 Registry tests

```text
test_registry_loads_complete_experiment
test_registry_handles_missing_best_metrics
test_registry_extracts_env_and_train_config_values
test_registry_exports_csv_parquet_markdown
```

### 11.2 Experiment runner tests

```text
test_experiment_matrix_expands_grid
test_experiment_matrix_run_ids_are_deterministic
test_experiment_runner_dry_run_does_not_train
test_experiment_runner_skips_completed_run
test_experiment_runner_smoke_train_1000_timesteps
```

### 11.3 Model-selection tests

```text
test_model_selection_ranks_by_primary_metric
test_model_selection_applies_hard_gates
test_model_selection_tie_breaks_by_turnover
test_model_selection_writes_selected_model_json
test_model_selection_report_states_test_not_used
```

### 11.4 Robustness tests

```text
test_cost_sensitivity_runs_multiple_bps_values
test_regime_windows_are_clipped_to_available_dates
test_robustness_report_contains_all_required_sections
test_robustness_does_not_touch_test_split_by_default
```

### 11.5 Diagnostics tests

```text
test_concentration_metrics_sum_by_date
test_detects_spy_only_collapse
test_detects_shy_only_collapse
test_turnover_distribution_is_finite
test_diagnostics_report_written
```

### 11.6 Finalization tests

```text
test_finalize_model_copies_required_artifacts
test_finalize_model_writes_manifest
test_finalize_model_fails_if_selected_model_missing
test_final_test_requires_confirmation
test_final_test_refuses_overwrite_without_force
```

---

## 12. Commands For Phase 3

### 12.1 Refresh Phase 1 and Phase 2 artifacts

```bash
python scripts/run_etl.py
python scripts/build_features.py
python scripts/validate_phase1.py
python scripts/smoke_test_env.py
python scripts/run_baselines.py
```

### 12.2 Run a PPO smoke experiment

```bash
python scripts/train_ppo.py \
  --total-timesteps 25000 \
  --run-id ppo_phase3_smoke_001
```

### 12.3 Run experiment matrix dry-run

```bash
python scripts/run_experiment_matrix.py \
  --config configs/experiments/ppo_phase3_seed_sweep.yaml \
  --dry-run
```

### 12.4 Run a small experiment matrix

```bash
python scripts/run_experiment_matrix.py \
  --config configs/experiments/ppo_phase3_default.yaml \
  --total-timesteps 25000
```

### 12.5 Build registry

```bash
python scripts/list_experiments.py \
  --experiment-root artifacts/experiments \
  --output artifacts/experiments/registry
```

### 12.6 Select model from validation

```bash
python scripts/select_model.py \
  --registry artifacts/experiments/registry.csv \
  --baseline-root artifacts/backtests/baselines_validation \
  --output-dir artifacts/model_selection
```

### 12.7 Run robustness checks

```bash
python scripts/run_robustness_checks.py \
  --selected-model artifacts/model_selection/selected_model.json \
  --config configs/evaluation.yaml \
  --output-dir artifacts/robustness
```

### 12.8 Analyze policy behavior

```bash
python scripts/analyze_policy_behavior.py \
  --selected-model artifacts/model_selection/selected_model.json \
  --output-dir artifacts/diagnostics
```

### 12.9 Finalize selected model package

```bash
python scripts/finalize_model.py \
  --selected-model artifacts/model_selection/selected_model.json \
  --model-version ppo_v1_selected_YYYYMMDD \
  --output-root artifacts/final_model
```

### 12.10 Run final test exactly once

```bash
python scripts/run_final_test.py \
  --model-version ppo_v1_selected_YYYYMMDD \
  --confirm-final-test
```

---

## 13. Common Phase 3 Failure Modes

### Failure mode 1 — Tuning on test

Symptom:

```text
Multiple test evaluations are run while changing hyperparameters.
```

Fix:

```text
Only validation drives selection. Test is run once after final selection.
```

### Failure mode 2 — Selecting by total return only

Symptom:

```text
High-return candidate has high turnover, high drawdown, and unstable seed behavior.
```

Fix:

```text
Use Sharpe as primary metric and apply drawdown/turnover/cost gates.
```

### Failure mode 3 — Ignoring seed variance

Symptom:

```text
One lucky PPO run beats baselines, but most seeds fail.
```

Fix:

```text
Run seed sweeps and rank median behavior, not only best seed.
```

### Failure mode 4 — Action-temperature regression

Symptom:

```text
A sharper softmax improves validation return but explodes turnover and costs.
```

Fix:

```text
Evaluate cost drag, turnover distribution, and cost sensitivity before selection.
```

### Failure mode 5 — W&B-only reproducibility

Symptom:

```text
Metrics exist in W&B, but local artifacts cannot reproduce the run.
```

Fix:

```text
Every run must write local manifests, configs, feature spec, metrics, and model artifacts.
```

### Failure mode 6 — Candidate cannot be served later

Symptom:

```text
Final model exists, but feature spec/config/version information is missing.
```

Fix:

```text
Final model package must include model.zip, env.yaml, train_ppo.yaml, feature_spec_v1.json, manifest.json, and selected_model.json.
```

---

## 14. Definition of Done

Phase 3 is complete when:

```text
1. Phase 3 configs and docs are checked in.
2. Experiment registry scans all experiment bundles.
3. Experiment matrix runner can dry-run and execute small runs.
4. W&B tracked mode and offline/local mode both work.
5. At least one seed sweep is run for the current default PPO config.
6. At least one action-temperature or entropy sweep is run.
7. Candidate ranking report is generated from validation only.
8. Selected model is justified against baselines.
9. Robustness checks cover transaction-cost sensitivity and regime windows.
10. Diagnostics identify concentration, turnover, and asset-allocation behavior.
11. Final model package is created with all required artifacts.
12. Final test evaluation is run once, with explicit confirmation, after model selection is frozen.
13. Model card documents data version, feature version, environment config, training config, validation results, robustness results, test results, known limitations, and serving assumptions.
```

---

## 15. Recommended Phase 3 Model Card Template

```markdown
# Model Card — {model_version}

## Summary
- Algorithm:
- Feature version:
- Asset universe:
- Training period:
- Validation period:
- Test period:
- Rebalance frequency:
- Transaction cost assumption:

## Selected Model
- Source run ID:
- Source artifact path:
- Selection metric:
- Selection rationale:
- Was test used for selection? No.

## Validation Results
| Strategy | Total Return | CAGR | Sharpe | Max Drawdown | Turnover | Cost Drag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Robustness Summary
- Cost sensitivity:
- Regime results:
- Seed stability:

## Behavior Diagnostics
- Average allocation:
- Concentration:
- Turnover distribution:
- Notable regime behavior:

## Final Test Results
- Test run date:
- Test command:
- Metrics:

## Known Limitations
- ETF-only simulator.
- No intraday execution model.
- Static universe.
- Simplified transaction-cost model.
- No tax, borrow, slippage, or liquidity constraints beyond current cost assumption.

## Serving Assumptions
- API must use structured payloads.
- API must flatten inputs by feature_spec order.
- API must use the same action transform and action_temperature as training.
```

---

## 16. Guidance To Junior Developers

The key Phase 3 discipline is this:

> We are no longer proving that the environment can train. We are proving that a selected model deserves to be trusted.

That means every Phase 3 PR should improve one of four things:

1. **Reproducibility** — Can we recreate the run?
2. **Comparability** — Can we compare PPO fairly to baselines?
3. **Robustness** — Does the policy survive realistic changes in costs, regimes, and seeds?
4. **Explainability** — Can we explain what the policy is doing?

If a proposed change does not improve one of those four, it probably does not belong in Phase 3.
