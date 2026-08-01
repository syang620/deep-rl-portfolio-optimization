# Phase 3 Remaining Implementation Plan — Candidate Qualification Before Final Test

**Project:** RL Dynamic Portfolio Allocation
**Document purpose:** implementation plan and engineering instructions for the remaining Phase 3 work
**Audience:** junior engineers, coding-agent operators, ML reviewers, and portfolio-management reviewers
**Repository baseline:** `syang620/deep-rl-portfolio-optimization`
**Latest reviewed Phase 3 commit:** `7ef06fa82a5a01e265f24c239a5e20f6db194947` — `Complete Phase 3 candidate validation and packaging`
**Date:** 2026-07-27
**Status:** Phase 3A remains open. PR 11 through PR 16 are implemented; PR 14
and PR 15 retain all four predeclared partial-rebalancing alphas for later
walk-forward analysis rather than selecting from 2024. PR 16 builds data
artifacts only; training begins in PR 17. The turnover-v2 campaign has not
accessed 2025+, but a legacy Phase 2 model has. Phase 3B is blocked until the
remaining work and gates in this document are completed and PM/ML reviewers
approve a new independent holdout.

---

## 0. Executive decision

The current PPO campaign is directionally encouraging but is not yet sufficient to support a claim of persistent outperformance or replacement of weekly equal weight.

The remaining Phase 3 objective is not another broad PPO hyperparameter sweep. It is:

> Convert the current five-seed research result into one precisely defined, executable policy and determine whether its state-dependent decisions produce repeatable net value beyond transparent allocation rules after controlling for initialization, turnover, historical regime variation, and execution assumptions.

The work is divided into two subphases:

```text
Phase 3A — Candidate qualification and pre-test freeze
Phase 3B — One-time final test evaluation after PM/ML sign-off
```

This document primarily governs Phase 3A. Phase 3B remains blocked until the final candidate and acceptance criteria are frozen.

---

## 1. Current evidence and correct interpretation

The existing research supports the following provisional statement:

> PPO is a diversified adaptive candidate with modest 2024 development-period alpha, but current evidence does not establish statistically reliable superiority over equal weight after accounting for checkpoint/configuration selection, seed variance, turnover, and the short evaluation history.

The current campaign has already established:

- five completed PPO seeds;
- modest positive active return versus weekly equal weight in 2024;
- four of five seeds ahead on Sharpe;
- no drawdown improvement;
- substantially higher turnover and transaction-cost drag;
- valid cost monotonicity;
- diversified allocations rather than SPY-only, SHY-only, or single-asset collapse;
- material policy disagreement and 2022 seed instability;
- paired bootstrap intervals that still include zero;
- no 2025+ access by the turnover-v2 campaign;
- one declared legacy 2025+ evaluation by a different Phase 2 model.

The 2024 period has been used for checkpoint selection, configuration decisions, seed comparison, model packaging, diagnostics, and statistical analysis. From this point forward, reports must label it as:

```text
2024 consumed development/selection period
```

It may still be used for development and diagnosis, but not presented as independent proof.

---

## 2. The investment claim Phase 3 must test

The project is not trying to prove that PPO can produce a high backtest return.

The target claim is:

> A reproducibly trained, state-dependent PPO allocation policy produces economically meaningful and sufficiently robust net improvement over transparent portfolio rules, while remaining diversified, implementable, and stable enough to justify its additional model and operating complexity.

That claim has five components:

1. **Adaptiveness**
   The agent's actions must respond meaningfully to current market state rather than reproduce a static asset mix.

2. **Economic value**
   Net performance after costs must improve on declared simple baselines.

3. **Robustness**
   The result must survive seeds, initialization changes, historical holdouts, costs, and execution delay.

4. **Implementability**
   Turnover, capacity, and execution assumptions must be credible.

5. **Reproducibility and governance**
   One exact executable policy must be frozen without consulting the final test set.

---

## 3. Locked conventions for the remaining work

These rules are non-negotiable unless a senior PM/ML review explicitly changes them.

### 3.1 Common starting allocation for primary comparisons

The primary comparison must start all rebalancing strategies from equal weight:

```text
PPO
five-seed PPO ensemble
inverse volatility
weekly equal weight
buy-and-hold equal weight
momentum
SPY-only
SHY-only
```

Inverse volatility is a separate competing policy. It is not the default starting allocation for PPO.

This isolates the policy effect:

```text
same initial portfolio
+ same dates
+ same returns
+ same cost and drift mechanics
= fair comparison of decision rules
```

### 3.2 Initialization sensitivity is separate

The frozen PPO candidate must also be evaluated from multiple initial portfolios:

```text
equal weight
inverse volatility
100% SHY
```

This is a model-risk diagnostic. Results must not be used to choose whichever initial condition makes PPO look best.

The official reported result remains the equal-weight initialization unless the investment mandate is formally changed before the final-test freeze.

### 3.3 Campaign medians are not executable policies

The five-seed median performance is a distribution summary, not a portfolio.

Phase 3 must construct and evaluate actual executable candidates:

```text
A. Representative single-seed policy
B. Mean-weight ensemble of all five seed policies
C. Mean-weight ensemble with one transparent turnover-control overlay
```

### 3.4 Preserve the current financial contracts

Continue to use:

```text
weekly rebalance every 5 trading days
long-only and fully invested
softmax temperature 0.5
10 bps base transaction cost
one-way turnover = 0.5 × Σ|target - current drifted weight|
scaled log-growth reward for training
dynamic current weights in the observation
```

Any change to reward, features, action mapping, or training initialization creates a new model version and requires a new campaign.

### 3.5 No additional broad hyperparameter search

Do not restart temperature, entropy, network-size, or learning-rate sweeps unless the diagnostic work identifies a specific failure that those changes address.

The next work should reduce uncertainty about the existing candidate, not increase the number of selected configurations.

### 3.6 Independent holdout remains blocked

Do not evaluate 2025 onward during any task in Phase 3A. The repository's
2025+ test designation was previously accessed by a legacy Phase 2 model and
must not be described as a globally untouched holdout.

No script added below may default to the test split. Phase 3B requires an
explicit confirmation flag, a frozen candidate manifest, and PM/ML approval of
a new independent holdout.

---

## 4. Important current-code observation

The current backtest starts with an equal-weight vector. The inverse-volatility baseline receives only return history available inside its selected evaluation window. At the first row of a split, that history contains only one row, so estimated cross-asset volatilities are effectively equal and the initial inverse-volatility target becomes approximately equal weight.

Therefore, a real inverse-volatility **initial condition** requires an explicit past-only context window before the evaluation start date.

Do not solve this by weakening forward-return split boundaries. Add a narrow historical-context interface that can read rows strictly before the evaluation start while keeping reward and evaluation rows split-bounded.

---

# 5. Remaining workstreams and PR plan

The existing Phase 3 plan used PRs 1–10. Continue with the following sequence.

---

## PR 11 — Research freeze, terminology, and provenance audit

**Status:** Implemented. Campaign
`ppo_phase3_seed_sweep_turnover_v2_5ce122d9a926` is frozen and provenance
audited. The turnover-v2 campaign is test-free; Phase 3B remains blocked
because the declared legacy Phase 2 test access requires approval of a new
independent holdout.

### Goal

Freeze the current campaign as the starting point for the remaining research and verify that every selected artifact uses the corrected turnover-v2 mechanics.

### Files

```text
docs/planning/phase_3_remaining_regenerated.md
configs/research/phase3_candidate_qualification.yaml
src/portfolio_rl/training/provenance.py
scripts/freeze_phase3_campaign.py
tests/unit/test_provenance.py
```

### Required behavior

1. Record the exact Git commit.
2. Record the model run IDs and checkpoint paths for seeds:
   ```text
   7, 42, 101, 202, 999
   ```
3. Record data, feature-spec, environment, and training-config hashes.
4. Verify that each selected model was trained under the corrected one-way turnover convention.
5. Verify that no test artifact exists for the campaign.
6. Update reports to call 2024 the consumed development/selection period.
7. Write a frozen research manifest.

### Artifacts

```text
artifacts/research_freeze/{campaign_id}/
├── freeze_manifest.json
├── model_inventory.csv
├── hash_inventory.json
├── provenance_report.md
└── test_access_audit.json
```

### Acceptance criteria

```text
- Every selected model has traceable run and checkpoint metadata.
- All selected models are turnover-v2 compatible.
- The exact seed/checkpoint set is frozen.
- The turnover-v2 campaign test audit reports no access.
- Declared legacy project-level access is recorded and Phase 3B remains blocked.
- No model behavior changes are introduced.
```

### Stop condition

If any selected checkpoint was only re-evaluated under turnover-v2 but trained using inconsistent reward/cost mechanics, do not continue. Retrain the affected campaign first.

---

## PR 12 — Initial portfolio framework and initialization-sensitivity report

### Goal

**Status:** implemented for the frozen seed-42 policy.

Measure whether the frozen PPO policy's result is materially path-dependent on the portfolio present at the first decision.

### Files

```text
src/portfolio_rl/evaluation/initialization.py
src/portfolio_rl/evaluation/backtest.py
src/portfolio_rl/data/feature_store.py
scripts/run_initialization_sensitivity.py
tests/unit/evaluation/test_initialization.py
tests/unit/data/test_feature_store_context_history.py
tests/integration/test_initialization_sensitivity.py
```

### Design

Add an initializer protocol:

```python
class InitialPortfolioProvider(Protocol):
    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        ...
```

Implement:

```text
EqualWeightInitializer
SHYInitializer
InverseVolatilityInitializer
StaticWeightInitializer
```

The inverse-volatility initializer must use a configured past-only lookback, recommended:

```yaml
initialization:
  inverse_volatility_lookback_trading_days: 63
```

Add the narrow feature-store method:

```python
get_pre_window_log_returns(lookback: int) -> np.ndarray
```

Rules:

- return exactly `lookback` rows immediately before the evaluation store;
- rows must be strictly earlier than the evaluation start;
- earlier rows may cross a split boundary;
- raise when there is insufficient earlier history;
- no future evaluation rows;
- no forward-return API changes;
- no existing split-bounded trailing-return API changes;
- no general relaxation of split-bounded reward access.

Extend the backtest engine with an initializer object. Default behavior must
remain equal weight. Each scenario starts at NAV 1.0 already holding its
configured initial portfolio, with no establishment cost. The first policy
decision incurs normal half-L1 turnover and cost from that endowed portfolio
to the policy target.

Static weights must be ticker-mapped and flattened in feature-store asset
order. Missing or unexpected tickers and invalid weights must fail. The SHY
initializer must fail explicitly if SHY is absent.

### Run matrix

Evaluate the frozen seed-42 candidate from:

```text
equal weight
63-day inverse volatility
100% SHY
```

### Required outputs

```text
artifacts/initialization_sensitivity/{campaign_id}/
├── results_by_model_and_initializer.csv
├── allocation_paths.parquet
├── nav_paths.parquet
├── convergence_metrics.csv
├── convergence_summary.csv
├── run_manifest.json
└── initialization_sensitivity_report.md
```

Report:

- full-period return, Sharpe, drawdown, turnover, and cost drag;
- first 4-week and first 12-week performance;
- full-L1 and half-L1 distances for PPO targets and actual pre-trade paths;
- terminal and average half-L1 distance across initializations;
- time required for target and pre-trade paths to become similar, if they do.

The primary convergence rule is at most 5% one-way reallocation
(`half_l1 <= 0.05`) for four consecutive weekly decisions. Four- and
twelve-week diagnostics mean exactly 20 and 60 trading days and include four
and twelve cost rows. Short or zero-volatility Sharpe diagnostics return
`None`.

### Interpretation

Healthy:

```text
early differences are visible,
but allocations and full-period conclusions become broadly similar
```

Concerning:

```text
initial conditions produce persistently different allocations
or reverse the performance conclusion
```

### Acceptance criteria

```text
- Primary equal-weight result is unchanged.
- Initial endowments incur no establishment cost.
- Inverse-vol initialization uses only pre-start data.
- All initial weights are valid and sum to one.
- Every initializer run creates a fresh policy instance.
- Split- and date-based guards both prohibit test access.
- Freeze, model, configuration, and feature hashes are verified.
- No initialization is cherry-picked as the headline result.
- Findings are scoped only to seed 42; ensemble sensitivity is deferred to PR 13.
- Report clearly separates path-dependence diagnostics from policy comparison.
```

Before final walk-forward and final-test comparisons, the inverse-volatility
baseline must use a full past-only lookback across the evaluation-window
boundary. PR 12 tracks this requirement but does not change frozen baseline
results.

---

## PR 13 — Five-seed executable ensemble

### Goal

**Status:** implemented for the frozen five-seed turnover-v2 campaign.

Turn seed dispersion into one tradable policy rather than relying on a median statistic or one representative seed.

### Files

```text
src/portfolio_rl/policies/ensemble_policy.py
scripts/evaluate_policy_ensemble.py
tests/unit/policies/test_ensemble_policy.py
tests/integration/test_ensemble_backtest.py
```

### Canonical ensemble

At each decision date:

```python
member_targets = [
    policy.target_weights(observation, info)
    for policy in member_policies
]
ensemble_target = np.mean(member_targets, axis=0)
```

Because each member target is long-only and sums to one, the mean remains a valid portfolio.

### Rules

- Use all five frozen seed checkpoints.
- Feed each member the same observation and current live weights.
- Use deterministic inference.
- Use the same action temperature recorded with each model.
- Average target weights, not raw PPO actions.
- Do not select a subset of seeds based on 2024 results.
- Save member model hashes and ordering.

### Candidate comparison

```text
representative seed 42
five-seed mean-weight ensemble
weekly equal weight
inverse volatility
other existing baselines
```

All primary comparisons start equal weight.

### Artifacts

```text
artifacts/ensemble/{campaign_id}/
├── ensemble_manifest.json
├── member_targets.parquet
├── ensemble_targets.parquet
├── disagreement_metrics.parquet
├── backtest/
└── ensemble_report.md
```

Report:

- ensemble net performance;
- turnover and cost drag;
- seed-to-ensemble tracking error;
- cross-seed target-weight dispersion;
- dominant-asset agreement;
- 2022 ensemble behavior versus individual seeds.

### Acceptance criteria

```text
- Ensemble replay is deterministic.
- Target weights are finite, nonnegative, and sum to one.
- All five frozen models are included.
- Model hashes are recorded.
- Ensemble is evaluated as an actual policy, not inferred from median metrics.
```

---

## PR 14 — Transparent turnover-control overlay

**Implementation status:** Complete. The canonical study evaluates all four
predeclared alphas for the five-seed ensemble and seed-42 diagnostic, preserves
raw and executed targets, and does not select an alpha from the consumed 2024
development period.

### Goal

Determine whether the current signal can be retained with materially less trading.

### Initial scope

Implement only **partial rebalancing** first. Do not immediately create a large grid of overlay combinations.

### Files

```text
src/portfolio_rl/policies/overlays.py
scripts/run_turnover_overlay_study.py
tests/unit/policies/test_overlays.py
tests/integration/test_turnover_overlay_backtest.py
```

### Partial-rebalance rule

```python
executed_target = (
    current_drifted_weights
    + alpha * (policy_target - current_drifted_weights)
)
```

Evaluate:

```yaml
partial_rebalance_alpha:
  - 0.25
  - 0.50
  - 0.75
  - 1.00
```

This is a convex combination, so it stays long-only and sums to one.

### Candidate set

Apply the overlay to:

```text
five-seed mean-weight ensemble
representative seed 42, for diagnostic comparison
```

### Rules

- The overlay is evaluation-only in the first study.
- It must be applied after the model emits a target and before turnover/cost calculation.
- Do not change the PPO reward or retrain models in this PR.
- Do not select an alpha using one metric alone.
- Preserve all unmodified target weights for audit.

### Artifacts

```text
artifacts/turnover_overlays/{campaign_id}/
├── overlay_results.csv
├── raw_and_executed_targets.parquet
├── turnover_frontier.csv
├── cost_frontier.csv
└── turnover_overlay_report.md
```

### Selection rule

Do not choose the overlay solely from 2024. Carry all four predefined alphas into the walk-forward study.

After walk-forward results exist, prefer the lowest-turnover alpha that:

```text
- has positive median active return versus the declared hurdle baseline;
- has positive median active Sharpe;
- preserves an agreed fraction of the unmodified policy's gross advantage;
- does not materially worsen drawdown;
- passes execution-stress gates.
```

Thresholds must be written in config before aggregation.

### Deferred extensions

Only after reviewing partial rebalancing may the team add one of:

```text
no-trade band
equal-weight shrinkage
explicit turnover penalty with retraining
```

Adding more than one family simultaneously is prohibited because it creates another uncontrolled selection grid.

---

## PR 15 — Dynamic-value diagnostics

**Implementation status:** Complete. The frozen ensemble and all four
partial-rebalancing alphas are compared with a 2023 past-only static portfolio,
a non-deployable 2024 oracle static portfolio, a one-decision lag, and fixed
4-, 13-, and 26-decision circular shifts. No alpha is selected from 2024.

### Goal

Determine whether PPO's state-to-action mapping adds value beyond a static portfolio or a weakly timed sequence of allocations.

### Files

```text
src/portfolio_rl/evaluation/dynamic_value.py
scripts/run_dynamic_value_checks.py
tests/unit/evaluation/test_dynamic_value.py
tests/integration/test_dynamic_value_backtests.py
```

### Required diagnostics

#### A. Ex-ante static policy

Use average PPO targets estimated from a period available before the evaluated outer holdout.

For walk-forward folds, derive the static weights from the inner-validation period and hold those weights fixed in the outer holdout.

This is a deployable comparison.

#### B. Oracle static-average diagnostic

Compute the average PPO target over the same evaluation period and backtest that fixed portfolio.

This uses future information and is **not deployable**. Label it clearly as an attribution diagnostic only.

Purpose:

```text
If oracle static weights explain nearly all PPO performance,
the result is primarily strategic asset mix rather than dynamic timing.
```

#### C. One-rebalance lag

Apply the PPO target one weekly decision later.

Purpose:

```text
If lagged actions perform similarly,
timely state dependence may be weak.
```

#### D. Circularly shifted target sequences

Replay several predeclared circular shifts of the recorded target-weight path.

Purpose:

```text
Test whether matching the target sequence to the original market dates matters.
```

Use fixed shifts, for example:

```yaml
circular_shift_periods: [4, 13, 26]
```

Do not search many shifts and report only favorable ones.

### Artifacts

```text
artifacts/dynamic_value/{campaign_id}/
├── diagnostic_results.csv
├── target_sequences.parquet
├── active_return_decomposition.csv
└── dynamic_value_report.md
```

### Interpretation gates

Strong dynamic evidence:

```text
PPO/ensemble materially outperforms ex-ante static,
lagged, and shifted controls after costs
```

Weak dynamic evidence:

```text
static or mistimed controls capture most of the performance
```

If evidence is weak, the correct conclusion may be:

> The model discovered a useful diversified strategic allocation, but dynamic RL timing has not been demonstrated.

That is still a valid research result, but it changes the project claim.

---

## PR 16 — Leakage-safe nested walk-forward artifact builder

**Implementation status:** Complete. Four deterministic fold bundles are built
from pre-normalization features. Winsorization and scaling are fit on each
fold's inner-training rows only, and training/selection matrices are physically
separated from outer-evaluation matrices. No PPO training or checkpoint
selection occurs in this PR.

### Goal

Create multiple historical pseudo-out-of-sample periods without reusing the current 2010–2023 train-fitted scaler.

### Critical rule

Each fold must fit normalization only on its inner-training period.

Do not slice the existing normalized Phase 1 matrix and call it walk-forward. That would use a scaler fitted with information from later years.

### Files

```text
configs/walk_forward.yaml
src/portfolio_rl/data/walk_forward.py
src/portfolio_rl/features/fold_pipeline.py
scripts/build_walk_forward_artifacts.py
tests/unit/data/test_walk_forward_splits.py
tests/integration/test_walk_forward_artifacts.py
```

### Recommended nested folds

| Fold | Inner train | Inner validation | Outer pseudo-OOS evaluation |
|---|---|---|---|
| WF1 | 2010–2014 | 2015 | 2016–2017 |
| WF2 | 2010–2016 | 2017 | 2018–2019 |
| WF3 | 2010–2018 | 2019 | 2020–2021 |
| WF4 | 2010–2020 | 2021 | 2022–2023 |

The inner validation period selects the checkpoint. The outer period must never influence training, checkpoint selection, or per-fold configuration.

### Fold pipeline

For each fold:

1. Start from the unnormalized feature panel.
2. Assign fold-specific labels:
   ```text
   inner_train
   inner_validation
   outer_evaluation
   ```
3. Fit the scaler on `inner_train` only.
4. Transform all three periods with that scaler.
5. Build the fold model matrix in the same feature-spec order.
6. Save fold-specific hashes and quality reports.
7. Verify all rolling features use past-only data.
8. Preserve enough pre-period raw history for feature warm-up.

### Artifacts

```text
artifacts/walk_forward/data/{fold_id}/
├── model_matrix_daily.parquet
├── feature_spec.json
├── scaler.pkl
├── data_quality_report.json
├── fold_manifest.json
└── split_summary.json
```

### Required tests

```text
test_fold_dates_do_not_overlap
test_outer_evaluation_is_after_inner_validation
test_scaler_fit_rows_equal_inner_train_only
test_fold_feature_order_matches_main_feature_spec
test_fold_forward_returns_do_not_cross_outer_boundary
test_fold_artifact_hashes_are_stable
```

### Acceptance criteria

```text
- Four fold artifact bundles are reproducible.
- Scaler leakage tests pass.
- No 2024 or 2025+ rows enter any fold.
- Current feature definitions remain unchanged.
```

---

## PR 17 — Walk-forward training, baselines, ensemble, and aggregation

### Goal

Test whether the candidate family repeats across genuinely unseen historical periods.

### Files

```text
configs/experiments/ppo_walk_forward.yaml
src/portfolio_rl/training/walk_forward_runner.py
src/portfolio_rl/evaluation/walk_forward_report.py
scripts/run_walk_forward_campaign.py
scripts/aggregate_walk_forward.py
tests/unit/training/test_walk_forward_runner.py
tests/unit/evaluation/test_walk_forward_report.py
```

### Frozen training configuration

Use the current candidate PPO configuration:

```text
PPO MlpPolicy
temperature 0.5
500,000 steps
same network and PPO parameters
same reward
same 10 bps training cost
```

Do not tune separate hyperparameters for each fold.

### Seed protocol

Use two stages:

```text
Pilot: seeds [42, 101, 202]
Expansion: seeds [7, 42, 101, 202, 999]
```

Expand to five seeds if the pilot pipeline is valid and no implementation issue is found. Do not expand or stop based only on favorable pilot performance.

### Per-fold process

1. Train each seed on `inner_train`.
2. Select checkpoint using `inner_validation` only.
3. Evaluate selected checkpoint on `outer_evaluation`.
4. Construct the five-seed mean-weight ensemble on the outer period.
5. Apply each predeclared partial-rebalance alpha.
6. Run all declared baselines from equal weight.
7. Run initialization sensitivity separately.
8. Run dynamic-value controls.
9. Save complete artifacts.

### Baseline protocol

Before running the outer evaluations, declare one primary hurdle baseline in config.

Recommended reporting:

```text
Primary reference: weekly equal weight
Risk-aware hurdle: inverse volatility
Additional references: buy-and-hold equal weight, momentum, SPY-only, SHY-only
```

Do not redefine the hurdle separately in each fold.

### Artifacts

```text
artifacts/walk_forward/results/{campaign_id}/
├── fold_metrics.csv
├── seed_metrics.csv
├── ensemble_metrics.csv
├── overlay_metrics.csv
├── baseline_metrics.csv
├── active_returns.parquet
├── monthly_active_returns.csv
├── fold_manifests/
└── walk_forward_report.md
```

### Predeclared decision gates

Store thresholds in config, not hard-coded logic.

Suggested initial gates:

```yaml
walk_forward_gates:
  minimum_positive_active_return_folds: 3
  require_positive_median_active_return: true
  require_positive_median_active_sharpe: true
  maximum_single_fold_share_of_total_active_pnl: 0.60
  maximum_allowed_drawdown_deterioration_pp: 2.0
  maximum_average_weekly_turnover: null  # PM team must set
  maximum_transaction_cost_drag: null    # PM team must set
```

The PM team must fill the turnover and cost budgets before final aggregation.

### Required report questions

```text
- How many folds have positive active return?
- How many beat equal weight on Sharpe?
- How many beat inverse volatility?
- Is the median active return positive?
- Is performance dominated by one fold or month?
- What happens in 2022–2023?
- Does the ensemble reduce seed dispersion?
- Which partial-rebalance alpha lies on the best net-return/turnover frontier?
- Does initialization change the conclusion?
```

### Statistical interpretation

Use block bootstrap within folds and pooled active-return analysis as advisory evidence. Do not treat four folds as a large independent sample or overstate p-values.

The primary evidence is repeated directional performance and economic stability, not one significance threshold.

---

## PR 18 — Execution realism and 2022 root-cause analysis

### Goal

Test whether the candidate survives a minimally realistic implementation model and understand the most unstable historical regime.

### Files

```text
src/portfolio_rl/evaluation/execution_stress.py
src/portfolio_rl/evaluation/attribution.py
scripts/run_execution_stress.py
scripts/analyze_2022_instability.py
tests/unit/evaluation/test_execution_stress.py
tests/unit/evaluation/test_attribution.py
```

### A. One-close execution delay

With daily close-to-close data, define the stress clearly:

```text
decision is computed at close t
existing portfolio earns the next daily return
new target becomes effective at close t+1
```

This is an evaluation stress, not a claim about next-open execution.

Report:

```text
no-delay result
one-close-delay result
active-return loss from delay
turnover and cost changes
```

### B. Asset-specific cost tiers

Add an evaluation-only cost map, for example:

```yaml
asset_cost_bps:
  SPY: 5
  QQQ: 5
  IWM: 8
  EFA: 8
  EEM: 12
  TLT: 7
  IEF: 5
  SHY: 3
  LQD: 8
  HYG: 12
  GLD: 8
  DBC: 15
  VNQ: 10
  XLU: 8
```

These values are scenarios, not validated market estimates. The PM team may replace them with approved assumptions.

To preserve equivalence with the scalar one-way turnover convention:

```python
cost_fraction = (
    0.5
    * np.sum(np.abs(target - current) * asset_cost_bps)
    / 10_000.0
)
```

When all asset costs are equal, this must match the existing scalar calculation exactly.

### C. Optional capacity scenario

If volume data are sufficiently reliable, add scenario analysis for:

```text
portfolio AUM
weekly traded notional
percentage of average daily dollar volume
maximum asset-level participation
```

Treat this as a capacity approximation, not an execution simulator.

### D. 2022 root-cause analysis

For every seed and the ensemble, report:

```text
weekly target weights
equity, rates, credit, commodity, real-asset, and SHY exposure
asset-level return contribution
turnover and cost contribution
cross-seed policy disagreement
largest weekly active gains and losses
sensitivity to small observation perturbations
```

Answer:

1. Did losing seeds remain exposed to both equity and duration?
2. Which assets explain the approximately -9% to +9% seed range?
3. Was performance dominated by a few decisions?
4. Did the ensemble reduce the dispersion?
5. Did the partial-rebalance overlay reduce losses or only slow adaptation?

### Artifacts

```text
artifacts/execution_stress/{campaign_id}/
├── delay_results.csv
├── asset_cost_results.csv
├── capacity_scenarios.csv
└── execution_stress_report.md

artifacts/regime_attribution/{campaign_id}/
├── asset_contributions.parquet
├── exposure_paths.parquet
├── seed_disagreement.parquet
├── largest_active_periods.csv
└── rate_hike_2022_report.md
```

### Acceptance criteria

```text
- Scalar-cost equivalence test passes.
- No-delay reference reproduces existing metrics.
- One-close delay uses no future observation.
- Cost assumptions are versioned and labeled as scenarios.
- 2022 report identifies concrete drivers rather than only restating dispersion.
```

---

## PR 19 — Final candidate selection and pre-test freeze

### Goal

Choose one exact executable candidate and pre-register the final-test decision rules.

### Candidate must specify

```text
member model hashes
ensemble rule
initial portfolio
partial-rebalance alpha
action temperature
transaction-cost convention
feature version
data snapshot
checkpoint-selection rule
execution assumptions
baseline hurdle
```

Example candidate identity:

```text
five-seed mean-weight ensemble
equal-weight initial portfolio
partial-rebalance alpha = 0.50
weekly decisions
temperature 0.5 per member
10 bps base one-way turnover cost
```

The actual alpha must be selected from the predeclared walk-forward study, not from the final test.

### Selection method

Use lexicographic gates rather than one arbitrary weighted score:

```text
1. Must pass provenance and reproducibility checks.
2. Must have positive median walk-forward active return.
3. Must have positive median walk-forward active Sharpe.
4. Must beat the required number of folds.
5. Must pass turnover and cost budgets.
6. Must not materially worsen drawdown.
7. Must show meaningful dynamic value versus static/lagged controls.
8. Must pass initialization-sensitivity and execution-stress review.
9. Among survivors, prefer lower turnover and simpler implementation.
```

### Files

```text
configs/final_candidate_acceptance.yaml
src/portfolio_rl/training/pretest_freeze.py
scripts/freeze_final_candidate.py
tests/unit/training/test_pretest_freeze.py
```

### Artifacts

```text
artifacts/pretest_freeze/{model_version}/
├── frozen_candidate.json
├── acceptance_criteria.yaml
├── freeze_manifest.json
├── model_hashes.json
├── data_and_feature_hashes.json
├── commands.md
├── PM_review_packet.md
└── test_access_audit.json
```

### PM review packet

Include:

```text
2024 development result
walk-forward fold results
baseline comparison
initialization sensitivity
ensemble comparison
turnover frontier
dynamic-value diagnostics
execution-delay and cost scenarios
2022 attribution
known limitations
exact final-test pass/fail rules
```

### Acceptance criteria

```text
- One and only one executable candidate is frozen.
- No unresolved artifact hash or provenance gap remains.
- PM and ML reviewers approve the acceptance criteria.
- Final-test command is generated but not run.
- Test-access audit reports no use of the newly approved independent holdout.
```

---

## PR 20 — One-time final test evaluation

### Status

Deferred until the team reviews all Phase 3A evidence.

### Rules

Use the existing guarded evaluation path. Add no new candidate logic in this PR.

The command must:

```text
require explicit confirmation
require the frozen candidate manifest
verify all hashes
refuse to overwrite an existing result
write the exact command and timestamp
```

After the test:

```text
Pass:
    move to paper trading / forward monitoring and later serving work

Fail:
    do not tune against the same 2025+ period and call it validation
    create a new model version
    reserve future observations as the next independent holdout
```

---

# 6. Recommended order of execution

Run the remaining Phase 3 work in this exact order:

```text
1. PR 11 — Freeze and provenance audit
2. PR 12 — Initialization sensitivity
3. PR 13 — Five-seed executable ensemble
4. PR 14 — Partial-rebalance overlay
5. PR 15 — Dynamic-value diagnostics
6. PR 16 — Walk-forward artifact builder
7. PR 17 — Walk-forward training and aggregation
8. PR 18 — Execution stress and 2022 attribution
9. PM/ML evidence review
10. PR 19 — Freeze one final candidate and acceptance criteria
11. PR 20 — Run final test once, only after approval
```

Do not run PR 20 while earlier findings are still changing candidate construction.

---

# 7. Coding-agent instructions

Use one scoped prompt per PR. Provide only the relevant section of this document and the current file interfaces.

## Prompt for PR 12

```text
Implement an evaluation-only initial portfolio framework.

Requirements:
- Keep equal weight as the default.
- Add EqualWeightInitializer, SHYInitializer, InverseVolatilityInitializer, and StaticWeightInitializer.
- InverseVolatilityInitializer must use only returns strictly before the evaluation window.
- Add a narrow feature-store context-history accessor without weakening forward-return split boundaries.
- Extend run_weight_policy_backtest to accept explicit initial weights or an initializer.
- Add tests proving no future evaluation row is used.
- Write a CLI that compares frozen policies across equal-weight, inverse-volatility, and SHY initializations.
Do not retrain PPO and do not access the test split.
```

## Prompt for PR 13

```text
Implement a deterministic mean-weight ensemble for five saved SB3 PPO policies.

Requirements:
- Load all five frozen model paths from a manifest.
- Provide the same observation and current weights to every member.
- Convert each model output using its recorded action temperature.
- Average valid target weights, not raw actions.
- Validate finite, nonnegative weights summing to one.
- Record member hashes and order.
- Backtest the ensemble through the existing shared engine.
Do not select a subset of seeds based on performance.
```

## Prompt for PR 14

```text
Implement a partial-rebalance execution overlay.

Requirements:
- executed_target = current_weights + alpha * (policy_target - current_weights)
- Support alpha values 0.25, 0.50, 0.75, and 1.00.
- Apply after policy target generation and before turnover/cost calculation.
- Preserve raw and executed targets.
- Use evaluation only; do not change reward or retrain PPO.
- Produce return/Sharpe/drawdown/turnover/cost frontier artifacts.
- Add tests for alpha=0, alpha=1, convexity, and weight validity.
Do not add no-trade bands or shrinkage in this PR.
```

## Prompt for PR 15

```text
Implement dynamic-value diagnostics for a frozen target-weight policy.

Required controls:
- ex-ante static average weights from a prior period,
- oracle same-period static average labeled non-deployable,
- one-rebalance-lagged targets,
- fixed circular shifts of 4, 13, and 26 decisions.

Use the existing transaction-cost and drift mechanics.
Write a report comparing active return, Sharpe, drawdown, turnover, and cost.
Do not use the final test split and do not search over many shifts.
```

## Prompt for PR 16

```text
Implement leakage-safe nested walk-forward artifact generation.

Use four folds:
- inner train 2010-2014, inner validation 2015, outer evaluation 2016-2017
- inner train 2010-2016, inner validation 2017, outer evaluation 2018-2019
- inner train 2010-2018, inner validation 2019, outer evaluation 2020-2021
- inner train 2010-2020, inner validation 2021, outer evaluation 2022-2023

For every fold:
- start from unnormalized features,
- fit scaler on inner train only,
- transform inner validation and outer evaluation,
- preserve the feature-spec order,
- write model matrix, scaler, feature spec, quality report, and manifest,
- add explicit scaler leakage tests.
Do not reuse the current 2010-2023 fitted scaler.
```

## Prompt for PR 17

```text
Implement the nested walk-forward training and aggregation workflow.

Requirements:
- Same PPO configuration for every fold.
- Checkpoint selection uses inner validation only.
- Outer evaluation is never used for tuning.
- Run the declared seed set.
- Evaluate each seed, the five-seed mean-weight ensemble, partial-rebalance alphas, and all fixed baselines.
- All primary policy comparisons start equal weight.
- Produce fold-level and aggregate reports.
- Enforce configurable predeclared gates.
Do not access 2024 or the 2025+ test period.
```

## Prompt for PR 18

```text
Implement execution-stress and 2022 attribution reports.

Execution stress:
- one-close delayed execution using daily data,
- asset-specific cost map,
- exact equivalence to scalar cost when all asset costs are equal.

2022 attribution:
- asset contributions,
- asset-class exposures,
- turnover/cost contribution,
- seed disagreement,
- largest active gain/loss periods,
- ensemble comparison.

Do not change or retrain the policy.
```

---

# 8. What the team should return for senior review

After completing PRs 11–18, provide one evidence bundle with these summaries.

## A. Executable candidate table

| Candidate | Return | Sharpe | Max DD | Weekly turnover | Cost drag |
|---|---:|---:|---:|---:|---:|
| Seed 42 | | | | | |
| Five-seed ensemble | | | | | |
| Ensemble alpha 0.25 | | | | | |
| Ensemble alpha 0.50 | | | | | |
| Ensemble alpha 0.75 | | | | | |
| Weekly equal weight | | | | | |
| Inverse volatility | | | | | |

## B. Initialization sensitivity

| Candidate | Initial portfolio | Return | Sharpe | Max DD | Turnover |
|---|---|---:|---:|---:|---:|
| | Equal weight | | | | |
| | Inverse volatility | | | | |
| | SHY | | | | |

## C. Dynamic-value evidence

| Control | Active return vs PPO | Sharpe delta | Interpretation |
|---|---:|---:|---|
| Ex-ante static | | | |
| Oracle static | | | |
| One-week lag | | | |
| Shift 4 | | | |
| Shift 13 | | | |
| Shift 26 | | | |

## D. Walk-forward results

| Fold | PPO ensemble active return | Active Sharpe | DD difference | Turnover | Best fixed baseline |
|---|---:|---:|---:|---:|---|
| 2016–2017 | | | | | |
| 2018–2019 | | | | | |
| 2020–2021 | | | | | |
| 2022–2023 | | | | | |

## E. Execution stress

| Scenario | Return | Sharpe | Max DD | Cost drag |
|---|---:|---:|---:|---:|
| Base 10 bps | | | | |
| One-close delay | | | | |
| Asset-tier costs | | | | |
| 25 bps flat | | | | |
| 50 bps flat | | | | |

## F. PM conclusion

Answer explicitly:

```text
1. Is the policy's value genuinely dynamic?
2. Is it robust to initial allocation?
3. Does the ensemble improve seed stability?
4. Which overlay provides the best economic tradeoff?
5. Does it repeat across historical pseudo-OOS folds?
6. Does it survive execution delay and less favorable costs?
7. What exact candidate, if any, should be frozen for the final test?
```

---

# 9. Go / revise / stop decision framework

## Proceed to pre-test freeze

Proceed only if the evidence supports all of:

```text
- positive median walk-forward active return;
- positive median active Sharpe;
- repeated positive results across the required number of folds;
- no dependence on one month or one fold;
- meaningful value beyond ex-ante static and lagged controls;
- acceptable initialization sensitivity;
- acceptable turnover and cost budget;
- no catastrophic 2022-like behavior;
- one clearly specified executable candidate.
```

## Revise within Phase 3

Revise, without opening the test, if:

```text
- ensemble helps but turnover remains too high;
- one overlay produces a stable frontier but thresholds need PM input;
- initialization sensitivity is material but can be addressed by randomized initial-state training;
- execution delay weakens but does not eliminate the advantage;
- walk-forward evidence is mixed but diagnostically informative.
```

A revision that changes reward, features, or training initialization creates a new candidate version and must repeat the relevant campaign.

## Stop the RL replacement claim

Do not proceed with a replacement claim if:

```text
- static or lagged controls explain most of PPO performance;
- walk-forward active return is not directionally repeatable;
- realistic costs or delay eliminate the advantage;
- seed/initialization instability remains economically large;
- no candidate improves on simple baselines enough to justify the complexity.
```

The project may still conclude that the RL system is a valid research platform, while the current policy is not investment-ready.

---

# 10. Revised Phase 3 definition of done

## Phase 3A — Ready for final test

Phase 3A is complete when:

```text
1. Current campaign provenance is frozen and audited.
2. 2024 is labeled as consumed development/selection data.
3. Initialization sensitivity is measured without leakage.
4. A five-seed ensemble is evaluated as an actual policy.
5. Partial-rebalance turnover controls are evaluated.
6. Dynamic-value controls distinguish timing from static asset mix.
7. Four nested walk-forward folds are built with train-only scalers.
8. Candidate and baselines are evaluated across all folds.
9. Execution delay and asset-specific cost stresses are complete.
10. 2022 seed instability has a root-cause report.
11. One executable candidate passes predeclared PM/ML gates.
12. Final-test criteria and hashes are frozen.
13. The turnover-v2 campaign remains test-free and a new independent holdout
    is approved before Phase 3B.
```

## Phase 3B — Final test complete

Phase 3B is complete when:

```text
1. The frozen candidate is evaluated once on the newly approved independent
   holdout.
2. The command, hashes, and timestamp are recorded.
3. No post-test tuning is presented as independent evaluation.
4. The model card reports pass/fail against the predeclared gates.
5. The team decides whether to proceed to paper trading / serving, revise as a new version, or stop.
```

---

## Final guidance to junior engineers

The remaining Phase 3 work is not about finding a more favorable backtest.

It is about reducing uncertainty around one investment proposition:

> Does the frozen PPO policy contain repeatable, state-dependent allocation value that survives realistic implementation and is large enough to justify using RL instead of a simpler portfolio rule?

Every new test should make that proposition easier to accept or reject. If a proposed experiment cannot change that decision, it probably does not belong in the remaining Phase 3 scope.
