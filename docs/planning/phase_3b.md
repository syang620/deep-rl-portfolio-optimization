# Phase 3B — Forward Holdout Registration, Shadow Execution, and Final Evaluation

**Project:** RL Dynamic Portfolio Allocation

**Phase:** 3B — Forward holdout governance, immutable shadow execution, sealed performance recording, and authorized final evaluation

**Audience:** portfolio managers, quantitative researchers, model-risk reviewers, data/operations engineers, and coding-agent operators

**Phase 3A freeze commit:** `52ed068c404373eee7aea52e84fd6148fc2b6f72`

**Frozen candidate package:** `artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1`

**Frozen manifest SHA-256:** `1480c8de2323fa8555e5fa4e8f9f5adfd39b465a742ac0bc43bff066dcc39edd`

**Status:** Planning approved; execution remains blocked until a new independent forward holdout is formally registered
**Current 2025+ designation:** non-independent because of recorded legacy access; it must not be used as the Phase 3B final holdout

---

## 0. Executive Summary

Phase 3A produced and froze one final candidate:

```text
Policy:
    Five-seed mean-target PPO ensemble

Member seeds:
    7, 42, 101, 202, 999

Execution overlay:
    Partial rebalancing, alpha = 0.25

Initial portfolio:
    Equal weight

Rebalance frequency:
    Every 5 trading days

Primary execution convention:
    Recommendation generated using data through close t
    Executed at close t+1

Primary transaction-cost convention:
    Flat 10 bps per unit of one-way turnover

Execution-stress convention:
    Frozen asset-tier cost schedule, reported as an advisory diagnostic

Turnover:
    0.5 × sum(abs(target weights - live drifted current weights))
```

Phase 3B does not perform more model selection. Its purpose is to answer one final question using a truly independent forward period:

> Does the frozen, slow-moving PPO-derived allocation process deliver positive net active value versus weekly equal weight over a pre-registered 12-month forward holdout, without unacceptable drawdown, turnover, concentration, governance, or operational failures?

Phase 3B must preserve three principles:

1. **The candidate is immutable.**
2. **Performance remains sealed during the holdout.**
3. **The evaluation occurs once, under predeclared rules.**

The implementation is divided into four pull requests:

```text
PR 20 — Holdout registration and governance
PR 21 — Immutable candidate and baseline shadow runner
PR 22 — Operational monitoring and sealed performance ledger
PR 23 — Authorized one-time unseal and PM decision report
```

Before the holdout begins, the system must complete four consecutive weekly operational-certification cycles. Certification output is not part of the performance holdout.

---

# 1. Phase 3B Objective

The Phase 3B objective is:

> Register a new independent 12-month forward holdout, operate the frozen candidate and all baselines from identical point-in-time data, record recommendations and performance in an immutable sealed ledger, monitor operations and exposures without revealing performance, and conduct one authorized pass/mixed/fail evaluation at the end of the registered period.

Phase 3B should establish whether the candidate:

1. creates positive active return versus weekly equal weight;
2. creates positive active Sharpe versus weekly equal weight;
3. remains within approved drawdown, turnover, cost, and concentration limits;
4. survives realistic one-close delayed execution;
5. remains operationally reproducible and auditable;
6. is not dependent on one month, one asset, or one exceptional event;
7. deserves progression to controlled paper deployment or serving.

---

# 2. Non-Goals

Phase 3B does not include:

1. No model retraining.
2. No seed substitution.
3. No alpha tuning.
4. No feature changes.
5. No reward changes.
6. No universe changes.
7. No checkpoint reselection.
8. No live brokerage integration.
9. No discretionary override of model recommendations.
10. No interim performance-based stopping or extension.
11. No reuse of the contaminated 2025+ designation as independent evidence.
12. No tuning against the new holdout after unsealing.

Any change to the candidate creates a new model version and requires a new independent future holdout.

---

# 3. Inherited Frozen Candidate Contract

The following values are inherited from the Phase 3A package and must be verified before every Phase 3B decision run.

## 3.1 Candidate identity

```yaml
candidate:
  model_version: ppo_v1_ensemble5_alpha025_pretest_v1
  policy_type: five_seed_mean_target_ensemble
  member_seed_order: [7, 42, 101, 202, 999]
  partial_rebalance_alpha: 0.25
  initial_portfolio: equal_weight
  rebalance_frequency_trading_days: 5
  portfolio_constraints:
    long_only: true
    fully_invested: true
```

## 3.2 Ensemble rule

At each weekly decision:

```python
member_targets = [
    member_policy.target_weights(observation, info)
    for member_policy in member_policies
]

ensemble_target = np.mean(member_targets, axis=0)

executed_target = (
    current_drifted_weights
    + 0.25 * (ensemble_target - current_drifted_weights)
)
```

Every member receives:

```text
the same current market features
the same live ensemble portfolio weights
the same decision timestamp
```

The system must not simulate five separately drifting member portfolios.

## 3.3 Primary turnover and cost convention

```python
one_way_turnover = 0.5 * np.abs(
    executed_target - current_drifted_weights
).sum()
```

Primary Phase 3B costs use the flat 10 bps convention frozen in the PR 19
candidate package. The frozen asset-tier cost map remains a mandatory execution
stress and advisory diagnostic; it does not replace the primary cost convention.

## 3.4 Primary starting portfolio

The canonical holdout begins from equal weight at NAV 1.0.

The equal-weight allocation is treated as an existing endowment:

```text
No cost is charged to establish the initial portfolio.
Normal turnover and cost apply at the first candidate rebalance.
```

SHY, inverse-volatility, minimum-variance, Markowitz, or actual incumbent holdings may be studied later as deployment-transition scenarios. They do not replace the canonical Phase 3B initialization.

---

# 4. Governance State and Required Approvals

## 4.1 Current governance state

```yaml
governance:
  phase3a_status: complete
  candidate_status: frozen
  phase3b_status: blocked
  existing_test_designation: "2025+"
  existing_test_independent: false
  block_reason: recorded_legacy_access_before_final_candidate_freeze
  replacement_holdout_required: true
  pm_ml_approval_required: true
```

## 4.2 Approval committee

Use three named roles:

```text
1. Portfolio Manager / model user
2. Independent quantitative or model-risk reviewer
3. Data and operations custodian
```

The model developer may provide technical evidence but should not serve as the only independent reviewer.

## 4.3 Approval rules

Require unanimous approval for:

```text
registering the holdout
changing the registered holdout before it starts
unsealing performance
classifying a fatal governance incident
declaring the final Pass / Mixed / Fail result
```

Routine incident classification may use two of three approvers, but the independent reviewer must be one of them.

All approvals must be signed, timestamped, and included in the governance ledger.

---

# 5. New Independent Holdout Definition

## 5.1 Holdout type

The holdout must be a true forward period that begins only after:

1. the frozen candidate has been independently verified;
2. the Phase 3B runner has completed four operational-certification cycles;
3. the committee has registered the exact holdout dates;
4. the holdout configuration and container image have been hashed.

## 5.2 Duration

```text
Fixed duration: 12 calendar months
Early performance stopping: prohibited
Performance-driven extension: prohibited
```

The exact end date should be the last scheduled weekly decision date on or before the registered 12-month anniversary, with the final holding period completed according to the normal five-trading-day convention.

## 5.3 Registration fields

Create a registered holdout object containing:

```yaml
holdout:
  holdout_id: <approved identifier>
  status: registered
  start_decision_date: <YYYY-MM-DD>
  end_decision_date: <YYYY-MM-DD>
  registration_timestamp: <UTC timestamp>
  certification_end_date: <YYYY-MM-DD>
  minimum_valid_rebalance_decisions: 50
  expected_rebalance_frequency_trading_days: 5
  performance_unseal_not_before: <UTC timestamp>
  candidate_model_version: ppo_v1_ensemble5_alpha025_pretest_v1
  candidate_manifest_sha256: 1480c8de2323fa8555e5fa4e8f9f5adfd39b465a742ac0bc43bff066dcc39edd
  container_image_digest: <immutable OCI digest>
  git_commit: <Phase 3B runner commit>
  input_schema_version: <version>
  data_source_contract_version: <version>
```

## 5.4 Append-only snapshot hashing

Because the data do not exist at registration time, do not invent a final holdout snapshot hash.

Instead:

1. hash every daily point-in-time snapshot;
2. link each snapshot to the previous snapshot hash;
3. record recommendation and output hashes;
4. generate a final ledger root hash at the end of the holdout.

Example:

```json
{
  "snapshot_date": "2026-09-15",
  "snapshot_sha256": "...",
  "previous_snapshot_sha256": "...",
  "feature_payload_sha256": "...",
  "recommendation_sha256": "...",
  "container_image_digest": "...",
  "git_commit": "..."
}
```

---

# 6. Operational Certification

Before registering the holdout, run four consecutive weekly certification cycles.

## 6.1 Certification scope

Verify:

```text
point-in-time data ingestion
market-calendar handling
feature generation
five model loads
model and config hash checks
ensemble target construction
alpha-0.25 overlay
baseline generation
recommendation timestamps
one-close execution recording
flat 10 bps primary and asset-tier advisory cost calculation
sealed-ledger write
operational dashboard
alert generation
restart and replay recovery
```

## 6.2 Certification rules

Certification recommendations and returns:

```text
are excluded from Phase 3B performance
may be viewed for operational debugging
must not change the frozen candidate
must not be used to tune acceptance thresholds
```

If certification fails, fix operational code and restart the four-cycle certification count. Candidate weights, features, model files, and decision rules remain unchanged.

## 6.3 Certification artifact

```text
artifacts/phase3b/certification/{certification_id}/
├── certification_manifest.json
├── weekly_run_inventory.csv
├── hash_reconciliation.json
├── operational_incidents.csv
├── restart_recovery_report.md
└── certification_report.md
```

---

# 7. Point-in-Time Decision and Execution Contract

## 7.1 Decision timeline

Recommended schedule:

```text
Close t:
    Official market snapshot becomes eligible for ingestion.

After close t:
    Data and features are processed using information available through close t.

By 10:00 AM ET on t+1:
    Candidate and baseline recommendations are generated and signed.

Close t+1:
    Paper execution price is recorded.

After close t+1:
    Position transition, turnover, and cost are recorded.
    Subsequent returns are written to the sealed ledger.
```

The exact recommendation cutoff may be changed before registration, but it must be frozen.

## 7.2 One-close delay

The canonical candidate does not receive the t→t+1 return on its new target.

Instead:

```text
the pre-decision portfolio earns the next daily return
the new target becomes effective at close t+1
```

This convention must also apply consistently to relevant baselines.

## 7.3 Same input snapshot for all strategies

Candidate and baselines must be generated:

```text
in the same process or orchestration run
from the same point-in-time input snapshot
under the same market calendar
with the same execution timestamp
```

The system must reject mixed-snapshot comparisons.

---

# 8. Frozen Baselines

## 8.1 Primary hurdle

```text
weekly equal weight
```

It begins from equal weight and rebalances every five trading days under the same delayed execution and cost conventions.

## 8.2 Secondary baselines

```text
buy-and-hold equal weight
inverse volatility with full past-only lookback
momentum_63d_top3_equal_weight
SPY-only
SHY-only
```

## 8.3 Momentum rule

```text
At each weekly decision:
1. Use only the prior 63 completed trading days.
2. Calculate cumulative return for all 14 assets.
3. Rank descending.
4. Equal-weight the top 3.
5. Use canonical asset order for deterministic ties.
6. Apply the same delay, drift, and transaction-cost mechanics.
```

## 8.4 Inverse-volatility rule

```text
At each weekly decision:
1. Use the frozen past-only lookback.
2. Compute realized volatility per asset.
3. Apply the existing volatility floor.
4. Set weights proportional to inverse volatility.
5. Normalize to one.
```

The first holdout decision may use context before the registered start date because it is strictly past information.

## 8.5 Baseline immutability

Baseline definitions, lookbacks, tie-breakers, and costs must be hashed at registration.

No baseline may be altered after the holdout begins.

---

# 9. Performance Visibility and Access Control

## 9.1 Information visible during the holdout

Operational and risk reviewers may inspect:

```text
data freshness
schema validation
model and container hashes
candidate target weights
executed weights
asset-level exposures
asset-class exposures
turnover
cost estimate
concentration
recommendation timestamps
missed-decision incidents
system-health metrics
```

## 9.2 Information sealed until evaluation

The following remain inaccessible to PMs, developers, and normal operational users:

```text
candidate NAV
baseline NAV
cumulative return
active return
Sharpe
Sortino
candidate-versus-baseline rankings
monthly return tables
P&L attribution
drawdown comparisons
```

## 9.3 Performance access roles

```text
Automated service account:
    writes encrypted/sealed performance records

Data custodian:
    verifies file counts, hashes, encryption, and append-only status

PM and developers:
    cannot view sealed performance during the holdout
```

Unsealing requires two-person authorization:

```text
Portfolio Manager
+
Independent reviewer
```

The data custodian verifies the ledger root before unsealing.

## 9.4 No proxy leakage

Operational dashboards must not display proxy performance information that allows users to reconstruct results.

For example, do not expose:

```text
daily portfolio P&L
candidate daily return
baseline daily return
active contribution
cumulative transaction-cost-adjusted NAV
```

---

# 10. Acceptance Criteria

The authoritative acceptance contract remains the hashed PR 19 file:

```text
configs/final_candidate_acceptance.yaml
```

PR 20 must verify this exact file against the candidate package. Changing these
hard gates requires a new candidate version and a new pre-test freeze. Phase 3B
operational-validity rules and additional stress thresholds are recorded
separately and cannot silently replace the PR 19 gates.

Separate:

```text
hard gates
mixed-result diagnostics
advisory comparisons
operational validity gates
```

## 10.1 Hard performance gates

```yaml
performance:
  active_return_vs_equal_weight:
    minimum_exclusive: 0.0
    hard_gate: true

  active_sharpe_vs_equal_weight:
    minimum_exclusive: 0.0
    hard_gate: true

  final_nav:
    minimum_exclusive: 0.0
    hard_gate: true
```

Conventional statistical significance is reported but not a hard requirement because one 12-month market path is insufficient for a strong significance claim.

## 10.2 Drawdown gate

```yaml
risk:
  maximum_drawdown_deterioration_pp:
    maximum: 2.0
    hard_gate: true
```

Meaning:

```text
Candidate maximum drawdown may not be more than
2 percentage points worse than weekly equal weight.
```

## 10.3 Turnover gates

```yaml
turnover:
  average_weekly_turnover:
    maximum: 0.05
    hard_gate: true

  p95_weekly_turnover:
    maximum: 0.08
    hard_gate: false
    breach_classification: advisory

  maximum_single_week_turnover:
    maximum: 0.15
    hard_gate: false
    breach_classification: advisory
```

## 10.4 Cost gate

```yaml
costs:
  annualized_transaction_cost_drag:
    maximum: 0.003
    hard_gate: true
```

Maximum permitted annualized cost drag: 0.30% of NAV.

## 10.5 Concentration gates

```yaml
concentration:
  average_max_single_asset_weight:
    maximum: 0.15
    hard_gate: true

  peak_single_asset_weight:
    maximum: 0.25
    hard_gate: true

  average_hhi:
    maximum: 0.15
    hard_gate: false
    breach_classification: advisory

  maximum_single_month_share_of_absolute_active_pnl:
    maximum: 0.35
    hard_gate: true

  maximum_single_asset_share_of_absolute_active_pnl:
    maximum: 0.40
    hard_gate: true
```

## 10.6 Operational validity gates

```yaml
operations:
  minimum_valid_rebalance_decisions: 50
  maximum_missed_rebalance_decisions: 2
  maximum_consecutive_missing_trading_days: 5
  require_hash_reconciliation: true
  require_complete_snapshot_chain: true
  require_no_future_data_incident: true
  require_no_early_performance_access: true
```

A breach of operational validity classifies the holdout as invalid rather than an investment pass or fail.

---

# 11. Advisory and Mixed-Result Diagnostics

## 11.1 P&L concentration

```yaml
pnl_concentration:
  maximum_single_month_share_of_positive_active_pnl: 0.60
  maximum_single_asset_share_of_absolute_active_pnl: 0.50
  breach_classification: mixed
```

If cumulative active return is nonpositive or near zero, report these measures descriptively rather than forcing unstable ratios.

## 11.2 Secondary baseline dominance

The candidate is not required to beat every secondary baseline on every metric.

However:

```text
If two or more transparent secondary baselines
beat the candidate on both total return and Sharpe,
the result cannot be an unqualified Pass.
It is classified Mixed.
```

## 11.3 Exposure alerts

```yaml
alerts:
  equity_exposure_warning: 0.70
  equity_exposure_critical: 0.80
  single_asset_warning: 0.20
  single_asset_critical: 0.35
  rolling_13_week_average_turnover_warning: 0.04
```

Alerts are logged during the holdout but do not change the model.

---

# 12. Final Outcome Classification

## 12.1 Pass

Classify as Pass only when:

```text
all hard performance gates pass
all hard risk gates pass
all hard turnover and cost gates pass
all concentration gates pass
the holdout remains operationally valid
no severe P&L concentration issue exists
the candidate is not broadly dominated by transparent baselines
```

Recommended action:

```text
advance to controlled paper deployment and serving readiness
```

## 12.2 Mixed

Classify as Mixed when the primary return and Sharpe gates pass, but one or more of the following occurs:

```text
P&L is concentrated in one month or asset
multiple secondary baselines dominate
critical exposure alerts occur without hard-gate failure
active return is economically trivial
an important regime produces concerning behavior
nonfatal operational incidents reduce confidence
```

Recommended action:

```text
keep the candidate frozen
continue forward observation if approved
do not tune against the holdout
```

## 12.3 Fail

Classify as Fail when any investment hard gate fails:

```text
active return <= 0
active Sharpe <= 0
drawdown deterioration > 2 percentage points
average weekly turnover > 4%
annual cost drag > 0.50%
peak single-asset weight > 35%
```

Recommended action:

```text
close the candidate version
do not modify it and reuse the same holdout
any new candidate requires future independent data
```

## 12.4 Invalid

Classify the holdout as Invalid rather than Pass/Fail if a fatal governance or data incident occurs.

Examples:

```text
future-data leakage
candidate hash mismatch
early performance unsealing
ledger mutation
unapproved universe change
too many missed decisions
```

---

# 13. Data Corrections and Missing Data

## 13.1 Append-only correction policy

Never overwrite point-in-time decision inputs.

Vendor corrections must be stored as a new version containing:

```text
original value
corrected value
vendor/source
correction timestamp
reason
affected dates
affected decisions
```

The decision audit always uses the original point-in-time values.

The final report may include corrected official performance under the frozen correction policy.

## 13.2 Material correction thresholds

Require committee review when a correction changes:

```text
portfolio daily return by more than 10 bps
or
cumulative active return by more than 25 bps
```

If material, report both:

```text
point-in-time performance
corrected-data performance
```

## 13.3 Missing input policy

If required data are unavailable by the frozen decision cutoff:

```text
do not use future data
do not silently backfill
do not generate a new target
hold the current portfolio
apply the same no-trade treatment to applicable baselines
record a missed-decision incident
```

More than two missed scheduled rebalances invalidates the holdout.

---

# 14. Universe and Instrument Events

## 14.1 Temporary issue

For a temporary data or trading issue:

```text
apply the missing-input policy
hold the current portfolio
record the incident
```

## 14.2 Prolonged unavailability

If an asset is unavailable for more than five consecutive trading days:

```text
pause the Phase 3B validity classification
escalate to the approval committee
```

## 14.3 Permanent universe change

A merger, delisting, closure, or permanent instrument replacement changes the model’s fixed input/action dimensions.

Therefore:

```text
do not remove the asset
do not renormalize the remaining outputs
do not redirect weight to SHY
```

Unless such a fallback was frozen before holdout registration, a permanent universe change invalidates the final-performance holdout.

---

# 15. Incident Classification

## 15.1 Fatal incidents

Immediately invalidate the holdout for final-evidence purposes:

```text
candidate model hash mismatch
wrong seed order
wrong alpha
feature-spec mismatch
wrong container digest
future-data access
recommendation after cutoff
manual recommendation alteration
candidate and baselines using different snapshots
early access to sealed performance
ledger deletion or unverifiable mutation
unapproved universe change
candidate retraining or replacement
more than two missed scheduled rebalances
prolonged required-data failure
```

## 15.2 Nonfatal incidents

Log but do not invalidate automatically:

```text
delayed operational dashboard refresh
process retry producing identical output hash
nonmaterial vendor correction
one isolated missed rebalance
transient infrastructure outage with deterministic recovery
```

Every incident must contain:

```text
incident_id
timestamp
severity
affected decisions
root cause
remediation
hash evidence
approver disposition
```

---

# 16. Phase 3B Architecture

Recommended package additions:

```text
src/portfolio_rl/phase3b/
├── __init__.py
├── governance.py
├── holdout_registry.py
├── certification.py
├── frozen_candidate_loader.py
├── shadow_runner.py
├── baseline_runner.py
├── execution.py
├── snapshot_chain.py
├── sealed_ledger.py
├── operational_metrics.py
├── incidents.py
├── unseal.py
└── final_report.py
```

Thin CLI scripts:

```text
scripts/register_forward_holdout.py
scripts/run_phase3b_certification.py
scripts/run_phase3b_decision.py
scripts/verify_phase3b_ledger.py
scripts/show_phase3b_operations.py
scripts/unseal_phase3b_performance.py
scripts/build_phase3b_final_report.py
```

Configs:

```text
configs/phase3b/
├── holdout_registration.yaml
├── operations.yaml
├── execution.yaml
└── access_control.yaml

Authoritative candidate acceptance config:
configs/final_candidate_acceptance.yaml
```

---

# 17. PR 20 — Holdout Registration and Governance

## 17.1 Goal

Create the governance layer that can:

```text
verify the Phase 3A package
register certification
register one future holdout
freeze dates and rules
record approvals
block execution when governance is incomplete
```

## 17.2 Files

```text
src/portfolio_rl/phase3b/governance.py
src/portfolio_rl/phase3b/holdout_registry.py
src/portfolio_rl/phase3b/incidents.py
scripts/register_forward_holdout.py
configs/phase3b/holdout_registration.yaml
configs/phase3b/access_control.yaml
tests/unit/phase3b/test_governance.py
tests/unit/phase3b/test_holdout_registry.py
tests/integration/phase3b/test_holdout_registration.py
```

## 17.3 Required behavior

- Verify Phase 3A candidate manifest and all member hashes.
- Reject the existing 2025+ designation.
- Require successful operational certification.
- Require exact start and end dates.
- Require three named approvers.
- Freeze all Phase 3B configs.
- Refuse registration when working tree or container identity is unresolved.
- Write an immutable holdout-registration package.
- Refuse duplicate or overlapping registrations.

## 17.4 Artifacts

```text
artifacts/phase3b/registration/{holdout_id}/
├── holdout_registration.json
├── approval_record.json
├── candidate_verification.json
├── config_hashes.json
├── container_identity.json
├── access_policy.json
├── incident_policy.json
└── registration_manifest.json
```

## 17.5 Acceptance criteria

```text
- Existing 2025+ designation is rejected.
- Candidate manifest verifies.
- Registration cannot precede certification.
- Holdout dates span exactly the approved fixed horizon.
- Approver roles are complete.
- Performance unseal date is fixed.
- Output path is immutable.
- Registration manifest verifies in a fresh process.
```

---

# 18. PR 21 — Immutable Shadow Runner

## 18.1 Goal

Generate candidate and baseline recommendations from one point-in-time snapshot under the frozen execution contract.

**Implementation status:** code complete in PR 21; operational use remains
blocked. `configs/phase3b/execution.yaml` intentionally remains `draft` until
the normalized serving scaler and recommendation-signing identity receive
independent approval and a replacement holdout is registered after
certification. PR 21 does not register a holdout, begin certification, access
the contaminated 2025+ designation, or compute performance.

## 18.2 Files

```text
src/portfolio_rl/phase3b/frozen_candidate_loader.py
src/portfolio_rl/phase3b/shadow_runner.py
src/portfolio_rl/phase3b/baseline_runner.py
src/portfolio_rl/phase3b/execution.py
src/portfolio_rl/phase3b/snapshot_chain.py
scripts/run_phase3b_decision.py
configs/phase3b/execution.yaml
tests/unit/phase3b/test_frozen_candidate_loader.py
tests/unit/phase3b/test_snapshot_chain.py
tests/integration/phase3b/test_shadow_runner.py
```

## 18.3 Required behavior

1. Load the exact Phase 3A frozen candidate.
2. Verify all model, feature, config, and candidate hashes.
3. Load one point-in-time market snapshot.
4. Build the observation using live current weights.
5. Generate five member targets.
6. Average member targets.
7. Apply alpha 0.25.
8. Generate all baselines from the same snapshot.
9. Write signed recommendation files before the cutoff.
10. Record the delayed execution instruction.
11. Append the snapshot and recommendation hashes to the chain.
12. Refuse reruns that produce different outputs for the same snapshot.

## 18.4 Decision artifact

```text
artifacts/phase3b/holdouts/{holdout_id}/decisions/{decision_id}/
├── input_snapshot_manifest.json
├── input_live_state_manifest.json
├── feature_payload.parquet
├── trailing_log_returns.parquet
├── member_targets.parquet
├── ensemble_target.parquet
├── executed_target.parquet
├── baseline_targets.parquet
├── current_weights.parquet
├── execution_instructions.json
├── recommendation_manifest.json
├── recommendation_manifest.sig
└── incident_record.json
```

The feature payload contains exactly 302 normalized market features. The same
live candidate weights append the remaining 14 observation values passed to
all five frozen members. The snapshot also carries exactly 63 strictly past
return rows for inverse-volatility and momentum baselines. Each recommendation
manifest records physical and logical hashes, the approved candidate and
baseline identities, container identity, execution-config hash, prior chain
hash, and an SSH signature. Turnover and costs remain pending until close t+1,
when live drifted weights exist; the shadow runner does not fabricate those
quantities at decision time.

## 18.5 Acceptance criteria

```text
- All strategies use one snapshot hash.
- Ensemble mean exactly reconciles.
- Alpha overlay exactly reconciles.
- Weights are valid and sum to one.
- Recommendation is generated before cutoff.
- No future price is read.
- Same-snapshot retry is byte/logically identical.
- Candidate and baseline hashes are recorded.
```

---

# 19. PR 22 — Operational Monitoring and Sealed Ledger

## 19.1 Goal

Support continuous operational oversight without revealing investment performance.

## 19.2 Files

```text
src/portfolio_rl/phase3b/sealed_ledger.py
src/portfolio_rl/phase3b/operational_metrics.py
src/portfolio_rl/phase3b/certification.py
scripts/run_phase3b_certification.py
scripts/verify_phase3b_ledger.py
scripts/show_phase3b_operations.py
configs/phase3b/operations.yaml
tests/unit/phase3b/test_sealed_ledger.py
tests/unit/phase3b/test_operational_metrics.py
tests/integration/phase3b/test_certification.py
```

## 19.3 Visible operational metrics

```text
run success/failure
data freshness
feature completeness
model load status
hash reconciliation
recommendation timestamp
asset weights
asset-class exposures
turnover
cost estimate
concentration
missed decisions
incident counts
```

## 19.4 Sealed metrics

```text
portfolio returns
baseline returns
NAV
active return
Sharpe
drawdown comparison
monthly performance
P&L attribution
rankings
```

## 19.5 Ledger security contract

The sealed ledger should:

```text
be append-only
use authenticated encryption or equivalent access control
record every write hash
link to the prior ledger hash
refuse deletion or replacement
record authorized access attempts
support independent root verification
```

## 19.6 Certification command

```bash
python scripts/run_phase3b_certification.py   --candidate-package artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1   --weeks 4   --output-root artifacts/phase3b/certification
```

## 19.7 Acceptance criteria

```text
- Four consecutive certification decisions pass.
- Operations dashboard exposes no sealed metric.
- Ledger verification succeeds.
- Unauthorized unseal attempts fail.
- Restart recovery is deterministic.
- Incident policy is exercised in tests.
```

---

# 20. PR 23 — Authorized Unseal and Final Decision Report

## 20.1 Goal

After the registered holdout ends, authorize one unseal, calculate final metrics, apply frozen gates, and write the PM decision report.

## 20.2 Files

```text
src/portfolio_rl/phase3b/unseal.py
src/portfolio_rl/phase3b/final_report.py
scripts/unseal_phase3b_performance.py
scripts/build_phase3b_final_report.py
tests/unit/phase3b/test_unseal.py
tests/unit/phase3b/test_final_classification.py
tests/integration/phase3b/test_final_report.py
```

## 20.3 Preconditions

- Registered end date has passed.
- Minimum decision count is met.
- Ledger root verifies.
- Candidate and config hashes match registration.
- No fatal incident exists.
- Two-person unseal approval is present.
- Performance has not previously been unsealed.

## 20.4 Final metrics

Calculate:

```text
total return
CAGR
annualized volatility
Sharpe
Sortino
max drawdown
Calmar
average and annualized turnover
p95 and maximum weekly turnover
transaction-cost drag
active return
active Sharpe
information ratio
active drawdown difference
asset and month P&L concentration
baseline rankings
regime/event attribution
```

## 20.5 Final artifacts

```text
artifacts/phase3b/holdouts/{holdout_id}/final/
├── unseal_authorization.json
├── ledger_root_verification.json
├── candidate_metrics.json
├── baseline_metrics.json
├── active_metrics.json
├── risk_diagnostics.json
├── concentration_diagnostics.json
├── gate_results.json
├── final_classification.json
├── final_report.md
└── final_manifest.json
```

## 20.6 One-time rule

The unseal operation must:

```text
refuse to run twice
record the exact timestamp and approvers
record the code and container identity
preserve the original sealed ledger
```

---

# 21. Test Plan

## 21.1 Governance tests

```text
test_rejects_existing_2025_plus_designation
test_registration_requires_certification
test_registration_requires_three_approvers
test_registration_requires_exact_candidate_hash
test_registration_rejects_overlapping_holdout
test_registration_output_is_immutable
```

## 21.2 Candidate-loader tests

```text
test_requires_all_five_models
test_requires_exact_seed_order
test_requires_alpha_0_25
test_rejects_model_hash_mutation
test_rejects_feature_hash_mutation
test_rejects_wrong_container_digest
```

## 21.3 Point-in-time tests

```text
test_recommendation_uses_data_through_close_t_only
test_recommendation_cannot_read_t_plus_1_price
test_candidate_and_baselines_share_snapshot
test_recommendation_timestamp_precedes_execution_close
```

## 21.4 Execution tests

```text
test_one_close_delay_preserves_old_portfolio_for_first_day
test_asset_tier_costs_reconcile
test_flat_10_bps_secondary_result_reconciles
test_turnover_uses_half_l1
```

## 21.5 Ledger tests

```text
test_snapshot_hash_chain_is_append_only
test_missing_prior_hash_fails
test_ledger_mutation_is_detected
test_unauthorized_unseal_fails
test_operational_dashboard_excludes_performance
```

## 21.6 Incident tests

```text
test_missing_input_holds_current_portfolio
test_one_missed_decision_is_logged
test_three_missed_decisions_invalidate_holdout
test_future_data_incident_is_fatal
test_permanent_universe_change_invalidates_holdout
```

## 21.7 Final classification tests

```text
test_pass_requires_all_hard_gates
test_mixed_on_pnl_concentration
test_mixed_on_multiple_secondary_baseline_dominance
test_fail_on_nonpositive_active_return
test_fail_on_nonpositive_active_sharpe
test_fail_on_drawdown_deterioration
test_invalid_on_governance_breach
test_unseal_runs_once
```

---

# 22. Artifact Layout

```text
artifacts/phase3b/
├── certification/
│   └── {certification_id}/
├── registration/
│   └── {holdout_id}/
└── holdouts/
    └── {holdout_id}/
        ├── snapshots/
        ├── decisions/
        ├── executions/
        ├── operational/
        ├── incidents/
        ├── sealed_ledger/
        ├── chain_manifest.json
        └── final/
```

Generated performance artifacts must remain inaccessible until authorized unseal.

---

# 23. Recommended Commands

## 23.1 Verify the Phase 3A candidate

```bash
python scripts/freeze_final_candidate.py \
  --verify artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1/frozen_candidate.json
```

## 23.2 Run certification

```bash
python scripts/run_phase3b_certification.py   --candidate-package artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1   --weeks 4
```

## 23.3 Prepare and sign the registration challenge

```bash
python scripts/register_forward_holdout.py prepare \
  --config configs/phase3b/holdout_registration.yaml \
  --challenge-output artifacts/phase3b/registration_challenge.json

ssh-keygen -Y sign \
  -f <approver-private-key> \
  -n portfolio-rl-phase3b-registration-v1 \
  artifacts/phase3b/registration_challenge.json
```

Each of the three named approvers signs the same challenge with their own key and
places the detached signature in the approved role-named signature directory.

## 23.4 Register and independently verify the holdout

```bash
python scripts/register_forward_holdout.py register \
  --challenge artifacts/phase3b/registration_challenge.json \
  --approvals-dir artifacts/phase3b/approvals \
  --output-root artifacts/phase3b/registration

python scripts/register_forward_holdout.py verify \
  --registration-dir artifacts/phase3b/registration/<holdout_id>
```

## 23.5 Run a weekly decision

```bash
python scripts/run_phase3b_decision.py   --holdout-id <holdout_id>   --decision-date <YYYY-MM-DD>
```

## 23.6 Verify the ledger

```bash
python scripts/verify_phase3b_ledger.py   --holdout-id <holdout_id>
```

## 23.7 Authorized unseal

```bash
python scripts/unseal_phase3b_performance.py   --holdout-id <holdout_id>   --approval-record <path>   --confirm-one-time-unseal
```

---

# 24. Coding-Agent Prompt Guide

## PR 20 prompt

```text
Implement Phase 3B holdout registration and governance.

Requirements:
- Verify the Phase 3A frozen candidate package and manifest.
- Reject the existing 2025+ designation as non-independent.
- Require successful four-week operational certification.
- Register one exact 12-month forward holdout.
- Require PM, independent reviewer, and data/operations approvals.
- Freeze holdout dates, candidate hash, configs, container digest, access policy, and unseal date.
- Write immutable registration artifacts.
- Do not run candidate performance.
```

## PR 21 prompt

```text
Implement the immutable Phase 3B shadow runner.

Requirements:
- Load the exact five-seed alpha-0.25 frozen candidate.
- Verify all hashes on every decision.
- Generate candidate and frozen baselines from the same point-in-time snapshot.
- Use one live ensemble portfolio and common current weights for all members.
- Apply one-close delayed execution with flat 10 bps as primary cost.
- Retain the frozen asset-tier map as the mandatory advisory stress.
- Write signed recommendation and snapshot-chain artifacts.
- Do not expose or calculate cumulative performance in user-visible outputs.
```

## PR 22 prompt

```text
Implement operational monitoring and a sealed performance ledger.

Requirements:
- Expose data freshness, hashes, recommendations, exposures, turnover, concentration, and incidents.
- Seal candidate and baseline returns, NAV, active return, Sharpe, drawdown, rankings, and P&L attribution.
- Use an append-only linked hash chain.
- Require authorization for unseal.
- Support deterministic restart recovery.
- Implement four-week operational certification.
```

## PR 23 prompt

```text
Implement one-time authorized Phase 3B unseal and final reporting.

Requirements:
- Verify the holdout has ended and minimum decision count is satisfied.
- Verify the complete ledger root and all candidate/config hashes.
- Require PM plus independent-review approval.
- Refuse a second unseal.
- Compute candidate, baseline, active, risk, turnover, cost, concentration, and attribution metrics.
- Apply the frozen hard gates and Mixed diagnostics.
- Produce Pass, Mixed, Fail, or Invalid with a PM decision report.
- Do not modify the candidate.
```

---

# 25. Failure Modes

## Failure mode 1 — Interim performance leakage

Symptom:

```text
PM or developers view candidate NAV before the registered date.
```

Response:

```text
invalidate the holdout
```

## Failure mode 2 — Operational dashboard reveals proxy P&L

Symptom:

```text
visible daily returns or cumulative cost-adjusted NAV allow reconstruction.
```

Response:

```text
treat as early performance access
```

## Failure mode 3 — Baseline drift

Symptom:

```text
baseline definitions change during the holdout.
```

Response:

```text
fatal governance incident
```

## Failure mode 4 — Snapshot mismatch

Symptom:

```text
candidate and baselines use different input snapshots.
```

Response:

```text
fatal incident for that decision;
committee determines whether holdout remains valid
```

## Failure mode 5 — Candidate mutation

Symptom:

```text
model, alpha, seed order, feature spec, or container hash changes.
```

Response:

```text
invalidate the holdout
```

## Failure mode 6 — Performance-driven extension

Symptom:

```text
team extends the holdout because current performance appears weak.
```

Response:

```text
prohibited; fixed 12-month horizon remains binding
```

---

# 26. Definition of Done

## Phase 3B planning complete

```text
1. phase_3b.md is approved.
2. Candidate and baseline contracts are fixed.
3. Acceptance thresholds are approved.
4. Governance roles and access rules are approved.
5. Operational certification protocol is approved.
6. Holdout registration schema is approved.
```

## Phase 3B execution ready

```text
1. PRs 20–22 are implemented and tested.
2. One immutable container runs candidate and baselines.
3. Four weekly certification cycles pass.
4. Committee registers the exact holdout.
5. Performance ledger access is sealed.
```

## Phase 3B complete

```text
1. Twelve-month holdout finishes without fatal invalidation.
2. Minimum valid decision count is met.
3. Ledger root and all hashes verify.
4. Authorized one-time unseal occurs.
5. Frozen gates produce Pass, Mixed, Fail, or Invalid.
6. PM decision report is approved.
7. No post-holdout tuning is represented as independent evidence.
```

---

# 27. Final Guidance

The primary Phase 3B discipline is:

> Observe operations continuously, but observe performance only once.

The candidate has already been selected. Phase 3B is not an opportunity to improve it. It is an opportunity to determine whether the exact frozen process creates value in real forward time.

A successful result would justify progression to controlled paper deployment and serving readiness.

A failed result would not invalidate the engineering platform. It would show that the frozen investment candidate did not earn the right to proceed.
