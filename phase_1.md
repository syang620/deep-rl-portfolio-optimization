# Phase 1 Data Pipeline and Data Preparation Guide

Project: Deep RL for Constrained Resource Allocation (Dynamic Portfolio Optimization)  
Audience: junior machine learning engineers and coding-agent-assisted implementers  
Version: v3.0  
Date: 2026-04-25

---

## 0. What Phase 1 Must Deliver

Phase 1 is not "download ETF data." Phase 1 is to create a trustworthy, reproducible, leakage-safe market simulator dataset that can be consumed by the custom Gymnasium environment, the RL training scripts, the backtest engine, and the FastAPI serving layer.

By the end of Phase 1, the repo should produce:

```text
data/raw/prices_daily.parquet
data/raw/macro_daily.parquet

data/processed/features_daily.parquet
data/processed/global_features_daily.parquet
data/processed/model_matrix_daily.parquet

data/duckdb/portfolio.duckdb

artifacts/scalers/feature_scaler_v1.pkl
artifacts/feature_specs/feature_spec_v1.json
artifacts/reports/data_quality_report_v1.json
```

The milestone is complete only when the following is true:

```text
1. The pipeline is config-driven.
2. No ETF tickers, dates, feature windows, or costs are hardcoded in source files.
3. Raw, interim, and processed data are reproducible.
4. The final model matrix has no NaNs or infinite values.
5. The train/validation/test split is chronological.
6. Normalization is fit on train only.
7. The feature spec defines exact feature ordering.
8. The environment can later consume daily features and aggregate returns over weekly rebalance windows.
9. Unit and integration tests pass.
```

---

## 1. Repo Structure for Phase 1

Use the broader project layout below. Phase 1 mainly touches `configs/`, `src/portfolio_rl/data/`, `src/portfolio_rl/features/`, `scripts/`, `tests/`, `data/`, and `artifacts/`.

```text
rl-dynamic-portfolio-allocation/
├── configs/
│   ├── universe.yaml
│   ├── data.yaml
│   ├── features.yaml
│   └── env.yaml
├── data/                         # gitignored except README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── duckdb/
├── artifacts/                    # gitignored
│   ├── scalers/
│   ├── feature_specs/
│   └── reports/
├── scripts/
│   ├── run_etl.py
│   └── build_features.py
├── src/
│   └── portfolio_rl/
│       ├── config/
│       │   ├── schemas.py
│       │   └── loader.py
│       ├── data/
│       │   ├── sources.py
│       │   ├── yfinance_client.py
│       │   ├── fred_client.py
│       │   ├── etl.py
│       │   ├── validation.py
│       │   ├── storage.py
│       │   └── splits.py
│       └── features/
│           ├── returns.py
│           ├── technicals.py
│           ├── macro.py
│           ├── normalization.py
│           ├── builder.py
│           └── feature_spec.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 2. Config Files

### 2.1 `configs/universe.yaml`

Start with a static, liquid, diversified ETF universe. Use SHY as the cash-like proxy. Do not add a separate cash variable in v1.

```yaml
universe_name: liquid_global_etf_v1

assets:
  - ticker: SPY
    asset_class: us_large_cap_equity
  - ticker: QQQ
    asset_class: us_growth_equity
  - ticker: IWM
    asset_class: us_small_cap_equity
  - ticker: EFA
    asset_class: developed_ex_us_equity
  - ticker: EEM
    asset_class: emerging_market_equity
  - ticker: TLT
    asset_class: long_treasury
  - ticker: IEF
    asset_class: intermediate_treasury
  - ticker: SHY
    asset_class: short_treasury_cash_proxy
  - ticker: LQD
    asset_class: investment_grade_credit
  - ticker: HYG
    asset_class: high_yield_credit
  - ticker: GLD
    asset_class: gold
  - ticker: DBC
    asset_class: broad_commodities
  - ticker: VNQ
    asset_class: real_estate
  - ticker: XLU
    asset_class: defensive_equity_sector
```

### 2.2 `configs/data.yaml`

Use pre-2010 raw data for feature warm-up. The model period starts in 2010. Training includes the 2022 regime change.

```yaml
raw_start_date: "2007-01-01"
model_start_date: "2010-01-01"

train_start_date: "2010-01-01"
train_end_date: "2023-12-31"

validation_start_date: "2024-01-01"
validation_end_date: "2024-12-31"

test_start_date: "2025-01-01"
test_end_date: null

market_data_source: yfinance
macro_data_source: fred

storage:
  duckdb_path: data/duckdb/portfolio.duckdb
  raw_parquet_dir: data/raw
  interim_parquet_dir: data/interim
  processed_parquet_dir: data/processed
```

### 2.3 `configs/features.yaml`

```yaml
feature_version: v1

return_windows: [1, 5, 21, 63, 126, 252]
volatility_windows: [21, 63]
drawdown_windows: [63]
rsi_windows: [14]
price_z_windows: [20, 50]
correlation_windows: [63]

winsorization:
  enabled: true
  lower_quantile: 0.005
  upper_quantile: 0.995

normalization:
  method: standard_scaler
  fit_split: train
```

### 2.4 `configs/env.yaml`

This belongs to the environment phase, but Phase 1 must understand it because the data will be consumed in weekly five-trading-day windows.

```yaml
# env.yaml
# --- Time & Stepping Mechanics ---
rebalance_frequency_trading_days: 5
episode_length_trading_days: 260
max_episode_steps: 52  # derived exactly as (episode_length_trading_days / rebalance_frequency_trading_days)

# --- Action & Frictions ---
action_transform: softmax
action_temperature: 5.0
initial_weights: equal_weight
transaction_cost_bps: 10.0
```

---

## 3. What ETF Data We Need

For each ETF, ingest daily end-of-day data:

```text
date
ticker
open
high
low
close
adj_close
volume
dividends
stock_splits
source
downloaded_at
```

Use `adj_close` for returns because it approximates total-return behavior after splits and distributions. Use OHLCV for features such as range, ATR, liquidity, and volume z-scores.

Do not use leveraged ETFs, inverse ETFs, illiquid niche ETFs, or frequent universe changes in v1.

---

## 4. Macro and Risk Data

Use daily observable variables only. Do not use CPI, payrolls, GDP, or monthly data in v1 because these introduce release-date alignment problems.

Recommended global inputs:

```text
VIX or ^VIX
3-month Treasury yield
2-year Treasury yield
10-year Treasury yield
10Y - 2Y yield spread
High-yield credit spread
Investment-grade credit spread
Optional: oil or commodity proxy
Optional: dollar strength proxy
```

Store macro data in long format:

```text
date
series_id
value
source
downloaded_at
```

Forward-fill daily macro values only after the values are known. Never backfill.

---

## 5. Pipeline Stages

The pipeline should be deterministic and config-driven.

```text
Stage 1: Load config
Stage 2: Download ETF OHLCV
Stage 3: Download macro/risk data
Stage 4: Validate raw data
Stage 5: Align to common trading calendar
Stage 6: Engineer features
Stage 7: Normalize features
Stage 8: Create train/validation/test splits
Stage 9: Save DuckDB tables and Parquet snapshots
Stage 10: Emit data quality report
```

Recommended commands:

```bash
uv run python scripts/run_etl.py --config configs/data.yaml --universe configs/universe.yaml

uv run python scripts/build_features.py \
  --data-config configs/data.yaml \
  --feature-config configs/features.yaml \
  --universe configs/universe.yaml
```

---

## 6. Storage Design

Use both Parquet and DuckDB.

Parquet is useful for immutable snapshots. DuckDB is useful for local SQL queries and reproducible debugging.

Recommended DuckDB tables:

```sql
prices_daily(
    date DATE,
    ticker VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume DOUBLE,
    dividends DOUBLE,
    stock_splits DOUBLE,
    source VARCHAR,
    downloaded_at TIMESTAMP
);

macro_daily(
    date DATE,
    series_id VARCHAR,
    value DOUBLE,
    source VARCHAR,
    downloaded_at TIMESTAMP
);

features_daily(
    date DATE,
    ticker VARCHAR,
    feature_version VARCHAR,
    ret_1d DOUBLE,
    ret_5d DOUBLE,
    ret_21d DOUBLE,
    ret_63d DOUBLE,
    ret_126d DOUBLE,
    vol_21d DOUBLE,
    vol_63d DOUBLE,
    downside_vol_63d DOUBLE,
    drawdown_63d DOUBLE,
    rsi_14 DOUBLE,
    macd_12_26 DOUBLE,
    macd_signal_9 DOUBLE,
    price_z_20d DOUBLE,
    price_z_50d DOUBLE,
    volume_z_21d DOUBLE,
    rank_ret_21d DOUBLE,
    rank_ret_63d DOUBLE,
    rank_vol_21d DOUBLE,
    corr_to_spy_63d DOUBLE,
    beta_to_spy_63d DOUBLE
);

global_features_daily(
    date DATE,
    feature_version VARCHAR,
    vix_z_21d DOUBLE,
    vix_z_63d DOUBLE,
    dgs2_change_5d DOUBLE,
    dgs10_change_5d DOUBLE,
    yield_curve_10y_2y DOUBLE,
    credit_spread_z_63d DOUBLE,
    spy_vol_21d DOUBLE,
    spy_drawdown_63d DOUBLE
);

model_matrix_daily(
    date DATE,
    split VARCHAR,
    feature_version VARCHAR,
    -- flattened normalized features
    -- one-day asset returns for environment compounding
);
```

---

## 7. Feature Engineering Plan

The model needs features that answer five questions:

```text
1. What is each asset doing?
2. How risky is each asset right now?
3. Is the asset trending or mean-reverting?
4. How liquid/tradable is the asset?
5. What macro/risk regime are we in?
```

### 7.1 Per-asset v1 feature set

```text
ret_1d
ret_5d
ret_21d
ret_63d
ret_126d
vol_21d
vol_63d
downside_vol_63d
drawdown_63d
rsi_14
macd_12_26
macd_signal_9
price_z_20d
price_z_50d
volume_z_21d
rank_ret_21d
rank_ret_63d
rank_vol_21d
corr_to_spy_63d
beta_to_spy_63d
```

### 7.2 Global v1 feature set

```text
vix_z_21d
vix_z_63d
dgs2_change_5d
dgs10_change_5d
yield_curve_10y_2y
credit_spread_z_63d
spy_vol_21d
spy_drawdown_63d
```

### 7.3 Example formulas

```python
ret_1d = np.log(adj_close / adj_close.shift(1))
ret_21d = np.log(adj_close / adj_close.shift(21))
vol_21d = ret_1d.rolling(21).std() * np.sqrt(252)

price_z_20d = (
    adj_close - adj_close.rolling(20).mean()
) / adj_close.rolling(20).std()

dollar_volume = close * volume
volume_z_21d = (
    np.log1p(dollar_volume) - np.log1p(dollar_volume).rolling(21).mean()
) / np.log1p(dollar_volume).rolling(21).std()

corr_to_spy_63d = ret_asset.rolling(63).corr(ret_spy)
beta_to_spy_63d = (
    ret_asset.rolling(63).cov(ret_spy)
    / ret_spy.rolling(63).var()
)
```

---

## 8. Time Alignment and Leakage Prevention

Define timing as:

```text
At close of day t:
    observe features computed using data up to and including close t
    choose target weights for the next holding period

From close t to close t+5:
    portfolio earns compounded returns over the five trading days

At the next rebalance:
    calculate reward, apply transaction costs, update drifted weights, and emit next observation
```

Hard rules:

```text
observation[t] must never include return[t+1]
reward[t] may use future returns only inside the environment transition
normalization must be fit on train only
monthly macro is excluded in v1
never backfill missing values
```

The data pipeline should save one-day asset returns by date. The environment will compound those daily returns across each five-day rebalance window.

---

## 9. Normalization

Do not normalize using the full dataset.

Correct process:

```text
1. Build raw stationary features.
2. Assign chronological split labels.
3. Fit winsorization thresholds on train only.
4. Fit scaler on train only.
5. Transform train, validation, and test using the train-fitted artifacts.
6. Save the scaler and thresholds.
```

Incorrect process:

```text
Fit a scaler on 2010-2026 and then train on 2010-2023.
```

That leaks future information.

---

## 10. Data Validation Rules

### 10.1 Raw price checks

```text
date is unique per ticker
date is sorted ascending
open > 0
high > 0
low > 0
close > 0
adj_close > 0
volume >= 0
high >= low
```

### 10.2 Return checks

Flag suspicious rows but do not automatically delete them:

```text
abs(ret_1d) > 0.25
```

### 10.3 Final matrix checks

```text
no NaNs
no infinite values
all expected tickers present
all expected features present
feature columns match feature_spec.json
max(train_date) < min(validation_date)
max(validation_date) < min(test_date)
no target or future-return columns in observation features
```

---

## 11. Implementation Tasks

### Task 1: Config schemas and loader

Files:

```text
src/portfolio_rl/config/schemas.py
src/portfolio_rl/config/loader.py
```

Acceptance criteria:

```text
configs load into typed objects
invalid config fields fail fast
dates parse correctly
tickers are deduplicated and stable
```

### Task 2: ETF data client

File:

```text
src/portfolio_rl/data/yfinance_client.py
```

Acceptance criteria:

```text
downloads configured tickers
returns long-format DataFrame
standardizes column names
includes source and downloaded_at
unit tests pass using fixtures/mocks
```

### Task 3: Macro data client

File:

```text
src/portfolio_rl/data/fred_client.py
```

Acceptance criteria:

```text
downloads configured series
returns long-format DataFrame
numeric values are parsed safely
date coverage is validated
monthly macro is not included in v1 config
```

### Task 4: Storage layer

File:

```text
src/portfolio_rl/data/storage.py
```

Acceptance criteria:

```text
writes Parquet files
writes DuckDB tables
reads tables back
round-trip row counts match
```

### Task 5: Data validation

File:

```text
src/portfolio_rl/data/validation.py
```

Acceptance criteria:

```text
schema checks fail loudly
outliers are logged
final model matrix has zero NaNs and zero inf values
```

### Task 6: Feature builder

Files:

```text
src/portfolio_rl/features/returns.py
src/portfolio_rl/features/technicals.py
src/portfolio_rl/features/macro.py
src/portfolio_rl/features/builder.py
```

Acceptance criteria:

```text
features are calculated per ticker without leakage
warm-up rows are dropped
feature names match features.yaml
per-asset features and global features join cleanly by date
```

### Task 7: Split assignment

File:

```text
src/portfolio_rl/data/splits.py
```

Acceptance criteria:

```text
train = 2010-01-01 through 2023-12-31
validation = 2024-01-01 through 2024-12-31
test = 2025-01-01 onward
split boundaries are asserted
```

### Task 8: Normalization

File:

```text
src/portfolio_rl/features/normalization.py
```

Acceptance criteria:

```text
winsorization thresholds fit on train only
scaler fit on train only
all splits transformed using train artifacts
scaler is saved to artifacts/scalers/
```

### Task 9: Feature spec

File:

```text
src/portfolio_rl/features/feature_spec.py
```

Acceptance criteria:

```text
feature_spec_v1.json is emitted
asset_order is fixed
per_asset_features are fixed
global_features are fixed
observation_dim is explicit
serving and training can both use the same spec
```

---

## 12. Tests to Write First

Prioritize tests that prevent financial correctness bugs.

```text
tests/unit/test_data_validation.py
tests/unit/test_feature_builder.py
tests/unit/test_normalization.py
tests/unit/test_splits.py
tests/unit/test_feature_spec.py
tests/integration/test_etl_pipeline.py
tests/integration/test_build_features_pipeline.py
```

Critical test cases:

```text
1. Feature builder never emits raw close or adj_close as model features.
2. Normalizer fits only on train dates.
3. No feature column contains "future", "next", "target", or "label".
4. Final matrix has no NaNs after warm-up.
5. Feature order in model_matrix_daily matches feature_spec.json.
6. Pipeline fails if a configured ticker has insufficient data coverage.
7. Split boundaries are chronological and non-overlapping.
```

---

## 13. How to Prompt the Coding Agent

Coding agents are useful when given bounded, verifiable tasks. They are risky when asked to "build the whole pipeline" in one prompt.

### Rules for prompting the agent

```text
1. Give one bounded task at a time.
2. Specify exact files to create or modify.
3. Include acceptance criteria.
4. Ask for type hints and docstrings.
5. Require tests in the same prompt.
6. Tell the agent not to change unrelated files.
7. Tell the agent not to hardcode tickers, dates, or feature names.
8. Ask the agent to run or describe the exact pytest command.
9. Review generated code for leakage, calendar alignment, and config drift.
```

### Good prompt template

```text
Implement src/portfolio_rl/data/validation.py for the Phase 1 pipeline.

Requirements:
- Use type hints for all public functions.
- Validate the prices_daily schema: date, ticker, open, high, low, close,
  adj_close, volume, dividends, stock_splits, source, downloaded_at.
- Fail fast on duplicate ticker-date rows.
- Fail fast on non-positive prices.
- Warn, but do not delete, rows where abs(ret_1d) > 0.25.
- Do not hardcode ticker names.
- Do not modify unrelated files.

Also add tests in tests/unit/test_data_validation.py that cover:
- duplicate ticker-date detection
- non-positive price detection
- sorted date check
- high >= low check
- outlier warning behavior
```

### Bad prompt

```text
Build the whole data pipeline for me.
```

This is too broad. The agent will likely create inconsistent files, skip tests, hardcode assumptions, or leak future data.

### Prompt for feature engineering

```text
Implement src/portfolio_rl/features/returns.py and tests.

Inputs:
- Long-format prices DataFrame with date, ticker, adj_close.
- Feature config with return_windows.

Outputs:
- Long-format feature DataFrame with date, ticker, ret_1d, ret_5d,
  ret_21d, ret_63d, ret_126d, ret_252d.

Rules:
- Use log returns.
- Group by ticker before shifting.
- Do not backfill.
- Do not include future returns as features.
- Preserve date and ticker columns.
- Add unit tests using a small deterministic price fixture.
```

### Prompt for normalization

```text
Implement src/portfolio_rl/features/normalization.py and tests.

Requirements:
- Fit winsorization thresholds on train rows only.
- Fit StandardScaler on train rows only.
- Transform train, validation, and test using the train-fitted artifacts.
- Save scaler artifact to artifacts/scalers/feature_scaler_v1.pkl.
- Return a DataFrame with the same non-feature columns preserved.
- Add tests proving validation/test values do not influence train statistics.
```

### Prompt for feature spec

```text
Implement src/portfolio_rl/features/feature_spec.py.

Requirements:
- Build feature_spec_v1.json from configs/universe.yaml and features.yaml.
- Include asset_order, per_asset_features, global_features,
  current_weight_features, observation_dim, feature_version, created_at.
- Provide a function flatten_features(structured_features, feature_spec)
  that orders fields exactly according to the spec.
- Add tests that fail if asset order or feature order changes unexpectedly.
```

---

## 14. Common Implementation Mistakes

```text
Mistake: Backfilling missing prices.
Fix: Never backfill. Drop invalid windows or fail coverage checks.

Mistake: Fitting scaler on all dates.
Fix: Fit on train only.

Mistake: Feeding raw close prices to the model.
Fix: Use returns, z-scores, ranks, and normalized features.

Mistake: Hardcoding tickers inside feature functions.
Fix: Read asset_order from config or feature_spec.

Mistake: Building a flat API feature vector manually.
Fix: Use feature_spec-driven flattening.

Mistake: Precomputing weekly returns incorrectly.
Fix: Save one-day returns and let the environment compound across five daily rows.

Mistake: Using 2022 as validation only.
Fix: Training includes 2022 and 2023; validation is 2024.
```

---

## 15. Definition of Done

Phase 1 is complete when this command sequence succeeds:

```bash
make etl
make features
make test
```

And these artifacts exist:

```text
data/raw/prices_daily.parquet
data/raw/macro_daily.parquet
data/processed/model_matrix_daily.parquet
data/duckdb/portfolio.duckdb
artifacts/scalers/feature_scaler_v1.pkl
artifacts/feature_specs/feature_spec_v1.json
artifacts/reports/data_quality_report_v1.json
```

The data quality report should include:

```json
{
  "universe_name": "liquid_global_etf_v1",
  "feature_version": "v1",
  "n_assets": 14,
  "model_start_date": "2010-01-01",
  "train_end_date": "2023-12-31",
  "validation_start_date": "2024-01-01",
  "test_start_date": "2025-01-01",
  "nan_count_final": 0,
  "inf_count_final": 0,
  "normalization_fit_split": "train"
}
```

Do not begin RL training until Phase 1 passes this definition of done.
