# Phase 2 — Institutional RL Environment, Training, and Evaluation Blueprint

**Project:** RL Dynamic Portfolio Allocation  
**Phase:** 2 — Gymnasium Environment, Policy Training, Baseline Evaluation, and Experiment Artifacts  
**Audience:** Junior engineers, coding-agent operators, and project reviewers  
**Reviewed repo snapshot:** uploaded zip, Git commit `b779d86c87173a5171ebf9a51b3f04cbe69e6fe8` (`Harden Phase 1 validation checks`)  
**Status:** Ready to begin Phase 2 after the Phase 1 validation hardening changes are merged and the test environment is reproducible.  
**Reviewer update:** Phase 2 architecture approved by external MLE review. This revision incorporates reward scaling, VecNormalize constraints, aligned PPO rollout sizing, EOF-safe evaluation stepping, lightweight training `info` dictionaries, a dummy-agent environment smoke test, and stricter PR sequencing guidance.

---

## 0. Senior MLE Review of the Latest Phase 1 Codebase

The latest codebase is materially stronger than the earlier version. The most important Phase 1 concerns were addressed well enough to proceed with Phase 2 design.

### What is now in good shape

1. **Production logic lives under `src/portfolio_rl/`.**
   - The scripts are thin CLI wrappers.
   - Core logic lives in package modules such as `portfolio_rl.data.etl`, `portfolio_rl.features.pipeline`, `portfolio_rl.features.model_matrix`, and `portfolio_rl.features.feature_spec`.
   - This is the correct structure for Phase 2.

2. **Raw vs clean artifact validation is correctly separated.**
   - Raw artifacts can contain missingness and are summarized diagnostically.
   - Interim, processed, normalized, and model-matrix artifacts are required to be NaN/inf-free.
   - This is the right standard for model-ready data.

3. **Feature-spec-driven model matrix validation exists.**
   - `feature_spec_v1.json` defines asset order, feature order, current-weight feature order, and `observation_dim`.
   - `validate_phase1.py` checks exact model-matrix column set and order.
   - This prevents silent schema drift.

4. **Timing contract is now explicit.**
   - Phase 1 stores same-date close-to-close daily returns.
   - Phase 2 must consume future rows after the decision date.
   - For a decision at row `i`, the weekly reward must use return rows `i+1:i+6`, not row `i`.

5. **The repo confirms `.venv` is not tracked.**
   - `.gitignore` includes `.venv/`, `/data/`, `/artifacts/`, `.ipynb_checkpoints/`, `.DS_Store`.
   - `git ls-files .venv` returned zero files.

### Remaining hygiene notes before Phase 2 implementation

1. **The uploaded zip still contains `.venv/`, `.git/`, `data/`, `artifacts/`, `.pytest_cache/`, `.DS_Store`, and checkpoint files.**
   - This is not a Git tracking issue, but it is a handoff/archive issue.
   - For future reviews, export a clean zip with:

   ```bash
   git archive --format=zip --output deep-rl-portfolio-optimization-clean.zip HEAD
   ```

   If artifacts are needed, send them separately.

2. **I could not run the full test suite in my sandbox because runtime dependencies such as `duckdb`, `pyarrow`, and `yfinance` were not installed in the base interpreter, and the included `.venv` has a broken local Python symlink.**
   - I inspected the source code directly.
   - Your local CI or developer machine should run:

   ```bash
   make test
   make validate
   ```

3. **The Phase 1 model matrix includes static equal-weight current-weight columns.**
   - This is acceptable for Phase 1 artifact shape validation.
   - In Phase 2, the environment must not use those static weight columns as the live portfolio state.
   - The environment must replace the last `n_assets` observation entries with the actual drifted current weights at every step.

This last point is critical and is repeated below.

---

## 1. Phase 2 Objective

Phase 2 builds a controlled RL research system, not just a training script.

The deliverable is a modular, testable, reproducible portfolio-allocation system where:

1. Phase 1 data artifacts are loaded through a strict data interface.
2. A Gymnasium-compatible environment simulates weekly portfolio allocation.
3. The environment uses two clocks:
   - daily market data clock,
   - weekly agent decision clock.
4. Actions are transformed into long-only portfolio weights using temperature-scaled softmax.
5. Transaction costs and weight drift are modeled explicitly.
6. PPO is trained first as the baseline RL algorithm.
7. Learned policies are compared against deterministic baselines using the same backtest engine.
8. Every experiment is reproducible from saved config, data version, feature spec, scaler, model, and metrics.

The goal is to build an **institutional research platform** where an RL policy is one replaceable component inside a controlled simulation and evaluation framework.

---

## 2. Non-Negotiable Phase 2 Contracts

These are hard rules. Do not relax them without senior review.

### 2.1 Data timing contract

Phase 1 model matrix rows have this interpretation:

```text
row i date = close of trading day i
obs columns = information known at close of day i
return_*_1d columns = same-date close-to-close log returns ending on day i
```

Therefore, in Phase 2:

```text
Decision at row i:
    observe features known at close of day i
    choose target portfolio weights
    hold for next 5 trading days
    reward uses return rows i+1, i+2, i+3, i+4, i+5
    next observation is row i+5
```

Python slicing rule:

```python
forward_returns = returns.iloc[i + 1 : i + 1 + rebalance_frequency]
```

For weekly rebalancing with `rebalance_frequency = 5`:

```python
forward_returns = returns.iloc[i + 1 : i + 6]
```

Never use `returns.iloc[i]` as part of the reward for a decision made from `observation[i]`.

### 2.2 Two-clock environment contract

The environment must track two different clocks:

```python
self.current_data_idx  # daily market-row index
self.current_step      # weekly agent decision-step index
```

Each `step(action)` call advances:

```python
self.current_data_idx += rebalance_frequency_trading_days  # +5 rows
self.current_step += 1                                     # +1 decision
```

### 2.3 Weekly episode contract

Use the exact Phase 2 environment config:

```yaml
# configs/env.yaml
# --- Time & Stepping Mechanics ---
rebalance_frequency_trading_days: 5
episode_length_trading_days: 260
max_episode_steps: 52  # derived exactly as (episode_length_trading_days / rebalance_frequency_trading_days)

# --- Action & Frictions ---
action_transform: softmax
action_temperature: 5.0
initial_weights: equal_weight
transaction_cost_bps: 10.0

# --- Reward Scaling ---
reward_type: log_growth
reward_scale: 100.0
terminal_bad_gross_penalty: -100.0

# --- Diagnostics / Performance ---
record_arrays_in_info: false  # false for PPO training; true for backtest/reporting only
```

### 2.4 Action contract

The action space is normalized and symmetric:

```python
action_space = gymnasium.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(n_assets,),
    dtype=np.float32,
)
```

Convert raw action logits into target portfolio weights:

```python
target_weights = softmax(raw_action * action_temperature)
```

With `action_temperature = 5.0`, the policy can express high-conviction allocations while still receiving normalized actions from the RL algorithm.

### 2.5 Current-weight observation contract

`feature_spec_v1.json` says the observation consists of:

```text
[per-asset market features, global market features, current portfolio weights]
```

The current code builds `model_matrix_daily.parquet` using equal weights for the current-weight slice. This is useful for a fixed Phase 1 matrix shape, but Phase 2 must override the weight slice dynamically.

Correct Phase 2 behavior:

```python
market_features = model_matrix_obs[: market_feature_dim]
observation = concat(market_features, current_drifted_weights)
```

Do not pass the static equal-weight columns from `model_matrix_daily.parquet` directly into the agent after the first step.

### 2.6 Reward scaling and normalization contract

Weekly unlevered portfolio log returns are usually small. A raw weekly reward like `0.002` to `0.015` can make PPO's critic target scale unnecessarily tiny and slow learning. Therefore Phase 2 v1 uses an explicit reward scale:

```yaml
reward_scale: 100.0
```

Training reward is:

```python
reward = np.log(net_gross_return) * reward_scale
```

This scaled reward is only a learning signal. Evaluation metrics, NAV, CAGR, Sharpe, drawdown, turnover, and cost drag must always be computed from unscaled portfolio returns/NAV.

Do not simultaneously apply manual reward scaling and `VecNormalize(norm_reward=True)` unless there is a deliberate experiment with documented rationale. The default is manual reward scaling and no reward normalization wrapper.

### 2.7 VecNormalize contract

Do not use Stable-Baselines3 `VecNormalize(norm_obs=True)`.

Phase 1 already normalizes market features and the current-weight slice is mathematically meaningful in `[0.0, 1.0]`. Applying a rolling observation normalizer across the full observation would destroy both the feature contract and the weight semantics.

Allowed choices:

```python
# Preferred v1
vec_env = DummyVecEnv([...])  # no VecNormalize

# Optional reward-normalization experiment only if reward_scale is disabled or carefully audited
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
```

Never use:

```python
VecNormalize(vec_env, norm_obs=True, ...)
```


### 2.8 Split contract

Use Phase 1 split labels exactly:

```text
train:       2010-01-01 through 2023-12-31
validation: 2024-01-01 through 2024-12-31
test:       2025-01-01 through latest ingested trading date
```

Training environment:

```text
random contiguous 260-trading-day windows sampled from train split
```

Validation and test environments:

```text
chronological deterministic windows, no random start, no training updates
```

---

## 3. Phase 2 Repo Structure

Add the following modules.

```text
src/portfolio_rl/
│
├── data/
│   ├── dataset.py              # load Phase 1 model matrix and feature spec
│   ├── feature_store.py        # split-aware feature/return access API
│   └── schemas.py              # optional typed data contracts
│
├── env/
│   ├── __init__.py
│   ├── portfolio_env.py        # Gymnasium environment
│   ├── action.py               # action clipping/softmax/validation
│   ├── costs.py                # turnover and transaction cost math
│   ├── drift.py                # weight drift and return compounding
│   ├── reward.py               # reward functions
│   └── episode_sampler.py      # random train windows and fixed eval windows
│
├── policies/
│   ├── __init__.py
│   ├── base.py                 # common policy protocol
│   ├── sb3_policy.py           # wrapper around trained SB3 model
│   └── baseline_policies.py    # equal weight, SPY, SHY, inverse vol, momentum
│
├── training/
│   ├── __init__.py
│   ├── train_ppo.py
│   ├── callbacks.py
│   ├── experiment.py
│   └── registry.py
│
├── evaluation/
│   ├── __init__.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── attribution.py
│   └── reports.py
│
└── utils/
    ├── random.py
    ├── hashes.py
    └── logging.py
```

Add scripts only as thin CLI wrappers:

```text
scripts/train_ppo.py
scripts/evaluate_policy.py
scripts/run_baselines.py
```

Add configs:

```text
configs/train_ppo.yaml
configs/evaluation.yaml
```

Do not put business logic in scripts.

---

## 4. Required Dependencies

Update `pyproject.toml` for Phase 2.

Recommended additions:

```toml
[project]
dependencies = [
    "duckdb>=1.0",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "pydantic>=2.7",
    "PyYAML>=6.0",
    "requests>=2.31",
    "yfinance>=0.2",
    "numpy>=1.26",
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "torch>=2.2",
    "wandb>=0.16",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
]
```

Keep dependency versions bounded enough to avoid uncontrolled API changes.

---

## 5. Data Interface Design

### 5.1 Why we need a data interface

The environment should not directly know about Parquet files, DuckDB, raw column names, or artifact folders.

Instead, the environment should depend on a clean in-memory object:

```text
PortfolioFeatureStore
```

This makes the environment:

1. easy to unit test,
2. independent of storage format,
3. reusable for validation/test/backtest,
4. easier to serve later behind an API.

### 5.2 `PortfolioDataset`

Create:

```text
src/portfolio_rl/data/dataset.py
```

Responsibilities:

1. Load `data/processed/model_matrix_daily.parquet`.
2. Load `artifacts/feature_specs/feature_spec_v1.json`.
3. Validate exact schema using the same logic as `validate_phase1.py`.
4. Split the matrix into:
   - dates,
   - split labels,
   - market feature matrix,
   - return matrix.
5. Expose asset order and feature version.

Important: remove the static current-weight slice from Phase 1 observations when constructing market features.

Suggested class:

```python
@dataclass(frozen=True)
class PortfolioDataset:
    dates: pd.DatetimeIndex
    splits: np.ndarray
    market_features: np.ndarray      # shape: [n_dates, observation_dim - n_assets]
    returns: np.ndarray              # shape: [n_dates, n_assets], log returns
    asset_order: list[str]
    feature_version: str
    observation_dim: int

    @property
    def n_assets(self) -> int:
        return len(self.asset_order)

    @property
    def market_feature_dim(self) -> int:
        return self.observation_dim - self.n_assets
```

Implementation detail:

```python
obs_columns = [f"obs_{i:03d}" for i in range(feature_spec.observation_dim)]
return_columns = [f"return_{ticker.lower()}_1d" for ticker in feature_spec.asset_order]

obs = model_matrix[obs_columns].to_numpy(dtype=np.float32)
market_features = obs[:, : feature_spec.observation_dim - len(feature_spec.asset_order)]
returns = model_matrix[return_columns].to_numpy(dtype=np.float32)
```

Do not use:

```python
obs = model_matrix[obs_columns].to_numpy()
```

as the final environment observation, because the last `n_assets` columns are static equal weights from Phase 1.

### 5.3 `PortfolioFeatureStore`

Create:

```text
src/portfolio_rl/data/feature_store.py
```

Responsibilities:

1. Provide split-specific index ranges.
2. Provide observations for any market index.
3. Provide forward return windows using Phase 1 timing contract.
4. Enforce no cross-split leakage.

Suggested API:

```python
class PortfolioFeatureStore:
    def __init__(self, dataset: PortfolioDataset, split: str):
        ...

    @property
    def n_assets(self) -> int:
        ...

    @property
    def observation_dim(self) -> int:
        ...

    @property
    def market_feature_dim(self) -> int:
        ...

    def get_market_features(self, relative_idx: int) -> np.ndarray:
        ...

    def get_forward_log_returns(
        self,
        relative_idx: int,
        horizon: int,
    ) -> np.ndarray:
        """Return rows i+1 through i+horizon inclusive, shape [horizon, n_assets]."""
        ...

    def max_valid_start_index(self, episode_length_trading_days: int) -> int:
        ...

    def date_at(self, relative_idx: int) -> pd.Timestamp:
        ...
```

Implementation rule:

```python
start = absolute_index(relative_idx) + 1
end = start + horizon
return returns[start:end]
```

For `horizon=5`, this returns five rows.

---

## 6. Episode Sampler Design

Create:

```text
src/portfolio_rl/env/episode_sampler.py
```

### 6.1 Training sampler

Training should randomize episode start to reduce chronological memorization.

```python
class RandomWindowEpisodeSampler:
    def sample_start(self, store: PortfolioFeatureStore, episode_length: int, rng: np.random.Generator) -> int:
        max_start = store.max_valid_start_index(episode_length)
        return int(rng.integers(0, max_start + 1))
```

For train split:

```text
sample a contiguous 260-trading-day window
```

Implementation detail: a 260-trading-day holding window requires a valid decision observation at the start plus enough future return rows for 52 five-day steps. The sampler/store boundary checks must guarantee the final step can consume rows `i+1:i+6` and still return a valid final observation or truncate cleanly.

### 6.2 Evaluation sampler

Validation/test should be deterministic.

```python
class FixedStartEpisodeSampler:
    def sample_start(...):
        return 0
```

For validation/test, we usually run the full split in rolling weekly steps. Because the fixed split may be longer or shorter than exactly 260 trading days, allow evaluation to run until the split boundary. Do not randomize validation/test.

Recommended environment config option:

```yaml
episode_sampling:
  train_random_start: true
  eval_random_start: false
  train_episode_length_trading_days: 260
  eval_run_full_split: true
```

---

## 7. Financial Mechanics Modules

Do not put all math inside `PortfolioEnv.step()`. Create small, testable modules.

### 7.1 Action transform

File:

```text
src/portfolio_rl/env/action.py
```

Function:

```python
def action_to_weights(action: np.ndarray, temperature: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    logits = np.clip(action, -1.0, 1.0) * temperature
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()
    return weights.astype(np.float32)
```

Validation:

```python
assert np.all(weights >= 0.0)
assert np.isclose(weights.sum(), 1.0)
```

### 7.2 Transaction costs

File:

```text
src/portfolio_rl/env/costs.py
```

Function:

```python
def calculate_turnover(current_weights: np.ndarray, target_weights: np.ndarray) -> float:
    return float(np.abs(target_weights - current_weights).sum())


def calculate_transaction_cost_fraction(turnover: float, transaction_cost_bps: float) -> float:
    return turnover * transaction_cost_bps / 10_000.0
```

Interpretation:

```text
transaction_cost_bps is cost per dollar turned over.
If turnover = 1.0 and transaction_cost_bps = 10, cost fraction = 0.001.
```

### 7.3 Weight drift and compounding

File:

```text
src/portfolio_rl/env/drift.py
```

Remember: Phase 1 returns are **log returns**.

Function:

```python
def simulate_buy_and_hold_period(
    start_weights: np.ndarray,
    forward_log_returns: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Simulate one holding period without intra-period rebalancing.

    Args:
        start_weights: shape [n_assets], target weights after trade.
        forward_log_returns: shape [horizon, n_assets], rows i+1:i+1+horizon.

    Returns:
        period_gross_return: NAV multiplier before transaction cost.
        end_weights: drifted weights at end of holding period.
        daily_portfolio_simple_returns: shape [horizon].
    """
    weights = start_weights.astype(np.float64).copy()
    nav = 1.0
    daily_returns = []

    for daily_log_returns in forward_log_returns:
        asset_gross = np.exp(daily_log_returns)
        portfolio_gross = float(np.dot(weights, asset_gross))
        nav *= portfolio_gross
        daily_returns.append(portfolio_gross - 1.0)
        weights = weights * asset_gross / portfolio_gross

    return float(nav), weights.astype(np.float32), np.asarray(daily_returns, dtype=np.float32)
```

Why this is better than a shortcut:

```python
np.dot(start_weights, np.exp(forward_log_returns.sum(axis=0)))
```

The shortcut gives the same ending NAV for a buy-and-hold basket, but the daily simulation gives us:

1. daily path,
2. interim volatility,
3. drawdown diagnostics,
4. explicit end-of-period drifted weights.

### 7.4 Reward functions

File:

```text
src/portfolio_rl/env/reward.py
```

Start with a simple base reward, but scale it for PPO learning stability.

```python
def log_growth_reward(
    period_gross_return: float,
    transaction_cost_fraction: float,
    reward_scale: float = 100.0,
    bad_gross_penalty: float = -100.0,
) -> float:
    net_gross = (1.0 - transaction_cost_fraction) * period_gross_return
    if net_gross <= 0.0:
        return float(bad_gross_penalty)
    return float(np.log(net_gross) * reward_scale)
```

Why scale the reward:

```text
A normal weekly unlevered log return is often around 0.002 to 0.015.
PPO can train on small rewards, but tiny critic targets can slow learning and make diagnostics harder.
Scaling by 100 moves a typical useful reward into a healthier neural-network target range.
```

Important:

```text
Reward scaling affects only the learning signal.
It must not be used to compute reported NAV, CAGR, Sharpe, drawdown, or backtest metrics.
```

Risk penalty can be added later, but do not overcomplicate v1.

Recommended v1:

```yaml
reward:
  type: log_growth
  include_transaction_costs: true
  reward_scale: 100.0
  terminal_bad_gross_penalty: -100.0
  volatility_penalty_lambda: 0.0
```

Only add a volatility penalty after the simple scaled log-growth reward produces a working PPO baseline.

---

## 8. PortfolioEnv Design

Create:

```text
src/portfolio_rl/env/portfolio_env.py
```

### 8.1 Constructor inputs

```python
class PortfolioEnv(gymnasium.Env):
    def __init__(
        self,
        feature_store: PortfolioFeatureStore,
        env_config: EnvConfig,
        episode_sampler: EpisodeSampler,
        seed: int | None = None,
    ):
        ...
```

Do not pass raw file paths directly into the environment.

### 8.2 Spaces

```python
self.action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.n_assets,),
    dtype=np.float32,
)

self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(self.feature_store.observation_dim,),
    dtype=np.float32,
)
```

Observation dimension is:

```text
market_feature_dim + n_assets
```

For current v1 feature spec:

```text
market_feature_dim = 316 - 14 = 302
observation_dim = 316
```

### 8.3 State variables

```python
self.current_step: int        # agent decision clock
self.current_data_idx: int    # relative daily data index inside split
self.episode_start_idx: int
self.current_weights: np.ndarray
self.portfolio_value: float
```

### 8.4 Reset behavior

```python
def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    self.current_step = 0
    self.episode_start_idx = self.episode_sampler.sample_start(...)
    self.current_data_idx = self.episode_start_idx
    self.current_weights = equal_weight_vector(self.n_assets)
    self.portfolio_value = 1.0
    observation = self._build_observation()
    info = {
        "date": self.feature_store.date_at(self.current_data_idx),
        "portfolio_value": self.portfolio_value,
        "current_weights": self.current_weights.copy(),
    }
    return observation, info
```

During PPO training, keep reset info lightweight as well. Large arrays are allowed only when `record_arrays_in_info=True` for deterministic evaluation/reporting.

### 8.5 Observation builder

```python
def _build_observation(self) -> np.ndarray:
    market_features = self.feature_store.get_market_features(self.current_data_idx)
    obs = np.concatenate([market_features, self.current_weights]).astype(np.float32)
    if obs.shape != self.observation_space.shape:
        raise RuntimeError("observation shape mismatch")
    return obs
```

Do not use the static Phase 1 current-weight columns.

### 8.6 Step behavior

Canonical structure:

```python
def step(self, action: np.ndarray):
    # 1. Convert action to target portfolio weights.
    target_weights = action_to_weights(
        action,
        temperature=self.config.action_temperature,
    )

    # 2. Request future returns for the next 5 trading days.
    forward_log_returns = self.feature_store.get_forward_log_returns(
        self.current_data_idx,
        horizon=self.config.rebalance_frequency_trading_days,
    )

    # 3. Gracefully handle end-of-split / EOF.
    # Validation/test splits may not divide exactly into 5-day windows.
    if len(forward_log_returns) < self.config.rebalance_frequency_trading_days:
        observation = self._build_observation()
        info = {
            "date": self.feature_store.date_at(self.current_data_idx),
            "portfolio_value": self.portfolio_value,
            "eof_truncated": True,
        }
        return observation, 0.0, False, True, info

    # 4. Calculate turnover and transaction cost against DRIFTED current weights.
    turnover = calculate_turnover(self.current_weights, target_weights)
    cost_fraction = calculate_transaction_cost_fraction(
        turnover,
        self.config.transaction_cost_bps,
    )

    # 5. Simulate holding period and end-of-period weight drift.
    period_gross_return, drifted_weights, daily_portfolio_returns = (
        simulate_buy_and_hold_period(target_weights, forward_log_returns)
    )

    # 6. Compute net learning reward and update NAV using unscaled net gross.
    net_gross_return = (1.0 - cost_fraction) * period_gross_return
    reward = log_growth_reward(
        period_gross_return,
        cost_fraction,
        reward_scale=self.config.reward_scale,
        bad_gross_penalty=self.config.terminal_bad_gross_penalty,
    )
    self.portfolio_value *= net_gross_return

    # 7. Update current weights to drifted post-market weights.
    self.current_weights = drifted_weights

    # 8. Advance clocks.
    self.current_data_idx += self.config.rebalance_frequency_trading_days
    self.current_step += 1

    # 9. Termination/truncation.
    terminated = False
    truncated = bool(self.current_step >= self.config.max_episode_steps)

    # 10. Build next observation unless we have stepped beyond available data.
    observation = self._build_observation()

    # 11. Keep info lightweight during PPO training.
    info = {
        "date": self.feature_store.date_at(self.current_data_idx),
        "turnover": float(turnover),
        "transaction_cost_fraction": float(cost_fraction),
        "period_gross_return": float(period_gross_return),
        "net_gross_return": float(net_gross_return),
        "portfolio_value": float(self.portfolio_value),
        "reward_unscaled_log_growth": float(np.log(net_gross_return)) if net_gross_return > 0 else float("nan"),
    }

    if self.config.record_arrays_in_info:
        info.update(
            {
                "target_weights": target_weights.copy(),
                "current_weights": self.current_weights.copy(),
                "daily_portfolio_returns": daily_portfolio_returns.copy(),
            }
        )

    return observation, reward, terminated, truncated, info
```

EOF behavior is mandatory for validation and test. If fewer than `rebalance_frequency_trading_days` forward return rows are available, truncate early instead of slicing past the split boundary.

For full training episodes, the episode sampler should normally choose starts that have all 52 full forward windows available. The EOF branch is still required as a defensive guard and for evaluation splits whose length is not an exact multiple of five trading days.

`record_arrays_in_info` must be `False` during PPO training, especially when using vectorized environments. Large NumPy arrays inside `info` are serialized across processes in `SubprocVecEnv` and can become a major IPC/RAM bottleneck. Set it to `True` only for deterministic backtests and reporting runs.

Implementation gotcha:

```text
Turnover must compare new target weights against drifted current weights, not previous target weights.
```

Correct:

```python
turnover = abs(target_weights - drifted_current_weights).sum()
```

Incorrect:

```python
turnover = abs(target_weights - previous_target_weights).sum()
```

---

## 9. Baseline Policy Suite

Before training PPO, implement deterministic baselines.

Create:

```text
src/portfolio_rl/policies/baseline_policies.py
```

### 9.1 Policy protocol

```python
class Policy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...
```

For compatibility with the env, baseline policies can return raw actions or target weights. Prefer returning target weights through a common wrapper to avoid softmax ambiguity.

Recommended internal interface:

```python
class WeightPolicy(Protocol):
    def target_weights(self, observation: np.ndarray, info: dict) -> np.ndarray:
        ...
```

The backtest engine can use target weights directly. The RL environment uses actions.

### 9.2 Required baselines

Implement at least:

1. **EqualWeightWeeklyPolicy**
   - Rebalances to `1 / n_assets` every decision.

2. **BuyAndHoldEqualWeightPolicy**
   - Starts equal-weight and never trades again.
   - Useful to isolate cost impact and drift.

3. **SPYOnlyPolicy**
   - 100% SPY.

4. **SHYOnlyPolicy**
   - 100% SHY cash-like proxy.

5. **InverseVolatilityPolicy**
   - Uses selected volatility features or raw recent returns.
   - Target weight proportional to `1 / vol`.
   - Normalize to sum to one.

6. **MomentumPolicy** optional but useful.
   - Ranks assets by 63-day return feature.
   - Allocates to top `k` assets.

### 9.3 Why baselines are mandatory

Do not interpret PPO results until baselines exist.

If PPO cannot beat equal-weight after costs, the first debugging target is usually:

```text
data alignment, reward design, or environment mechanics
```

not PPO hyperparameters.

---

## 10. Backtest Engine

Create:

```text
src/portfolio_rl/evaluation/backtest.py
```

The backtest engine should evaluate both learned policies and baselines through the same mechanics.

### 10.1 Why not use the training environment directly?

The Gym environment is optimized for RL interaction. A backtest engine should be optimized for deterministic reporting:

1. daily NAV series,
2. weekly target weights,
3. drifted weights,
4. turnover,
5. transaction costs,
6. benchmark comparison,
7. metrics and plots.

The mechanics should call the same lower-level functions from `env/action.py`, `env/costs.py`, and `env/drift.py` to avoid math drift.

### 10.2 Backtest outputs

Produce these artifacts:

```text
artifacts/backtests/{run_id}/
├── nav.parquet
├── weights_target.parquet
├── weights_drifted.parquet
├── trades.parquet
├── costs.parquet
├── metrics.json
└── report.md
```

Required columns:

```text
nav.parquet:
    date
    strategy
    nav
    daily_return
    drawdown

weights_target.parquet:
    date
    strategy
    ticker
    target_weight

weights_drifted.parquet:
    date
    strategy
    ticker
    drifted_weight

trades.parquet:
    date
    strategy
    ticker
    pre_trade_weight
    target_weight
    trade_weight

costs.parquet:
    date
    strategy
    turnover
    transaction_cost_fraction
```

---

## 11. Evaluation Metrics

Create:

```text
src/portfolio_rl/evaluation/metrics.py
```

Required metrics:

```text
total_return
CAGR
annualized_volatility
Sharpe ratio
Sortino ratio
max_drawdown
Calmar ratio
average_weekly_turnover
annualized_turnover
transaction_cost_drag
hit_rate
best_month
worst_month
```

Use 252 trading days for annualization.

For weekly decision logs, be careful not to annualize weekly returns as if they were daily returns. Metrics should be computed from the daily NAV series when available.

---

## 12. PPO Training System

Create:

```text
src/portfolio_rl/training/train_ppo.py
```

### 12.1 Config

Create:

```text
configs/train_ppo.yaml
```

Initial config:

```yaml
algorithm: PPO
policy: MlpPolicy
total_timesteps: 500000
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
n_steps: 2080  # 40 complete 52-step episodes per environment
batch_size: 260  # divides 2080 cleanly; avoid truncated minibatch warnings
n_epochs: 10
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
seed: 42

network:
  pi: [256, 256]
  vf: [256, 256]

wandb:
  enabled: true
  project: rl-dynamic-portfolio-allocation
  group: phase2-ppo
  tags: [ppo, weekly-rebalance, v1-features]

checkpoints:
  save_freq_timesteps: 50000
  eval_freq_timesteps: 25000
```

### 12.2 Training flow

```text
1. Load configs.
2. Load Phase 1 dataset and feature spec.
3. Build train feature store.
4. Build validation feature store.
5. Construct train env with random window sampler.
6. Construct validation env with deterministic sampler.
7. Run Gymnasium/SB3 environment checker.
8. Train PPO.
9. Periodically run validation backtest.
10. Save best checkpoint by validation Sharpe or validation final NAV.
11. Save full experiment artifact bundle.
```

### 12.3 Vectorized environments

Once the single env passes tests, use vectorized training.

Recommended:

```python
from stable_baselines3.common.vec_env import DummyVecEnv
```

Start with `DummyVecEnv`. Move to `SubprocVecEnv` only after the environment is stable and picklable.


### 12.4 PPO rollout sizing

Because each episode is exactly 52 agent steps, set PPO `n_steps` to a clean multiple of 52.

Recommended v1:

```yaml
n_steps: 2080  # 52 * 40
batch_size: 260
```

This collects exactly 40 full weekly episodes per environment before each PPO update. It also avoids subtle GAE diagnostics confusion from repeatedly cutting rollouts mid-episode. If using multiple vectorized environments, remember SB3's total rollout buffer size is:

```text
n_steps * n_envs
```

Keep `batch_size` as a divisor of that product when practical.

### 12.5 VecNormalize rule

Do not use `VecNormalize(norm_obs=True)`.

Preferred v1:

```python
vec_env = DummyVecEnv(env_fns)
model = PPO(..., env=vec_env)
```

Optional reward-normalization experiment:

```python
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
```

Only use the optional reward-normalization experiment if manual `reward_scale` is disabled or the interaction is explicitly documented. Never stack uncontrolled manual scaling and VecNormalize reward normalization.

---

## 13. W&B Experiment Tracking

Log these items:

### 13.1 Scalars

```text
train/episode_reward_scaled
train/episode_reward_unscaled_log_growth
train/episode_nav
train/episode_turnover
train/episode_cost_drag
train/policy_entropy
train/explained_variance
validation/total_return
validation/sharpe
validation/max_drawdown
validation/turnover
```

### 13.2 Artifacts

```text
model.zip
config.yaml
env.yaml
train_ppo.yaml
feature_spec_v1.json
feature_spec_hash.txt
scaler_artifact_hash.txt
data_quality_report_v1.json
validation_metrics.json
validation_nav.parquet
validation_weights.parquet
```

### 13.3 Required run metadata

```text
git_commit
feature_version
feature_spec_hash
universe_name
asset_order
train_start_date
train_end_date
validation_start_date
validation_end_date
transaction_cost_bps
rebalance_frequency_trading_days
action_temperature
reward_type
seed
```

---

## 14. Experiment Artifact Layout

Each training run should create:

```text
artifacts/experiments/{run_id}/
├── model.zip
├── config.yaml
├── env.yaml
├── train_ppo.yaml
├── feature_spec_v1.json
├── feature_spec_hash.txt
├── scaler_artifact_hash.txt
├── data_quality_report_v1.json
├── metrics_validation.json
├── metrics_test.json              # only after final selected model
├── validation_nav.parquet
├── validation_weights.parquet
├── validation_trades.parquet
├── validation_costs.parquet
├── model_card.md
└── manifest.json
```

The manifest should include enough information to reproduce the run:

```json
{
  "run_id": "ppo_v1_2026_05_05_1430",
  "git_commit": "...",
  "feature_version": "v1",
  "feature_spec_hash": "...",
  "data_quality_report_hash": "...",
  "env_config_hash": "...",
  "train_config_hash": "...",
  "seed": 42,
  "created_at": "..."
}
```

---

## 15. Tests Required Before PPO Training

Do not start long PPO training until these tests pass.

### 15.1 Unit tests for `env/action.py`

```text
test_action_to_weights_sum_to_one
test_action_to_weights_nonnegative
test_action_to_weights_is_stable_for_large_logits
test_action_temperature_allows_concentration
test_zero_action_maps_to_equal_weight
```

### 15.2 Unit tests for `env/costs.py`

```text
test_zero_turnover_has_zero_cost
test_full_rotation_turnover
test_cost_fraction_bps_conversion
```

Example:

```python
current = np.array([1.0, 0.0])
target = np.array([0.0, 1.0])
turnover = 2.0
cost at 10 bps = 0.002
```

### 15.3 Unit tests for `env/drift.py`

```text
test_no_return_preserves_weights
test_positive_asset_return_increases_asset_weight
test_period_gross_return_matches_manual_calculation
test_log_returns_are_exponentiated_before_compounding
```

### 15.4 Unit tests for `data/feature_store.py`

```text
test_market_features_exclude_static_weight_slice
test_forward_returns_use_rows_i_plus_1_through_i_plus_horizon
test_forward_returns_do_not_cross_split_boundary
test_store_reports_correct_asset_order
test_store_reports_correct_observation_dim
```

### 15.5 Unit tests for `env/portfolio_env.py`

```text
test_reset_returns_valid_observation_and_info
test_step_advances_two_clocks
test_step_uses_forward_returns_not_same_row_returns
test_step_charges_cost_against_drifted_current_weights
test_step_truncates_gracefully_when_forward_window_hits_eof
test_episode_truncates_after_52_agent_steps
test_observation_replaces_current_weight_slice_dynamically
test_env_passes_gymnasium_check_env
```

### 15.6 Integration tests

```text
test_one_full_train_episode_runs_without_nan
test_validation_env_is_deterministic
test_dummy_random_agent_runs_52_steps_and_prints_final_nav
test_ppo_smoke_train_1000_timesteps
test_equal_weight_backtest_matches_manual_two_asset_case
test_backtest_outputs_required_artifacts
```

### 15.7 Mandatory dummy-agent smoke test before baselines

Before implementing baselines or PPO, initialize `PortfolioEnv` and run one full episode with random valid actions:

```python
obs, info = env.reset(seed=42)
for step in range(52):
    action = np.random.uniform(-1.0, 1.0, size=env.n_assets).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print({
    "steps": step + 1,
    "final_nav": info["portfolio_value"],
    "last_turnover": info.get("turnover"),
})
```

Acceptance criteria:

```text
- Loop reaches 52 steps on a full train episode.
- No NaNs in observation, reward, NAV, turnover, or weights.
- Turnover is non-negative and finite.
- Portfolio NAV is positive and finite.
- Final output prints a sensible final NAV.
```

This is the critical bridge test between pure unit tests and real baseline/backtest work. If the dummy agent cannot complete a 52-step episode cleanly, do not proceed to PR 6.

---

## 16. Implementation Sequence

Implement Phase 2 in this order.

### Milestone 1 — Data interface

Files:

```text
src/portfolio_rl/data/dataset.py
src/portfolio_rl/data/feature_store.py
```

Acceptance criteria:

```text
- Loads model_matrix_daily.parquet and feature_spec_v1.json.
- Exposes market_features without static equal-weight slice.
- Exposes returns matrix in feature_spec asset order.
- get_forward_log_returns(i, 5) returns rows i+1:i+6.
- Split boundaries are enforced.
```

### Milestone 2 — Financial mechanics

Files:

```text
src/portfolio_rl/env/action.py
src/portfolio_rl/env/costs.py
src/portfolio_rl/env/drift.py
src/portfolio_rl/env/reward.py
```

Acceptance criteria:

```text
- All financial functions are pure and unit tested.
- Softmax weights sum to one.
- Transaction cost conversion is correct.
- Drift math works on simple manually checked examples.
```

### Milestone 3 — Episode sampler

Files:

```text
src/portfolio_rl/env/episode_sampler.py
```

Acceptance criteria:

```text
- Training windows are random but reproducible with seed.
- Evaluation windows are deterministic.
- Sampled windows do not require returns outside split boundary.
```

### Milestone 4 — PortfolioEnv

Files:

```text
src/portfolio_rl/env/portfolio_env.py
```

Acceptance criteria:

```text
- Gymnasium-compatible reset/step signatures.
- Two-clock mechanics implemented.
- Weekly rewards use five future rows.
- EOF-safe truncation works on short validation/test tails.
- Observation appends dynamic current weights.
- `record_arrays_in_info=False` keeps training info lightweight.
- check_env passes.
- Dummy random-agent 52-step smoke test passes.
```

### Milestone 5 — Dummy-agent environment smoke test

Files:

```text
scripts/smoke_test_env.py
tests/integration/test_dummy_agent_env.py
```

Acceptance criteria:

```text
- Runs PortfolioEnv with random actions sampled from Uniform[-1, 1].
- Completes 52 agent steps on a full training episode.
- Prints final NAV, cumulative turnover, and final date.
- Does not crash or emit NaN/inf values.
- Must pass before baseline/backtest implementation begins.
```

### Milestone 6 — Baselines and backtest

Files:

```text
src/portfolio_rl/policies/baseline_policies.py
src/portfolio_rl/evaluation/backtest.py
src/portfolio_rl/evaluation/metrics.py
```

Acceptance criteria:

```text
- Equal-weight, buy-and-hold, SPY-only, SHY-only, inverse-vol baselines exist.
- Same mechanics used for all strategies.
- Backtest emits NAV, weights, trades, costs, metrics.
```

### Milestone 7 — PPO training harness

Files:

```text
src/portfolio_rl/training/train_ppo.py
src/portfolio_rl/training/callbacks.py
scripts/train_ppo.py
```

Acceptance criteria:

```text
- PPO smoke test trains for 1,000 timesteps.
- Full training writes experiment artifact folder.
- W&B logs config, metrics, and artifacts.
- Best validation checkpoint is saved.
```

### Milestone 8 — Validation report

Files:

```text
src/portfolio_rl/evaluation/reports.py
scripts/evaluate_policy.py
```

Acceptance criteria:

```text
- Validation report compares PPO vs baselines.
- Includes returns, risk, drawdown, turnover, cost drag.
- Includes warning if PPO underperforms equal-weight.
```

---

## 17. Coding-Agent Prompt Guide

Juniors should use small, scoped prompts. Do not ask the coding agent to build all of Phase 2 at once.

When using these prompts, paste only the relevant blueprint sections into the LLM context window. For example, paste Section 5 for the Data Interface PR, Section 7 for the Financial Mechanics PR, and Section 8 for the PortfolioEnv PR. Do not paste the entire TDD unless the agent genuinely needs global context. Focused context reduces hallucinated dependencies and keeps code changes small.

### Bad prompt

```text
Implement Phase 2 RL training for portfolio optimization.
```

This will likely create tightly coupled code, duplicate math, weak tests, and hidden lookahead bugs.

### Good prompt 1 — Data interface

```text
Implement src/portfolio_rl/data/dataset.py and src/portfolio_rl/data/feature_store.py.

Requirements:
- Load data/processed/model_matrix_daily.parquet and artifacts/feature_specs/feature_spec_v1.json.
- Validate model_matrix column order using the feature spec.
- Split obs columns into market_features and static weight features.
- Exclude the static weight slice from market_features.
- Expose returns in feature_spec asset_order.
- Implement get_forward_log_returns(i, horizon) using rows i+1:i+1+horizon.
- Enforce train/validation/test split boundaries.
- Write unit tests for the data timing contract.
Do not implement the Gym environment yet.
```

### Good prompt 2 — Financial mechanics

```text
Implement pure financial mechanics modules:
- src/portfolio_rl/env/action.py
- src/portfolio_rl/env/costs.py
- src/portfolio_rl/env/drift.py
- src/portfolio_rl/env/reward.py

Use NumPy only. Do not import Gymnasium or Stable-Baselines3.
Implement action_to_weights with temperature-scaled softmax.
Implement transaction cost from L1 turnover and bps.
Implement buy-and-hold weight drift using log returns.
Implement scaled log growth reward after transaction costs with reward_scale=100.0.
Write unit tests with small two-asset manual examples.
```

### Good prompt 3 — Episode sampler

```text
Implement src/portfolio_rl/env/episode_sampler.py.

Requirements:
- RandomWindowEpisodeSampler samples random valid starts inside the split.
- FixedStartEpisodeSampler returns deterministic start index 0.
- Sampling must guarantee that a 260-trading-day episode with 52 weekly steps has enough forward returns.
- Add deterministic seed tests.
Do not implement PortfolioEnv yet.
```

### Good prompt 4 — PortfolioEnv skeleton

```text
Implement src/portfolio_rl/env/portfolio_env.py skeleton.

Requirements:
- Inherit from gymnasium.Env.
- Use PortfolioFeatureStore, EnvConfig, and EpisodeSampler.
- Define action_space Box[-1, 1] with shape n_assets.
- Define observation_space with shape feature_store.observation_dim.
- Implement reset() only.
- _build_observation must concatenate market_features with dynamic current_weights.
- Do not implement step() yet.
- Write reset and observation-shape tests.
```

### Good prompt 5 — PortfolioEnv step

```text
Add PortfolioEnv.step().

Requirements:
- Convert action to target weights using action_to_weights.
- Calculate turnover against drifted current_weights.
- Fetch forward returns with get_forward_log_returns(current_data_idx, 5).
- Simulate buy-and-hold period using drift.py.
- Apply transaction costs to NAV.
- Reward is scaled log net growth using reward_scale=100.0.
- If fewer than 5 forward return rows remain, truncate gracefully instead of crashing.
- Keep info dictionaries scalar-only unless record_arrays_in_info=True.
- Advance current_data_idx by 5 and current_step by 1.
- truncated is true at current_step >= 52.
- Return observation, reward, terminated, truncated, info.
- Write tests proving row i reward uses returns i+1:i+6, not i:i+5.
```

### Good prompt 5.5 — Dummy random-agent environment smoke test

```text
Implement scripts/smoke_test_env.py and tests/integration/test_dummy_agent_env.py.

Requirements:
- Initialize PortfolioEnv using the train split and env.yaml.
- Sample random actions with np.random.uniform(-1.0, 1.0, size=n_assets).
- Run the environment for up to 52 agent steps.
- Track final NAV, cumulative turnover, final date, and whether truncation occurred.
- Assert observations, rewards, turnover, and NAV are finite.
- Assert the loop reaches 52 steps on a full training episode.
- Do not implement baselines or PPO in this PR.
```

### Good prompt 6 — Baselines

```text
Implement baseline policies and deterministic backtest engine.

Requirements:
- Equal-weight weekly rebalance.
- Equal-weight buy-and-hold.
- SPY-only.
- SHY-only.
- Inverse-volatility.
- Use the same transaction cost and drift functions as PortfolioEnv.
- Output nav, weights, trades, costs, and metrics.
- Write a manual two-asset test to verify NAV and turnover.
```

### Good prompt 7 — PPO training

```text
Implement PPO training harness.

Requirements:
- Load configs/data.yaml, configs/env.yaml, configs/train_ppo.yaml.
- Load PortfolioDataset and create train/validation feature stores.
- Build train env with RandomWindowEpisodeSampler.
- Build validation env/backtest for periodic evaluation.
- Use Stable-Baselines3 PPO.
- Save model.zip, configs, feature_spec, hashes, validation metrics, and W&B logs.
- Add a 1,000-timestep smoke test.
```

---

## 18. Common Failure Modes and How to Detect Them

### Failure mode 1 — Using same-row returns for reward

Symptom:

```text
validation performance looks unrealistically good
```

Detection:

```text
test_step_uses_forward_returns_not_same_row_returns
```

Fix:

```python
returns.iloc[i + 1 : i + 1 + horizon]
```

### Failure mode 2 — Static weight slice used as current weights

Symptom:

```text
agent observation always says portfolio is equal weight
```

Fix:

```python
observation = concat(market_features, current_drifted_weights)
```

### Failure mode 3 — Cost calculated against previous target weights

Symptom:

```text
turnover and cost are understated after large market moves
```

Fix:

```python
turnover = abs(new_target_weights - drifted_current_weights).sum()
```

### Failure mode 4 — Treating log returns as simple returns

Symptom:

```text
NAV drift inconsistent with manual calculation
```

Fix:

```python
asset_gross = np.exp(log_returns)
```

### Failure mode 5 — Validation episodes randomized

Symptom:

```text
validation metrics change across repeated evaluation runs
```

Fix:

```text
randomize train only; validation/test deterministic
```

### Failure mode 6 — PPO underperforms equal weight

Do not immediately tune hyperparameters.

Check, in order:

```text
1. return alignment
2. dynamic current-weight observation
3. transaction cost math
4. reward scale
5. action temperature
6. baseline engine correctness
7. PPO hyperparameters
```


### Failure mode 7 — Observation destroyed by VecNormalize

Symptom:

```text
policy behavior becomes unstable or blind despite clean Phase 1 features
```

Cause:

```text
VecNormalize(norm_obs=True) rescales already-normalized market features and destroys current-weight semantics.
```

Fix:

```python
VecNormalize(vec_env, norm_obs=False, norm_reward=True)  # optional reward-only experiment
```

or no `VecNormalize` wrapper for the preferred v1 path.

### Failure mode 8 — Training slows due to large info dictionaries

Symptom:

```text
SubprocVecEnv training is CPU/IPC-bound and RAM usage grows quickly
```

Cause:

```text
large NumPy arrays are returned in info at every step and pickled across worker processes
```

Fix:

```yaml
record_arrays_in_info: false
```

Use rich arrays only in deterministic backtest/reporting runs.

### Failure mode 9 — Validation crashes near split end

Symptom:

```text
IndexError, shape mismatch, or empty return arrays near the end of validation/test
```

Cause:

```text
get_forward_log_returns requests a full 5-day window when fewer than 5 rows remain
```

Fix:

```text
truncate the episode early when len(forward_log_returns) < rebalance_frequency_trading_days
```

---

## 19. Definition of Done for Phase 2

Phase 2 is complete only when all of the following are true:

```text
1. PortfolioDataset and PortfolioFeatureStore are implemented and tested.
2. PortfolioEnv passes Gymnasium env checks.
3. Two-clock weekly stepping is tested.
4. The environment uses dynamic current weights, not static Phase 1 weight columns.
5. Forward reward rows are i+1:i+6 for a decision at row i.
6. Transaction costs are charged against drifted current weights.
7. Equal-weight, buy-and-hold, SPY-only, SHY-only, and inverse-vol baselines are implemented.
8. Dummy random-agent environment smoke test completes 52 steps and prints final NAV.
9. One common backtest engine evaluates both baselines and learned policies.
10. `VecNormalize(norm_obs=True)` is not used anywhere in the training path.
11. Training `info` dictionaries are scalar-only unless `record_arrays_in_info=True`.
12. PPO smoke test runs successfully.
13. Full PPO training produces a saved experiment bundle.
14. W&B logs configs, metrics, and artifacts.
15. Validation report compares PPO against baselines after costs.
16. Test backtest is run only once for the final selected model.
```

---

## 20. Recommended Pull Request Plan

Use small PRs.

```text
PR 1: Phase 2 data interface
PR 2: Financial mechanics modules and tests
PR 3: Episode sampler
PR 4: PortfolioEnv reset and observation
PR 5: PortfolioEnv step and two-clock tests
PR 5.5: Dummy random-agent environment smoke test
PR 6: Baseline policies and backtest engine
PR 7: Evaluation metrics and reports
PR 8: PPO training harness and W&B
PR 9: Full validation run and model card
```

Each PR should include tests. No PR should contain both environment mechanics and PPO training code. Follow this PR sequence strictly. Do not start PR 6 until the dummy-agent smoke test in PR 5.5 runs 52 steps, reports finite turnover and final NAV, and exits cleanly.

---

## 21. Final Guidance to the Team

The fastest way to fail Phase 2 is to start training too early.

The correct order is:

```text
data contract → financial mechanics → environment tests → dummy random-agent smoke test → baselines → PPO smoke test → full PPO training
```

The RL agent is the last component, not the first.

A clean Phase 2 system should make the following statement true:

```text
If PPO underperforms, we can tell whether the problem is data, environment mechanics, reward design, baseline comparison, or policy learning.
```

That is what makes the system institutional-grade rather than a fragile research notebook.
