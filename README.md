# Roostoo Bot v2

Research and live-trading environment for a long-only crypto spot bot that:

- executes through the Roostoo mock exchange API
- collects live market data from Binance public spot klines
- centralizes feature, scoring, risk, and execution infrastructure so notebooks, backtests, and live trading do not drift

## Current Live Deployment

Research is paused. The current live policy is the V1 fixed-regime deployment:
train one ridge artifact on the latest 3 months, train one learned cluster
regime gate on the latest 24 months, then trade live from those saved artifacts
without automatic refit promotion.

The canonical feature file is:

```text
data/features/live_1h_features_30m.csv
```

V1 freezes ridge and regime artifacts:

```text
ridge train:  T - 3 months  -> T
regime train: T - 24 months -> T
live trade:   use saved artifacts until explicitly replaced
```

The rejected scheduled-refit candidate was:

```text
ridge refit:  every 7 days, using refit_time - 3 months -> refit_time
regime refit: every 1 month, using refit_time - 24 months -> refit_time
deploy:       immediately replace active artifacts after each refit
```

The 10-fold walk-forward panel showed that scheduled refitting traded more but
did not improve risk-adjusted results. V1 mostly won by staying out of bad
regimes, so live deployment should prioritize capital preservation and avoid
automatic model replacement.

Rank-decay exits and direct Roll-impact alpha overlays did not beat the plain
TP/SL momentum baseline in prior short-sample studies, so they remain disabled
unless a future 30-month OOS test shows a clear improvement after costs.

```text
universe = fixed_25_full_history_pairs
model = momentum_only
score = beta_momentum * z_momentum
horizon = 24
top_k = 1
max_new_entries = 1
max_positions = 3
position_fraction = 0.25
take_profit = 0.03
stop_loss = 0.015
exits = TP/SL only
```

The fixed deployment universe is the 25-pair full-history set used by the
strict walk-forward panel:

```text
AAVE/USD, ADA/USD, APT/USD, AVAX/USD, BNB/USD, BTC/USD, DOGE/USD,
DOT/USD, ETH/USD, FET/USD, FIL/USD, HBAR/USD, ICP/USD, LINK/USD,
LTC/USD, NEAR/USD, PAXG/USD, PEPE/USD, SOL/USD, SUI/USD, TRX/USD,
UNI/USD, WLD/USD, XRP/USD, ZEC/USD
```

In CLI commands, `--pairs all` means this deployment universe.

Generated backtest artifacts are intentionally local and disposable. Stale
short-sample CSV outputs should not be used for the current V1/V2 comparison.

## Research Log

The original hypothesis was:

> Market microstructure statistics may directly predict cross-sectional rankings of forward crypto returns.

The evidence from initial 1h-candle IC checks was weak. Microstructure measures, including Roll measure, Roll impact, VPIN-like measures, and related liquidity proxies, did not show strong standalone predictive power for 1h, 6h, or 24h forward returns.

Prior live-style portfolio backtests also failed to validate microstructure as a linear alpha overlay. Roll-aware models sometimes improved in-sample IC, but did not improve traded out-of-sample performance after portfolio rules and costs.

The working alpha hypothesis has therefore evolved:

> Recent momentum is the primary alpha. Microstructure, especially Roll impact / low-liquidity stress, is secondary execution-quality or risk context rather than a direct alpha.

The research baseline became:

```text
baseline = momentum_only
score = beta_momentum * z_momentum
```

Any microstructure layer must improve this baseline out-of-sample after fees and realistic slippage, preferably by reducing drawdown, stop-hit rate, bad pair exposure, left-tail losses, or realized slippage without destroying return.

The shared scoring code still supports these model specifications for research:

- `momentum_only`
- `momentum_plus_roll`
- `momentum_roll_interaction`
- `momentum_plus_roll_plus_interaction`

The deployment profile uses:

```text
momentum_only
```

The code defaults, `.env.example`, `train-live-models`, and live commands are
aligned to this deployment profile. Environment variables can still override
them, so check `.env` before starting a live process.

The combined score uses:

```text
z_momentum
z_low_roll_impact
z_momentum_x_low_roll_impact
```

## What Has Been Built

### Data

- Binance public kline client in `bot/binance_data.py`
- Roostoo pair to Binance symbol mapping in `bot/config.py`
- Optional CSV candle persistence for backtesting / research only
- Live trading fetches Binance data directly

### Features And Execution Quality

- Microstructure / execution-quality measures in `bot/microstructure.py`
- Alpha features in `bot/features.py`
- Forward return and IC helpers in `bot/forward_ic.py`
- Shared ridge score implementation in `bot/strategy/ridge.py`

### Research Notebooks

Research artifacts are grouped under:

```text
notebooks/feature/
notebooks/microstructure/
```

Existing checks include:

- archival microstructure IC checks on 1h candles
- feature IC checks on 1h candles
- momentum + Roll impact risk/execution-quality filter checks
- ridge score walk-forward checks
- portfolio backtest outputs

### Live Trading

Live trading is implemented in:

```text
bot/live_trader.py
bot/main.py
bot/live_state.py
bot/executor.py
bot/telemetry.py
bot/monitoring.py
```

Live behavior:

- dry-run by default
- `--execute` required for real Roostoo orders
- waits until next UTC hour + delay in continuous mode
- uses market orders
- long-only spot
- local state file stores entry price, entry score, quantity, and order id
- exits happen only through TP/SL unless manual liquidation is requested
- every cycle logs score snapshots and order attribution for local monitoring

Local live state:

```text
data/live_state.json
```

This file is intentionally ignored by git.

Live cycle flow when saved V1 artifacts are used:

```text
1. Fetch recent Binance 1h candles for the configured pair universe.
2. Compute microstructure, alpha, and ridge score features.
3. Load the saved ridge and cluster-regime artifacts from `data/models`.
4. Fetch Roostoo wallet balances and ticker prices.
5. Reconcile local live state against the Roostoo wallet.
6. Build exit and entry intents with the shared strategy logic.
7. Log the full score/rank snapshot for the cycle.
8. If dry-run, log intents only and place no orders.
9. If --execute, place market sell exits first, then market buy entries.
10. Update local live state after successful orders.
11. Write cycle, score, order, closed-trade, and monitor logs.
```

### Live Monitoring

Local monitoring is implemented in:

```text
bot/telemetry.py
bot/monitoring.py
```

Monitoring is self-contained and file-based. It does not change trading
decisions. The event format is structured so Telegram or another alert
transport can be added later without changing the live-trading telemetry.

Raw monitoring logs:

```text
logs/live_cycles.jsonl      cycle-level portfolio, config, intents, and orders
logs/live_scores.jsonl      per-pair score/rank snapshots for each cycle
logs/trades.jsonl           order/fill telemetry and realized slippage
logs/closed_trades.jsonl    closed-trade attribution for successful exits
logs/monitor_events.jsonl   health/risk findings, local-only for now
```

Generated monitoring reports:

```text
reports/live_monitoring/summary.csv
reports/live_monitoring/pair_attribution.csv
reports/live_monitoring/exit_reason_attribution.csv
reports/live_monitoring/slippage.csv
reports/live_monitoring/forward_ic.csv
reports/live_monitoring/health.json
reports/live_monitoring/regime_snapshot.csv
reports/live_monitoring/rank_persistence.csv
reports/live_monitoring/score_gap_report.csv
reports/live_monitoring/same_pair_reentry.csv
reports/live_monitoring/post_exit_returns.csv
```

### Backtest Infrastructure

The backtest was refactored to better mimic live trading:

- shared cycle intent builder in `bot/strategy/ridge.py`
- Roostoo-like simulated executor in `bot/backtest/simulated_executor.py`
- portfolio accounting in `bot/backtest/portfolio.py`
- generic rolling portfolio backtest in `bot/backtest/ridge_score_portfolio.py`
- fixed-vs-scheduled OOS refit comparison in `bot/backtest/refit_policy_experiment.py`

Backtest validity improvements:

- live and backtest use the same entry / exit intent logic
- supports asymmetric `--tp` and `--sl`
- supports `--top-k`, `--max-new-entries`, and `--max-positions`
- simulated executor applies amount precision
- simulated executor enforces minimum order notional
- simulated executor supports configurable slippage
- exits and entries are decided before portfolio mutation, matching live behavior

Known remaining mismatch:

- backtest fills against candle prices
- live fills against Roostoo market liquidity at runtime
- spread and realized market-order slippage are approximated only with `--slippage-bps`

Backtest flow:

```text
1. Load the feature CSV, usually data/features/live_1h_features_30m.csv.
2. Recompute ridge signal columns and forward-return targets.
3. Filter to the requested pair universe.
4. Build rolling IS/OS folds, or use `refit_policy_experiment` for the fixed V1 versus scheduled V2 deployment-style comparison.
5. For each fold or refit segment, train ridge only on data available before that segment and score the OOS window.
6. Step through OS candles one hourly bar at a time.
7. Mark current portfolio value from candle close prices.
8. Build exit and entry intents with the same shared strategy logic used live.
9. Simulate sells first, then buys, using Roostoo-like precision/min-order rules.
10. Apply configured fees and slippage.
11. Record equity, trades, closed-trade reasons, pair attribution, and fold summary.
```

## Setup

```bash
cp .env.example .env
poetry install
```

Fill in `.env`:

```text
ROOSTOO_API_KEY=...
ROOSTOO_API_SECRET=...
ROOSTOO_BASE_URL=https://mock-api.roostoo.com
```

Do not commit `.env`.

## Basic Commands

Check public Roostoo and Binance connectivity:

```bash
.venv/bin/python -m bot.main smoke
```

Check signed Roostoo balance:

```bash
.venv/bin/python -m bot.main balance
```

Collect candles to local CSV for backtesting:

```bash
.venv/bin/python -m bot.main collect \
  --pairs BTC/USD,ETH/USD,SOL/USD \
  --interval 1h \
  --limit 1000
```

## Live Trading Commands

Deployment pair universe:

```bash
PAIRS=all
```

Dry-run one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs "$PAIRS" \
  --model-dir data/models \
  --cluster-regime-gate
```

Execute one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs "$PAIRS" \
  --model-dir data/models \
  --cluster-regime-gate \
  --execute
```

Run continuously on the hour:

```bash
.venv/bin/python -m bot.main live \
  --pairs "$PAIRS" \
  --model-dir data/models \
  --cluster-regime-gate \
  --execute
```

Meaning of the main risk knobs:

```text
--top-k 1             only the highest-ranked candidate is eligible
--max-new-entries 1   open at most one new position per cycle
--max-positions 3     hold at most three total positions
--position-fraction   fraction of portfolio value allocated per new trade
--tp                  take-profit return threshold
--sl                  stop-loss return threshold
```

Example risk math:

```text
position_fraction = 0.25
sl = 0.015
portfolio loss at stop ~= 0.25 * 0.015 = 0.375% per stopped position
```

Fees, slippage, and hourly stop checks can make realized loss worse.

## Liquidation

Dry-run liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs "$PAIRS"
```

Execute liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs "$PAIRS" \
  --cancel-pending \
  --execute
```

Liquidation sells only free non-USD balances. Locked balances are reserved by pending orders unless pending orders are cancelled first.

## Backtesting

Run the current deployment baseline backtest:

```bash
.venv/bin/python -m bot.backtest.rank_decay_exit_experiment
```

This writes the older TP/SL baseline plus the rejected rank-decay
candidate for comparison.

Run the generic live-like portfolio backtest manually:

```bash
.venv/bin/python -m bot.backtest.ridge_score_portfolio \
  --pairs "$PAIRS" \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --fee-bps 10 \
  --slippage-bps 5
```

The generic backtest writes:

```text
<prefix>_summary.csv
<prefix>_equity.csv
<prefix>_trades.csv
<prefix>_metadata.json
```

Run no-lookahead regime diagnostics on the deployment baseline backtest:

```bash
.venv/bin/python -m bot.backtest.regime_diagnostics \
  --features data/features/live_1h_features_30m.csv \
  --trades reports/backtests/<prefix>_trades.csv \
  --equity reports/backtests/<prefix>_equity.csv \
  --metadata reports/backtests/<prefix>_metadata.json \
  --output-dir reports/backtests/regime_diagnostics \
  --prefix momentum_only_slip5
```

This writes:

```text
momentum_only_slip5_entry_regime.csv
momentum_only_slip5_trade_outcome_attribution.csv
momentum_only_slip5_rank_persistence.csv
momentum_only_slip5_score_gap_report.csv
momentum_only_slip5_same_pair_reentry.csv
momentum_only_slip5_post_exit_returns.csv
momentum_only_slip5_shadow_regime_filters.csv
```

Regime diagnostics are analysis-only. Candidate filters are evaluated offline
and do not change live execution.

### Learned Regime Cluster Gate

The learned cluster gate is a default-off backtest candidate. It trains on a
larger prior historical window than the ridge IS window, clusters only
decision-time entry features, labels historically profitable clusters, and then
allows future entries only when the current top-ranked candidate looks like one
of those profitable clusters. Exits are never blocked.

Run the deployment-style momentum backtest with the learned gate:

```bash
.venv/bin/python -m bot.backtest.ridge_score_portfolio \
  --features data/features/live_1h_features_30m.csv \
  --output-dir reports/backtests/cluster_regime \
  --prefix momentum_only_cluster_gate_slip5 \
  --pairs "$PAIRS" \
  --model momentum_only \
  --is-months 3 \
  --os-months 2 \
  --step-months 1 \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --fee-bps 10 \
  --slippage-bps 5 \
  --cluster-regime-gate \
  --cluster-lookback-months 24 \
  --cluster-n-clusters 4 \
  --cluster-min-trades 50
```

Or run baseline and learned-gate variants together:

```bash
.venv/bin/python -m bot.backtest.cluster_regime_gate_experiment \
  --features data/features/live_1h_features_30m.csv \
  --output-dir reports/backtests/cluster_regime \
  --prefix momentum_only_cluster_gate \
  --pairs "$PAIRS"
```

Additional reports are written when the gate is enabled:

```text
<prefix>_cluster_gate_decisions.csv
<prefix>_cluster_gate_summary.csv
<prefix>_cluster_gate_profiles.csv
```

Live execution uses this gate only when `--cluster-regime-gate` is passed with
a `--model-dir` containing saved artifacts. The current startup fetch remains
limited to `LIVE_HISTORY_LIMIT=1000` hourly candles, so production-style usage
should train artifacts from local paginated history first.

## Operational Notes

### Fixed V1 Artifacts

The intended live policy is explicit:

```text
ridge / alpha model:    train once from the latest 3 months of 1h data
regime cluster library: train once from the latest 24 months of 1h data
hourly live trading:    use saved artifacts; do not refit or promote automatically
```

The old `--history-limit` startup training path is still available, but it is
not the preferred path for production-style runs because Binance single kline
requests are capped at 1000 candles.

Collect a long local candle history:

```bash
.venv/bin/python -m bot.main collect-history \
  --pairs "$PAIRS" \
  --interval 1h \
  --start 2024-01-12 \
  --end 2026-07-12 \
  --output-dir data/candles
```

Build one shared feature file from the local candles:

```bash
.venv/bin/python -m bot.main build-features \
  --pairs "$PAIRS" \
  --interval 1h \
  --input-dir data/candles \
  --output data/features/live_1h_features_30m.csv
```

Run the fixed-vs-scheduled OOS refit comparison:

```bash
.venv/bin/python -m bot.backtest.refit_policy_experiment \
  --features data/features/live_1h_features_30m.csv \
  --pairs "$PAIRS" \
  --os-months 2 \
  --ridge-train-months 3 \
  --regime-train-months 24 \
  --ridge-refit-days 7 \
  --regime-refit-months 1 \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --fee-bps 10 \
  --slippage-bps 5 \
  --cluster-n-clusters 4 \
  --cluster-min-trades 50
```

For the walk-forward panel used to select V1 over scheduled refit, add:

```bash
  --folds 10 \
  --fold-step-days 7 \
  --common-history-universe \
  --workers 2
```

Train the saved live artifacts:

```bash
.venv/bin/python -m bot.main train-live-models \
  --pairs "$PAIRS" \
  --model-dir data/models
```

This writes:

```text
data/models/ridge_selection.json
data/models/cluster_regime_gate.json
```

Run live from saved artifacts:

```bash
.venv/bin/python -m bot.main live \
  --pairs "$PAIRS" \
  --model-dir data/models \
  --cluster-regime-gate \
  --execute
```

Operational cadence:

```text
before deploy: train-live-models writes ridge_selection.json and cluster_regime_gate.json
hourly:        live bot scores/trades with the saved artifacts
replacement:   refresh data and retrain only after reviewing backtest/live diagnostics
```

For live testing, use `tmux` so the bot keeps running if the terminal disconnects:

```bash
tmux new -s roostoo-bot
```

Monitor logs:

```bash
tail -f logs/live_cycles.jsonl
tail -f logs/trades.jsonl
tail -f logs/closed_trades.jsonl
```

Check wallet:

```bash
.venv/bin/python -m bot.main balance
```

Check merged wallet and local state:

```bash
.venv/bin/python -m bot.main positions --pairs "$PAIRS"
```

Check raw local state:

```bash
cat data/live_state.json
```

Generate local monitoring reports:

```bash
.venv/bin/python -m bot.main monitor-health --pairs "$PAIRS"

.venv/bin/python -m bot.main monitor-summary \
  --since-hours 168 \
  --output-dir reports/live_monitoring

.venv/bin/python -m bot.main monitor-forward \
  --since-hours 720 \
  --horizons 1,6,24 \
  --output-dir reports/live_monitoring

.venv/bin/python -m bot.main monitor-regime \
  --since-hours 168 \
  --horizons 1,3,6,24 \
  --output-dir reports/live_monitoring
```

## Tests

Run the test suite:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Current coverage includes:

- Binance client behavior
- Roostoo signing helpers
- microstructure measures
- feature engineering
- forward return / IC helpers
- live state reconciliation
- scheduler behavior
- shared strategy intent generation
- simulated executor behavior
- portfolio backtest behavior
- liquidation planning
- live telemetry and monitoring reports
- no-lookahead backtest regime diagnostics

## Next Work

Research is paused. Deployment, operations, and live diagnostics are now the priority:

1. Run the deployment baseline in dry-run mode and inspect `logs/live_cycles.jsonl`.
2. Confirm Roostoo balances, minimum order handling, and local `data/live_state.json`.
3. Execute one live cycle, then verify orders and local state reconciliation.
4. Move production operation from raw terminal / tmux to `systemd` or another process manager.
5. Review `monitor-summary`, `monitor-forward`, and `monitor-regime` after enough live cycles accumulate.
6. Add Telegram delivery for high-severity `monitor_events.jsonl` findings.
7. Resume research only after live operation and attribution are stable.

The core principle going forward is:

> Backtest and live trading should differ only by data source and executor implementation.
