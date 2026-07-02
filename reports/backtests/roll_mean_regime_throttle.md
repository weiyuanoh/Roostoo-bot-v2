# Backtest: universe roll-impact regime throttle

Run date: 2026-06-17

## Hypothesis

High universe-wide `roll_impact` is a liquidity-stress regime. Momentum entries opened during this regime should have worse realised trade outcomes. Blocking new entries during this regime should improve the momentum-only baseline.

## Baseline score math

```text
score[i,t] = beta_momentum * z_momentum[i,t]
```

## Regime math

```text
market_roll_impact[t] = mean_i(roll_impact[i,t])

threshold[t] = 80th percentile of prior 720 hourly market_roll_impact values

is_stressed[t] = market_roll_impact[t] >= threshold[t]
```

The threshold uses only prior bars. If history is insufficient, `is_stressed = False`.

## Trading rule

```text
if is_stressed[t]:
    no new entries
else:
    trade normally
```

Existing positions still exit normally through TP/SL.

## Parameters

- Pairs: `BNB/USD, BTC/USD, ETH/USD, SOL/USD, TAO/USD, TRX/USD, XRP/USD, ZEC/USD`
- Model: `momentum_only`
- Horizon: `24`
- IS/OS: `4m/4m`
- Top K: `1`
- Max new entries: `1`
- Max positions: `3`
- Position fraction: `0.25`
- TP/SL: `0.03` / `0.015`
- Fee bps: `10.0`
- Slippage bps: `0,5`
- Regime aggregation: `mean`

## Comparison

| Run | Slippage | Return | Max DD | Ret/DD | Sharpe | Sortino | Win | Stop | TP | Avg stopped loss | Median hold | Trades | Stressed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 4.94% | 14.66% | 0.337 | 0.421 | 0.873 | 40.23% | 59.48% | 39.65% | -2.21% | 7.00h | 686 | 0.00% |
| candidate | 0 | -0.67% | 15.09% | -0.044 | -0.210 | -0.410 | 38.96% | 60.74% | 38.34% | -2.20% | 7.00h | 652 | 19.93% |
| baseline | 5 | -4.14% | 15.28% | -0.271 | -0.576 | -1.185 | 39.26% | 60.46% | 38.68% | -2.24% | 6.50h | 698 | 0.00% |
| candidate | 5 | -10.32% | 15.77% | -0.654 | -1.434 | -2.726 | 38.23% | 61.47% | 37.61% | -2.24% | 6.00h | 654 | 19.93% |

## Pair attribution

### baseline slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 111 | 44.14% | 0.54% | -1.72% | 154,536 |
| TRX/USD | 23 | 56.52% | 0.94% | 2.90% | 53,190 |
| TAO/USD | 77 | 41.56% | 0.15% | -1.71% | 21,496 |
| BTC/USD | 27 | 44.44% | 0.35% | -1.74% | 17,581 |
| SOL/USD | 21 | 38.10% | 0.15% | -1.64% | 6,865 |
| BNB/USD | 33 | 39.39% | 0.08% | -1.68% | 5,134 |
| ETH/USD | 28 | 25.00% | -0.82% | -1.85% | -58,997 |
| XRP/USD | 23 | 17.39% | -1.13% | -1.80% | -64,207 |

### candidate slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 102 | 43.14% | 0.49% | -1.70% | 125,273 |
| TRX/USD | 23 | 56.52% | 0.92% | 2.90% | 51,656 |
| TAO/USD | 76 | 40.79% | 0.11% | -1.69% | 15,489 |
| SOL/USD | 20 | 40.00% | 0.25% | -1.65% | 11,443 |
| BTC/USD | 26 | 38.46% | 0.05% | -1.74% | -2,377 |
| BNB/USD | 32 | 37.50% | -0.05% | -1.68% | -5,428 |
| XRP/USD | 20 | 15.00% | -1.23% | -1.78% | -59,437 |
| ETH/USD | 27 | 22.22% | -0.92% | -1.84% | -62,851 |

### baseline slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 111 | 43.24% | 0.42% | -1.71% | 110,221 |
| TRX/USD | 25 | 52.00% | 0.67% | 0.82% | 39,959 |
| SOL/USD | 21 | 42.86% | 0.31% | -1.72% | 14,872 |
| BTC/USD | 27 | 44.44% | 0.28% | -1.71% | 14,591 |
| BNB/USD | 35 | 37.14% | -0.08% | -1.75% | -8,383 |
| TAO/USD | 78 | 39.74% | -0.06% | -1.76% | -14,232 |
| XRP/USD | 25 | 20.00% | -0.82% | -1.81% | -49,836 |
| ETH/USD | 27 | 22.22% | -1.00% | -1.76% | -64,948 |

### candidate slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 99 | 41.41% | 0.23% | -1.71% | 51,975 |
| TRX/USD | 25 | 52.00% | 0.64% | 0.35% | 37,940 |
| SOL/USD | 20 | 40.00% | 0.20% | -1.73% | 8,352 |
| BNB/USD | 34 | 38.24% | -0.02% | -1.75% | -3,968 |
| TAO/USD | 75 | 40.00% | -0.04% | -1.74% | -10,645 |
| BTC/USD | 26 | 34.62% | -0.21% | -1.83% | -15,485 |
| XRP/USD | 23 | 21.74% | -0.77% | -1.82% | -42,721 |
| ETH/USD | 25 | 24.00% | -0.88% | -1.76% | -52,086 |

## Artifacts

- `reports/backtests/roll_mean_regime_throttle_baseline_slip0_summary.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip0_equity.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip0_trades.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip0_metadata.json`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip0_summary.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip0_equity.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip0_trades.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip0_metadata.json`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip5_summary.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip5_equity.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip5_trades.csv`
- `reports/backtests/roll_mean_regime_throttle_baseline_slip5_metadata.json`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip5_summary.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip5_equity.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip5_trades.csv`
- `reports/backtests/roll_mean_regime_throttle_candidate_slip5_metadata.json`
