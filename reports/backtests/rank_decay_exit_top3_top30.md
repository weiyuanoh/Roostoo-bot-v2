# Backtest: rank-decay exit for momentum-only

Run date: 2026-06-18

## Experiment

Baseline uses the current momentum-only score with TP/SL exits. Candidate uses the same score, sizing, entries, fees, and slippage, but also exits a held pair when its current score rank is worse than the configured threshold.

```text
score[i,t] = beta_momentum * z_momentum[i,t]

if held_pair_rank[t] > 3:
    exit position
```

The rank-decay exit is evaluated before TP/SL. Same-cycle re-entry of the exiting pair is blocked by the existing held-pair entry logic.

## Parameters

- Universe name: `fixed_top30_median_dollar_volume`
- Universe size: `30`
- Universe selection: top 30 by median hourly dollar volume from feature file; fixed before backtest
- Pairs: `BTC/USD, ETH/USD, SOL/USD, XRP/USD, ZEC/USD, BNB/USD, DOGE/USD, SUI/USD, TRX/USD, ADA/USD, PEPE/USD, PAXG/USD, LINK/USD, TAO/USD, AVAX/USD, NEAR/USD, LTC/USD, ENA/USD, UNI/USD, WLD/USD, AAVE/USD, HBAR/USD, FET/USD, FIL/USD, TRUMP/USD, TON/USD, DOT/USD, ICP/USD, APT/USD, VIRTUAL/USD`
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
- Rank exit threshold: `3`

## Comparison

| Run | Slippage | Return | Max DD | Ret/DD | Sharpe | Sortino | Win | Stop | TP | Rank exits | Median hold | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 15.18% | 12.51% | 1.214 | 1.531 | 2.679 | 41.46% | 58.04% | 41.21% | 0 | 5.00h | 796 |
| candidate | 0 | -12.59% | 17.16% | -0.734 | -1.462 | -2.650 | 37.11% | 21.25% | 19.83% | 414 | 3.00h | 1412 |
| baseline | 5 | 0.14% | 12.78% | 0.011 | 0.029 | 0.050 | 39.56% | 59.71% | 39.56% | 0 | 5.00h | 814 |
| candidate | 5 | -30.07% | 32.05% | -0.938 | -3.778 | -6.838 | 33.99% | 22.14% | 19.18% | 414 | 3.00h | 1418 |

## Pair attribution

### baseline slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| TON/USD | 45 | 57.78% | 1.51% | 3.01% | 176,394 |
| ZEC/USD | 40 | 50.00% | 0.84% | 1.43% | 89,173 |
| NEAR/USD | 40 | 47.50% | 0.67% | -1.63% | 70,682 |
| TRX/USD | 20 | 50.00% | 0.66% | -0.80% | 32,343 |
| BNB/USD | 11 | 54.55% | 0.97% | 2.90% | 25,434 |
| DOGE/USD | 10 | 50.00% | 0.78% | 0.68% | 20,066 |
| SOL/USD | 5 | 60.00% | 1.24% | 3.05% | 15,163 |
| HBAR/USD | 8 | 37.50% | 0.49% | -1.81% | 14,421 |
| AAVE/USD | 8 | 50.00% | 0.76% | 0.59% | 13,556 |
| FET/USD | 8 | 50.00% | 0.70% | 0.19% | 12,349 |
| WLD/USD | 34 | 41.18% | 0.12% | -1.74% | 9,576 |
| ADA/USD | 1 | 100.00% | 2.97% | 2.97% | 7,959 |
| LINK/USD | 4 | 50.00% | 0.50% | 0.48% | 5,166 |
| ETH/USD | 4 | 50.00% | 0.40% | 0.38% | 3,921 |
| TAO/USD | 32 | 43.75% | 0.05% | -1.63% | 3,374 |
| DOT/USD | 8 | 37.50% | -0.02% | -1.80% | -1,500 |
| APT/USD | 10 | 30.00% | -0.04% | -1.71% | -2,040 |
| XRP/USD | 1 | 0.00% | -1.73% | -1.73% | -4,056 |
| TRUMP/USD | 17 | 35.29% | -0.10% | -1.76% | -4,520 |
| PEPE/USD | 3 | 33.33% | -0.66% | -2.52% | -4,790 |
| AVAX/USD | 1 | 0.00% | -2.27% | -2.27% | -5,123 |
| VIRTUAL/USD | 2 | 0.00% | -2.66% | -2.66% | -13,931 |
| BTC/USD | 12 | 25.00% | -0.48% | -1.75% | -16,956 |
| FIL/USD | 5 | 20.00% | -1.42% | -2.10% | -17,752 |
| SUI/USD | 7 | 28.57% | -0.89% | -2.28% | -18,148 |
| LTC/USD | 5 | 0.00% | -2.04% | -2.02% | -25,414 |
| PAXG/USD | 22 | 27.27% | -0.43% | -1.69% | -26,640 |
| UNI/USD | 7 | 14.29% | -1.52% | -1.81% | -27,399 |
| ENA/USD | 11 | 18.18% | -1.42% | -1.81% | -38,765 |
| ICP/USD | 17 | 23.53% | -0.75% | -1.86% | -40,322 |

### candidate slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| TON/USD | 66 | 46.97% | 0.77% | -0.03% | 113,752 |
| NEAR/USD | 56 | 42.86% | 0.63% | -0.57% | 76,069 |
| TAO/USD | 48 | 54.17% | 0.67% | 0.13% | 74,359 |
| ZEC/USD | 71 | 42.25% | 0.44% | -0.29% | 65,550 |
| FET/USD | 15 | 46.67% | 0.81% | -0.79% | 30,462 |
| HBAR/USD | 17 | 41.18% | 0.48% | -0.10% | 15,397 |
| DOT/USD | 18 | 38.89% | 0.38% | -0.88% | 13,961 |
| SOL/USD | 7 | 57.14% | 0.04% | 0.16% | 660 |
| BNB/USD | 18 | 33.33% | 0.00% | -0.32% | -516 |
| WLD/USD | 37 | 40.54% | 0.01% | -1.75% | -1,324 |
| DOGE/USD | 17 | 29.41% | -0.04% | -0.64% | -2,082 |
| AVAX/USD | 3 | 33.33% | -0.64% | -0.86% | -4,312 |
| ADA/USD | 2 | 0.00% | -0.96% | -0.96% | -4,425 |
| AAVE/USD | 19 | 31.58% | -0.17% | -0.28% | -6,864 |
| PEPE/USD | 8 | 25.00% | -0.37% | -0.73% | -7,190 |
| LINK/USD | 7 | 28.57% | -0.62% | -0.42% | -9,971 |
| ETH/USD | 13 | 30.77% | -0.35% | -0.55% | -10,598 |
| XRP/USD | 6 | 33.33% | -0.82% | -0.61% | -11,708 |
| TRX/USD | 63 | 39.68% | -0.08% | -0.10% | -12,724 |
| APT/USD | 17 | 17.65% | -0.34% | -0.48% | -12,799 |
| LTC/USD | 11 | 45.45% | -0.52% | -0.17% | -13,010 |
| BTC/USD | 19 | 26.32% | -0.32% | -0.24% | -13,982 |
| PAXG/USD | 50 | 28.00% | -0.15% | -0.23% | -17,791 |
| FIL/USD | 12 | 33.33% | -0.85% | -1.06% | -23,928 |
| TRUMP/USD | 26 | 30.77% | -0.41% | -0.98% | -24,511 |
| UNI/USD | 12 | 16.67% | -0.97% | -0.57% | -27,021 |
| SUI/USD | 12 | 33.33% | -1.04% | -1.58% | -29,189 |
| VIRTUAL/USD | 6 | 0.00% | -2.19% | -1.74% | -30,481 |
| ICP/USD | 27 | 25.93% | -0.61% | -1.61% | -36,408 |
| ENA/USD | 23 | 26.09% | -1.04% | -1.71% | -55,975 |

### baseline slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| TON/USD | 43 | 55.81% | 1.32% | 2.91% | 137,401 |
| NEAR/USD | 41 | 48.78% | 0.84% | -1.65% | 82,775 |
| ZEC/USD | 41 | 48.78% | 0.69% | -0.20% | 68,674 |
| TRX/USD | 21 | 42.86% | 0.49% | -1.67% | 24,218 |
| AAVE/USD | 8 | 50.00% | 1.23% | 0.78% | 21,761 |
| BNB/USD | 12 | 50.00% | 0.78% | 0.59% | 20,708 |
| DOGE/USD | 10 | 50.00% | 0.68% | 0.58% | 16,024 |
| SOL/USD | 5 | 60.00% | 1.14% | 2.94% | 13,422 |
| ADA/USD | 1 | 100.00% | 2.86% | 2.86% | 7,284 |
| LINK/USD | 4 | 50.00% | 0.54% | 0.60% | 5,602 |
| ETH/USD | 4 | 50.00% | 0.36% | 0.28% | 3,552 |
| WLD/USD | 34 | 41.18% | 0.04% | -1.81% | 1,374 |
| DOT/USD | 8 | 37.50% | -0.12% | -1.90% | -3,208 |
| XRP/USD | 1 | 0.00% | -1.82% | -1.82% | -4,192 |
| AVAX/USD | 1 | 0.00% | -2.37% | -2.37% | -5,313 |
| PEPE/USD | 3 | 33.33% | -0.76% | -2.61% | -5,491 |
| TRUMP/USD | 17 | 35.29% | -0.17% | -1.73% | -7,431 |
| APT/USD | 11 | 27.27% | -0.32% | -1.74% | -9,122 |
| TAO/USD | 33 | 42.42% | -0.12% | -1.73% | -9,667 |
| BTC/USD | 14 | 28.57% | -0.38% | -1.70% | -14,912 |
| HBAR/USD | 10 | 20.00% | -0.67% | -1.91% | -15,218 |
| FET/USD | 8 | 25.00% | -0.74% | -2.25% | -15,755 |
| SUI/USD | 7 | 28.57% | -0.99% | -2.37% | -18,130 |
| LTC/USD | 4 | 0.00% | -2.01% | -1.92% | -19,162 |
| VIRTUAL/USD | 3 | 0.00% | -2.89% | -3.14% | -21,497 |
| FIL/USD | 6 | 16.67% | -1.59% | -2.07% | -22,705 |
| UNI/USD | 7 | 14.29% | -1.48% | -1.89% | -25,209 |
| PAXG/USD | 22 | 27.27% | -0.52% | -1.78% | -28,849 |
| ICP/USD | 17 | 23.53% | -0.85% | -1.96% | -39,107 |
| ENA/USD | 11 | 18.18% | -1.52% | -1.91% | -40,049 |

### candidate slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| TON/USD | 65 | 43.08% | 0.55% | -0.20% | 69,212 |
| NEAR/USD | 56 | 42.86% | 0.63% | -0.66% | 67,021 |
| TAO/USD | 50 | 52.00% | 0.47% | 0.01% | 50,901 |
| ZEC/USD | 71 | 42.25% | 0.34% | -0.39% | 42,430 |
| FET/USD | 15 | 40.00% | 0.51% | -0.89% | 21,182 |
| DOT/USD | 18 | 38.89% | 0.28% | -0.98% | 9,921 |
| SOL/USD | 7 | 57.14% | -0.06% | 0.06% | -462 |
| ADA/USD | 2 | 0.00% | -1.06% | -1.06% | -4,303 |
| AVAX/USD | 3 | 33.33% | -0.73% | -0.96% | -4,658 |
| BNB/USD | 18 | 27.78% | -0.10% | -0.42% | -4,910 |
| DOGE/USD | 17 | 29.41% | -0.14% | -0.74% | -5,878 |
| WLD/USD | 37 | 40.54% | -0.07% | -1.75% | -6,295 |
| HBAR/USD | 18 | 33.33% | -0.08% | -0.35% | -7,513 |
| PEPE/USD | 8 | 25.00% | -0.47% | -0.83% | -8,659 |
| LINK/USD | 7 | 28.57% | -0.72% | -0.52% | -9,803 |
| AAVE/USD | 19 | 31.58% | -0.27% | -0.38% | -10,464 |
| XRP/USD | 6 | 33.33% | -0.92% | -0.71% | -11,855 |
| ETH/USD | 13 | 15.38% | -0.45% | -0.65% | -12,715 |
| LTC/USD | 11 | 18.18% | -0.62% | -0.27% | -13,859 |
| BTC/USD | 19 | 21.05% | -0.42% | -0.34% | -16,416 |
| APT/USD | 18 | 11.11% | -0.49% | -0.64% | -19,131 |
| FIL/USD | 12 | 33.33% | -0.95% | -1.16% | -24,466 |
| TRX/USD | 63 | 33.33% | -0.18% | -0.20% | -24,996 |
| PAXG/USD | 50 | 26.00% | -0.25% | -0.33% | -26,438 |
| SUI/USD | 12 | 33.33% | -1.14% | -1.68% | -27,638 |
| TRUMP/USD | 26 | 23.08% | -0.51% | -1.08% | -27,652 |
| UNI/USD | 12 | 16.67% | -1.07% | -0.67% | -28,334 |
| VIRTUAL/USD | 6 | 0.00% | -2.29% | -1.84% | -28,355 |
| ICP/USD | 27 | 22.22% | -0.71% | -1.71% | -36,382 |
| ENA/USD | 23 | 26.09% | -1.14% | -1.81% | -56,530 |

## Artifacts

- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip0_equity.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip0_trades.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip0_metadata.json`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip0_equity.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip0_trades.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip0_metadata.json`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip5_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip5_equity.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip5_trades.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip5_metadata.json`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip5_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip5_equity.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip5_trades.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip5_metadata.json`
