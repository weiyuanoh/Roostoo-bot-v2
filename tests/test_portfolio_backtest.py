import pandas as pd
import pytest

from bot.backtest.portfolio import Portfolio, compute_equity_metrics
from bot.backtest.ridge_score_portfolio import run_portfolio_backtest


def test_portfolio_buy_sell_updates_cash_and_realized_pnl():
    portfolio = Portfolio(initial_cash=1_000.0, fee_bps=10.0)

    buy = portfolio.buy("BTC/USD", 100.0, 10.0, 1, 0.5)

    assert buy is not None
    assert portfolio.cash == pytest.approx(899.9)
    assert portfolio.positions["BTC/USD"].quantity == pytest.approx(10.0)

    sell = portfolio.sell("BTC/USD", 15.0, 2, "gain_threshold")

    assert sell is not None
    assert portfolio.cash == pytest.approx(1049.75)
    assert sell["pnl"] == pytest.approx(49.85)
    assert "BTC/USD" not in portfolio.positions


def test_compute_equity_metrics_reports_drawdown_and_return():
    metrics = compute_equity_metrics([100.0, 110.0, 90.0, 120.0], 100.0)

    assert metrics["total_return_pct"] == pytest.approx(20.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(18.1818181818)


def test_ridge_portfolio_backtest_runs_on_synthetic_hourly_frame():
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=24 * 62, freq="h", tz="UTC")
    pairs = [f"P{idx}/USD" for idx in range(8)]
    for t_idx, timestamp in enumerate(timestamps):
        for p_idx, pair in enumerate(pairs):
            base = 100 + p_idx * 10
            close = base + (t_idx * (p_idx + 1) * 0.05)
            rows.append(
                {
                    "open_time": int(timestamp.timestamp() * 1000),
                    "timestamp": timestamp,
                    "pair": pair,
                    "close": close,
                    "target_z": (p_idx - 3.5) * 0.1,
                    "z_momentum": (p_idx - 3.5) * 0.2,
                    "z_low_roll_impact": (3.5 - p_idx) * 0.1,
                    "z_momentum_x_low_roll_impact": (p_idx - 3.5) * 0.05,
                    "forward_return_1": 0.01 * (p_idx - 3.5),
                }
            )
    frame = pd.DataFrame(rows)

    summary, equity, trades = run_portfolio_backtest(
        frame,
        model="momentum_plus_roll_plus_interaction",
        horizon=1,
        is_months=1,
        os_months=1,
        step_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.25,
        exit_threshold=0.5,
        fee_bps=0.0,
        max_positions=2,
    )

    assert not summary.empty
    assert not equity.empty
    assert not trades.empty
    assert summary["closed_trades"].iloc[0] >= 1


def test_ridge_portfolio_backtest_supports_asymmetric_tp_sl():
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=24 * 62, freq="h", tz="UTC")
    pairs = [f"P{idx}/USD" for idx in range(8)]
    os_start = pd.Timestamp("2025-02-01", tz="UTC")
    for timestamp in timestamps:
        for p_idx, pair in enumerate(pairs):
            close = 100.0 + p_idx
            if pair == "P7/USD" and timestamp >= os_start + pd.Timedelta(hours=1):
                close = 79.0
            rows.append(
                {
                    "open_time": int(timestamp.timestamp() * 1000),
                    "timestamp": timestamp,
                    "pair": pair,
                    "close": close,
                    "target_z": float(p_idx),
                    "z_momentum": float(p_idx),
                    "z_low_roll_impact": 0.0,
                    "z_momentum_x_low_roll_impact": 0.0,
                    "forward_return_1": float(p_idx),
                }
            )
    frame = pd.DataFrame(rows)

    _, _, trades = run_portfolio_backtest(
        frame,
        model="momentum_only",
        horizon=1,
        is_months=1,
        os_months=1,
        step_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.25,
        take_profit=0.5,
        stop_loss=0.2,
        fee_bps=0.0,
        max_positions=1,
    )

    assert "loss_threshold" in set(trades["reason"])


def test_ridge_portfolio_backtest_respects_top_k_and_max_new_entries():
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=24 * 62, freq="h", tz="UTC")
    pairs = [f"P{idx}/USD" for idx in range(8)]
    for timestamp in timestamps:
        for p_idx, pair in enumerate(pairs):
            rows.append(
                {
                    "open_time": int(timestamp.timestamp() * 1000),
                    "timestamp": timestamp,
                    "pair": pair,
                    "close": 100.0 + p_idx,
                    "target_z": float(p_idx),
                    "z_momentum": float(p_idx),
                    "z_low_roll_impact": 0.0,
                    "z_momentum_x_low_roll_impact": 0.0,
                    "forward_return_1": float(p_idx),
                }
            )
    frame = pd.DataFrame(rows)

    _, equity, trades = run_portfolio_backtest(
        frame,
        model="momentum_only",
        horizon=1,
        is_months=1,
        os_months=1,
        step_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.1,
        take_profit=10.0,
        stop_loss=10.0,
        fee_bps=0.0,
        max_positions=5,
        top_k=1,
        max_new_entries=1,
    )

    buys = trades[trades["side"].eq("BUY")]
    assert buys["pair"].nunique() == 1
    assert buys["pair"].iloc[0] == "P7/USD"
    assert equity["positions"].max() == 1
