import pytest

from bot.backtest.portfolio import Portfolio
from bot.backtest.simulated_executor import SimulatedExecutor


def test_simulated_executor_rounds_quantity_and_returns_roostoo_shape():
    portfolio = Portfolio(initial_cash=1_000.0, fee_bps=10.0)
    executor = SimulatedExecutor(
        portfolio,
        exchange_info={"BTC/USD": {"AmountPrecision": 2, "MiniOrder": 1}},
    )
    executor.set_context(timestamp=1, score=0.7, reason="ridge_entry")

    result = executor.buy("BTC/USD", 123.45, 10.0, use_limit=False)

    assert result is not None
    assert result["Success"] is True
    detail = result["OrderDetail"]
    assert detail["Status"] == "FILLED"
    assert detail["FilledQuantity"] == pytest.approx(12.35)
    assert detail["FilledAverPrice"] == pytest.approx(10.0)
    assert portfolio.positions["BTC/USD"].quantity == pytest.approx(12.35)


def test_simulated_executor_rejects_below_min_order():
    portfolio = Portfolio(initial_cash=1_000.0, fee_bps=0.0)
    executor = SimulatedExecutor(
        portfolio,
        exchange_info={"BTC/USD": {"AmountPrecision": 4, "MiniOrder": 50}},
    )
    executor.set_context(timestamp=1, score=0.7, reason="ridge_entry")

    result = executor.buy("BTC/USD", 10.0, 10.0, use_limit=False)

    assert result is None
    assert "BTC/USD" not in portfolio.positions


def test_simulated_executor_applies_buy_and_sell_slippage():
    portfolio = Portfolio(initial_cash=1_000.0, fee_bps=0.0)
    executor = SimulatedExecutor(
        portfolio,
        exchange_info={"BTC/USD": {"AmountPrecision": 4, "MiniOrder": 1}},
        slippage_bps=100.0,
    )
    executor.set_context(timestamp=1, score=0.7, reason="ridge_entry")

    buy = executor.buy("BTC/USD", 100.0, 10.0, use_limit=False)

    assert buy is not None
    assert buy["OrderDetail"]["FilledAverPrice"] == pytest.approx(10.1)

    executor.set_context(timestamp=2, reason="gain_threshold")
    sell = executor.sell("BTC/USD", portfolio.positions["BTC/USD"].quantity, 20.0, use_limit=False)

    assert sell is not None
    assert sell["OrderDetail"]["FilledAverPrice"] == pytest.approx(19.8)
