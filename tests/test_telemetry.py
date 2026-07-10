import pytest

from bot.telemetry import closed_trade_record, json_safe, order_telemetry_record, slippage_bps, stable_id


class Intent:
    pair = "BTC/USD"
    side = "BUY"
    reason = "ridge_entry"
    quantity = 0.0
    notional_usd = 100.0


class Position:
    quantity = 1.0
    entry_price = 100.0
    entry_time = "2026-01-01T00:00:00+00:00"
    entry_score = 0.8
    order_id = 11


def test_slippage_bps_uses_adverse_direction_for_buy_and_sell():
    assert slippage_bps("BUY", 100.0, 101.0) == pytest.approx(100.0)
    assert slippage_bps("SELL", 100.0, 99.0) == pytest.approx(100.0)
    assert slippage_bps("BUY", 0.0, 101.0) is None


def test_stable_id_is_deterministic():
    assert stable_id("a", 1, None) == stable_id("a", 1, None)
    assert stable_id("a", 1, None) != stable_id("a", 2, None)


def test_order_telemetry_record_extracts_fill_and_slippage():
    result = {
        "Success": True,
        "OrderDetail": {
            "OrderID": 123,
            "Status": "FILLED",
            "FilledQuantity": 1.0,
            "FilledAverPrice": 101.0,
            "CommissionChargeValue": 0.1,
        },
    }

    record = order_telemetry_record(
        cycle_id="cycle",
        intent=Intent(),
        result=result,
        reference_price=100.0,
    )

    assert record["success"] is True
    assert record["order_id"] == 123
    assert record["slippage_bps"] == pytest.approx(100.0)


def test_closed_trade_record_reports_positive_tp_return():
    intent = Intent()
    intent.side = "SELL"
    intent.reason = "gain_threshold"
    order = {
        "success": True,
        "filled_quantity": 1.0,
        "filled_avg_price": 103.0,
        "reference_price": 103.0,
        "order_id": 44,
        "slippage_bps": 0.0,
        "commission": 0.1,
    }

    record = closed_trade_record(
        cycle_id="cycle",
        position=Position(),
        intent=intent,
        order_record=order,
        latest_rank=1,
    )

    assert record is not None
    assert record["gross_return"] == pytest.approx(0.03)
    assert record["gross_pnl"] == pytest.approx(3.0)


def test_json_safe_converts_non_json_values():
    payload = json_safe({"value": float("nan")})
    assert payload["value"] is None
