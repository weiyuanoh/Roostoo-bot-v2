import pytest

from bot.live_state import LivePositionState, LiveState, wallet_pair_quantities


def test_wallet_pair_quantities_maps_roostoo_wallet_to_pairs():
    wallet = {
        "USD": {"Free": 1000, "Lock": 0},
        "BTC": {"Free": 0.2, "Lock": 0.1},
        "UNKNOWN": {"Free": 5, "Lock": 0},
    }

    assert wallet_pair_quantities(wallet)["BTC/USD"] == pytest.approx(0.3)


def test_live_state_reconciles_closed_and_wallet_only_positions(tmp_path):
    state = LiveState(tmp_path / "state.json")
    state.upsert_position(
        LivePositionState(
            pair="ETH/USD",
            quantity=2.0,
            entry_price=100.0,
            entry_time="2026-01-01T00:00:00+00:00",
            entry_score=1.5,
        )
    )

    state.reconcile_wallet(
        {
            "BTC": {"Free": 0.1, "Lock": 0},
            "ETH": {"Free": 0, "Lock": 0},
        }
    )

    assert "ETH/USD" not in state.positions
    assert state.positions["BTC/USD"].quantity == 0.1
    assert state.positions["BTC/USD"].entry_price is None


def test_live_state_round_trips_json(tmp_path):
    path = tmp_path / "live_state.json"
    state = LiveState(path)
    state.upsert_position(
        LivePositionState(
            pair="SOL/USD",
            quantity=3.0,
            entry_price=50.0,
            entry_time="2026-01-01T00:00:00+00:00",
            entry_score=0.7,
            order_id=123,
        )
    )
    state.save()

    loaded = LiveState(path)

    assert loaded.positions["SOL/USD"].quantity == 3.0
    assert loaded.positions["SOL/USD"].entry_price == 50.0
    assert loaded.positions["SOL/USD"].order_id == 123
