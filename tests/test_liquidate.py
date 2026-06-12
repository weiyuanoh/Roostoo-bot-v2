from bot.liquidate import liquidation_intents


def test_liquidation_intents_use_free_balances_and_valid_prices():
    wallet = {
        "USD": {"Free": 1000, "Lock": 0},
        "BTC": {"Free": 0.1, "Lock": 0.2},
        "ETH": {"Free": 2.0, "Lock": 0},
        "SOL": {"Free": 3.0, "Lock": 0},
    }
    ticker = {
        "BTC/USD": {"LastPrice": 100.0},
        "ETH/USD": {"LastPrice": 0},
    }

    intents = liquidation_intents(wallet, ticker, pairs=["BTC/USD", "ETH/USD", "SOL/USD"])

    assert len(intents) == 1
    assert intents[0].pair == "BTC/USD"
    assert intents[0].quantity == 0.1
    assert intents[0].notional_usd == 10.0


def test_liquidation_intents_respect_pair_filter():
    wallet = {
        "BTC": {"Free": 0.1, "Lock": 0},
        "ETH": {"Free": 2.0, "Lock": 0},
    }
    ticker = {
        "BTC/USD": {"LastPrice": 100.0},
        "ETH/USD": {"LastPrice": 200.0},
    }

    intents = liquidation_intents(wallet, ticker, pairs=["ETH/USD"])

    assert [intent.pair for intent in intents] == ["ETH/USD"]
