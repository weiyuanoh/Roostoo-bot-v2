from bot.data_store import CandleStore


def test_write_csv_sorts_and_deduplicates(tmp_path):
    store = CandleStore(tmp_path)
    candles = [
        {"open_time": 2, "open": 2, "high": 2, "low": 2, "close": 2, "volume": 2, "close_time": 3},
        {"open_time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "close_time": 2},
        {"open_time": 2, "open": 9, "high": 9, "low": 9, "close": 9, "volume": 9, "close_time": 4},
    ]

    path = store.write_csv("BTC/USD", "1h", candles)

    assert path.name == "BTC_USD_1h.csv"
    lines = path.read_text().splitlines()
    assert lines[1].startswith("1,")
    assert lines[2].startswith("2,9")
    assert len(lines) == 3


def test_read_many_adds_pair_and_combines(tmp_path):
    store = CandleStore(tmp_path)
    candles = [
        {"open_time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "close_time": 2},
    ]
    store.write_csv("BTC/USD", "1h", candles)
    store.write_csv("ETH/USD", "1h", candles)

    frame = store.read_many(("BTC/USD", "ETH/USD"), "1h")

    assert set(frame["pair"]) == {"BTC/USD", "ETH/USD"}
    assert len(frame) == 2
