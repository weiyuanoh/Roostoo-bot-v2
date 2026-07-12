from bot.binance_data import BinanceData


class FakeResponse:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise RuntimeError("boom")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_fetch_klines_parses_binance_payload():
    session = FakeSession(
        [
            FakeResponse(
                [
                    [
                        1000,
                        "1.0",
                        "2.0",
                        "0.5",
                        "1.5",
                        "123.4",
                        2000,
                        "0",
                        0,
                        "0",
                        "0",
                        "0",
                    ]
                ]
            )
        ]
    )
    client = BinanceData(base_urls=["https://example.test"], session=session)

    candles = client.fetch_klines("BTC/USD", interval="1h", limit=1)

    assert candles == [
        {
            "open_time": 1000,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 123.4,
            "close_time": 2000,
        }
    ]
    assert session.calls[0][1]["params"]["symbol"] == "BTCUSDT"


def test_fetch_klines_falls_back_to_second_endpoint():
    session = FakeSession(
        [
            FakeResponse([], fail=True),
            FakeResponse(
                [[1000, "1.0", "2.0", "0.5", "1.5", "123.4", 2000]]
            ),
        ]
    )
    client = BinanceData(
        base_urls=["https://bad.example", "https://good.example"],
        session=session,
    )

    candles = client.fetch_klines("BTC/USD")

    assert candles and candles[0]["close"] == 1.5
    assert len(session.calls) == 2


def test_fetch_klines_unknown_pair_returns_none():
    client = BinanceData(base_urls=["https://example.test"], session=FakeSession([]))

    assert client.fetch_klines("UNKNOWN/USD") is None


def test_fetch_klines_paginated_advances_start_time():
    session = FakeSession(
        [
            FakeResponse(
                [
                    [1000, "1", "1", "1", "1", "1", 1999],
                    [2000, "2", "2", "2", "2", "2", 2999],
                ]
            ),
            FakeResponse([[3000, "3", "3", "3", "3", "3", 3999]]),
        ]
    )
    client = BinanceData(base_urls=["https://example.test"], session=session)

    candles = client.fetch_klines_paginated(
        "BTC/USD",
        interval="1h",
        start_time=1000,
        end_time=5000,
        limit=2,
        sleep_seconds=0,
    )

    assert [candle["open_time"] for candle in candles or []] == [1000, 2000, 3000]
    assert session.calls[0][1]["params"]["startTime"] == 1000
    assert session.calls[1][1]["params"]["startTime"] == 2001
