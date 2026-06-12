"""Binance public candle data client."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import requests

from bot.config import (
    BINANCE_BASE_URL,
    BINANCE_FALLBACK_URLS,
    BINANCE_SYMBOL_MAP,
    REQUEST_TIMEOUT_SECONDS,
    TRADEABLE_COINS,
)
from bot.logger import get_logger

log = get_logger("binance_data")

Candle = dict[str, int | float]


class BinanceData:
    """Fetch and cache spot klines from Binance-compatible public endpoints."""

    def __init__(
        self,
        base_urls: list[str] | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        urls = base_urls or [BINANCE_BASE_URL, *BINANCE_FALLBACK_URLS]
        self.base_urls = self._dedupe_urls(urls)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.candles: dict[str, list[Candle]] = defaultdict(list)

    @staticmethod
    def _dedupe_urls(urls: list[str] | tuple[str, ...]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for url in urls:
            cleaned = url.rstrip("/")
            if cleaned and cleaned not in seen:
                deduped.append(cleaned)
                seen.add(cleaned)
        return deduped

    def binance_symbol(self, pair: str) -> str | None:
        return BINANCE_SYMBOL_MAP.get(pair)

    def fetch_klines(
        self,
        pair: str,
        interval: str = "1h",
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Candle] | None:
        """Fetch klines for a Roostoo pair like `BTC/USD`."""
        symbol = self.binance_symbol(pair)
        if not symbol:
            log.warning("No Binance symbol mapping for %s", pair)
            return None

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        raw: list[list[Any]] | None = None
        last_error: Exception | None = None
        for base_url in self.base_urls:
            try:
                response = self.session.get(
                    f"{base_url}/api/v3/klines",
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                parsed = response.json()
                if not isinstance(parsed, list):
                    log.error("klines returned non-list for %s: %r", pair, parsed)
                    return None
                raw = parsed
                break
            except Exception as exc:
                last_error = exc
                log.warning("klines failed via %s for %s: %s", base_url, pair, exc)

        if raw is None:
            log.error("klines failed for %s on all endpoints: %s", pair, last_error)
            return None

        return [
            {
                "open_time": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": int(kline[6]),
            }
            for kline in raw
        ]

    def load_history(
        self,
        pairs: list[str] | None = None,
        interval: str = "1h",
        limit: int = 1000,
        sleep_seconds: float = 0.1,
    ) -> int:
        """Load historical candles into memory and return loaded pair count."""
        selected = pairs or TRADEABLE_COINS
        loaded = 0
        for pair in selected:
            candles = self.fetch_klines(pair, interval=interval, limit=limit)
            if candles:
                self.candles[pair] = candles
                loaded += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        log.info("Loaded history for %s/%s pairs", loaded, len(selected))
        return loaded

    def update_latest(self, pairs: list[str] | None = None, interval: str = "1h") -> None:
        """Fetch the newest candles and upsert them into the in-memory cache."""
        selected = pairs or TRADEABLE_COINS
        for pair in selected:
            candles = self.fetch_klines(pair, interval=interval, limit=3)
            if not candles:
                continue

            existing = self.candles[pair]
            index = {candle["open_time"]: idx for idx, candle in enumerate(existing)}
            for candle in candles:
                open_time = candle["open_time"]
                if open_time in index:
                    existing[index[open_time]] = candle
                else:
                    existing.append(candle)

            if len(existing) > 2000:
                self.candles[pair] = existing[-2000:]

    def get_closes(self, pair: str, n: int | None = None) -> np.ndarray:
        return self._series(pair, "close", n)

    def get_highs(self, pair: str, n: int | None = None) -> np.ndarray:
        return self._series(pair, "high", n)

    def get_lows(self, pair: str, n: int | None = None) -> np.ndarray:
        return self._series(pair, "low", n)

    def get_volumes(self, pair: str, n: int | None = None) -> np.ndarray:
        return self._series(pair, "volume", n)

    def _series(self, pair: str, field: str, n: int | None = None) -> np.ndarray:
        candles = self.candles.get(pair, [])
        values = np.array([candle[field] for candle in candles], dtype=float)
        if n is not None:
            values = values[-n:]
        return values

