"""Roostoo REST API client with HMAC-SHA256 signing."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping
from typing import Any

import requests

from bot.config import API_KEY, API_SECRET, BASE_URL, REQUEST_TIMEOUT_SECONDS
from bot.logger import get_logger

log = get_logger("roostoo_client")


def timestamp_ms() -> str:
    return str(int(time.time() * 1000))


def encode_params(params: Mapping[str, Any]) -> str:
    """Encode params in the order Roostoo expects for signatures."""
    return "&".join(f"{key}={params[key]}" for key in sorted(params))


def sign_params(
    params: Mapping[str, Any],
    api_secret: str,
    now_ms: str | None = None,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    """Return signed headers, encoded params, and the timestamped payload."""
    signed = {**params, "timestamp": now_ms or timestamp_ms()}
    total_params = encode_params(signed)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {"MSG-SIGNATURE": signature}, total_params, signed


class RoostooClient:
    """Thin wrapper around the Roostoo mock exchange API."""

    def __init__(
        self,
        api_key: str = API_KEY,
        api_secret: str = API_SECRET,
        base_url: str = BASE_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _signed_headers(self, params: Mapping[str, Any]) -> tuple[dict[str, str], str, dict[str, Any]]:
        headers, total_params, signed = sign_params(params, self.api_secret)
        headers["RST-API-KEY"] = self.api_key
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers, total_params, signed

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error("%s %s failed: %s", method.upper(), path, exc)
            return None

    # ---- Public endpoints ----

    def server_time(self) -> int | None:
        data = self._request("GET", "/v3/serverTime")
        if not data:
            return None
        return data.get("ServerTime")

    def exchange_info(self) -> dict[str, Any] | None:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {"timestamp": timestamp_ms()}
        if pair:
            params["pair"] = pair
        data = self._request("GET", "/v3/ticker", params=params)
        if not data:
            return None
        if not data.get("Success"):
            log.warning("ticker error: %s", data.get("ErrMsg"))
            return None
        return data.get("Data", {})

    # ---- Signed endpoints ----

    def balance(self) -> dict[str, Any] | None:
        headers, _, params = self._signed_headers({})
        data = self._request("GET", "/v3/balance", headers=headers, params=params)
        if not data:
            return None
        if not data.get("Success"):
            log.warning("balance error: %s", data.get("ErrMsg"))
            return None
        return data.get("SpotWallet", data.get("Wallet", {}))

    def pending_count(self) -> dict[str, Any] | None:
        headers, _, params = self._signed_headers({})
        return self._request("GET", "/v3/pending_count", headers=headers, params=params)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            payload["price"] = str(price)

        headers, body, _ = self._signed_headers(payload)
        data = self._request("POST", "/v3/place_order", headers=headers, data=body)
        if data and not data.get("Success"):
            log.warning("place_order error [%s %s]: %s", pair, side, data.get("ErrMsg"))
        return data

    def query_order(
        self,
        order_id: int | None = None,
        pair: str | None = None,
        pending_only: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)

        headers, body, _ = self._signed_headers(payload)
        data = self._request("POST", "/v3/query_order", headers=headers, data=body)
        if not data:
            return None
        if not data.get("Success"):
            return []
        return data.get("OrderMatched", [])

    def cancel_order(
        self,
        order_id: int | None = None,
        pair: str | None = None,
    ) -> list[dict[str, Any]] | None:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair

        headers, body, _ = self._signed_headers(payload)
        data = self._request("POST", "/v3/cancel_order", headers=headers, data=body)
        if not data:
            return None
        return data.get("CanceledList", [])

