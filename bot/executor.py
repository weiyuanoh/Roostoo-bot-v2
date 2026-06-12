"""Translate portfolio decisions into Roostoo orders."""

from __future__ import annotations

import time
from typing import Any

from bot.config import LIMIT_ORDER_TIMEOUT_SECONDS
from bot.logger import get_logger, log_trade
from bot.roostoo_client import RoostooClient

log = get_logger("executor")


class Executor:
    """Execute market or midpoint limit orders on Roostoo."""

    def __init__(self, client: RoostooClient, exchange_info: dict[str, Any]) -> None:
        self.client = client
        self.exchange_info = exchange_info
        self.pending_orders: dict[int, dict[str, Any]] = {}

    def _precision(self, pair: str) -> tuple[int, int, float]:
        info = self.exchange_info.get(pair, {})
        return (
            int(info.get("PricePrecision", 4)),
            int(info.get("AmountPrecision", 2)),
            float(info.get("MiniOrder", 1.0)),
        )

    def _round_quantity(self, pair: str, quantity: float) -> float:
        _, amount_precision, _ = self._precision(pair)
        if amount_precision == 0:
            return float(int(quantity))
        return round(quantity, amount_precision)

    def _round_price(self, pair: str, price: float) -> float:
        price_precision, _, _ = self._precision(pair)
        return round(price, price_precision)

    def _passes_min_order(self, pair: str, quantity: float, price: float) -> bool:
        _, _, minimum = self._precision(pair)
        return quantity * price >= minimum

    def buy(
        self,
        pair: str,
        quantity_usd: float,
        current_price: float,
        bid: float = 0,
        ask: float = 0,
        use_limit: bool = True,
    ) -> dict[str, Any] | None:
        if current_price <= 0:
            return None
        quantity = self._round_quantity(pair, quantity_usd / current_price)
        return self._place(pair, "BUY", quantity, current_price, bid, ask, use_limit)

    def sell(
        self,
        pair: str,
        coin_quantity: float,
        current_price: float,
        bid: float = 0,
        ask: float = 0,
        use_limit: bool = True,
    ) -> dict[str, Any] | None:
        quantity = self._round_quantity(pair, coin_quantity)
        return self._place(pair, "SELL", quantity, current_price, bid, ask, use_limit)

    def _place(
        self,
        pair: str,
        side: str,
        quantity: float,
        current_price: float,
        bid: float,
        ask: float,
        use_limit: bool,
    ) -> dict[str, Any] | None:
        if quantity <= 0 or current_price <= 0:
            return None
        if not self._passes_min_order(pair, quantity, current_price):
            log.debug("Order too small: %s %s qty=%s price=%s", side, pair, quantity, current_price)
            return None

        if use_limit and bid > 0 and ask > 0:
            limit_price = self._round_price(pair, (bid + ask) / 2)
            result = self.client.place_order(pair, side, quantity, "LIMIT", limit_price)
            if result and result.get("Success"):
                self._track_and_log(result, pair, side, quantity, limit_price)
                return result
            err = (result or {}).get("ErrMsg", "").lower()
            if "permission" in err or "unauthorized" in err:
                return result

        result = self.client.place_order(pair, side, quantity, "MARKET")
        if result and result.get("Success"):
            self._track_and_log(result, pair, side, quantity, current_price)
        return result

    def manage_pending_orders(self) -> list[dict[str, Any]]:
        """Cancel stale limits and return fill events discovered this cycle."""
        fill_events: list[dict[str, Any]] = []
        now = time.time()
        for order_id, info in list(self.pending_orders.items()):
            matches = self.client.query_order(order_id=order_id)
            if matches:
                order = matches[0]
                status = str(order.get("Status", "")).upper()
                filled_qty = float(order.get("FilledQuantity", 0) or 0)
                if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED"}:
                    if filled_qty > 0:
                        fill_events.append(
                            {
                                "pair": info["pair"],
                                "side": info["side"],
                                "filled_qty": filled_qty,
                                "filled_avg_price": float(order.get("FilledAverPrice", 0) or 0),
                            }
                        )
                    self.pending_orders.pop(order_id, None)
                    continue

            if now - float(info["time_placed"]) > LIMIT_ORDER_TIMEOUT_SECONDS:
                log.info("Cancelling stale order %s (%s %s)", order_id, info["pair"], info["side"])
                self.client.cancel_order(order_id=order_id)
                self.pending_orders.pop(order_id, None)
        return fill_events

    def cancel_all_pending(self) -> None:
        if not self.pending_orders:
            return
        self.client.cancel_order()
        self.pending_orders.clear()

    def _track_and_log(
        self,
        result: dict[str, Any],
        pair: str,
        side: str,
        quantity: float,
        price: float,
    ) -> None:
        detail = result.get("OrderDetail", {})
        order_id = detail.get("OrderID")
        if detail.get("Status") == "PENDING" and order_id is not None:
            self.pending_orders[int(order_id)] = {
                "pair": pair,
                "side": side,
                "time_placed": time.time(),
                "price": price,
                "quantity": quantity,
            }
        log_trade(
            {
                "pair": pair,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                "status": detail.get("Status"),
                "role": detail.get("Role"),
                "filled_qty": detail.get("FilledQuantity"),
                "filled_avg_price": detail.get("FilledAverPrice"),
                "commission": detail.get("CommissionChargeValue"),
            }
        )

