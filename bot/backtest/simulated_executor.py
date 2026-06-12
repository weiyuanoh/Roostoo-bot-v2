"""Roostoo-like market-order simulator for portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.backtest.portfolio import Portfolio


@dataclass(frozen=True)
class ExecutionContext:
    timestamp: int | str
    score: float | None = None
    reason: str = ""


class SimulatedExecutor:
    """Match the live Executor surface while filling against backtest prices."""

    def __init__(
        self,
        portfolio: Portfolio,
        exchange_info: dict[str, Any] | None = None,
        *,
        slippage_bps: float = 0.0,
    ) -> None:
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self.portfolio = portfolio
        self.exchange_info = exchange_info or {}
        self.slippage_rate = float(slippage_bps) / 10_000
        self.context = ExecutionContext(timestamp="")
        self.order_id = 0
        self.events: list[dict[str, Any]] = []

    def set_context(
        self,
        *,
        timestamp: int | str,
        score: float | None = None,
        reason: str = "",
    ) -> None:
        self.context = ExecutionContext(timestamp=timestamp, score=score, reason=reason)

    def buy(
        self,
        pair: str,
        quantity_usd: float,
        current_price: float,
        bid: float = 0,
        ask: float = 0,
        use_limit: bool = True,
    ) -> dict[str, Any] | None:
        del bid, ask, use_limit
        if current_price <= 0:
            return None
        fill_price = self._fill_price("BUY", current_price)
        quantity = self._round_quantity(pair, quantity_usd / fill_price)
        return self._place(pair, "BUY", quantity, fill_price)

    def sell(
        self,
        pair: str,
        coin_quantity: float,
        current_price: float,
        bid: float = 0,
        ask: float = 0,
        use_limit: bool = True,
    ) -> dict[str, Any] | None:
        del bid, ask, use_limit
        if current_price <= 0:
            return None
        fill_price = self._fill_price("SELL", current_price)
        quantity = self._round_quantity(pair, coin_quantity)
        return self._place(pair, "SELL", quantity, fill_price)

    def _place(
        self,
        pair: str,
        side: str,
        quantity: float,
        fill_price: float,
    ) -> dict[str, Any] | None:
        if quantity <= 0 or fill_price <= 0:
            return None
        if not self._passes_min_order(pair, quantity, fill_price):
            return None

        if side == "BUY":
            event = self.portfolio.buy_quantity(
                pair,
                quantity,
                fill_price,
                self.context.timestamp,
                float(self.context.score or 0.0),
                self.context.reason or "ridge_entry",
            )
        else:
            event = self.portfolio.sell(
                pair,
                fill_price,
                self.context.timestamp,
                self.context.reason or "exit",
            )
        if not event:
            return None

        self.order_id += 1
        self.events.append(event)
        return self._order_result(event)

    def _order_result(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": self.order_id,
                "Status": "FILLED",
                "Role": "TAKER",
                "FilledQuantity": event["quantity"],
                "FilledAverPrice": event["price"],
                "CommissionChargeValue": event["fee"],
            },
        }

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

    def _passes_min_order(self, pair: str, quantity: float, price: float) -> bool:
        _, _, minimum = self._precision(pair)
        return quantity * price >= minimum

    def _fill_price(self, side: str, current_price: float) -> float:
        if side == "BUY":
            return current_price * (1 + self.slippage_rate)
        return current_price * (1 - self.slippage_rate)
