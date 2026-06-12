"""Minimal long-only portfolio accounting for research backtests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    pair: str
    quantity: float
    entry_price: float
    entry_time: int | str
    entry_value: float
    entry_score: float

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def return_pct(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return price / self.entry_price - 1.0


class Portfolio:
    """Cash plus long spot positions with simple market-order fills."""

    def __init__(self, initial_cash: float = 1_000_000.0, fee_bps: float = 10.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = float(fee_bps) / 10_000
        self.positions: dict[str, Position] = {}

    def value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for pair, position in self.positions.items():
            price = prices.get(pair)
            if price is not None and price > 0:
                total += position.market_value(price)
        return total

    def buy(
        self,
        pair: str,
        notional: float,
        price: float,
        timestamp: int | str,
        score: float,
    ) -> dict | None:
        if notional <= 0 or price <= 0:
            return None
        return self.buy_quantity(pair, float(notional) / price, price, timestamp, score, "entry")

    def buy_quantity(
        self,
        pair: str,
        quantity: float,
        price: float,
        timestamp: int | str,
        score: float,
        reason: str = "entry",
    ) -> dict | None:
        if pair in self.positions or quantity <= 0 or price <= 0:
            return None
        max_quantity = self.cash / ((1 + self.fee_rate) * price)
        fill_quantity = min(float(quantity), max_quantity)
        fill_notional = fill_quantity * price
        if fill_quantity <= 0 or fill_notional <= 0:
            return None
        fee = fill_notional * self.fee_rate
        self.cash -= fill_notional + fee
        self.positions[pair] = Position(
            pair=pair,
            quantity=fill_quantity,
            entry_price=price,
            entry_time=timestamp,
            entry_value=fill_notional,
            entry_score=score,
        )
        return {
            "timestamp": timestamp,
            "pair": pair,
            "side": "BUY",
            "price": price,
            "quantity": fill_quantity,
            "notional": fill_notional,
            "fee": fee,
            "cash_after": self.cash,
            "score": score,
            "reason": reason,
        }

    def sell(
        self,
        pair: str,
        price: float,
        timestamp: int | str,
        reason: str,
    ) -> dict | None:
        position = self.positions.get(pair)
        if position is None or price <= 0:
            return None
        notional = position.quantity * price
        fee = notional * self.fee_rate
        self.cash += notional - fee
        pnl = notional - fee - position.entry_value
        ret = pnl / position.entry_value if position.entry_value > 0 else 0.0
        self.positions.pop(pair, None)
        return {
            "timestamp": timestamp,
            "pair": pair,
            "side": "SELL",
            "price": price,
            "quantity": position.quantity,
            "notional": notional,
            "fee": fee,
            "cash_after": self.cash,
            "score": position.entry_score,
            "reason": reason,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "entry_value": position.entry_value,
            "pnl": pnl,
            "return_pct": ret,
        }


def compute_equity_metrics(equity: list[float], initial_cash: float) -> dict[str, float | int]:
    if len(equity) < 2:
        return {}
    import numpy as np

    values = np.array(equity, dtype=float)
    total_return = values[-1] / initial_cash - 1.0
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    daily = values[::24]
    daily_returns = np.diff(daily) / daily[:-1] if len(daily) > 1 else np.array([])
    sharpe = (
        float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))
        if len(daily_returns) > 1 and np.std(daily_returns) > 0
        else 0.0
    )
    downside = daily_returns[daily_returns < 0]
    sortino = (
        float(np.mean(daily_returns) / np.std(downside) * np.sqrt(365))
        if len(downside) > 1 and np.std(downside) > 0
        else 0.0
    )
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0
    return {
        "initial_cash": float(initial_cash),
        "final_value": float(values[-1]),
        "total_return_pct": float(total_return * 100),
        "max_drawdown_pct": float(max_drawdown * 100),
        "sharpe": sharpe,
        "sortino": sortino,
        "equity_points": int(len(values)),
    }
