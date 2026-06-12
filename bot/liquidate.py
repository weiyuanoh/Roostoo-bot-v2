"""Liquidate Roostoo spot positions safely."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.config import LIVE_STATE_PATH, TRADEABLE_COINS
from bot.executor import Executor
from bot.live_state import LiveState
from bot.roostoo_client import RoostooClient


@dataclass(frozen=True)
class LiquidationIntent:
    pair: str
    quantity: float
    price: float
    notional_usd: float


def liquidation_intents(
    wallet: dict[str, Any],
    ticker: dict[str, Any],
    *,
    pairs: list[str] | None = None,
) -> list[LiquidationIntent]:
    """Build sell intents for free non-USD balances."""
    allowed = set(pairs or TRADEABLE_COINS)
    intents: list[LiquidationIntent] = []
    for pair in sorted(allowed):
        asset = pair.split("/")[0]
        balances = wallet.get(asset, {})
        if not isinstance(balances, dict):
            continue
        quantity = float(balances.get("Free", 0) or 0)
        if quantity <= 0:
            continue
        price = float((ticker.get(pair, {}) or {}).get("LastPrice", 0) or 0)
        if price <= 0:
            continue
        intents.append(
            LiquidationIntent(
                pair=pair,
                quantity=quantity,
                price=price,
                notional_usd=quantity * price,
            )
        )
    return intents


def liquidate_positions(
    *,
    pairs: list[str] | None = None,
    execute: bool = False,
    cancel_pending: bool = False,
    state_path: str | Path = LIVE_STATE_PATH,
    client: RoostooClient | None = None,
) -> dict[str, Any]:
    """Dry-run or execute market liquidation for selected spot positions."""
    roostoo = client or RoostooClient()
    exchange = roostoo.exchange_info() or {}
    trade_pairs = exchange.get("TradePairs", exchange)
    executor = Executor(roostoo, trade_pairs if isinstance(trade_pairs, dict) else {})

    if execute and cancel_pending:
        executor.cancel_all_pending()

    wallet = roostoo.balance()
    ticker = roostoo.ticker()
    if wallet is None:
        raise RuntimeError("Roostoo balance unavailable")
    if ticker is None:
        raise RuntimeError("Roostoo ticker unavailable")

    intents = liquidation_intents(wallet, ticker, pairs=pairs)
    orders = []
    state = LiveState(state_path)

    if execute:
        for intent in intents:
            result = executor.sell(intent.pair, intent.quantity, intent.price, use_limit=False)
            orders.append({"intent": intent.__dict__, "result": result})
            if result and result.get("Success"):
                state.remove_position(intent.pair)
        state.save()

    return {
        "execute": execute,
        "cancel_pending": cancel_pending,
        "intents": [intent.__dict__ for intent in intents],
        "orders": orders,
        "estimated_notional_usd": sum(intent.notional_usd for intent in intents),
    }
