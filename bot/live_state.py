"""Local live-trading state for position entry metadata.

Roostoo wallet balances tell us what we hold, but not the strategy entry score
or entry price needed for the +/-50% exit policy. This module stores that small
piece of bot-owned metadata in JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bot.config import TRADEABLE_COINS


@dataclass
class LivePositionState:
    pair: str
    quantity: float
    entry_price: float | None
    entry_time: str | None
    entry_score: float | None = None
    order_id: int | None = None


class LiveState:
    """Persist and reconcile live position metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.positions: dict[str, LivePositionState] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.positions = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        positions = raw.get("positions", {})
        self.positions = {
            pair: LivePositionState(
                pair=str(data.get("pair", pair)),
                quantity=float(data.get("quantity", 0) or 0),
                entry_price=_optional_float(data.get("entry_price")),
                entry_time=data.get("entry_time"),
                entry_score=_optional_float(data.get("entry_score")),
                order_id=_optional_int(data.get("order_id")),
            )
            for pair, data in positions.items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "positions": {
                pair: asdict(position)
                for pair, position in sorted(self.positions.items())
                if position.quantity > 0
            }
        }
        tmp = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def held_pairs(self) -> set[str]:
        return {pair for pair, position in self.positions.items() if position.quantity > 0}

    def upsert_position(self, position: LivePositionState) -> None:
        if position.quantity <= 0:
            self.positions.pop(position.pair, None)
            return
        self.positions[position.pair] = position

    def remove_position(self, pair: str) -> None:
        self.positions.pop(pair, None)

    def reconcile_wallet(self, wallet: dict[str, Any]) -> None:
        """Drop closed local positions and add placeholders for wallet-only holdings."""
        wallet_quantities = wallet_pair_quantities(wallet)
        for pair in list(self.positions):
            quantity = wallet_quantities.get(pair, 0.0)
            if quantity <= 0:
                self.positions.pop(pair, None)
            else:
                self.positions[pair].quantity = quantity

        for pair, quantity in wallet_quantities.items():
            if quantity <= 0 or pair in self.positions:
                continue
            self.positions[pair] = LivePositionState(
                pair=pair,
                quantity=quantity,
                entry_price=None,
                entry_time=None,
                entry_score=None,
                order_id=None,
            )


def wallet_pair_quantities(wallet: dict[str, Any]) -> dict[str, float]:
    pairs_by_asset = {pair.split("/")[0]: pair for pair in TRADEABLE_COINS}
    quantities: dict[str, float] = {}
    for asset, balances in wallet.items():
        pair = pairs_by_asset.get(str(asset).upper())
        if not pair:
            continue
        if isinstance(balances, dict):
            free = float(balances.get("Free", 0) or 0)
            locked = float(balances.get("Lock", balances.get("Locked", 0)) or 0)
        else:
            free = float(balances or 0)
            locked = 0.0
        quantities[pair] = free + locked
    return quantities


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)
