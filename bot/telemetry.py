"""Local telemetry helpers for live trading diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.config import LOG_DIR
from bot.logger import log_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def slippage_bps(side: str, reference_price: float | None, fill_price: float | None) -> float | None:
    if reference_price is None or fill_price is None:
        return None
    if reference_price <= 0 or fill_price <= 0:
        return None
    side_upper = side.upper()
    if side_upper == "BUY":
        return (fill_price / reference_price - 1.0) * 10_000
    if side_upper == "SELL":
        return (1.0 - fill_price / reference_price) * 10_000
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def log_monitor_event(
    event_type: str,
    severity: str,
    title: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "created_at": utc_now_iso(),
        "event_type": event_type,
        "severity": severity,
        "title": title,
        "message": message,
        "data": data or {},
    }
    log_jsonl("monitor_events.jsonl", json_safe(event))
    return event


def order_detail(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    detail = result.get("OrderDetail", {})
    return detail if isinstance(detail, dict) else {}


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def log_score_snapshot(
    *,
    cycle_id: str,
    scores: pd.DataFrame,
    price_map: dict[str, float],
    held_pairs: set[str],
    entry_pairs: set[str],
    exit_pairs: set[str],
    model: str,
    alpha: float | None,
    strategy_params: dict[str, Any],
    entry_gate_decisions: list[dict[str, Any]] | None = None,
) -> None:
    ranked = scores.copy()
    if "ridge_score" in ranked.columns:
        ranked["rank"] = ranked["ridge_score"].rank(method="first", ascending=False)
    else:
        ranked["rank"] = np.nan
    gate_by_pair = {
        str(decision.get("pair")): decision
        for decision in entry_gate_decisions or []
        if decision.get("pair") is not None
    }
    records = []
    for row in ranked.itertuples(index=False):
        pair = str(getattr(row, "pair"))
        gate_decision = gate_by_pair.get(pair, {})
        records.append(
            {
                "logged_at": utc_now_iso(),
                "cycle_id": cycle_id,
                "pair": pair,
                "open_time": getattr(row, "open_time", None),
                "timestamp": getattr(row, "timestamp", None),
                "close": getattr(row, "close", None),
                "ticker_price": price_map.get(pair),
                "score": getattr(row, "ridge_score", None),
                "rank": optional_int(getattr(row, "rank", None)),
                "held": pair in held_pairs,
                "intended_entry": pair in entry_pairs,
                "intended_exit": pair in exit_pairs,
                "cluster_gate_checked": bool(gate_decision),
                "cluster_gate_allowed": gate_decision.get("cluster_gate_allowed"),
                "cluster_id": gate_decision.get("cluster_id"),
                "cluster_distance": gate_decision.get("cluster_distance"),
                "cluster_reason": gate_decision.get("cluster_reason"),
                "model": model,
                "alpha": alpha,
                **strategy_params,
            }
        )
    for record in records:
        log_jsonl("live_scores.jsonl", json_safe(record))


def order_telemetry_record(
    *,
    cycle_id: str,
    intent: Any,
    result: dict[str, Any] | None,
    reference_price: float,
) -> dict[str, Any]:
    detail = order_detail(result)
    side = str(getattr(intent, "side", "")).upper()
    pair = str(getattr(intent, "pair", ""))
    filled_quantity = optional_float(detail.get("FilledQuantity"))
    filled_avg_price = optional_float(
        detail.get("FilledAverPrice", detail.get("FilledAvgPrice"))
    )
    order_id = optional_int(detail.get("OrderID"))
    status = detail.get("Status")
    success = bool(result and result.get("Success"))
    intent_id = stable_id(cycle_id, pair, side, getattr(intent, "reason", ""), reference_price)
    return {
        "logged_at": utc_now_iso(),
        "cycle_id": cycle_id,
        "intent_id": intent_id,
        "pair": pair,
        "side": side,
        "reason": getattr(intent, "reason", ""),
        "intended_quantity": getattr(intent, "quantity", 0.0),
        "intended_notional_usd": getattr(intent, "notional_usd", 0.0),
        "reference_price": reference_price,
        "order_id": order_id,
        "status": status,
        "success": success,
        "filled_quantity": filled_quantity,
        "filled_avg_price": filled_avg_price,
        "commission": optional_float(detail.get("CommissionChargeValue")),
        "role": detail.get("Role"),
        "error": (result or {}).get("ErrMsg") if isinstance(result, dict) else None,
        "slippage_bps": slippage_bps(side, reference_price, filled_avg_price),
    }


def log_order_telemetry(record: dict[str, Any]) -> None:
    log_jsonl("trades.jsonl", json_safe(record))


def closed_trade_record(
    *,
    cycle_id: str,
    position: Any,
    intent: Any,
    order_record: dict[str, Any],
    latest_rank: int | None,
) -> dict[str, Any] | None:
    if not order_record.get("success"):
        return None
    exit_price = optional_float(order_record.get("filled_avg_price")) or optional_float(
        order_record.get("reference_price")
    )
    entry_price = optional_float(getattr(position, "entry_price", None))
    quantity = optional_float(order_record.get("filled_quantity")) or optional_float(
        getattr(position, "quantity", None)
    )
    if exit_price is None or entry_price is None or quantity is None:
        gross_return = None
        gross_pnl = None
    else:
        gross_return = exit_price / entry_price - 1.0 if entry_price > 0 else None
        gross_pnl = (exit_price - entry_price) * quantity
    return {
        "logged_at": utc_now_iso(),
        "cycle_id": cycle_id,
        "pair": getattr(intent, "pair", None),
        "exit_reason": getattr(intent, "reason", ""),
        "entry_time": getattr(position, "entry_time", None),
        "entry_price": entry_price,
        "entry_score": getattr(position, "entry_score", None),
        "entry_order_id": getattr(position, "order_id", None),
        "exit_price": exit_price,
        "exit_quantity": quantity,
        "exit_order_id": order_record.get("order_id"),
        "latest_rank": latest_rank,
        "gross_return": gross_return,
        "gross_pnl": gross_pnl,
        "slippage_bps": order_record.get("slippage_bps"),
        "commission": order_record.get("commission"),
    }


def log_closed_trade(record: dict[str, Any] | None) -> None:
    if record is not None:
        log_jsonl("closed_trades.jsonl", json_safe(record))


def log_path(filename: str) -> Path:
    return LOG_DIR / filename
