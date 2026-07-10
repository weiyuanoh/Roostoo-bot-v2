"""Local monitoring reports for live trading telemetry."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.binance_data import BinanceData
from bot.config import LOG_DIR, LIVE_STATE_PATH, TRADEABLE_COINS
from bot.forward_ic import spearman_ic
from bot.live_state import LiveState, wallet_pair_quantities
from bot.telemetry import log_monitor_event


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def since_filter(frame: pd.DataFrame, since_hours: int, time_col: str = "logged_at") -> pd.DataFrame:
    if frame.empty or time_col not in frame.columns:
        return frame
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    out = frame.copy()
    times = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    return out[times >= cutoff].copy()


def positions_rows(
    wallet: dict[str, Any],
    state: LiveState,
    prices: dict[str, float] | None = None,
    pairs: list[str] | None = None,
) -> list[dict[str, Any]]:
    prices = prices or {}
    pair_filter = set(pairs or TRADEABLE_COINS)
    wallet_quantities = wallet_pair_quantities(wallet)
    all_pairs = sorted((set(wallet_quantities) | set(state.positions)) & pair_filter)
    rows = []
    for pair in all_pairs:
        position = state.positions.get(pair)
        wallet_qty = float(wallet_quantities.get(pair, 0.0) or 0.0)
        state_qty = float(getattr(position, "quantity", 0.0) or 0.0) if position else 0.0
        price = prices.get(pair)
        entry_price = getattr(position, "entry_price", None) if position else None
        rows.append(
            {
                "pair": pair,
                "wallet_quantity": wallet_qty,
                "state_quantity": state_qty,
                "quantity_diff": wallet_qty - state_qty,
                "price": price,
                "market_value": wallet_qty * price if price else None,
                "entry_price": entry_price,
                "entry_time": getattr(position, "entry_time", None) if position else None,
                "entry_score": getattr(position, "entry_score", None) if position else None,
                "order_id": getattr(position, "order_id", None) if position else None,
                "return_pct": (
                    price / entry_price - 1.0
                    if price and entry_price and entry_price > 0
                    else None
                ),
                "missing_entry_metadata": bool(wallet_qty > 0 and (not position or not entry_price)),
                "state_only": bool(state_qty > 0 and wallet_qty <= 0),
                "wallet_only": bool(wallet_qty > 0 and not position),
            }
        )
    return rows


def health_report(
    *,
    wallet: dict[str, Any] | None,
    state: LiveState,
    log_dir: Path = LOG_DIR,
    pairs: list[str] | None = None,
    max_cycle_age_minutes: int = 90,
    emit_events: bool = True,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    wallet = wallet or {}
    positions = positions_rows(wallet, state, pairs=pairs)
    for row in positions:
        if row["state_only"]:
            issues.append(_issue("state_only_position", "warning", f"{row['pair']} in state but not wallet", row))
        if row["wallet_only"]:
            issues.append(_issue("wallet_only_position", "warning", f"{row['pair']} in wallet but not state", row))
        if row["missing_entry_metadata"]:
            issues.append(
                _issue("missing_entry_metadata", "warning", f"{row['pair']} missing entry metadata", row)
            )
        if abs(float(row["quantity_diff"] or 0.0)) > 1e-12:
            issues.append(_issue("quantity_mismatch", "warning", f"{row['pair']} quantity mismatch", row))

    cycles = read_jsonl(log_dir / "live_cycles.jsonl")
    if cycles.empty:
        issues.append(_issue("no_live_cycles", "warning", "No live cycle logs found", {}))
        latest_cycle_at = None
    else:
        time_col = "logged_at" if "logged_at" in cycles.columns else "timestamp"
        times = pd.to_datetime(cycles[time_col], utc=True, errors="coerce")
        latest = times.max()
        latest_cycle_at = latest.isoformat() if pd.notna(latest) else None
        if pd.notna(latest):
            age_minutes = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 60
            if age_minutes > max_cycle_age_minutes:
                issues.append(
                    _issue(
                        "stale_live_cycle",
                        "critical",
                        f"Latest live cycle is {age_minutes:.1f} minutes old",
                        {"latest_cycle_at": latest_cycle_at, "age_minutes": age_minutes},
                    )
                )

    trades = read_jsonl(log_dir / "trades.jsonl")
    failed_orders = 0
    if not trades.empty and "success" in trades.columns:
        failed_orders = int((trades["success"] == False).sum())  # noqa: E712
        if failed_orders:
            issues.append(
                _issue("failed_orders", "warning", f"{failed_orders} failed order records", {})
            )

    status = "ok"
    if any(issue["severity"] == "critical" for issue in issues):
        status = "critical"
    elif issues:
        status = "warning"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "latest_cycle_at": latest_cycle_at,
        "positions": positions,
        "failed_orders": failed_orders,
        "issues": issues,
    }
    if emit_events:
        for issue in issues:
            log_monitor_event(
                issue["event_type"],
                issue["severity"],
                issue["title"],
                issue["message"],
                data=issue["data"],
            )
    return report


def write_health_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "health.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def summary_reports(
    *,
    log_dir: Path = LOG_DIR,
    since_hours: int = 168,
) -> dict[str, pd.DataFrame]:
    cycles = since_filter(read_jsonl(log_dir / "live_cycles.jsonl"), since_hours)
    trades = since_filter(read_jsonl(log_dir / "trades.jsonl"), since_hours)
    closed = since_filter(read_jsonl(log_dir / "closed_trades.jsonl"), since_hours)

    summary = pd.DataFrame(
        [
            {
                "since_hours": since_hours,
                "cycles": int(len(cycles)),
                "executed_cycles": _sum_bool(cycles, "execute"),
                "orders": int(len(trades)),
                "successful_orders": _sum_bool(trades, "success"),
                "closed_trades": int(len(closed)),
                "mean_slippage_bps": _mean(trades, "slippage_bps"),
                "gross_pnl": _sum(closed, "gross_pnl"),
                "mean_gross_return_pct": _mean(closed, "gross_return", multiplier=100),
            }
        ]
    )

    pair_attr = _group_attr(closed, "pair")
    exit_attr = _group_attr(closed, "exit_reason")
    slippage = _slippage_report(trades)
    return {
        "summary": summary,
        "pair_attribution": pair_attr,
        "exit_reason_attribution": exit_attr,
        "slippage": slippage,
    }


def write_summary_reports(reports: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in reports.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def forward_report(
    *,
    log_dir: Path = LOG_DIR,
    since_hours: int = 720,
    horizons: tuple[int, ...] = (1, 6, 24),
    binance: BinanceData | None = None,
) -> pd.DataFrame:
    scores = since_filter(read_jsonl(log_dir / "live_scores.jsonl"), since_hours)
    if scores.empty:
        return pd.DataFrame(
            columns=["horizon", "observations", "spearman_ic", "top_rank_mean_return", "intent_mean_return"]
        )
    feed = binance or BinanceData()
    scores = scores.dropna(subset=["pair", "open_time", "close"])
    if scores.empty:
        return pd.DataFrame()
    candle_map = _fetch_candle_map(feed, sorted(scores["pair"].unique()), max(horizons) + 2)
    rows = []
    for row in scores.itertuples(index=False):
        pair = str(row.pair)
        open_time = _to_int(getattr(row, "open_time"))
        close = _to_float(getattr(row, "close"))
        if open_time is None or close is None or close <= 0:
            continue
        for horizon in horizons:
            future = candle_map.get(pair, {}).get(open_time + horizon * 3_600_000)
            if future is None or future <= 0:
                continue
            rows.append(
                {
                    "horizon": horizon,
                    "pair": pair,
                    "score": _to_float(getattr(row, "score", None)),
                    "rank": _to_int(getattr(row, "rank", None)),
                    "intended_entry": bool(getattr(row, "intended_entry", False)),
                    "forward_return": future / close - 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["horizon", "observations", "spearman_ic", "top_rank_mean_return", "intent_mean_return"]
        )
    out = []
    for horizon, group in frame.groupby("horizon", observed=True):
        valid = group.dropna(subset=["score", "forward_return"])
        top_rank = group[group["rank"].eq(1)]
        intents = group[group["intended_entry"]]
        out.append(
            {
                "horizon": int(horizon),
                "observations": int(len(group)),
                "spearman_ic": (
                    spearman_ic(valid["score"], valid["forward_return"])[0]
                    if len(valid) >= 3
                    else np.nan
                ),
                "top_rank_mean_return": float(top_rank["forward_return"].mean()) if not top_rank.empty else np.nan,
                "intent_mean_return": float(intents["forward_return"].mean()) if not intents.empty else np.nan,
                "intent_hit_rate": (
                    float((intents["forward_return"] > 0).mean()) if not intents.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(out).sort_values("horizon")


def write_forward_report(frame: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "forward_ic.csv"
    frame.to_csv(path, index=False)
    return path


def _issue(event_type: str, severity: str, message: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "severity": severity,
        "title": event_type.replace("_", " ").title(),
        "message": message,
        "data": data,
    }


def _sum_bool(frame: pd.DataFrame, col: str) -> int:
    if frame.empty or col not in frame.columns:
        return 0
    return int(frame[col].fillna(False).astype(bool).sum())


def _mean(frame: pd.DataFrame, col: str, multiplier: float = 1.0) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.mean() * multiplier) if not values.empty else np.nan


def _sum(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum())


def _group_attr(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    columns = [col, "closed_trades", "gross_pnl", "mean_return_pct", "win_rate"]
    if frame.empty or col not in frame.columns:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    data["gross_pnl"] = pd.to_numeric(data.get("gross_pnl"), errors="coerce")
    data["gross_return"] = pd.to_numeric(data.get("gross_return"), errors="coerce")
    grouped = data.groupby(col, dropna=False, observed=True)
    out = grouped.agg(
        closed_trades=("pair", "size"),
        gross_pnl=("gross_pnl", "sum"),
        mean_return_pct=("gross_return", lambda values: float(values.mean() * 100)),
        win_rate=("gross_return", lambda values: float((values > 0).mean())),
    ).reset_index()
    return out.sort_values("gross_pnl", ascending=False)


def _slippage_report(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["pair", "side", "orders", "successful_orders", "mean_slippage_bps", "max_slippage_bps"]
    if frame.empty or "slippage_bps" not in frame.columns:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    data["slippage_bps"] = pd.to_numeric(data["slippage_bps"], errors="coerce")
    grouped = data.groupby(["pair", "side"], dropna=False, observed=True)
    return grouped.agg(
        orders=("pair", "size"),
        successful_orders=("success", lambda values: int(values.fillna(False).astype(bool).sum())),
        mean_slippage_bps=("slippage_bps", "mean"),
        max_slippage_bps=("slippage_bps", "max"),
    ).reset_index()


def _fetch_candle_map(feed: BinanceData, pairs: list[str], min_limit: int) -> dict[str, dict[int, float]]:
    limit = max(100, min(1000, min_limit))
    out: dict[str, dict[int, float]] = {}
    for pair in pairs:
        candles = feed.fetch_klines(pair, interval="1h", limit=limit) or []
        out[pair] = {int(candle["open_time"]): float(candle["close"]) for candle in candles}
    return out


def _to_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def default_output_dir() -> Path:
    return Path("reports/live_monitoring")


def default_state() -> LiveState:
    return LiveState(LIVE_STATE_PATH)
