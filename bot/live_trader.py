"""Live trading loop for the shared ridge momentum + roll-impact strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.binance_data import BinanceData
from bot.config import (
    LIVE_FORWARD_HORIZON,
    LIVE_HISTORY_LIMIT,
    LIVE_INTERVAL,
    LIVE_MIN_HISTORY_BARS,
    LIVE_MODEL,
    LIVE_MAX_NEW_ENTRIES,
    LIVE_MAX_POSITIONS,
    LIVE_POSITION_FRACTION,
    LIVE_STATE_PATH,
    LIVE_STOP_LOSS,
    LIVE_TAKE_PROFIT,
    LIVE_TOP_K,
    TRADEABLE_COINS,
)
from bot.executor import Executor
from bot.live_state import LivePositionState, LiveState
from bot.logger import get_logger, log_jsonl
from bot.roostoo_client import RoostooClient
from bot.model_store import load_cluster_gate, load_ridge_selection
from bot.telemetry import (
    closed_trade_record,
    log_closed_trade,
    log_order_telemetry,
    log_score_snapshot,
    order_telemetry_record,
    stable_id,
)
from bot.strategy.ridge import (
    DEFAULT_RIDGE_ALPHAS,
    RidgeSelection,
    add_training_target,
    build_cycle_intents,
    build_feature_frame,
    latest_scores,
    score_frame,
    score_ranks,
    select_ridge_model,
)
from bot.strategy.regime import ClusterRegimeGate, RegimeThrottleConfig, add_decision_time_regime_features, add_roll_impact_regime

log = get_logger("live_trader")


@dataclass(frozen=True)
class RidgeLiveConfig:
    pairs: tuple[str, ...] = tuple(TRADEABLE_COINS)
    interval: str = LIVE_INTERVAL
    history_limit: int = LIVE_HISTORY_LIMIT
    horizon: int = LIVE_FORWARD_HORIZON
    model: str = LIVE_MODEL
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    position_fraction: float = LIVE_POSITION_FRACTION
    take_profit: float = LIVE_TAKE_PROFIT
    stop_loss: float = LIVE_STOP_LOSS
    top_k: int | None = LIVE_TOP_K or None
    max_new_entries: int | None = LIVE_MAX_NEW_ENTRIES or None
    max_positions_override: int | None = LIVE_MAX_POSITIONS or None
    regime_config: RegimeThrottleConfig | None = None
    min_history_bars: int = LIVE_MIN_HISTORY_BARS
    state_path: Path = LIVE_STATE_PATH
    model_dir: Path | None = None
    cluster_regime_gate: bool = False

    @property
    def max_positions(self) -> int:
        if self.max_positions_override is not None:
            return self.max_positions_override
        return max(1, int(np.floor(1 / self.position_fraction)))

    def __post_init__(self) -> None:
        if self.take_profit <= 0:
            raise ValueError("take_profit must be positive")
        if self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive")
        if self.position_fraction <= 0:
            raise ValueError("position_fraction must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if self.max_new_entries is not None and self.max_new_entries <= 0:
            raise ValueError("max_new_entries must be positive when set")
        if self.max_positions_override is not None and self.max_positions_override <= 0:
            raise ValueError("max_positions_override must be positive when set")
        if self.cluster_regime_gate and self.model_dir is None:
            raise ValueError("cluster_regime_gate requires model_dir")


class RidgeLiveTrader:
    """Train, score, and execute the hourly ridge strategy."""

    def __init__(
        self,
        roostoo: RoostooClient,
        binance: BinanceData,
        executor: Executor,
        state: LiveState,
        config: RidgeLiveConfig,
    ) -> None:
        self.roostoo = roostoo
        self.binance = binance
        self.executor = executor
        self.state = state
        self.config = config
        self.selection: RidgeSelection | None = None
        self.selection_metadata: dict[str, Any] = {}
        self.cluster_gate: ClusterRegimeGate | None = None
        self.cluster_gate_metadata: dict[str, Any] = {}

    @classmethod
    def from_clients(
        cls,
        *,
        config: RidgeLiveConfig,
        roostoo: RoostooClient | None = None,
        binance: BinanceData | None = None,
    ) -> "RidgeLiveTrader":
        roostoo_client = roostoo or RoostooClient()
        exchange = roostoo_client.exchange_info() or {}
        trade_pairs = exchange.get("TradePairs", exchange)
        executor = Executor(roostoo_client, trade_pairs if isinstance(trade_pairs, dict) else {})
        return cls(
            roostoo=roostoo_client,
            binance=binance or BinanceData(),
            executor=executor,
            state=LiveState(config.state_path),
            config=config,
        )

    def initialize(self) -> RidgeSelection:
        """Fetch recent candles and fit the ridge score used by live cycles."""
        if self.config.model_dir is not None:
            self.selection, self.selection_metadata = load_ridge_selection(self.config.model_dir)
            if self.config.cluster_regime_gate:
                self.cluster_gate, self.cluster_gate_metadata = load_cluster_gate(self.config.model_dir)
            log.info(
                "Loaded live ridge model=%s alpha=%s terms=%s train_start=%s train_end=%s",
                self.selection.model,
                self.selection.alpha,
                ",".join(self.selection.terms),
                self.selection_metadata.get("train_start"),
                self.selection_metadata.get("train_end"),
            )
            return self.selection

        candles = self.fetch_history_frame()
        features = build_feature_frame(candles)
        train = add_training_target(features, horizon=self.config.horizon)
        target_col = f"forward_return_{self.config.horizon}"
        train = train.dropna(subset=[target_col, "target_z"])
        if train.empty:
            raise RuntimeError("not enough non-null target rows to train live ridge model")
        self.selection = select_ridge_model(
            train,
            self.config.model,
            target_col=target_col,
            ridge_alphas=self.config.ridge_alphas,
        )
        log.info(
            "Trained live ridge model=%s alpha=%s terms=%s IS spearman=%.6f",
            self.selection.model,
            self.selection.alpha,
            ",".join(self.selection.terms),
            self.selection.is_mean_spearman,
        )
        return self.selection

    def fetch_history_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for pair in self.config.pairs:
            candles = self.binance.fetch_klines(
                pair,
                interval=self.config.interval,
                limit=self.config.history_limit,
            )
            if not candles or len(candles) < self.config.min_history_bars:
                log.warning("Skipping %s: only %s history bars", pair, len(candles or []))
                continue
            rows.extend({**candle, "pair": pair} for candle in candles)
        if not rows:
            raise RuntimeError("no Binance history loaded for live scoring")
        return pd.DataFrame(rows).sort_values(["pair", "open_time"]).reset_index(drop=True)

    def score_latest(self, candles: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.selection is None:
            self.initialize()
        assert self.selection is not None
        source = candles if candles is not None else self.fetch_history_frame()
        features = build_feature_frame(source)
        if self.config.regime_config is not None:
            features = add_roll_impact_regime(features, self.config.regime_config)
        if self.cluster_gate is not None:
            scored = score_frame(features, self.selection.terms, self.selection.beta, score_col="ridge_score")
            scored = add_decision_time_regime_features(scored)
            latest_time = scored["open_time"].max()
            latest = scored[scored["open_time"].eq(latest_time)].copy()
            return latest.sort_values("ridge_score", ascending=False, na_position="last")
        return latest_scores(features, self.selection)

    def run_cycle(self, *, execute: bool = False) -> dict[str, Any]:
        """Run one hourly cycle. Dry-run is the default unless execute=True."""
        if self.selection is None:
            self.initialize()

        wallet = self.roostoo.balance()
        ticker = self.roostoo.ticker()
        if wallet is None:
            raise RuntimeError("Roostoo balance unavailable")
        if ticker is None:
            raise RuntimeError("Roostoo ticker unavailable")

        self.state.reconcile_wallet(wallet)
        scores = self.score_latest()
        price_map = live_price_map(ticker, scores)
        portfolio_value = compute_portfolio_value(wallet, price_map)
        available_cash = wallet_usd_cash(wallet)
        cycle_timestamp = datetime.now(timezone.utc).isoformat()
        latest_open_time = _latest_open_time(scores)
        cycle_id = stable_id(cycle_timestamp, latest_open_time, self.config.model, execute)

        cycle = build_cycle_intents(
            scores,
            self.state.positions,
            price_map,
            portfolio_value=portfolio_value,
            available_cash=available_cash,
            position_fraction=self.config.position_fraction,
            max_positions=self.config.max_positions,
            top_k=self.config.top_k,
            max_new_entries=self.config.max_new_entries,
            regime_config=self.config.regime_config,
            cluster_gate=self.cluster_gate,
            take_profit=self.config.take_profit,
            stop_loss=self.config.stop_loss,
        )
        exits = cycle.exits
        entries = cycle.entries
        ranks = score_ranks(scores)
        strategy_params = {
            "take_profit": self.config.take_profit,
            "stop_loss": self.config.stop_loss,
            "top_k": self.config.top_k,
            "max_new_entries": self.config.max_new_entries,
            "max_positions": self.config.max_positions,
            "position_fraction": self.config.position_fraction,
            "execute": execute,
            "ridge_model_trained_at": self.selection_metadata.get("as_of"),
            "ridge_train_start": self.selection_metadata.get("train_start"),
            "ridge_train_end": self.selection_metadata.get("train_end"),
            "regime_model_trained_at": self.cluster_gate_metadata.get("as_of"),
            "regime_train_start": self.cluster_gate_metadata.get("train_start"),
            "regime_train_end": self.cluster_gate_metadata.get("train_end"),
        }
        log_score_snapshot(
            cycle_id=cycle_id,
            scores=scores,
            price_map=price_map,
            held_pairs=set(self.state.positions),
            entry_pairs={intent.pair for intent in entries},
            exit_pairs={intent.pair for intent in exits},
            model=self.selection.model if self.selection else self.config.model,
            alpha=self.selection.alpha if self.selection else None,
            strategy_params=strategy_params,
            entry_gate_decisions=cycle.entry_gate_decisions,
        )
        gate_checks = len(cycle.entry_gate_decisions)
        gate_allowed = sum(
            bool(decision.get("cluster_gate_allowed")) for decision in cycle.entry_gate_decisions
        )

        results = {
            "timestamp": cycle_timestamp,
            "cycle_id": cycle_id,
            "latest_open_time": latest_open_time,
            "execute": execute,
            "portfolio_value": portfolio_value,
            "available_cash": available_cash,
            "model": self.selection.model if self.selection else self.config.model,
            "alpha": self.selection.alpha if self.selection else None,
            "take_profit": self.config.take_profit,
            "stop_loss": self.config.stop_loss,
            "top_k": self.config.top_k,
            "max_new_entries": self.config.max_new_entries,
            "max_positions": self.config.max_positions,
            "regime": cycle.regime.__dict__ if cycle.regime is not None else None,
            "cluster_gate": {
                "enabled": self.cluster_gate is not None,
                "trained_at": self.cluster_gate_metadata.get("as_of"),
                "train_start": self.cluster_gate_metadata.get("train_start"),
                "train_end": self.cluster_gate_metadata.get("train_end"),
                "allowed_clusters": sorted(self.cluster_gate.allowed_clusters) if self.cluster_gate else [],
                "checks": gate_checks,
                "allowed_checks": gate_allowed,
                "blocked_checks": gate_checks - gate_allowed,
            },
            "entry_gate_decisions": cycle.entry_gate_decisions,
            "entries": [intent.__dict__ for intent in entries],
            "exits": [intent.__dict__ for intent in exits],
            "orders": [],
        }

        if execute:
            results["orders"].extend(self._execute_exits(exits, price_map, cycle_id, ranks))
            results["orders"].extend(self._execute_entries(entries, price_map, cycle_id))
            self.state.save()
        else:
            regime_text = ""
            if cycle.regime is not None:
                regime_text = (
                    " regime_stressed=%s market_roll_impact=%s threshold=%s entries_blocked=%s"
                    % (
                        cycle.regime.is_stressed,
                        cycle.regime.market_roll_impact,
                        cycle.regime.threshold,
                        cycle.regime.entries_blocked,
                    )
                )
            log.info("DRY RUN: %s exits, %s entries%s", len(exits), len(entries), regime_text)

        log_jsonl("live_cycles.jsonl", results)
        return results

    def _execute_exits(
        self,
        intents,
        price_map: dict[str, float],
        cycle_id: str,
        ranks: dict[str, int],
    ) -> list[dict[str, Any]]:
        orders = []
        for intent in intents:
            price = price_map[intent.pair]
            position = self.state.positions.get(intent.pair)
            result = self.executor.sell(intent.pair, intent.quantity, price, use_limit=False)
            telemetry = order_telemetry_record(
                cycle_id=cycle_id,
                intent=intent,
                result=result,
                reference_price=price,
            )
            log_order_telemetry(telemetry)
            log_closed_trade(
                closed_trade_record(
                    cycle_id=cycle_id,
                    position=position,
                    intent=intent,
                    order_record=telemetry,
                    latest_rank=ranks.get(intent.pair),
                )
                if position is not None
                else None
            )
            orders.append({"intent": intent.__dict__, "result": result, "telemetry": telemetry})
            if result and result.get("Success"):
                self.state.remove_position(intent.pair)
        return orders

    def _execute_entries(self, intents, price_map: dict[str, float], cycle_id: str) -> list[dict[str, Any]]:
        orders = []
        for intent in intents:
            price = price_map[intent.pair]
            result = self.executor.buy(intent.pair, intent.notional_usd, price, use_limit=False)
            telemetry = order_telemetry_record(
                cycle_id=cycle_id,
                intent=intent,
                result=result,
                reference_price=price,
            )
            log_order_telemetry(telemetry)
            orders.append({"intent": intent.__dict__, "result": result, "telemetry": telemetry})
            if result and result.get("Success"):
                quantity, avg_price, order_id = order_fill_details(
                    result,
                    fallback_notional=intent.notional_usd,
                    fallback_price=price,
                )
                self.state.upsert_position(
                    LivePositionState(
                        pair=intent.pair,
                        quantity=quantity,
                        entry_price=avg_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        entry_score=intent.score,
                        order_id=order_id,
                    )
                )
        return orders


def wallet_usd_cash(wallet: dict[str, Any]) -> float:
    balances = wallet.get("USD", {})
    if isinstance(balances, dict):
        return float(balances.get("Free", 0) or 0)
    return float(balances or 0)


def compute_portfolio_value(wallet: dict[str, Any], prices: dict[str, float]) -> float:
    value = wallet_usd_cash(wallet)
    usd_locked = wallet.get("USD", {}).get("Lock", 0) if isinstance(wallet.get("USD"), dict) else 0
    value += float(usd_locked or 0)
    for asset, balances in wallet.items():
        asset_name = str(asset).upper()
        if asset_name == "USD":
            continue
        pair = f"{asset_name}/USD"
        price = prices.get(pair)
        if price is None:
            continue
        if isinstance(balances, dict):
            quantity = float(balances.get("Free", 0) or 0) + float(
                balances.get("Lock", balances.get("Locked", 0)) or 0
            )
        else:
            quantity = float(balances or 0)
        value += quantity * price
    return value


def live_price_map(ticker: dict[str, Any], scores: pd.DataFrame) -> dict[str, float]:
    prices = {
        str(pair): float(data.get("LastPrice", 0) or 0)
        for pair, data in ticker.items()
        if isinstance(data, dict) and float(data.get("LastPrice", 0) or 0) > 0
    }
    for row in scores[["pair", "close"]].dropna().itertuples(index=False):
        prices.setdefault(str(row.pair), float(row.close))
    return prices


def order_fill_details(
    result: dict[str, Any],
    *,
    fallback_notional: float,
    fallback_price: float,
) -> tuple[float, float, int | None]:
    detail = result.get("OrderDetail", {}) if isinstance(result, dict) else {}
    filled_qty = float(detail.get("FilledQuantity", 0) or 0)
    avg_price = float(detail.get("FilledAverPrice", detail.get("FilledAvgPrice", 0)) or 0)
    if filled_qty <= 0 and fallback_price > 0:
        filled_qty = fallback_notional / fallback_price
    if avg_price <= 0:
        avg_price = fallback_price
    order_id = detail.get("OrderID")
    return filled_qty, avg_price, int(order_id) if order_id not in {None, ""} else None


def _latest_open_time(scores: pd.DataFrame) -> int | str | None:
    if scores.empty or "open_time" not in scores.columns:
        return None
    value = scores["open_time"].max()
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value
