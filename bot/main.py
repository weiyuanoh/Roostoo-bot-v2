"""Command-line utilities for the new Roostoo bot environment."""

from __future__ import annotations

import argparse
import signal
from datetime import datetime, timezone
from collections.abc import Iterable
from pathlib import Path

from bot.binance_data import BinanceData
from bot.config import (
    DATA_DIR,
    LIVE_CYCLE_DELAY_SECONDS,
    LIVE_FORWARD_HORIZON,
    LIVE_HISTORY_LIMIT,
    LIVE_INTERVAL,
    LIVE_MAX_NEW_ENTRIES,
    LIVE_MAX_POSITIONS,
    LIVE_MODEL,
    LIVE_POSITION_FRACTION,
    LIVE_STATE_PATH,
    LIVE_STOP_LOSS,
    LIVE_TAKE_PROFIT,
    LIVE_TOP_K,
    TRADEABLE_COINS,
)
from bot.data_store import CandleStore
from bot.live_trader import RidgeLiveConfig, RidgeLiveTrader
from bot.liquidate import liquidate_positions
from bot.live_state import LiveState
from bot.monitoring import (
    default_output_dir,
    forward_report,
    health_report,
    positions_rows,
    regime_reports,
    summary_reports,
    write_forward_report,
    write_health_report,
    write_regime_reports,
    write_summary_reports,
)
from bot.roostoo_client import RoostooClient
from bot.scheduler import next_hour_boundary, sleep_until
from bot.strategy.regime import RegimeThrottleConfig


def parse_pairs(raw: str | None) -> list[str]:
    if not raw:
        return ["BTC/USD", "ETH/USD", "SOL/USD"]
    if raw.lower() == "all":
        return TRADEABLE_COINS
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def collect_candles(
    pairs: Iterable[str],
    interval: str,
    limit: int,
    output_dir=DATA_DIR,
) -> list[tuple[str, int, str]]:
    feed = BinanceData()
    store = CandleStore(output_dir)
    results: list[tuple[str, int, str]] = []
    for pair in pairs:
        candles = feed.fetch_klines(pair, interval=interval, limit=limit)
        if not candles:
            results.append((pair, 0, ""))
            continue
        path = store.append_csv(pair, interval, candles)
        results.append((pair, len(candles), str(path)))
    return results


def smoke() -> int:
    roostoo = RoostooClient()
    server_time = roostoo.server_time()
    exchange = roostoo.exchange_info()

    binance = BinanceData()
    candles = binance.fetch_klines("BTC/USD", interval="1h", limit=1)

    print("Roostoo server time:", server_time or "unavailable")
    print("Roostoo exchange pairs:", len((exchange or {}).get("TradePairs", {})))
    if candles:
        print("Binance BTC/USD latest close:", candles[-1]["close"])
    else:
        print("Binance BTC/USD latest close: unavailable")

    return 0 if server_time and candles else 1


def show_balance() -> int:
    wallet = RoostooClient().balance()
    if wallet is None:
        print("Balance unavailable. Check ROOSTOO_API_KEY and ROOSTOO_API_SECRET.")
        return 1
    for asset, balances in sorted(wallet.items()):
        free = balances.get("Free", 0)
        locked = balances.get("Lock", 0)
        if free or locked:
            print(f"{asset}: free={free} locked={locked}")
    return 0


def _parse_alphas(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _live_config(args: argparse.Namespace) -> RidgeLiveConfig:
    take_profit = args.tp
    stop_loss = args.sl
    if args.exit_threshold is not None:
        take_profit = args.exit_threshold
        stop_loss = args.exit_threshold
    return RidgeLiveConfig(
        pairs=tuple(parse_pairs(args.pairs)),
        interval=args.interval,
        history_limit=args.history_limit,
        horizon=args.horizon,
        model=args.model,
        ridge_alphas=_parse_alphas(args.ridge_alphas),
        position_fraction=args.position_fraction,
        take_profit=take_profit,
        stop_loss=stop_loss,
        top_k=_positive_or_none(args.top_k),
        max_new_entries=_positive_or_none(args.max_new_entries),
        max_positions_override=_positive_or_none(args.max_positions),
        regime_config=_regime_config(args),
        state_path=Path(args.state_path),
    )


def _positive_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _regime_config(args: argparse.Namespace) -> RegimeThrottleConfig | None:
    if not args.regime_throttle:
        return None
    return RegimeThrottleConfig(
        aggregation=args.regime_aggregation,
        lookback_bars=args.regime_lookback_bars,
        percentile=args.regime_percentile,
        min_history_bars=args.regime_min_history_bars,
    )


def live_once(args: argparse.Namespace) -> int:
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Starting live-once cycle ({mode}).", flush=True)
    if not args.execute:
        print("Dry-run will fetch data and produce trade intents, but will not place orders.", flush=True)
    trader = RidgeLiveTrader.from_clients(config=_live_config(args))
    try:
        result = trader.run_cycle(execute=args.execute)
    except RuntimeError as exc:
        print(f"live-once failed: {exc}", flush=True)
        print("Check Roostoo connectivity/credentials with: .venv/bin/python -m bot.main balance", flush=True)
        return 1
    print(
        "execute={execute} portfolio_value={portfolio_value:.2f} "
        "exits={exits} entries={entries}{regime}".format(
            execute=result["execute"],
            portfolio_value=result["portfolio_value"],
            exits=len(result["exits"]),
            entries=len(result["entries"]),
            regime=_format_regime(result.get("regime")),
        )
    )
    return 0


def live_loop(args: argparse.Namespace) -> int:
    running = True
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Starting continuous live loop ({mode}).", flush=True)
    if not args.execute:
        print("Dry-run will fetch data and produce trade intents, but will not place orders.", flush=True)

    def _stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    trader = RidgeLiveTrader.from_clients(config=_live_config(args))
    try:
        trader.initialize()
    except RuntimeError as exc:
        print(f"live initialization failed: {exc}", flush=True)
        return 1
    while running:
        target = next_hour_boundary(delay_seconds=args.delay_seconds)
        target_dt = datetime.fromtimestamp(target, tz=timezone.utc).isoformat()
        print(f"Waiting until {target_dt} for next 1h cycle")
        if not sleep_until(target, keep_running=lambda: running):
            break
        try:
            result = trader.run_cycle(execute=args.execute)
        except RuntimeError as exc:
            print(f"live cycle failed: {exc}", flush=True)
            print("The loop will wait for the next scheduled cycle.", flush=True)
            continue
        print(
            "cycle execute={execute} portfolio_value={portfolio_value:.2f} "
            "exits={exits} entries={entries}{regime}".format(
                execute=result["execute"],
                portfolio_value=result["portfolio_value"],
                exits=len(result["exits"]),
                entries=len(result["entries"]),
                regime=_format_regime(result.get("regime")),
            )
        )
    return 0


def _format_regime(regime: dict | None) -> str:
    if not regime:
        return ""
    return (
        " regime_stressed={is_stressed} market_roll_impact={market_roll_impact} "
        "threshold={threshold} entries_blocked={entries_blocked}"
    ).format(**regime)


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pairs", default="all", help="Comma list or 'all'")
    parser.add_argument("--interval", default=LIVE_INTERVAL, help="Binance interval; strategy expects 1h")
    parser.add_argument("--history-limit", type=int, default=LIVE_HISTORY_LIMIT)
    parser.add_argument("--horizon", type=int, default=LIVE_FORWARD_HORIZON)
    parser.add_argument("--model", default=LIVE_MODEL)
    parser.add_argument("--ridge-alphas", default="0.1,1,10,100")
    parser.add_argument("--position-fraction", type=float, default=LIVE_POSITION_FRACTION)
    parser.add_argument("--top-k", type=int, default=LIVE_TOP_K, help="Only consider the top N ranked names; 0 disables")
    parser.add_argument(
        "--max-new-entries",
        type=int,
        default=LIVE_MAX_NEW_ENTRIES,
        help="Maximum new positions opened per cycle; 0 disables",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=LIVE_MAX_POSITIONS,
        help="Maximum total open positions; 0 derives from position fraction",
    )
    parser.add_argument("--tp", type=float, default=LIVE_TAKE_PROFIT, help="Take-profit return, e.g. 0.5")
    parser.add_argument("--sl", type=float, default=LIVE_STOP_LOSS, help="Stop-loss return, e.g. 0.2")
    parser.add_argument(
        "--regime-throttle",
        action="store_true",
        help="Block new entries during high universe-wide roll-impact regimes.",
    )
    parser.add_argument("--regime-aggregation", choices=("median", "mean"), default="median")
    parser.add_argument("--regime-lookback-bars", type=int, default=720)
    parser.add_argument("--regime-percentile", type=float, default=0.80)
    parser.add_argument("--regime-min-history-bars", type=int, default=168)
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help="Deprecated: set both --tp and --sl to the same value.",
    )
    parser.add_argument("--state-path", default=str(LIVE_STATE_PATH))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Place Roostoo market orders. Omit for dry-run planning only.",
    )


def liquidate(args: argparse.Namespace) -> int:
    mode = "EXECUTE" if args.execute else "DRY RUN"
    selected_pairs = None if args.pairs.lower() == "all" else parse_pairs(args.pairs)
    print(f"Starting liquidation ({mode}).", flush=True)
    if not args.execute:
        print("Dry-run will list sell intents, but will not place orders.", flush=True)
    try:
        result = liquidate_positions(
            pairs=selected_pairs,
            execute=args.execute,
            cancel_pending=args.cancel_pending,
            state_path=Path(args.state_path),
        )
    except RuntimeError as exc:
        print(f"liquidation failed: {exc}", flush=True)
        return 1

    intents = result["intents"]
    print(f"estimated_notional_usd={result['estimated_notional_usd']:.2f} positions={len(intents)}")
    for intent in intents:
        print(
            "{pair}: qty={quantity:.12g} price={price:.8g} notional={notional_usd:.2f}".format(
                **intent
            )
        )
    if args.execute:
        successes = sum(1 for order in result["orders"] if (order.get("result") or {}).get("Success"))
        print(f"orders={len(result['orders'])} successful={successes}")
    return 0


def positions(args: argparse.Namespace) -> int:
    roostoo = RoostooClient()
    wallet = roostoo.balance()
    if wallet is None:
        print("Balance unavailable. Check Roostoo connectivity/credentials.")
        return 1
    ticker = roostoo.ticker() or {}
    prices = {
        pair: float(data.get("LastPrice", 0) or 0)
        for pair, data in ticker.items()
        if isinstance(data, dict) and float(data.get("LastPrice", 0) or 0) > 0
    }
    state = LiveState(args.state_path)
    rows = positions_rows(wallet, state, prices=prices, pairs=parse_pairs(args.pairs))
    if not rows:
        print("No positions.")
        return 0
    for row in rows:
        ret = row["return_pct"]
        ret_text = "n/a" if ret is None else f"{ret * 100:.2f}%"
        print(
            "{pair}: wallet_qty={wallet_quantity:.12g} state_qty={state_quantity:.12g} "
            "price={price} value={market_value} entry={entry_price} return={ret}".format(
                pair=row["pair"],
                wallet_quantity=row["wallet_quantity"],
                state_quantity=row["state_quantity"],
                price=_fmt_optional(row["price"]),
                market_value=_fmt_optional(row["market_value"]),
                entry_price=_fmt_optional(row["entry_price"]),
                ret=ret_text,
            )
        )
        if row["missing_entry_metadata"] or row["state_only"] or row["wallet_only"]:
            print(
                "  flags: missing_entry_metadata={missing} state_only={state_only} wallet_only={wallet_only}".format(
                    missing=row["missing_entry_metadata"],
                    state_only=row["state_only"],
                    wallet_only=row["wallet_only"],
                )
            )
    return 0


def monitor_health(args: argparse.Namespace) -> int:
    wallet = RoostooClient().balance()
    state = LiveState(args.state_path)
    report = health_report(
        wallet=wallet,
        state=state,
        pairs=parse_pairs(args.pairs),
        max_cycle_age_minutes=args.max_cycle_age_minutes,
    )
    path = write_health_report(report, Path(args.output_dir))
    print(f"status={report['status']} issues={len(report['issues'])} -> {path}")
    for issue in report["issues"]:
        print(f"{issue['severity']}: {issue['message']}")
    return 0 if report["status"] != "critical" else 1


def monitor_summary(args: argparse.Namespace) -> int:
    reports = summary_reports(since_hours=args.since_hours)
    paths = write_summary_reports(reports, Path(args.output_dir))
    summary = reports["summary"]
    if not summary.empty:
        row = summary.iloc[0].to_dict()
        print(
            "cycles={cycles} orders={orders} closed_trades={closed_trades} "
            "mean_slippage_bps={mean_slippage_bps} gross_pnl={gross_pnl}".format(**row)
        )
    for path in paths:
        print(path)
    return 0


def monitor_forward(args: argparse.Namespace) -> int:
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    frame = forward_report(since_hours=args.since_hours, horizons=horizons)
    path = write_forward_report(frame, Path(args.output_dir))
    print(frame.to_string(index=False) if not frame.empty else "No forward observations.")
    print(path)
    return 0


def monitor_regime(args: argparse.Namespace) -> int:
    reports = regime_reports(since_hours=args.since_hours)
    paths = write_regime_reports(reports, Path(args.output_dir))
    for path in paths:
        print(path)
    return 0


def _fmt_optional(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Roostoo bot utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Check Roostoo public API and Binance data access")
    subparsers.add_parser("balance", help="Fetch signed Roostoo wallet balances")

    collect = subparsers.add_parser("collect", help="Collect Binance candles to local CSV files")
    collect.add_argument("--pairs", default="BTC/USD,ETH/USD,SOL/USD", help="Comma list or 'all'")
    collect.add_argument("--interval", default="1h", help="Binance interval, e.g. 1m, 5m, 1h, 1d")
    collect.add_argument("--limit", type=int, default=1000, help="Candles per pair, max 1000 per request")
    collect.add_argument("--output-dir", default=str(DATA_DIR), help="Directory for candle CSV files")

    live_once_parser = subparsers.add_parser("live-once", help="Run one live ridge cycle; dry-run by default")
    add_live_args(live_once_parser)

    live_parser = subparsers.add_parser("live", help="Run continuously on the next UTC hour + delay")
    add_live_args(live_parser)
    live_parser.add_argument("--delay-seconds", type=int, default=LIVE_CYCLE_DELAY_SECONDS)

    liquidate_parser = subparsers.add_parser("liquidate", help="Liquidate spot positions; dry-run by default")
    liquidate_parser.add_argument("--pairs", default="all", help="Comma list or 'all'")
    liquidate_parser.add_argument("--state-path", default=str(LIVE_STATE_PATH))
    liquidate_parser.add_argument(
        "--cancel-pending",
        action="store_true",
        help="Cancel pending Roostoo orders before liquidation. Only used with --execute.",
    )
    liquidate_parser.add_argument(
        "--execute",
        action="store_true",
        help="Place market sell orders. Omit for dry-run planning only.",
    )

    positions_parser = subparsers.add_parser("positions", help="Show wallet merged with local live state")
    positions_parser.add_argument("--pairs", default="all", help="Comma list or 'all'")
    positions_parser.add_argument("--state-path", default=str(LIVE_STATE_PATH))

    health_parser = subparsers.add_parser("monitor-health", help="Write local live health report")
    health_parser.add_argument("--pairs", default="all", help="Comma list or 'all'")
    health_parser.add_argument("--state-path", default=str(LIVE_STATE_PATH))
    health_parser.add_argument("--output-dir", default=str(default_output_dir()))
    health_parser.add_argument("--max-cycle-age-minutes", type=int, default=90)

    summary_parser = subparsers.add_parser("monitor-summary", help="Write local live attribution reports")
    summary_parser.add_argument("--since-hours", type=int, default=168)
    summary_parser.add_argument("--output-dir", default=str(default_output_dir()))

    forward_parser = subparsers.add_parser("monitor-forward", help="Write score-vs-forward-return report")
    forward_parser.add_argument("--since-hours", type=int, default=720)
    forward_parser.add_argument("--horizons", default="1,6,24")
    forward_parser.add_argument("--output-dir", default=str(default_output_dir()))

    regime_parser = subparsers.add_parser("monitor-regime", help="Write live regime diagnostics reports")
    regime_parser.add_argument("--since-hours", type=int, default=168)
    regime_parser.add_argument("--horizons", default="1,3,6,24", help="Reserved for report compatibility")
    regime_parser.add_argument("--output-dir", default=str(default_output_dir()))

    args = parser.parse_args()
    if args.command == "smoke":
        return smoke()
    if args.command == "balance":
        return show_balance()
    if args.command == "collect":
        results = collect_candles(
            parse_pairs(args.pairs),
            args.interval,
            args.limit,
            args.output_dir,
        )
        for pair, count, path in results:
            status = path if path else "failed"
            print(f"{pair}: {count} candles -> {status}")
        return 0 if all(count > 0 for _, count, _ in results) else 1
    if args.command == "live-once":
        return live_once(args)
    if args.command == "live":
        return live_loop(args)
    if args.command == "liquidate":
        return liquidate(args)
    if args.command == "positions":
        return positions(args)
    if args.command == "monitor-health":
        return monitor_health(args)
    if args.command == "monitor-summary":
        return monitor_summary(args)
    if args.command == "monitor-forward":
        return monitor_forward(args)
    if args.command == "monitor-regime":
        return monitor_regime(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
