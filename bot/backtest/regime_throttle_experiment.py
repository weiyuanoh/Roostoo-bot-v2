"""Compare momentum-only baseline against the universe roll-impact regime throttle."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest.ridge_score_portfolio import (
    DEFAULT_INITIAL_CASH,
    filter_pairs,
    parse_alphas,
    parse_pairs,
    run_portfolio_backtest,
    write_outputs,
)
from bot.strategy.regime import RegimeThrottleConfig
from bot.strategy.ridge import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_IS_MONTHS,
    DEFAULT_OS_MONTHS,
    DEFAULT_RIDGE_ALPHAS,
    DEFAULT_STEP_MONTHS,
    load_model_frame,
)


DEFAULT_PAIRS = "BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD,TRX/USD,ZEC/USD,TAO/USD"
DEFAULT_OUTPUT_DIR = Path("reports/backtests")


def main() -> int:
    parser = argparse.ArgumentParser(description="Universe roll-impact regime throttle experiment")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="roll_regime_throttle")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--is-months", type=int, default=DEFAULT_IS_MONTHS)
    parser.add_argument("--os-months", type=int, default=DEFAULT_OS_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--ridge-alphas", default=",".join(str(alpha) for alpha in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-new-entries", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--position-fraction", type=float, default=0.25)
    parser.add_argument("--tp", type=float, default=0.03)
    parser.add_argument("--sl", type=float, default=0.015)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", default="0,5")
    parser.add_argument("--regime-aggregation", choices=("median", "mean"), default="median")
    parser.add_argument("--regime-lookback-bars", type=int, default=720)
    parser.add_argument("--regime-percentile", type=float, default=0.80)
    parser.add_argument("--regime-min-history-bars", type=int, default=168)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feature_path = Path(args.features)
    pairs = parse_pairs(args.pairs)
    frame = filter_pairs(load_model_frame(feature_path, horizon=args.horizon), pairs)
    ridge_alphas = parse_alphas(args.ridge_alphas)
    slippage_values = tuple(float(part.strip()) for part in args.slippage_bps.split(",") if part.strip())
    regime_config = RegimeThrottleConfig(
        aggregation=args.regime_aggregation,
        lookback_bars=args.regime_lookback_bars,
        percentile=args.regime_percentile,
        min_history_bars=args.regime_min_history_bars,
    )

    results = []
    artifacts: list[str] = []
    for slippage_bps in slippage_values:
        for label, config in (("baseline", None), ("candidate", regime_config)):
            prefix = f"{args.prefix}_{label}_slip{_tag_number(slippage_bps)}"
            summary, equity, trades = run_portfolio_backtest(
                frame,
                model="momentum_only",
                horizon=args.horizon,
                is_months=args.is_months,
                os_months=args.os_months,
                step_months=args.step_months,
                ridge_alphas=ridge_alphas,
                initial_cash=args.initial_cash,
                position_fraction=args.position_fraction,
                take_profit=args.tp,
                stop_loss=args.sl,
                fee_bps=args.fee_bps,
                slippage_bps=slippage_bps,
                max_positions=args.max_positions,
                top_k=args.top_k,
                max_new_entries=args.max_new_entries,
                regime_config=config,
            )
            metadata = {
                "label": label,
                "feature_path": str(feature_path),
                "model": "momentum_only",
                "horizon": args.horizon,
                "is_months": args.is_months,
                "os_months": args.os_months,
                "step_months": args.step_months,
                "ridge_alphas": ridge_alphas,
                "initial_cash": args.initial_cash,
                "position_fraction": args.position_fraction,
                "take_profit": args.tp,
                "stop_loss": args.sl,
                "fee_bps": args.fee_bps,
                "slippage_bps": slippage_bps,
                "max_positions": args.max_positions,
                "top_k": args.top_k,
                "max_new_entries": args.max_new_entries,
                "regime_throttle": config is not None,
                "regime_aggregation": args.regime_aggregation,
                "regime_lookback_bars": args.regime_lookback_bars,
                "regime_percentile": args.regime_percentile,
                "regime_min_history_bars": args.regime_min_history_bars,
                "pair_universe": sorted(frame["pair"].unique()),
                "rows": int(len(frame)),
                "start": str(frame["timestamp"].min()),
                "end": str(frame["timestamp"].max()),
            }
            write_outputs(summary, equity, trades, output_dir, prefix, metadata)
            artifacts.extend(
                [
                    f"{prefix}_summary.csv",
                    f"{prefix}_equity.csv",
                    f"{prefix}_trades.csv",
                    f"{prefix}_metadata.json",
                ]
            )
            results.append(
                {
                    "label": label,
                    "slippage_bps": slippage_bps,
                    "summary": summary,
                    "equity": equity,
                    "trades": trades,
                    "prefix": prefix,
                }
            )

    report = build_report(args, frame, results, artifacts)
    report_path = output_dir / f"{args.prefix}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


def build_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    results: list[dict],
    artifacts: list[str],
) -> str:
    comparison_rows = [_comparison_row(result) for result in results]
    lines = [
        "# Backtest: universe roll-impact regime throttle",
        "",
        f"Run date: {date.today().isoformat()}",
        "",
        "## Hypothesis",
        "",
        "High universe-wide `roll_impact` is a liquidity-stress regime. Momentum entries opened during this regime should have worse realised trade outcomes. Blocking new entries during this regime should improve the momentum-only baseline.",
        "",
        "## Baseline score math",
        "",
        "```text",
        "score[i,t] = beta_momentum * z_momentum[i,t]",
        "```",
        "",
        "## Regime math",
        "",
        "```text",
        f"market_roll_impact[t] = {args.regime_aggregation}_i(roll_impact[i,t])",
        "",
        f"threshold[t] = {_ordinal_percentile(args.regime_percentile)} percentile of prior {args.regime_lookback_bars} hourly market_roll_impact values",
        "",
        "is_stressed[t] = market_roll_impact[t] >= threshold[t]",
        "```",
        "",
        "The threshold uses only prior bars. If history is insufficient, `is_stressed = False`.",
        "",
        "## Trading rule",
        "",
        "```text",
        "if is_stressed[t]:",
        "    no new entries",
        "else:",
        "    trade normally",
        "```",
        "",
        "Existing positions still exit normally through TP/SL.",
        "",
        "## Parameters",
        "",
        f"- Pairs: `{', '.join(sorted(frame['pair'].unique()))}`",
        "- Model: `momentum_only`",
        f"- Horizon: `{args.horizon}`",
        f"- IS/OS: `{args.is_months}m/{args.os_months}m`",
        f"- Top K: `{args.top_k}`",
        f"- Max new entries: `{args.max_new_entries}`",
        f"- Max positions: `{args.max_positions}`",
        f"- Position fraction: `{args.position_fraction}`",
        f"- TP/SL: `{args.tp}` / `{args.sl}`",
        f"- Fee bps: `{args.fee_bps}`",
        f"- Slippage bps: `{args.slippage_bps}`",
        f"- Regime aggregation: `{args.regime_aggregation}`",
        "",
        "## Comparison",
        "",
        _markdown_table(
            [
                "Run",
                "Slippage",
                "Return",
                "Max DD",
                "Ret/DD",
                "Sharpe",
                "Sortino",
                "Win",
                "Stop",
                "TP",
                "Avg stopped loss",
                "Median hold",
                "Trades",
                "Stressed",
            ],
            comparison_rows,
        ),
        "",
        "## Pair attribution",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result['label']} slippage={result['slippage_bps']:g} bps",
                "",
                _pair_attribution_table(result["trades"]),
                "",
            ]
        )

    lines.extend(["## Artifacts", ""])
    lines.extend(f"- `{Path('reports/backtests') / artifact}`" for artifact in artifacts)
    lines.append("")
    return "\n".join(lines)


def _comparison_row(result: dict) -> list[str]:
    summary = result["summary"]
    trades = result["trades"]
    row = summary.iloc[-1] if not summary.empty else pd.Series(dtype=float)
    exits = trades[trades["side"].eq("SELL")] if not trades.empty else pd.DataFrame()
    win_rate = float((exits["pnl"] > 0).mean() * 100) if not exits.empty else np.nan
    stop_rate = float((exits["reason"].eq("loss_threshold")).mean() * 100) if not exits.empty else np.nan
    tp_rate = float((exits["reason"].eq("gain_threshold")).mean() * 100) if not exits.empty else np.nan
    return [
        result["label"],
        f"{result['slippage_bps']:g}",
        _pct(row.get("total_return_pct")),
        _pct(row.get("max_drawdown_pct")),
        _num(row.get("return_over_max_drawdown")),
        _num(row.get("sharpe")),
        _num(row.get("sortino")),
        _pct(win_rate),
        _pct(stop_rate),
        _pct(tp_rate),
        _pct(row.get("avg_stopped_loss_pct")),
        _hours(row.get("median_holding_hours")),
        str(int(row.get("trades", 0) or 0)),
        _pct(row.get("stressed_hour_pct")),
    ]


def _pair_attribution_table(trades: pd.DataFrame) -> str:
    if trades.empty or "side" not in trades.columns:
        return "_No trades._"
    exits = trades[trades["side"].eq("SELL")].copy()
    if exits.empty:
        return "_No closed trades._"
    grouped = exits.groupby("pair", observed=True)
    rows = []
    for pair, group in grouped:
        rows.append(
            [
                str(pair),
                str(len(group)),
                _pct(float((group["pnl"] > 0).mean() * 100)),
                _pct(float(group["return_pct"].mean() * 100)),
                _pct(float(group["return_pct"].median() * 100)),
                _usd(float(group["pnl"].sum())),
            ]
        )
    rows.sort(key=lambda item: float(item[-1].replace(",", "")), reverse=True)
    return _markdown_table(["Pair", "Closed", "Win", "Avg return", "Median return", "PnL"], rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}%"


def _num(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.3f}"


def _hours(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}h"


def _usd(value: float) -> str:
    return f"{value:,.0f}"


def _tag_number(value: float) -> str:
    return str(value).replace(".", "p").rstrip("0").rstrip("p")


def _ordinal_percentile(value: float) -> str:
    percentile = int(round(value * 100))
    suffix = "th"
    if percentile % 100 not in {11, 12, 13}:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")
    return f"{percentile}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
