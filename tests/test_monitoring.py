import json
from pathlib import Path

import pytest

from bot.live_state import LivePositionState, LiveState
from bot.monitoring import (
    forward_report,
    health_report,
    positions_rows,
    summary_reports,
    write_summary_reports,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_positions_rows_flags_wallet_only_missing_entry_metadata(tmp_path):
    state = LiveState(tmp_path / "state.json")
    wallet = {"BTC": {"Free": 0.5, "Lock": 0}}

    rows = positions_rows(wallet, state, prices={"BTC/USD": 100.0}, pairs=["BTC/USD"])

    assert rows[0]["wallet_only"] is True
    assert rows[0]["missing_entry_metadata"] is True
    assert rows[0]["market_value"] == pytest.approx(50.0)


def test_health_report_flags_state_wallet_mismatch_without_emitting_events(tmp_path):
    state = LiveState(tmp_path / "state.json")
    state.upsert_position(
        LivePositionState(
            pair="ETH/USD",
            quantity=1.0,
            entry_price=100.0,
            entry_time="2026-01-01T00:00:00+00:00",
        )
    )

    report = health_report(
        wallet={},
        state=state,
        log_dir=tmp_path,
        pairs=["ETH/USD"],
        emit_events=False,
    )

    assert report["status"] == "warning"
    assert {issue["event_type"] for issue in report["issues"]} >= {
        "state_only_position",
        "quantity_mismatch",
        "no_live_cycles",
    }


def test_summary_reports_writes_expected_csvs(tmp_path):
    write_jsonl(
        tmp_path / "live_cycles.jsonl",
        [{"logged_at": "2099-01-01T00:00:00+00:00", "execute": True}],
    )
    write_jsonl(
        tmp_path / "trades.jsonl",
        [
            {
                "logged_at": "2099-01-01T00:00:00+00:00",
                "pair": "BTC/USD",
                "side": "BUY",
                "success": True,
                "slippage_bps": 5.0,
            }
        ],
    )
    write_jsonl(
        tmp_path / "closed_trades.jsonl",
        [
            {
                "logged_at": "2099-01-01T00:00:00+00:00",
                "pair": "BTC/USD",
                "exit_reason": "gain_threshold",
                "gross_pnl": 10.0,
                "gross_return": 0.03,
            }
        ],
    )

    reports = summary_reports(log_dir=tmp_path, since_hours=24 * 365 * 100)
    paths = write_summary_reports(reports, tmp_path / "reports")

    assert reports["summary"]["orders"].iloc[0] == 1
    assert reports["pair_attribution"]["gross_pnl"].iloc[0] == pytest.approx(10.0)
    assert {path.name for path in paths} == {
        "summary.csv",
        "pair_attribution.csv",
        "exit_reason_attribution.csv",
        "slippage.csv",
    }


class FakeBinance:
    def fetch_klines(self, pair, interval="1h", limit=1000):
        assert pair == "BTC/USD"
        return [
            {"open_time": 1000, "close": 100.0},
            {"open_time": 3_601_000, "close": 101.0},
            {"open_time": 21_601_000, "close": 106.0},
        ]


def test_forward_report_computes_horizon_returns(tmp_path):
    write_jsonl(
        tmp_path / "live_scores.jsonl",
        [
            {
                "logged_at": "2099-01-01T00:00:00+00:00",
                "pair": "BTC/USD",
                "open_time": 1000,
                "close": 100.0,
                "score": 1.0,
                "rank": 1,
                "intended_entry": True,
            }
        ],
    )

    report = forward_report(
        log_dir=tmp_path,
        since_hours=24 * 365 * 100,
        horizons=(1, 6),
        binance=FakeBinance(),
    )

    assert set(report["horizon"]) == {1, 6}
    assert report.loc[report["horizon"].eq(1), "intent_mean_return"].iloc[0] == pytest.approx(0.01)
    assert report.loc[report["horizon"].eq(6), "intent_mean_return"].iloc[0] == pytest.approx(0.06)
