#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAIRS="${PAIRS:-BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD}"
MODEL="${MODEL:-momentum_plus_roll_plus_interaction}"
POSITION_FRACTION="${POSITION_FRACTION:-0.10}"
TP="${TP:-0.5}"
SL="${SL:-0.2}"
TOP_K="${TOP_K:-1}"
MAX_NEW_ENTRIES="${MAX_NEW_ENTRIES:-1}"
MAX_POSITIONS="${MAX_POSITIONS:-1}"

echo "Dry-run liquidation plan:"
.venv/bin/python -m bot.main liquidate --pairs "$PAIRS"

if [[ "${CONFIRM_LIQUIDATE:-}" != "YES" ]]; then
  echo
  echo "Refusing to execute liquidation."
  echo "Set CONFIRM_LIQUIDATE=YES to sell positions and restart live trading."
  exit 1
fi

echo
echo "Executing liquidation:"
.venv/bin/python -m bot.main liquidate --pairs "$PAIRS" --cancel-pending --execute

echo
echo "Restarting live bot:"
exec .venv/bin/python -m bot.main live \
  --pairs "$PAIRS" \
  --model "$MODEL" \
  --top-k "$TOP_K" \
  --max-new-entries "$MAX_NEW_ENTRIES" \
  --max-positions "$MAX_POSITIONS" \
  --position-fraction "$POSITION_FRACTION" \
  --tp "$TP" \
  --sl "$SL" \
  --execute
