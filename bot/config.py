"""Runtime configuration for the Roostoo bot.

Environment variables are loaded from `.env` when present. Values are exported
as module constants so legacy modules from `quant-competition` can be ported
without changing every import at once.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip().rstrip("/")
        for item in os.getenv(name, default).split(",")
        if item.strip()
    )


# --- Roostoo API ---
API_KEY = os.getenv("ROOSTOO_API_KEY", "")
API_SECRET = os.getenv("ROOSTOO_API_SECRET", "")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))

# --- Binance market data ---
# data-api.binance.vision is the default because api.binance.com can return
# HTTP 451 from some networks while the public data mirror still serves klines.
BINANCE_BASE_URL = os.getenv(
    "BINANCE_BASE_URL", "https://data-api.binance.vision"
).rstrip("/")
BINANCE_FALLBACK_URLS = _csv_env(
    "BINANCE_FALLBACK_URLS",
    "https://data-api.binance.vision,https://api.binance.com",
)

# --- Local storage/logging ---
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "candles")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_CONSOLE = os.getenv("BOT_LOG_CONSOLE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# --- Execution defaults ---
LIMIT_ORDER_TIMEOUT_SECONDS = int(os.getenv("LIMIT_ORDER_TIMEOUT_SECONDS", "90"))
LIMIT_ORDER_OFFSET_BPS = float(os.getenv("LIMIT_ORDER_OFFSET_BPS", "1"))

# --- Live ridge score strategy defaults ---
LIVE_STATE_PATH = Path(os.getenv("LIVE_STATE_PATH", str(PROJECT_ROOT / "data" / "live_state.json")))
LIVE_INTERVAL = os.getenv("LIVE_INTERVAL", "1h")
LIVE_HISTORY_LIMIT = int(os.getenv("LIVE_HISTORY_LIMIT", "1000"))
LIVE_FORWARD_HORIZON = int(os.getenv("LIVE_FORWARD_HORIZON", "24"))
LIVE_MODEL = os.getenv("LIVE_MODEL", "momentum_plus_roll_plus_interaction")
LIVE_POSITION_FRACTION = float(os.getenv("LIVE_POSITION_FRACTION", str(1 / 20)))
LIVE_TAKE_PROFIT = float(os.getenv("LIVE_TAKE_PROFIT", os.getenv("LIVE_EXIT_THRESHOLD", "0.50")))
LIVE_STOP_LOSS = float(os.getenv("LIVE_STOP_LOSS", os.getenv("LIVE_EXIT_THRESHOLD", "0.50")))
LIVE_CYCLE_DELAY_SECONDS = int(os.getenv("LIVE_CYCLE_DELAY_SECONDS", "5"))
LIVE_MIN_HISTORY_BARS = int(os.getenv("LIVE_MIN_HISTORY_BARS", "200"))
LIVE_TOP_K = int(os.getenv("LIVE_TOP_K", "0"))
LIVE_MAX_NEW_ENTRIES = int(os.getenv("LIVE_MAX_NEW_ENTRIES", "0"))
LIVE_MAX_POSITIONS = int(os.getenv("LIVE_MAX_POSITIONS", "0"))

# Keep the old 43-coin universe available as a starting point. New strategy work
# should test any narrower universe explicitly before trading it live.
TRADEABLE_COINS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "BNB/USD",
    "XRP/USD",
    "DOGE/USD",
    "ADA/USD",
    "AVAX/USD",
    "LINK/USD",
    "DOT/USD",
    "SUI/USD",
    "NEAR/USD",
    "LTC/USD",
    "TON/USD",
    "UNI/USD",
    "FET/USD",
    "HBAR/USD",
    "XLM/USD",
    "FIL/USD",
    "APT/USD",
    "ARB/USD",
    "SEI/USD",
    "PEPE/USD",
    "SHIB/USD",
    "FLOKI/USD",
    "WIF/USD",
    "BONK/USD",
    "TRX/USD",
    "ICP/USD",
    "AAVE/USD",
    "WLD/USD",
    "ONDO/USD",
    "CRV/USD",
    "PENDLE/USD",
    "ENA/USD",
    "TAO/USD",
    "POL/USD",
    "ZEC/USD",
    "TRUMP/USD",
    "EIGEN/USD",
    "VIRTUAL/USD",
    "CAKE/USD",
    "PAXG/USD",
]

BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT",
    "DOGE/USD": "DOGEUSDT",
    "ADA/USD": "ADAUSDT",
    "AVAX/USD": "AVAXUSDT",
    "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT",
    "SUI/USD": "SUIUSDT",
    "NEAR/USD": "NEARUSDT",
    "LTC/USD": "LTCUSDT",
    "TON/USD": "TONUSDT",
    "UNI/USD": "UNIUSDT",
    "FET/USD": "FETUSDT",
    "HBAR/USD": "HBARUSDT",
    "XLM/USD": "XLMUSDT",
    "FIL/USD": "FILUSDT",
    "APT/USD": "APTUSDT",
    "ARB/USD": "ARBUSDT",
    "SEI/USD": "SEIUSDT",
    "PEPE/USD": "PEPEUSDT",
    "SHIB/USD": "SHIBUSDT",
    "FLOKI/USD": "FLOKIUSDT",
    "WIF/USD": "WIFUSDT",
    "BONK/USD": "BONKUSDT",
    "TRX/USD": "TRXUSDT",
    "ICP/USD": "ICPUSDT",
    "AAVE/USD": "AAVEUSDT",
    "WLD/USD": "WLDUSDT",
    "ONDO/USD": "ONDOUSDT",
    "CRV/USD": "CRVUSDT",
    "PENDLE/USD": "PENDLEUSDT",
    "ENA/USD": "ENAUSDT",
    "TAO/USD": "TAOUSDT",
    "POL/USD": "POLUSDT",
    "ZEC/USD": "ZECUSDT",
    "TRUMP/USD": "TRUMPUSDT",
    "EIGEN/USD": "EIGENUSDT",
    "VIRTUAL/USD": "VIRTUALUSDT",
    "CAKE/USD": "CAKEUSDT",
    "PAXG/USD": "PAXGUSDT",
}
