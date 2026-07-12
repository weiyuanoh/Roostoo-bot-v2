"""Local candle storage helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.binance_data import Candle
from bot.config import DATA_DIR


class CandleStore:
    """Persist candle batches as CSV files under `DATA_DIR`."""

    def __init__(self, root: Path = DATA_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, pair: str, interval: str) -> Path:
        safe_pair = pair.replace("/", "_")
        return self.root / f"{safe_pair}_{interval}.csv"

    def write_csv(self, pair: str, interval: str, candles: list[Candle]) -> Path:
        path = self.path_for(pair, interval)
        frame = pd.DataFrame(candles)
        if frame.empty:
            frame.to_csv(path, index=False)
            return path

        frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last")
        frame.to_csv(path, index=False)
        return path

    def append_csv(self, pair: str, interval: str, candles: list[Candle]) -> Path:
        path = self.path_for(pair, interval)
        if path.exists():
            existing = pd.read_csv(path)
            frame = pd.concat([existing, pd.DataFrame(candles)], ignore_index=True)
            frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last")
            frame.to_csv(path, index=False)
            return path
        return self.write_csv(pair, interval, candles)

    def read_csv(self, pair: str, interval: str) -> pd.DataFrame:
        path = self.path_for(pair, interval)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        if frame.empty:
            return frame
        frame["pair"] = pair
        return frame.sort_values("open_time").drop_duplicates("open_time", keep="last")

    def read_many(self, pairs: list[str] | tuple[str, ...], interval: str) -> pd.DataFrame:
        frames = [self.read_csv(pair, interval) for pair in pairs]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["pair", "open_time"]).reset_index(drop=True)
