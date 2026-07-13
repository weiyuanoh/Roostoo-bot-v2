import pandas as pd
import pytest

from bot.live_models import _validate_pair_history


def test_live_artifact_history_validation_requires_full_pair_coverage():
    frame = pd.DataFrame(
        [
            {"pair": "AAA/USD", "timestamp": pd.Timestamp("2024-01-01", tz="UTC")},
            {"pair": "AAA/USD", "timestamp": pd.Timestamp("2026-01-01", tz="UTC")},
            {"pair": "BBB/USD", "timestamp": pd.Timestamp("2025-01-01", tz="UTC")},
            {"pair": "BBB/USD", "timestamp": pd.Timestamp("2026-01-01", tz="UTC")},
        ]
    )

    with pytest.raises(ValueError, match="BBB/USD"):
        _validate_pair_history(
            frame,
            pairs=("AAA/USD", "BBB/USD"),
            required_start=pd.Timestamp("2024-01-01", tz="UTC"),
            required_end=pd.Timestamp("2026-01-01", tz="UTC"),
        )
