from bot.config import (
    DEPLOYMENT_PAIRS,
    DEPLOYMENT_POLICY,
    LIVE_MAX_NEW_ENTRIES,
    LIVE_MAX_POSITIONS,
    LIVE_MODEL,
    LIVE_POSITION_FRACTION,
    LIVE_STOP_LOSS,
    LIVE_TAKE_PROFIT,
    LIVE_TOP_K,
    REGIME_TRAIN_MONTHS,
    RIDGE_TRAIN_MONTHS,
    TRADEABLE_COINS,
)
from bot.main import parse_pairs


def test_live_defaults_match_v1_fixed_regime_deployment_profile():
    assert DEPLOYMENT_POLICY == "v1_fixed_regime"
    assert LIVE_MODEL == "momentum_only"
    assert LIVE_TOP_K == 1
    assert LIVE_MAX_NEW_ENTRIES == 1
    assert LIVE_MAX_POSITIONS == 3
    assert LIVE_POSITION_FRACTION == 0.25
    assert LIVE_TAKE_PROFIT == 0.03
    assert LIVE_STOP_LOSS == 0.015
    assert RIDGE_TRAIN_MONTHS == 3
    assert REGIME_TRAIN_MONTHS == 24


def test_all_pair_parser_uses_deployment_universe():
    assert parse_pairs("all") == TRADEABLE_COINS == DEPLOYMENT_PAIRS
    assert len(TRADEABLE_COINS) == 25
