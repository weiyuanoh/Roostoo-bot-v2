from bot.roostoo_client import encode_params, sign_params


def test_encode_params_sorts_keys():
    assert encode_params({"z": 1, "a": 2}) == "a=2&z=1"


def test_sign_params_is_deterministic():
    headers, body, signed = sign_params(
        {"pair": "BTC/USD", "side": "BUY"},
        api_secret="secret",
        now_ms="123456",
    )

    assert body == "pair=BTC/USD&side=BUY&timestamp=123456"
    assert signed["timestamp"] == "123456"
    assert headers["MSG-SIGNATURE"] == (
        "090f951be619c50c0e387e8c1e5e86752440393ea555f72a3568abfc92123c78"
    )
