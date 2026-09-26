from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trader.config import GatewayConfig
from trader.gateway_client import GatewayError, PythonGetawayClient


def _config():
    return GatewayConfig(base_url="http://gw.local", timeout_seconds=5, login=123, password="pw", server="Demo")


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_login_requires_credentials():
    client = PythonGetawayClient(GatewayConfig(base_url="http://gw.local", timeout_seconds=5))
    with pytest.raises(GatewayError, match="missing"):
        client.login()


@patch("trader.gateway_client.requests.request")
def test_login_stores_session_token(mock_request):
    mock_request.return_value = _mock_response({"token": "tok123", "slot_id": "slot-1", "expires_in": 3600})
    client = PythonGetawayClient(_config())

    session = client.login()

    assert session.token == "tok123"
    called_kwargs = mock_request.call_args.kwargs
    assert called_kwargs["json"] == {"login": 123, "password": "pw", "server": "Demo"}


@patch("trader.gateway_client.requests.request")
def test_candles_corrects_broker_server_time_to_utc(mock_request):
    client = PythonGetawayClient(_config())
    client._session = type("S", (), {"token": "tok"})()

    # Broker server clock reads 2 hours ahead of real UTC (a plausible EET-style offset).
    real_utc_now = pd.Timestamp.now(tz="UTC")
    server_epoch = int((real_utc_now + pd.Timedelta(hours=2)).timestamp())
    bar_epoch = server_epoch - 3600  # one server-hour before "now"

    def side_effect(method, url, **kwargs):
        if "/market/tick/" in url:
            return _mock_response({"time": server_epoch, "bid": 1.1, "ask": 1.1002})
        if "/market/candles/" in url:
            return _mock_response({"candles": [{"time": bar_epoch, "open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1005}]})
        raise AssertionError(f"unexpected url {url}")

    mock_request.side_effect = side_effect

    df = client.candles("EURUSD", "H1", 1)

    expected_utc = pd.Timestamp(bar_epoch, unit="s", tz="UTC") - pd.Timedelta(hours=2)
    assert df["timestamp"].iloc[0] == expected_utc
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close"]


@patch("trader.gateway_client.requests.request")
def test_open_position_maps_direction_to_broker_side(mock_request):
    mock_request.return_value = _mock_response({"retcode": 10009})
    client = PythonGetawayClient(_config())
    client._session = type("S", (), {"token": "tok"})()

    client.open_position("EURUSD", "long", volume=0.1, sl=1.0, tp=1.1)

    called_kwargs = mock_request.call_args.kwargs
    assert called_kwargs["json"]["side"] == "BUY"
    assert mock_request.call_args.args[1] == "http://gw.local/trading/open"

    client.open_position("EURUSD", "short", volume=0.1, sl=1.0, tp=1.1)
    assert mock_request.call_args.kwargs["json"]["side"] == "SELL"


@patch("trader.gateway_client.requests.request")
def test_account_hits_account_endpoint_with_auth(mock_request):
    mock_request.return_value = _mock_response({"balance": 10_000.0, "equity": 9_950.0, "trade_mode": 0})
    client = PythonGetawayClient(_config())
    client._session = type("S", (), {"token": "tok"})()

    data = client.account()

    assert data["equity"] == 9_950.0
    assert mock_request.call_args.args[1] == "http://gw.local/account"
    assert mock_request.call_args.kwargs["headers"] == {"Authorization": "Bearer tok"}


@patch("trader.gateway_client.requests.request")
def test_position_history_hits_the_right_ticket_endpoint(mock_request):
    mock_request.return_value = _mock_response({"position_id": 555, "realized_pnl": 12.3, "deals": []})
    client = PythonGetawayClient(_config())
    client._session = type("S", (), {"token": "tok"})()

    data = client.position_history(555)

    assert data["position_id"] == 555
    assert mock_request.call_args.args[1] == "http://gw.local/history/position/555"


def _client_with_tick_age(seconds_behind_server_clock: float) -> PythonGetawayClient:
    """Servidor a UTC+3; el último tick tiene `seconds_behind...` de antigüedad."""
    client = PythonGetawayClient(_config())
    tick_time = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3) - pd.Timedelta(seconds=seconds_behind_server_clock)).timestamp()
    client._request = MagicMock(return_value={"time": tick_time})
    return client


def test_offset_from_fresh_tick():
    assert _client_with_tick_age(1).server_utc_offset() == pd.Timedelta(hours=3)


def test_offset_rejects_stale_weekend_tick_and_keeps_last_good():
    # bug 2026-09-19: tick del viernes leído el sábado -> offset -20h -> CSVs corridos +23h
    stale = _client_with_tick_age(23 * 3600 + 17 * 60)
    with pytest.raises(RuntimeError, match="no determinable"):
        stale.server_utc_offset()

    stale._last_good_offset = pd.Timedelta(hours=3)
    assert stale.server_utc_offset() == pd.Timedelta(hours=3)


def test_offset_rejects_tick_a_few_minutes_old():
    with pytest.raises(RuntimeError):
        _client_with_tick_age(7 * 60).server_utc_offset()
