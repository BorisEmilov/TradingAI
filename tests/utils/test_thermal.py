from unittest.mock import patch

from tradingai.utils.thermal import wait_for_safe_temp


def test_returns_immediately_when_temp_unavailable():
    with patch("tradingai.utils.thermal.read_cpu_temp_c", return_value=None), \
         patch("tradingai.utils.thermal.time.sleep") as mock_sleep:
        wait_for_safe_temp(max_temp_c=80.0)
    mock_sleep.assert_not_called()


def test_returns_immediately_when_already_safe():
    with patch("tradingai.utils.thermal.read_cpu_temp_c", return_value=60.0), \
         patch("tradingai.utils.thermal.time.sleep") as mock_sleep:
        wait_for_safe_temp(max_temp_c=80.0)
    mock_sleep.assert_not_called()


def test_sleeps_until_temperature_drops_below_threshold():
    readings = iter([95.0, 90.0, 85.0, 78.0])
    with patch("tradingai.utils.thermal.read_cpu_temp_c", side_effect=lambda: next(readings)), \
         patch("tradingai.utils.thermal.time.sleep") as mock_sleep:
        wait_for_safe_temp(max_temp_c=80.0, poll_seconds=5.0, max_wait_seconds=300.0)
    assert mock_sleep.call_count == 3  # 95->90->85 siguen calientes, 78 ya esta bien


def test_gives_up_after_max_wait_without_hanging_forever():
    # Temperatura constante (99C, nunca baja): 0->10->20->30s esperados; se corta en
    # cuanto "waited" (30) ya no es < max_wait_seconds (25), tras 3 pausas de 10s.
    with patch("tradingai.utils.thermal.read_cpu_temp_c", return_value=99.0), \
         patch("tradingai.utils.thermal.time.sleep") as mock_sleep:
        wait_for_safe_temp(max_temp_c=80.0, poll_seconds=10.0, max_wait_seconds=25.0)
    assert mock_sleep.call_count == 3
