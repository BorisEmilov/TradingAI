import pandas as pd
import pytest

from tradingai.mt5.data_feed import CandleCloseWatcher


class _FakeConnector:
    def __init__(self, candles: pd.DataFrame) -> None:
        self.candles = candles

    def get_candles(self, symbol, timeframe, n_candles):
        return self.candles.tail(n_candles).reset_index(drop=True)


def _candles(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "close": range(len(timestamps))})


def test_first_call_without_state_file_returns_immediately():
    connector = _FakeConnector(_candles(["2026-08-28 07:00", "2026-08-28 07:15"]))
    watcher = CandleCloseWatcher(connector, "EURUSD", "M15", poll_seconds=0)
    row = watcher.wait_for_new_candle()
    assert row["timestamp"] == pd.Timestamp("2026-08-28 07:00")


def test_persists_last_seen_candle_to_state_file(tmp_path):
    state_file = tmp_path / "EURUSD.txt"
    connector = _FakeConnector(_candles(["2026-08-28 07:00", "2026-08-28 07:15"]))
    watcher = CandleCloseWatcher(connector, "EURUSD", "M15", poll_seconds=0, state_file=state_file)
    watcher.wait_for_new_candle()

    assert state_file.exists()
    assert pd.Timestamp(state_file.read_text().strip()) == pd.Timestamp("2026-08-28 07:00")


def test_fresh_process_with_existing_state_file_does_not_reevaluate_same_candle(tmp_path):
    # Caso real 2026-08-28: sin esto, reiniciar el proceso hacia que la vela YA vista
    # antes del reinicio se tratara como "nueva" y disparara un ciclo de mas.
    state_file = tmp_path / "EURUSD.txt"
    state_file.write_text("2026-08-28 07:00:00")

    connector = _FakeConnector(_candles(["2026-08-28 06:45", "2026-08-28 07:00"]))
    watcher = CandleCloseWatcher(connector, "EURUSD", "M15", poll_seconds=0, state_file=state_file)

    # La misma vela de siempre sigue siendo la ultima cerrada -- no debe considerarse
    # nueva. Usamos un timeout via una conexion que nunca avanza para probarlo sin
    # bloquear el test: forzamos manualmente la comparacion en vez de loopear.
    assert watcher._last_timestamp == pd.Timestamp("2026-08-28 07:00:00")


def test_fresh_process_with_existing_state_file_still_fires_on_a_genuinely_new_candle(tmp_path):
    # El estado persistido es mas viejo que la ultima vela cerrada disponible ahora
    # -- debe dispararse igual, la persistencia solo evita re-evaluar la MISMA vela.
    state_file = tmp_path / "EURUSD.txt"
    state_file.write_text("2026-08-28 06:45:00")

    connector = _FakeConnector(_candles(["2026-08-28 07:00", "2026-08-28 07:15"]))
    watcher = CandleCloseWatcher(connector, "EURUSD", "M15", poll_seconds=0, state_file=state_file)
    row = watcher.wait_for_new_candle()

    assert row["timestamp"] == pd.Timestamp("2026-08-28 07:00")


def test_corrupted_state_file_is_ignored_not_fatal(tmp_path):
    state_file = tmp_path / "EURUSD.txt"
    state_file.write_text("no es una fecha valida")

    connector = _FakeConnector(_candles(["2026-08-28 07:00", "2026-08-28 07:15"]))
    watcher = CandleCloseWatcher(connector, "EURUSD", "M15", poll_seconds=0, state_file=state_file)
    row = watcher.wait_for_new_candle()

    assert row["timestamp"] == pd.Timestamp("2026-08-28 07:00")
