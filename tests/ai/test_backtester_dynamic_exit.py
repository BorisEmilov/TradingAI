import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from tradingai.ai.evaluation.backtester import Backtester
from tradingai.core.signal import Direction, TradingSignal


def _ohlc(closes: np.ndarray, high_pad: float = 0.0002, low_pad: float = 0.0002) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="15min"),
            "open": closes,
            "high": closes + high_pad,
            "low": closes - low_pad,
            "close": closes,
        }
    )


def _ohlc_from_lows(lows: list[float]) -> pd.DataFrame:
    highs = [low + 0.0010 for low in lows]
    closes = [low + 0.0005 for low in lows]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(lows), freq="15min"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _long_signal(entry: float, sl: float, tp: float) -> TradingSignal:
    return TradingSignal(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.8,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def test_dynamic_exit_closes_early_on_a_break_of_structure():
    # Un swing low confirmado (1.0995) antes de la entrada, y otro MAS BAJO (1.0970)
    # confirmado despues -- una ruptura de estructura clasica (BOS/CHoCH) en contra
    # de un LONG. SL/TP fijados MUY lejos a proposito para que la unica salida
    # posible dentro de la ventana sea la invalidacion de estructura.
    lows = [
        1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035,  # swing1 confirmado en idx3
        1.1045, 1.1050,  # idx8-9, entrada en idx9
        1.1040, 1.1030, 1.1020, 1.0970, 1.0980, 1.0990, 1.1000,  # idx10-16, swing2 (1.0970) confirmado en idx16
    ]
    candles = _ohlc_from_lows(lows)
    entry = lows[9] + 0.0005

    signal = _long_signal(entry=entry, sl=entry - 0.5, tp=entry + 0.5)
    backtester = Backtester(
        confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=10, dynamic_exit=True,
    )
    trade = backtester.run(candles, [(9, signal)])[0]

    assert trade.exit_reason == "estructura"


def test_dynamic_exit_matches_static_when_disabled_by_default():
    lows = [
        1.1030, 1.1020, 1.1010, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035,
        1.1045, 1.1050,
        1.1040, 1.1030, 1.1020, 1.0970, 1.0980, 1.0990, 1.1000,
    ]
    candles = _ohlc_from_lows(lows)
    entry = lows[9] + 0.0005

    signal = _long_signal(entry=entry, sl=entry - 0.5, tp=entry + 0.5)
    backtester = Backtester(confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=10)
    trade = backtester.run(candles, [(9, signal)])[0]

    # Sin dynamic_exit, no hay invalidacion de estructura: el SL/TP estan tan lejos
    # que la operacion agota la ventana de holding ("timeout"), no "estructura".
    assert trade.exit_reason == "timeout"


def test_dynamic_exit_extends_tp_beyond_pre_existing_swing():
    # Swing alcista confirmado en el historial (indices 198-204) por ENCIMA de donde
    # quedara el TP fijo original -- disponible desde la primera vela futura porque
    # el lookback de swings (100 velas) lo alcanza a cubrir. El TP dinamico debe
    # extenderse a ese nivel en vez de cerrar en el TP fijo (2:1) cuando el precio lo
    # alcanza en el camino hacia el swing.
    n_pre_swing = 198
    pre = 1.1000 + 0.0004 * np.arange(n_pre_swing)  # hasta ~1.1788
    swing_block = np.array([1.1850, 1.1800, 1.1790, 1.1780])  # pico confirmado + 3 velas mas bajas
    pullback = 1.1780 - 0.0004 * np.arange(1, 46)  # retrocede hasta ~1.1600 (zona de entrada)
    warmup = np.concatenate([pre, swing_block, pullback])

    entry_idx = len(warmup) - 1
    entry = warmup[entry_idx]
    rally = entry + 0.0015 * np.arange(1, 30)  # sube directo, cruzando el TP fijo y el swing
    closes = np.concatenate([warmup, rally])
    candles = _ohlc(closes, high_pad=0.0002, low_pad=0.0002)

    original_tp = entry + 0.0020  # 2:1 con sl=entry-0.0010... (ver sl abajo)
    sl = entry - 0.0010
    signal = _long_signal(entry=entry, sl=sl, tp=original_tp)

    static_bt = Backtester(confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=30)
    dynamic_bt = Backtester(
        confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=30,
        dynamic_exit=True, structure_lookback_candles=100,
    )

    static_trade = static_bt.run(candles, [(entry_idx, signal)])[0]
    dynamic_trade = dynamic_bt.run(candles, [(entry_idx, signal)])[0]

    assert static_trade.exit_reason == "tp"
    assert static_trade.exit_price == pytest.approx(original_tp)

    assert dynamic_trade.exit_reason == "tp"
    assert dynamic_trade.exit_price > original_tp
    assert dynamic_trade.pnl_pct > static_trade.pnl_pct
