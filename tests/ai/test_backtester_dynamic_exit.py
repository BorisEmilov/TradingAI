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


def test_dynamic_exit_closes_early_when_bias_reverses():
    # 250 velas alcistas (sesgo bullish claro al momento de entrar) seguidas de una
    # caida fuerte y sostenida -- suficiente para que el sesgo de EMAs se voltee a
    # bearish antes de llegar al SL/TP (fijados MUY lejos a proposito para que la
    # unica salida posible dentro de la ventana sea la invalidacion de estructura).
    warmup = 1.1000 + 0.0004 * np.arange(250)
    entry = warmup[-1]
    decline = entry - 0.0010 * np.arange(1, 61)
    closes = np.concatenate([warmup, decline])
    candles = _ohlc(closes)

    signal = _long_signal(entry=entry, sl=entry - 0.5, tp=entry + 0.5)
    backtester = Backtester(
        confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=60,
        dynamic_exit=True, structure_bias_lookback_candles=250,
    )
    trade = backtester.run(candles, [(249, signal)])[0]

    assert trade.exit_reason == "estructura"


def test_dynamic_exit_matches_static_when_disabled_by_default():
    warmup = 1.1000 + 0.0004 * np.arange(250)
    entry = warmup[-1]
    decline = entry - 0.0010 * np.arange(1, 61)
    closes = np.concatenate([warmup, decline])
    candles = _ohlc(closes)

    signal = _long_signal(entry=entry, sl=entry - 0.5, tp=entry + 0.5)
    backtester = Backtester(confidence_threshold=0.5, spread_pips=0, slippage_pips=0, max_holding_bars=60)
    trade = backtester.run(candles, [(249, signal)])[0]

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
        dynamic_exit=True, structure_bias_lookback_candles=250, structure_lookback_candles=100,
    )

    static_trade = static_bt.run(candles, [(entry_idx, signal)])[0]
    dynamic_trade = dynamic_bt.run(candles, [(entry_idx, signal)])[0]

    assert static_trade.exit_reason == "tp"
    assert static_trade.exit_price == pytest.approx(original_tp)

    assert dynamic_trade.exit_reason == "tp"
    assert dynamic_trade.exit_price > original_tp
    assert dynamic_trade.pnl_pct > static_trade.pnl_pct
