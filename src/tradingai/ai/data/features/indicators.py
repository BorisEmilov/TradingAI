"""Indicadores tecnicos clasicos (via pandas-ta) como contexto adicional al SMC.

pandas_ta devuelve `None` (no una Serie/DataFrame de NaN) cuando la entrada es mas
corta que el periodo del indicador (p.ej. EMA-200 con menos de 200 velas — algo que
pasa en la practica cerca del inicio del historico disponible de una temporalidad,
como M5 en este proyecto). Asignar ese `None` directamente a una columna la tipa
como `object`, y el `dropna()` de `normalize_ohlcv` la ignora en silencio (solo mira
columnas numericas) — el hueco se cuela hasta el modelo como NaN dentro de un tensor
"valido". Este modulo normaliza siempre a NaN float64 explicito para que el dropna
lo detecte como cualquier otro warm-up.

Ademas, EMA/MACD/Bollinger salen de pandas_ta como niveles de precio absolutos
(p.ej. ~1.10 en EURUSD, ~2000+ en XAUUSD). Si se usan tal cual como feature del
modelo, entrenar con varios simbolos a la vez mezcla escalas completamente distintas
en la misma columna y confunde al modelo (ver `atr_14` vs `atr_pct` mas abajo: el
mismo problema que ya se soluciono para fvg_top/fvg_bottom al principio del proyecto).
Por eso aqui se normalizan a distancia % respecto al cierre antes de exponerlos.
`atr_14` es la excepcion: se mantiene en precio absoluto porque `predictor.py` y
`training/dataset.py` lo necesitan asi para calcular TP/SL reales; el modelo consume
en cambio `atr_pct` (ver NON_FEATURE_COLS en train.py/backtest.py, que excluye
`atr_14` de las features del modelo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

# Columnas que devuelve pandas_ta para macd()/bbands() con los parametros que usamos,
# calculadas una vez sobre una serie sintetica con historia de sobra. Se usan solo
# para reconstruir la forma esperada cuando pandas_ta devuelve None por falta de datos.
_DUMMY_CLOSE = pd.Series(np.linspace(1.0, 1.1, 300))
_MACD_COLUMNS = list(ta.macd(_DUMMY_CLOSE).columns)
_BBANDS_COLUMNS = list(ta.bbands(_DUMMY_CLOSE, length=20).columns)

# De las bandas de Bollinger, BBL/BBM/BBU son niveles de precio (se normalizan);
# BBB (bandwidth) y BBP (percent-b) ya son relativos y se dejan tal cual.
_BBANDS_PRICE_COLUMNS = [c for c in _BBANDS_COLUMNS if c.startswith(("BBL", "BBM", "BBU"))]


def _safe_series(result: pd.Series | None, index: pd.Index) -> pd.Series:
    if result is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return result.astype("float64")


def _safe_frame(result: pd.DataFrame | None, index: pd.Index, columns: list[str]) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame(np.nan, index=index, columns=columns, dtype="float64")
    return result.astype("float64")


def _pct_of_close(level: pd.Series, close: pd.Series) -> pd.Series:
    """Convierte un nivel de precio absoluto en distancia % al cierre (escala-invariante)."""
    return (level - close) / close


def add_indicators(df: pd.DataFrame, include: list[str] | None = None) -> pd.DataFrame:
    include = include or ["ema", "rsi", "atr", "macd", "bollinger"]
    out = df.copy()
    close = out["close"]

    if "ema" in include:
        out["ema_20_pct"] = _pct_of_close(_safe_series(ta.ema(close, length=20), out.index), close)
        out["ema_50_pct"] = _pct_of_close(_safe_series(ta.ema(close, length=50), out.index), close)
        out["ema_200_pct"] = _pct_of_close(_safe_series(ta.ema(close, length=200), out.index), close)

    if "rsi" in include:
        out["rsi_14"] = _safe_series(ta.rsi(close, length=14), out.index)

    if "atr" in include:
        atr = _safe_series(ta.atr(out["high"], out["low"], close, length=14), out.index)
        out["atr_14"] = atr  # precio absoluto: lo usan predictor.py/training/dataset.py para TP/SL
        out["atr_pct"] = atr / close  # version normalizada: la que ve el modelo

    if "macd" in include:
        macd = _safe_frame(ta.macd(close), out.index, _MACD_COLUMNS)
        for col in macd.columns:
            macd[col] = macd[col] / close  # MACD es una diferencia de precios; normaliza por precio
        out = pd.concat([out, macd], axis=1)

    if "bollinger" in include:
        bbands = _safe_frame(ta.bbands(close, length=20), out.index, _BBANDS_COLUMNS)
        for col in _BBANDS_PRICE_COLUMNS:
            if col in bbands.columns:
                bbands[col] = _pct_of_close(bbands[col], close)
        out = pd.concat([out, bbands], axis=1)

    return out
