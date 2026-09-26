"""NUEVO por completo -- no existía nada de esto en el código (verificado:
`grep -rl "PDH\\|PDL\\|PWH\\|PWL"` sobre trader/trader/ no encontró nada antes
de este módulo).

PDH/PDL/PWH/PWL = high/low del último día/semana de calendario YA CERRADO,
nunca el día/semana en curso -- causal por construcción: se calculan sobre un
DataFrame D1 ya truncado con `closed_candles_as_of`/`TimeframeAnalysis.as_of`,
igual que cualquier otro detector de este proyecto, así que basta con mirar
la ÚLTIMA fila de ese D1 truncado (para PDH/PDL) o resamplear a semanal y
tomar la penúltima semana completa (para PWH/PWL, ya que la última fila de un
resample semanal representa una semana que puede seguir en curso).

Semana = lunes a domingo (`W-SUN` de pandas, ancla el cierre de semana en
domingo, la etiqueta de cada bucket es su lunes de inicio) -- no la semana de
trading forex domingo-tarde-a-viernes-tarde exacta, simplificación
documentada: el error práctico es de horas alrededor del cierre del viernes,
irrelevante para un nivel de referencia como PWH/PWL (a diferencia de un
timestamp de ejecución, donde sí importaría).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.events import TF_DURATION, MarketEvent
from trader.sessions import classify_session


@dataclass(frozen=True)
class LiquidityLevels:
    pdh: float | None
    pdl: float | None
    pwh: float | None
    pwl: float | None


def compute_liquidity_levels(d1_candles_as_of: pd.DataFrame) -> LiquidityLevels:
    """`d1_candles_as_of` debe venir ya truncada causalmente (p.ej.
    `closed_candles_as_of(d1_df, "D1", as_of)` o `d1_analysis.as_of(as_of).df`)
    -- esta función no trunca nada por su cuenta, solo lee la última fila."""
    if len(d1_candles_as_of) == 0:
        return LiquidityLevels(None, None, None, None)

    last_day = d1_candles_as_of.iloc[-1]
    pdh, pdl = float(last_day["high"]), float(last_day["low"])

    weekly = (
        d1_candles_as_of.set_index("timestamp")
        .resample("W-SUN", label="left", closed="left")
        .agg({"high": "max", "low": "min"})
    )
    # la ÚLTIMA fila del resample puede ser una semana todavía en curso (la
    # semana que contiene `last_day`) -- PWH/PWL es la semana anterior a esa,
    # completa. Si solo hay 1 fila (menos de 2 semanas de historia), no hay
    # semana previa completa: None, no un valor a medias.
    if len(weekly) < 2:
        pwh, pwl = None, None
    else:
        prev_week = weekly.iloc[-2]
        pwh, pwl = float(prev_week["high"]), float(prev_week["low"])

    return LiquidityLevels(pdh=pdh, pdl=pdl, pwh=pwh, pwl=pwl)


def compute_asia_session_high_low(m15_or_h1_candles_as_of: pd.DataFrame, sessions_config) -> tuple[float | None, float | None]:
    """High/low de la sesión Asia MÁS RECIENTE ya completamente cerrada --
    nunca la sesión Asia del día en curso (si `as_of` cae dentro de una
    sesión Asia todavía corriendo, esa se descarta entera, se usa la
    anterior). Reutiliza `sessions.classify_session` (DST-aware) tal cual,
    sin reimplementar ventanas horarias."""
    if len(m15_or_h1_candles_as_of) == 0:
        return None, None

    ts = m15_or_h1_candles_as_of["timestamp"]
    in_asia = ts.apply(lambda t: classify_session(t, sessions_config).asia)
    if not in_asia.any():
        return None, None

    # agrupar por "dia de sesion Asia" -- una racha continua de barras in_asia=True
    session_id = (in_asia != in_asia.shift(1)).cumsum()
    asia_groups = m15_or_h1_candles_as_of[in_asia.to_numpy()].groupby(session_id[in_asia.to_numpy()])

    last_bar_ts = ts.iloc[-1]
    complete_groups = [g for _, g in asia_groups if g["timestamp"].iloc[-1] < last_bar_ts or not in_asia.iloc[-1]]
    if not complete_groups:
        return None, None
    # si la sesion Asia mas reciente sigue en curso en la ultima barra, se
    # descarta (últ. grupo) y se usa la completa anterior
    if in_asia.iloc[-1]:
        complete_groups = complete_groups[:-1] if len(complete_groups) > 1 else []
        if not complete_groups:
            return None, None
    last_complete = complete_groups[-1]
    return float(last_complete["high"].max()), float(last_complete["low"].min())


def detect_named_level_sweep(
    df: pd.DataFrame, timeframe: str, level_price: float, direction: str, min_wick_pct: float
) -> MarketEvent | None:
    """Sweep de un nivel NOMBRADO arbitrario (PDL, Asia High, etc.), no de un
    swing -- generaliza la misma lógica de mecha+cierre que
    `detectors/liquidity.py::detect_liquidity_sweeps` (rota el nivel con la
    mecha, cierra de vuelta del lado correcto), parametrizada por un precio
    cualquiera en vez de derivarlo de la lista de swings. Devuelve el sweep
    MÁS RECIENTE dentro de `df` (ya truncado causalmente por el llamador).
    `direction`: "bullish" = barrido de liquidez INFERIOR (mecha bajo el
    nivel, cierre de vuelta arriba); "bearish" = liquidez SUPERIOR."""
    close_ts = df["timestamp"] + TF_DURATION[timeframe]
    o, h, l, c = (df[col].to_numpy() for col in ("open", "high", "low", "close"))
    best: MarketEvent | None = None

    for i in range(len(df)):
        candle_range = h[i] - l[i]
        if candle_range <= 0:
            continue
        if direction == "bullish" and l[i] < level_price and c[i] > level_price:
            wick = min(o[i], c[i]) - l[i]
            if wick / candle_range * 100.0 >= min_wick_pct:
                best = MarketEvent.point(kind="named_level_sweep", timeframe=timeframe, timestamp=close_ts.iloc[i], direction="bullish", price=level_price)
        elif direction == "bearish" and h[i] > level_price and c[i] < level_price:
            wick = h[i] - max(o[i], c[i])
            if wick / candle_range * 100.0 >= min_wick_pct:
                best = MarketEvent.point(kind="named_level_sweep", timeframe=timeframe, timestamp=close_ts.iloc[i], direction="bearish", price=level_price)

    return best
