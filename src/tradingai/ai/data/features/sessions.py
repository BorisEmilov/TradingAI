"""Sesiones de mercado, killzones (ICT) y estrategias institucionales basadas en
sesiones: rango asiatico + ruptura, y opening range breakout de NY.

Los flags de sesion/killzone (session_asia, is_killzone, etc.) solo dependen del
timestamp de cada vela -> sin riesgo de look-ahead. Las features de rango (asian_*,
ny_or_*) SI dependen de precio, pero se construyen con maximo/minimo ACUMULADO dentro
de la ventana (cummax/cummin) y se propagan (ffill) solo dentro del mismo dia UTC --
nunca miran una vela futura de esa misma ventana, y el rango queda "congelado" en
cuanto la ventana se cierra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _in_range(hour: np.ndarray, start: int, end: int) -> np.ndarray:
    """Bool array: hour en [start, end), soportando rangos que cruzan medianoche."""
    if start <= end:
        return (hour >= start) & (hour < end)
    return (hour >= start) | (hour < end)


def _session_range_features(
    df: pd.DataFrame, hour: np.ndarray, date: pd.Series, start: int, end: int, prefix: str
) -> pd.DataFrame:
    """Rango (alto/bajo) de una ventana horaria fija dentro del dia, y ruptura de ese
    rango por el precio de cierre durante el resto del dia.

    Ej. `prefix="asian"` con la ventana de la sesion asiatica = Asian Range Breakout
    (institucional: rango de baja volatilidad que suele romperse direccionalmente al
    abrir Londres/NY). `prefix="ny_or"` con la primera hora de NY = Opening Range
    Breakout. El rango se construye con cummax/cummin (nunca ve una vela futura de la
    ventana) y se congela (ffill) una vez la ventana se cierra ese dia.
    """
    in_window = _in_range(hour, start, end)
    high_in_window = df["high"].where(in_window)
    low_in_window = df["low"].where(in_window)

    range_high = high_in_window.groupby(date).cummax().groupby(date).ffill()
    range_low = low_in_window.groupby(date).cummin().groupby(date).ffill()

    # Fallback para velas sin ningun rango conocido todavia -- el primer dia del
    # dataset antes de que la ventana ocurra, o timeframes gruesos (D1/H4) cuya vela
    # nunca cae dentro de la ventana horaria (ej. D1 sellado a las 00:00 UTC nunca
    # coincide con la ventana de NY). Sin esto la columna queda NaN y
    # `normalize_ohlcv` descarta la fila entera aguas abajo -- ver bug real del
    # 2026-08-25 (D1 se quedaba con 0 velas tras el pipeline). Usar el propio
    # high/low de la vela como rango neutro (width=0, sin ruptura posible).
    range_high = range_high.fillna(df["high"])
    range_low = range_low.fillna(df["low"])

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_range_high"] = range_high
    out[f"{prefix}_range_low"] = range_low
    out[f"{prefix}_range_width_pct"] = (range_high - range_low) / df["close"] * 100
    out[f"broke_{prefix}_range_high"] = df["close"] > range_high
    out[f"broke_{prefix}_range_low"] = df["close"] < range_low
    return out


def add_session_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Anade columnas de sesion/killzone/rango segun la hora UTC del timestamp.

    `config` (opcional) puede sobreescribir las ventanas por defecto, ej.:
        {"asia_utc": [0, 7], "london_utc": [7, 16], "ny_utc": [12, 21],
         "london_killzone_utc": [7, 10], "ny_killzone_utc": [12, 15],
         "ny_opening_range_utc": [12, 13]}
    """
    cfg = config or {}
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"])
    hour = timestamps.dt.hour.to_numpy()
    date = timestamps.dt.date

    asia = cfg.get("asia_utc", [0, 7])
    london = cfg.get("london_utc", [7, 16])
    ny = cfg.get("ny_utc", [12, 21])
    london_kz = cfg.get("london_killzone_utc", [7, 10])
    ny_kz = cfg.get("ny_killzone_utc", [12, 15])
    ny_or = cfg.get("ny_opening_range_utc", [ny[0], ny[0] + 1])

    session_asia = _in_range(hour, *asia)
    session_london = _in_range(hour, *london)
    session_ny = _in_range(hour, *ny)
    kz_london = _in_range(hour, *london_kz)
    kz_ny = _in_range(hour, *ny_kz)

    out["session_asia"] = session_asia
    out["session_london"] = session_london
    out["session_ny"] = session_ny
    out["session_overlap_london_ny"] = session_london & session_ny
    out["is_killzone"] = kz_london | kz_ny

    out = out.join(_session_range_features(out, hour, date, *asia, prefix="asian"))
    out = out.join(_session_range_features(out, hour, date, *ny_or, prefix="ny_or"))

    return out
