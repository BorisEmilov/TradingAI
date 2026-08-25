"""VWAP (Volume Weighted Average Price) anclado a sesion, con bandas de desviacion.

El benchmark de ejecucion mas usado por mesas institucionales: el precio medio
ponderado por volumen desde el inicio de la sesion, con bandas de +-1/+-2 desviaciones
estandar que se usan tanto para mean-reversion (rebote hacia VWAP) como para medir
fuerza direccional (precio sostenido fuera de banda = presion institucional real).

Ancla diaria a las 00:00 UTC (no hay "apertura de mercado" unica en forex/CFDs 24h,
asi que se usa la medianoche UTC como reset, consistente con como los brokers cierran
la sesion diaria). Solo usa sumas acumuladas desde el inicio de cada dia -> causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamp = pd.to_datetime(out["timestamp"])
    day = timestamp.dt.floor("D")

    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    volume = out["volume"]

    tp_vol = typical_price * volume
    cum_tp_vol = tp_vol.groupby(day).cumsum()
    cum_vol = volume.groupby(day).cumsum().replace(0, np.nan)

    vwap = cum_tp_vol / cum_vol
    vwap = vwap.bfill().ffill()  # primeras velas del historico sin volumen previo, relleno neutro

    # Desviacion estandar acumulada del precio tipico respecto al VWAP, dentro de la sesion.
    sq_dev = (typical_price - vwap) ** 2 * volume
    cum_sq_dev = sq_dev.groupby(day).cumsum()
    variance = (cum_sq_dev / cum_vol).clip(lower=0)
    std = np.sqrt(variance).fillna(0)

    # Con pocas velas acumuladas en la sesion, `std` es casi cero (varianza calculada
    # sobre muy pocos puntos) y dividir por ese valor casi-cero dispara el z-score a
    # magnitudes absurdas. Se exige un minimo de velas dentro de la sesion antes de
    # confiar en el z-score; antes de eso, neutro (0.0).
    bars_in_session = volume.groupby(day).cumcount() + 1
    min_bars = 5
    std_floor = out["close"] * 1e-6  # umbral relativo al precio, evita division por ~0
    reliable = (bars_in_session >= min_bars) & (std > std_floor)

    close = out["close"]
    out["vwap_dist_pct"] = (close - vwap) / close
    out["vwap_zscore"] = np.where(reliable, (close - vwap) / std, 0.0)
    out["above_vwap_upper_1std"] = reliable & (close > (vwap + std))
    out["below_vwap_lower_1std"] = reliable & (close < (vwap - std))
    out["above_vwap_upper_2std"] = reliable & (close > (vwap + 2 * std))
    out["below_vwap_lower_2std"] = reliable & (close < (vwap - 2 * std))
    return out
