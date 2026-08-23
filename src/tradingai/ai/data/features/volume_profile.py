"""Zonas de acumulacion/distribucion via perfil de volumen (Volume Profile).

POC (Point of Control): nivel de precio con mayor volumen negociado.
Value Area: rango de precio que concentra el `value_area_pct` del volumen total.
Acumulacion/distribucion: rango estrecho con volumen creciente -> posible zona
de posicionamiento institucional antes de un movimiento direccional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_profile(df: pd.DataFrame, bins: int = 50, value_area_pct: float = 70.0) -> dict:
    """Calcula POC y Value Area para el rango de velas dado (uso en ventanas moviles)."""
    price_min, price_max = df["low"].min(), df["high"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    volume_per_bin = np.zeros(bins)

    for _, row in df.iterrows():
        # Reparte el volumen de la vela proporcionalmente entre los bins que toca su rango.
        lo_idx = np.searchsorted(edges, row["low"], side="right") - 1
        hi_idx = np.searchsorted(edges, row["high"], side="right") - 1
        lo_idx, hi_idx = np.clip([lo_idx, hi_idx], 0, bins - 1)
        span = hi_idx - lo_idx + 1
        volume_per_bin[lo_idx : hi_idx + 1] += row["volume"] / span

    poc_idx = int(np.argmax(volume_per_bin))
    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    order = np.argsort(volume_per_bin)[::-1]
    total_volume = volume_per_bin.sum()
    target = total_volume * value_area_pct / 100
    cum, selected = 0.0, []
    for idx in order:
        cum += volume_per_bin[idx]
        selected.append(idx)
        if cum >= target:
            break

    value_area_high = edges[max(selected) + 1]
    value_area_low = edges[min(selected)]

    return {
        "poc": poc_price,
        "value_area_high": value_area_high,
        "value_area_low": value_area_low,
        "volume_per_bin": volume_per_bin,
        "bin_edges": edges,
    }


def rolling_accumulation_distribution(df: pd.DataFrame, window: int = 20, range_pct_threshold: float = 0.5) -> pd.DataFrame:
    """Marca ventanas de rango estrecho + volumen creciente como posible acumulacion/distribucion."""
    out = df.copy()
    price_range_pct = (out["high"].rolling(window).max() - out["low"].rolling(window).min()) / out["close"] * 100
    volume_trend = out["volume"].rolling(window).mean().pct_change(window)

    out["is_accum_dist_zone"] = (price_range_pct < range_pct_threshold) & (volume_trend > 0)
    out["range_pct"] = price_range_pct
    return out
