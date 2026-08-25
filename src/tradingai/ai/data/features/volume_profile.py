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


def rolling_volume_profile_features(df: pd.DataFrame, window: int = 200, bins: int = 20, value_area_pct: float = 70.0) -> pd.DataFrame:
    """POC / Value Area recalculados en una ventana movil, como features por vela.

    Version vectorizada con `np.histogram` (usa el precio tipico de cada vela, no el
    reparto proporcional bin-a-bin de `compute_volume_profile`) para que sea viable
    sobre historicos grandes (~90k velas M5) sin el coste de un `iterrows()` por vela.
    Usa solo las `window` velas ANTERIORES a la actual (excluye la vela actual) -> causal.
    """
    out = df.copy()
    n = len(out)
    high = out["high"].to_numpy()
    low = out["low"].to_numpy()
    close = out["close"].to_numpy()
    volume = out["volume"].to_numpy()
    typical = (high + low) / 2

    dist_to_poc_pct = np.full(n, 0.0)
    dist_to_va_high_pct = np.full(n, 0.0)
    dist_to_va_low_pct = np.full(n, 0.0)
    inside_value_area = np.zeros(n, dtype=bool)

    for i in range(window, n):
        w_typical = typical[i - window : i]
        w_volume = volume[i - window : i]
        price_min, price_max = w_typical.min(), w_typical.max()
        if price_max <= price_min:
            continue

        hist, edges = np.histogram(w_typical, bins=bins, range=(price_min, price_max), weights=w_volume)
        total = hist.sum()
        if total <= 0:
            continue

        poc_idx = int(hist.argmax())
        poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2

        order = np.argsort(hist)[::-1]
        target = total * value_area_pct / 100
        cum, selected = 0.0, []
        for idx in order:
            cum += hist[idx]
            selected.append(idx)
            if cum >= target:
                break

        va_high = edges[max(selected) + 1]
        va_low = edges[min(selected)]

        c = close[i]
        dist_to_poc_pct[i] = (c - poc_price) / c
        dist_to_va_high_pct[i] = (c - va_high) / c
        dist_to_va_low_pct[i] = (c - va_low) / c
        inside_value_area[i] = va_low <= c <= va_high

    out["dist_to_poc_pct"] = dist_to_poc_pct
    out["dist_to_va_high_pct"] = dist_to_va_high_pct
    out["dist_to_va_low_pct"] = dist_to_va_low_pct
    out["inside_value_area"] = inside_value_area
    return out


def rolling_accumulation_distribution(df: pd.DataFrame, window: int = 20, range_pct_threshold: float = 0.5) -> pd.DataFrame:
    """Marca ventanas de rango estrecho + volumen creciente como posible acumulacion/distribucion."""
    out = df.copy()
    price_range_pct = (out["high"].rolling(window).max() - out["low"].rolling(window).min()) / out["close"] * 100
    volume_trend = out["volume"].rolling(window).mean().pct_change(window)

    out["is_accum_dist_zone"] = (price_range_pct < range_pct_threshold) & (volume_trend > 0)
    out["range_pct"] = price_range_pct
    return out
