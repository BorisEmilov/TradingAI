"""Combina todos los extractores de features en un unico pipeline configurable."""

from __future__ import annotations

import pandas as pd

from tradingai.ai.data.features import (
    divergence,
    elliott,
    gaps,
    indicators,
    market_structure,
    sessions,
    smc,
    volume_profile,
    vwap,
    wyckoff,
)

# Columnas que build_feature_pipeline produce pero NO deben pasarse al modelo como
# feature: las OHLCV crudas, metadatos, "atr_14" (precio absoluto, mantenido solo
# para que predictor.py/training/dataset.py calculen TP/SL reales — el modelo consume
# la version normalizada "atr_pct" en su lugar), e igual con los niveles de rango de
# sesion (asian_range_high/low, ny_or_range_high/low): precios absolutos que solo
# sirven para calcular las versiones normalizadas (*_width_pct, broke_*_range_high/low),
# esas si son features. Usar esta constante en vez de redefinirla por separado en
# train.py/backtest.py/tests evita que se desincronicen.
NON_FEATURE_COLUMNS = {
    "timestamp", "open", "high", "low", "close", "volume", "structure_event", "atr_14",
    "asian_range_high", "asian_range_low", "ny_or_range_high", "ny_or_range_low",
}


def select_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Columnas numericas de `features_df` que el modelo debe consumir como input."""
    return [
        c
        for c in features_df.columns
        if c not in NON_FEATURE_COLUMNS and features_df[c].dtype.kind in "fib"
    ]


def build_feature_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aplica los extractores habilitados en `config["features"]` sobre el OHLCV."""
    features_cfg = config.get("features", {})
    out = df.copy()

    ms_cfg = features_cfg.get("market_structure", {})
    out = market_structure.detect_swings(out, lookback=ms_cfg.get("swing_lookback", 5))
    out = market_structure.detect_structure_breaks(out)

    smc_cfg = features_cfg.get("smc", {})
    if smc_cfg.get("enabled", True):
        out = smc.detect_order_blocks(out, lookback=smc_cfg.get("ob_lookback", 50))
        out = smc.detect_liquidity_pools(out, lookback=smc_cfg.get("liquidity_lookback", 100))
        out = smc.detect_liquidity_sweeps(
            out,
            lookback=smc_cfg.get("liquidity_lookback", 100),
            tolerance_pct=smc_cfg.get("liquidity_sweep_tolerance_pct", 0.05),
        )
        out = smc.premium_discount_zone(out, lookback=smc_cfg.get("ob_lookback", 50))

    fvg_cfg = features_cfg.get("fvg", {})
    if fvg_cfg.get("enabled", True):
        out = gaps.detect_fvg(out, min_gap_size_pips=fvg_cfg.get("min_gap_size_pips", 2.0))
        if fvg_cfg.get("detect_inverted", True):
            out = gaps.detect_inverted_fvg(out, min_gap_size_pips=fvg_cfg.get("min_gap_size_pips", 2.0))

    sessions_cfg = features_cfg.get("sessions", {})
    if sessions_cfg.get("enabled", True):
        out = sessions.add_session_features(out, config=sessions_cfg)

        # Combinacion "Turtle Soup" (institucional): barrido de liquidez que ocurre
        # DENTRO de una killzone ICT -- el barrido aislado o la killzone aislada ya
        # existen como columnas propias (liquidity_sweep_*, is_killzone), esta
        # columna es la combinacion explicita de ambas para que el modelo pueda usar
        # cualquiera de las tres segun el contexto. Requiere smc habilitado (arriba).
        if smc_cfg.get("enabled", True):
            out["killzone_liquidity_sweep_bearish"] = out["liquidity_sweep_bearish"] & out["is_killzone"]
            out["killzone_liquidity_sweep_bullish"] = out["liquidity_sweep_bullish"] & out["is_killzone"]

    vp_cfg = features_cfg.get("volume_profile", {})
    if vp_cfg.get("enabled", True):
        out = volume_profile.rolling_accumulation_distribution(out)
        out = volume_profile.rolling_volume_profile_features(
            out,
            window=vp_cfg.get("rolling_window", 200),
            bins=vp_cfg.get("bins", 20),
            value_area_pct=vp_cfg.get("value_area_pct", 70.0),
        )

    vwap_cfg = features_cfg.get("vwap", {})
    if vwap_cfg.get("enabled", True):
        out = vwap.add_vwap_features(out)

    wyckoff_cfg = features_cfg.get("wyckoff", {})
    if wyckoff_cfg.get("enabled", True):
        out = wyckoff.detect_wyckoff_events(
            out,
            range_window=wyckoff_cfg.get("range_window", 20),
            tolerance_pct=wyckoff_cfg.get("tolerance_pct", 0.05),
        )

    elliott_cfg = features_cfg.get("elliott", {})
    if elliott_cfg.get("enabled", True):
        out = elliott.compute_elliott_features(out, deviation_pct=elliott_cfg.get("deviation_pct", 0.5))

    ind_cfg = features_cfg.get("indicators", {})
    if ind_cfg.get("enabled", True):
        out = indicators.add_indicators(out, include=ind_cfg.get("include"))

    # Requiere swing_high/swing_low (market_structure, arriba) y rsi_14 (indicators,
    # justo arriba) -> va al final del pipeline.
    divergence_cfg = features_cfg.get("divergence", {})
    if divergence_cfg.get("enabled", True):
        out = divergence.detect_momentum_divergence(out)

    return out
