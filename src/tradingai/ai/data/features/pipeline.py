"""Combina todos los extractores de features en un unico pipeline configurable."""

from __future__ import annotations

import pandas as pd

from tradingai.ai.data.features import gaps, indicators, market_structure, smc, volume_profile

# Columnas que build_feature_pipeline produce pero NO deben pasarse al modelo como
# feature: las OHLCV crudas, metadatos, y "atr_14" (precio absoluto, mantenido solo
# para que predictor.py/training/dataset.py calculen TP/SL reales — el modelo consume
# la version normalizada "atr_pct" en su lugar). Usar esta constante en vez de
# redefinirla por separado en train.py/backtest.py/tests evita que se desincronicen.
NON_FEATURE_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "structure_event", "atr_14"}


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
        out = smc.premium_discount_zone(out, lookback=smc_cfg.get("ob_lookback", 50))

    fvg_cfg = features_cfg.get("fvg", {})
    if fvg_cfg.get("enabled", True):
        out = gaps.detect_fvg(out, min_gap_size_pips=fvg_cfg.get("min_gap_size_pips", 2.0))

    vp_cfg = features_cfg.get("volume_profile", {})
    if vp_cfg.get("enabled", True):
        out = volume_profile.rolling_accumulation_distribution(out)

    ind_cfg = features_cfg.get("indicators", {})
    if ind_cfg.get("enabled", True):
        out = indicators.add_indicators(out, include=ind_cfg.get("include"))

    return out
