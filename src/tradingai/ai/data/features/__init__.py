"""Extractores de features de analisis tecnico usados por el modelo de IA.

Cada submodulo anade columnas a un DataFrame OHLCV:
- market_structure: swing highs/lows, tendencia, BOS/CHoCH
- smc: order blocks, liquidity pools (equal highs/lows), premium/discount
- gaps: fair value gaps (FVG) / imbalances
- volume_profile: zonas de acumulacion/distribucion (POC, value area)
- indicators: EMA, RSI, ATR, MACD, Bollinger Bands, etc.
"""

from tradingai.ai.data.features.pipeline import build_feature_pipeline

__all__ = ["build_feature_pipeline"]
