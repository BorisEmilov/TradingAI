"""Extractores de features de analisis tecnico usados por el modelo de IA.

Cada submodulo anade columnas a un DataFrame OHLCV:
- market_structure: swing highs/lows, tendencia, BOS/CHoCH
- smc: order blocks, liquidity pools (equal highs/lows), liquidity sweeps
  (barrido + rechazo), premium/discount
- gaps: fair value gaps (FVG) / imbalances, e inversion de FVG (invalidado y volteado)
- sessions: sesion de mercado (Asia/Londres/NY) y killzones ICT segun hora UTC
- volume_profile: zonas de acumulacion/distribucion (POC, value area)
- indicators: EMA, RSI, ATR, MACD, Bollinger Bands, etc.

Todas estas estrategias se combinan como features de un unico modelo (GBM por
simbolo, ver ai/inference/gbm_predictor.py) en vez de generar senales discretas
independientes: el modelo aprende que combinacion de estrategias importa en cada
contexto, en vez de reglas de confluencia fijas escritas a mano.
"""

from tradingai.ai.data.features.pipeline import build_feature_pipeline

__all__ = ["build_feature_pipeline"]
