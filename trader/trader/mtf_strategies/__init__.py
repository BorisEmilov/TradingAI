"""prompt-2-continuacion-reversion-mtf.md: Estrategia 1 (Continuación MTF SMC,
4H/1H/15M) y Estrategia 2 (Reversión MTF SMC) -- completamente aisladas del
pipeline de producción (`trader/pipeline/`, el sistema de scoring que corre
hoy) y del piloto RSI en XPTUSD (`trader/quant/`, `scripts/run_pilot_rsi_xpt.py`).
Nada aquí importa de ni es importado por esos dos sistemas.

Reutiliza sin cambios los detectores de bajo nivel ya auditados
(`trader/detectors/`, `trader/events.py::TimeframeAnalysis` vía
`trader/pipeline/engine.py` -- se importa esa clase, no se reimplementa) --
swings, BOS/CHoCH, sweeps, equal highs/lows, FVG, order blocks, ATR, sesiones
con DST. Todo lo demás en este paquete es nuevo o una adaptación explícita,
documentada módulo por módulo.
"""
