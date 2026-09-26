"""NUEVO -- `pipeline/engine.py::TimeframeAnalysis.bias()` ya existe pero
responde una pregunta distinta ("¿cuál fue la ÚLTIMA ruptura de estructura,
BOS o CHoCH?"), no la que Estrategia 1 necesita: una secuencia explícita de
swings HH-HL-HH (o LL-LH-LL) con el último punto relevante todavía intacto.
Reutiliza los swings ya detectados (`detectors/structure.py::detect_swings`,
vía `TimeframeAnalysis.swings`) -- no hay detección de swings nueva aquí,
solo una lectura distinta de la misma lista.

Reusado tal cual, sin duplicar, para la negación de Estrategia 2 Setup A
("4H no fuertemente bullish" = `classify_htf_structure(...) != "bullish"`, confirmado
en el prompt de decisiones de diseño) -- un solo detector, dos consumidores.
"""

from __future__ import annotations

from typing import Literal

from trader.events import MarketEvent

StructureBias = Literal["bullish", "bearish", "none"]


def _alternating(swings: list[MarketEvent]) -> list[MarketEvent]:
    """Swings en orden cronológico, colapsando corridas del mismo tipo a su
    extremo (el más alto de una corrida de swing_highs, el más bajo de una
    corrida de swing_lows) -- sin esto, dos swing_highs seguidos (sin un low
    real entre medio) romperían la lectura de "el low ENTRE dos highs"."""
    ordered = sorted(swings, key=lambda e: e.timestamp)
    out: list[MarketEvent] = []
    for s in ordered:
        if out and out[-1].kind == s.kind:
            better = (s.price > out[-1].price) if s.kind == "swing_high" else (s.price < out[-1].price)
            if better:
                out[-1] = s
        else:
            out.append(s)
    return out


def classify_htf_structure(swings: list[MarketEvent], latest_close: float) -> StructureBias:
    """HH->HL->HH (bullish) o LL->LH->LL (bearish) sobre los últimos 5 swings
    alternados (2 highs + 1 pivote intermedio de cada lado, o el espejo).
    Exige que la secuencia TERMINE en el punto que confirma el patrón (un
    High fresco para bullish, un Low fresco para bearish) -- si el swing más
    reciente todavía no llegó a ese punto, el patrón no está confirmado
    todavía: "none" (NO TRADE), no una extrapolación optimista. Exige además
    que el HL/LH intermedio no haya sido roto: `latest_close` todavía del
    lado correcto de ese nivel. Cualquier cosa que no encaje exactamente
    (estructura confusa, rango comprimido, rupturas falsas, menos de 5
    swings) es "none" -- el NO TRADE de la regla 1 es el valor por defecto,
    no una excepción a manejar aparte.
    """
    alt = _alternating(swings)
    if len(alt) < 5:
        return "none"

    last5 = alt[-5:]
    kinds = [s.kind for s in last5]

    if kinds == ["swing_high", "swing_low", "swing_high", "swing_low", "swing_high"]:
        h0, l0, h1, l1, h2 = (s.price for s in last5)  # H0,L0,H1,L1,H2
        if h2 > h1 and l1 > l0 and latest_close > l1:
            return "bullish"
        return "none"

    if kinds == ["swing_low", "swing_high", "swing_low", "swing_high", "swing_low"]:
        l0, h0, l1, h1, l2 = (s.price for s in last5)  # L0,H0,L1,H1,L2
        if l2 < l1 and h1 < h0 and latest_close < h1:
            return "bearish"
        return "none"

    return "none"


_OPPOSITE = {"bullish": "bearish", "bearish": "bullish"}


def entry_4h_condition_holds(strategy: str, setup: str | None, direction4h: str, structure_4h: StructureBias) -> bool:
    """Condición de 4H de ENTRADA de cada setup -- única fuente, usada por el
    gate de reversión y por la cancelación proactiva del piloto (que debe
    revalidar exactamente lo mismo, ni más ni menos -- bug 2026-09-25: el
    piloto aplicaba la de continuación a todo).
    - continuación: 4H == dirección (continuation.py deriva la dirección del
      propio sesgo, "none" = NO TRADE).
    - reversión setup A (Asia): 4H "no fuertemente" en contra (none vale).
    - reversión setup B (PDH/PDL): sin condición de 4H."""
    if strategy == "continuation":
        return structure_4h == direction4h
    if (setup or "").startswith("setup_a"):
        return structure_4h != _OPPOSITE[direction4h]
    return True
