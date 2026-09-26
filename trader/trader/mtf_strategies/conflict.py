"""NUEVO -- decisión de diseño confirmada: si Estrategia 1 y Estrategia 2
generan señal sobre el MISMO símbolo (al mismo momento de evaluación), se
descartan las DOS, se loguea el conflicto para revisión posterior. Sin
prioridad fija, sin ejecutar ninguna. No hay lógica de "cuál gana" acá a
propósito -- eso sería inventar una regla que la spec explícitamente no da.

CORRECCIÓN (Prompt 5, 2026-09-21): scopeado por SÍMBOLO únicamente, no por
símbolo+sesión -- la primera versión exigía que también coincidiera la
sesión, un remanente de cuando la concurrencia se pensaba a nivel de sesión
global (ver `session_risk.py`). Con la concurrencia corregida a nivel de
símbolo, la sesión ya no pertenece a esta comparación.
"""

from __future__ import annotations

from trader.mtf_strategies.signal import ConflictDiscarded, MTFSignal, NoSignal


def resolve(
    continuation_result: list[MTFSignal] | NoSignal, reversal_result: list[MTFSignal] | NoSignal
) -> tuple[list[MTFSignal], list[ConflictDiscarded]]:
    """Devuelve (señales_ejecutables, conflictos_descartados). Si una de las
    2 estrategias no dio señal, la otra pasa sin conflicto -- el conflicto
    solo existe cuando AMBAS generaron algo sobre el mismo símbolo."""
    cont_signals = continuation_result if isinstance(continuation_result, list) else []
    rev_signals = reversal_result if isinstance(reversal_result, list) else []

    if not cont_signals or not rev_signals:
        return cont_signals + rev_signals, []

    conflicts = [
        ConflictDiscarded(symbol=c.symbol, session=c.session, continuation_signal=c, reversal_signal=r)
        for c in cont_signals
        for r in rev_signals
        if c.symbol == r.symbol
    ]
    if not conflicts:
        return cont_signals + rev_signals, []

    conflicted_symbols = {c.symbol for c in cont_signals} & {r.symbol for r in rev_signals}
    survivors = [s for s in (cont_signals + rev_signals) if s.symbol not in conflicted_symbols]
    return survivors, conflicts
