"""ADAPTADO de `pipeline/confluence.py::evaluate_confluences` -- ese gate
cuenta FAMILIAS de confluencia genéricas (liquidez/estructura/FVG/order block/
...) contra un mínimo configurable, pensado para features ICT amplias y
reemplazado en producción por el score ponderado (`pipeline/scoring.py`,
2026-09-17) precisamente porque un conteo de familias resultó menos preciso
que un score. Estrategia 1/2 piden explícitamente lo contrario del score:
"todo-o-nada", pero sobre condiciones NOMBRADAS y específicas de esta
secuencia (liquidez HTF + sweep 1H + MSS 15M + FVG + R:R>=2), no un conteo de
familias genéricas -- por eso esto es una adaptación, no una reconexión
literal de `evaluate_confluences`: mismo espíritu binario ("sin puntaje
parcial"), estructura distinta porque las condiciones son distintas.

Nunca se importa `pipeline/scoring.py` ni `pipeline/confluence.py` desde este
paquete -- aislamiento explícito del sistema en producción.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failed_condition: str | None  # nombre de la PRIMERA condición que fallo -- todo-o-nada, no hace falta saber cuántas más fallaron
    conditions: dict[str, bool] = field(default_factory=dict)


def evaluate_all_or_nothing(conditions: dict[str, bool]) -> GateResult:
    """`conditions` en el ORDEN en que deben cumplirse (dict de Python 3.7+
    preserva orden de inserción) -- se reporta la primera que falla, mismo
    principio que el algoritmo secuencial de las reglas 13 (Estrategia 1) y
    el equivalente de Estrategia 2: cada paso debe cumplirse, sin score
    parcial que compense uno faltante con otros sobrantes."""
    for name, ok in conditions.items():
        if not ok:
            return GateResult(passed=False, failed_condition=name, conditions=conditions)
    return GateResult(passed=True, failed_condition=None, conditions=conditions)
