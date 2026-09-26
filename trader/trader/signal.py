from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.events import MarketEvent
from trader.pipeline.confluence import ConfluenceCheck
from trader.pipeline.scoring import ScoreBreakdown
from trader.risk.levels import TradeLevels
from trader.sessions import SessionState


@dataclass(frozen=True)
class TradingSignal:
    symbol: str
    direction: str  # "long" | "short"
    category: str  # "trend" | "reversal" -- mandatory label, see prompt-eliminar-gate-d1-h4.md
    bias_1d: str  # "up" | "down"
    session: SessionState
    poi: MarketEvent
    confirmation: MarketEvent
    confluences: ConfluenceCheck
    score: ScoreBreakdown  # composite weighted score, see logs/scoring_system_weights_proposal.md
    levels: TradeLevels
    partial_at_progress_pct: float
    generated_at: pd.Timestamp

    def partial_price(self) -> float:
        span = self.levels.tp - self.levels.entry
        return self.levels.entry + span * self.partial_at_progress_pct

    def to_report(self) -> str:
        bias_label = "alcista" if self.direction == "long" else "bajista"
        session_label = "/".join(self.session.active_labels) or "fuera de sesion"
        confluence_list = ", ".join(sorted(self.confluences.families)) or "ninguna"
        category_label = "tendencia" if self.category == "trend" else "reversion"
        return (
            f"Par: {self.symbol}\n"
            f"Categoria: {category_label}\n"
            f"Bias 1D: {bias_label}\n"
            f"Sesion activa: {session_label}\n"
            f"POI 1H: {self.poi.kind} [{self.poi.price_low:.5f}, {self.poi.price_high:.5f}]\n"
            f"Confirmacion 15min: {self.confirmation.kind}\n"
            f"Confluencias usadas: {confluence_list}\n"
            f"Score: {self.score.total:.1f} (bias={self.score.bias:.1f}, poi={self.score.poi:.1f}, "
            f"confirmacion={self.score.confirmation:.1f}, sweep={self.score.sweep:.1f}, "
            f"sesion={self.score.session:.1f}, diversidad={self.score.diversity:.1f})\n"
            f"Entrada: {self.levels.entry:.5f}\n"
            f"SL: {self.levels.sl:.5f} - nivel estructural del POI 1H\n"
            f"TP: {self.levels.tp:.5f} - proxima liquidez/estructura relevante\n"
            f"Ratio R:R: 1:{self.levels.risk_reward:.2f}\n"
            f"Regla de parcial: cerrar 50% en {self.partial_price():.5f} + mover SL a BE\n"
        )


@dataclass(frozen=True)
class ReformedSignal:
    """Senal de la arquitectura de 4 capas (logs/reformulacion_diseno_capas.md).

    NO extiende `TradingSignal` -- se evaluo y se descarto: los campos
    obligatorios de `TradingSignal` (`poi`/`confirmation` como MarketEvent
    unicos, `confluences: ConfluenceCheck`, `score: ScoreBreakdown`,
    `levels.tp` como take-profit FIJO) no tienen equivalente real en el
    diseno reformulado -- la razon dominante puede ser un sweep de liquidez
    sin ninguna zona asociada, y la capa 4 de gestion (`trader/management.py`)
    deliberadamente NO usa un TP fijo. Forzar esos campos con valores
    inventados solo para "reusar" el tipo seria auditoria falsa, no real.
    """

    symbol: str
    direction: str  # "long" | "short"
    generated_at: pd.Timestamp
    regime: str
    regime_efficiency_ratio: float
    dominant_reason_kind: str  # "liquidity_taken" | "htf_zone" | "key_level"
    dominant_reason_strength_percentile: float
    confirmation_score_percentile: float
    confirmation_followthrough: bool
    named_pattern_classification: str | None
    conviction_multiplier: float
    risk_pct_applied: float
    entry_reference_price: float
    original_sl: float
    partial_target: float | None
    final_target: float  # objetivo de TP real usado en gestion -- ver trader/management.py; agregado sobre la
    # lista minima de logs/reformulacion_diseno_capas.md por auditabilidad (la senal por si sola no permite
    # reconstruir contra que TP se gestiono la operacion sin este campo)
    planned_risk_reward: float | None  # solo para el gate de elegibilidad, no un TP de gestion -- ver management.py

    def to_report(self) -> str:
        direction_label = "largo" if self.direction == "long" else "corto"
        pattern = self.named_pattern_classification or "sin clasificar"
        partial = f"{self.partial_target:.5f}" if self.partial_target is not None else "ninguno (sin nivel intermedio)"
        planned_rr = f"1:{self.planned_risk_reward:.2f}" if self.planned_risk_reward is not None else "n/d"
        return (
            f"Par: {self.symbol}\n"
            f"Direccion: {direction_label}\n"
            f"Regimen D1: {self.regime} (ratio de eficiencia={self.regime_efficiency_ratio:.2f})\n"
            f"Razon dominante: {self.dominant_reason_kind} (percentil={self.dominant_reason_strength_percentile:.1f})\n"
            f"Confirmacion M15: percentil={self.confirmation_score_percentile:.1f}, "
            f"followthrough={self.confirmation_followthrough}, patron={pattern}\n"
            f"Conviccion: x{self.conviction_multiplier:.2f} -> riesgo aplicado {self.risk_pct_applied:.2f}%\n"
            f"Entrada de referencia: {self.entry_reference_price:.5f}\n"
            f"SL estructural inicial: {self.original_sl:.5f}\n"
            f"Objetivo de parcial: {partial}\n"
            f"TP final: {self.final_target:.5f}\n"
            f"R:R planeado (gate de elegibilidad): {planned_rr}\n"
        )
