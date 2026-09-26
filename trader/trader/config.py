from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_ENV_PATH = DEFAULT_CONFIG_PATH.parent / ".env"


@dataclass(frozen=True)
class SessionWindowConfig:
    timezone: str
    start_hour: int
    end_hour: int


@dataclass(frozen=True)
class SessionsConfig:
    asia: SessionWindowConfig
    london: SessionWindowConfig
    new_york: SessionWindowConfig
    killzone_london: SessionWindowConfig
    killzone_new_york: SessionWindowConfig
    # False (default): any of Asia/London/NY active is enough to pass the session
    # gate. True: only the London-NY overlap or a killzone passes -- a strictly
    # narrower gate than what the strategy doc's own "prioritize the overlap"
    # language might suggest is already the default.
    require_overlap_or_killzone: bool = False


@dataclass(frozen=True)
class RiskConfig:
    min_risk_reward: float
    partial_at_progress_pct: float
    min_risk_atr_multiple: float = 1.0  # reject (never widen) a structural SL closer than this many H1 ATRs


@dataclass(frozen=True)
class ConfluenceConfig:
    min_confluences: int
    min_timeframes: int
    min_confluences_reversal: int = 4


@dataclass(frozen=True)
class PoiConfig:
    tolerance_pct: float = 0.1


@dataclass(frozen=True)
class StructureConfig:
    swing_left_bars: int
    swing_right_bars: int
    displacement_atr_multiple: float
    atr_period: int = 14


@dataclass(frozen=True)
class ZoneLifecycleConfig:
    # A touched OB/FVG zone only counts as a genuinely invalidated (vs. merely
    # respected) once it's held for this long past the first touch without a
    # closing break -- see prompt-fix-invalidacion-obfvg.md. Measured in M15
    # candles regardless of the zone's own timeframe, since M15 is this
    # system's execution clock.
    invalidation_grace_m15_candles: int = 6


@dataclass(frozen=True)
class M15ConfirmationConfig:
    # How many trailing M15 candles (inclusive of the one that just closed)
    # count as "fresh enough" for a confirmation event to still be tradeable.
    # 1 = only the candle that just closed (the original, strict behavior);
    # N>1 allows a confirmation up to N-1 candles stale. See
    # prompt-auditoria-m15-confirmation.md / prompt-implementar-m15-n2.md --
    # `_candidate_confirmation_timestamps` in backtest/engine.py MUST read
    # this same value, or the backtest silently under-counts relative to what
    # `_latest_confirmation` would actually allow live.
    window_candles: int = 1


@dataclass(frozen=True)
class NewsEventConfig:
    timestamp_utc: str  # ISO-8601, e.g. "2025-07-30T18:00:00+00:00"
    label: str


@dataclass(frozen=True)
class ScoringConfig:
    # Minimum total score (out of 90 -- see
    # logs/scoring_system_weights_proposal.md) required to generate a signal,
    # once all hard gates (R:R, SL floor, news, POI existence, M15
    # confirmation existence) already passed. Two thresholds, trend vs
    # reversal, replacing the old min_confluences/min_confluences_reversal
    # 3:4 ratio -- these are exactly what prompt-sistema-puntuacion-ponderada.md's
    # threshold sweep varies; the 50.0/67.0 defaults are the sweep's CENTER,
    # not a locked-in final answer.
    min_score_trend: float = 50.0
    min_score_reversal: float = 67.0


@dataclass(frozen=True)
class NewsFilterConfig:
    # No operar en ventanas de noticias de alto impacto -- ya especificado en
    # la estrategia original, nunca implementado hasta prompt-implementar-m15-n2.md.
    # NFP se calcula (primer viernes del mes, 8:30 ET) sin necesidad de una
    # lista manual. FOMC es una lista curada de trader/config.yaml (fecha del
    # segundo dia de cada reunion, 14:00 ET), tomada de la pagina oficial de
    # la Fed. CPI y bancos centrales no-USD (ECB/BOE/BOJ) NO estan cubiertos
    # todavia -- ver el docstring de trader/news_filter.py.
    enabled: bool = True
    buffer_minutes_before: int = 30
    buffer_minutes_after: int = 60
    nfp_auto: bool = True
    manual_events: list[NewsEventConfig] = field(default_factory=list)


@dataclass(frozen=True)
class LiquidityConfig:
    equal_level_tolerance_pct: float
    turtle_soup_lookback_bars: int
    sweep_wick_min_pct: float


@dataclass(frozen=True)
class FvgConfig:
    min_gap_pct: float


@dataclass(frozen=True)
class SupportResistanceConfig:
    cluster_tolerance_pct: float
    min_touches: int


@dataclass(frozen=True)
class ElliottConfig:
    zigzag_deviation_pct: float


@dataclass(frozen=True)
class GatewayConfig:
    base_url: str
    timeout_seconds: int
    login_timeout_seconds: int = 90
    login: int | None = None
    password: str | None = None
    server: str | None = None


@dataclass(frozen=True)
class TraderConfig:
    gateway: GatewayConfig
    symbols: list[str]
    structure: StructureConfig
    liquidity: LiquidityConfig
    fvg: FvgConfig
    support_resistance: SupportResistanceConfig
    elliott: ElliottConfig
    risk: RiskConfig
    confluence: ConfluenceConfig
    sessions: SessionsConfig
    poi: PoiConfig
    zone_lifecycle: ZoneLifecycleConfig
    m15_confirmation: M15ConfirmationConfig
    news_filter: NewsFilterConfig
    scoring: ScoringConfig
    signals_log_path: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _session_window(raw: dict[str, Any]) -> SessionWindowConfig:
    return SessionWindowConfig(
        timezone=raw["timezone"],
        start_hour=int(raw["start_hour"]),
        end_hour=int(raw["end_hour"]),
    )


def _news_filter_config(raw: dict[str, Any]) -> NewsFilterConfig:
    return NewsFilterConfig(
        enabled=bool(raw.get("enabled", True)),
        buffer_minutes_before=int(raw.get("buffer_minutes_before", 30)),
        buffer_minutes_after=int(raw.get("buffer_minutes_after", 60)),
        nfp_auto=bool(raw.get("nfp_auto", True)),
        manual_events=[NewsEventConfig(**e) for e in raw.get("manual_events", [])],
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TraderConfig:
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    sessions_raw = raw["sessions"]
    kz = sessions_raw["killzones"]

    return TraderConfig(
        gateway=GatewayConfig(
            base_url=raw["gateway"]["base_url"],
            timeout_seconds=int(raw["gateway"]["timeout_seconds"]),
            login_timeout_seconds=int(raw["gateway"].get("login_timeout_seconds", 90)),
            login=int(os.environ["PYGW_LOGIN"]) if os.environ.get("PYGW_LOGIN") else None,
            password=os.environ.get("PYGW_PASSWORD"),
            server=os.environ.get("PYGW_SERVER"),
        ),
        symbols=list(raw["symbols"]),
        structure=StructureConfig(**raw["structure"]),
        liquidity=LiquidityConfig(**raw["liquidity"]),
        fvg=FvgConfig(**raw["fvg"]),
        support_resistance=SupportResistanceConfig(**raw["support_resistance"]),
        elliott=ElliottConfig(**raw["elliott"]),
        risk=RiskConfig(**raw["risk"]),
        confluence=ConfluenceConfig(**raw["confluence"]),
        sessions=SessionsConfig(
            asia=_session_window(sessions_raw["asia"]),
            london=_session_window(sessions_raw["london"]),
            new_york=_session_window(sessions_raw["new_york"]),
            killzone_london=_session_window(kz["london"]),
            killzone_new_york=_session_window(kz["new_york"]),
            require_overlap_or_killzone=bool(sessions_raw.get("require_overlap_or_killzone", False)),
        ),
        poi=PoiConfig(**raw.get("poi", {"tolerance_pct": 0.1})),
        zone_lifecycle=ZoneLifecycleConfig(**raw.get("zone_lifecycle", {"invalidation_grace_m15_candles": 6})),
        m15_confirmation=M15ConfirmationConfig(**raw.get("m15_confirmation", {"window_candles": 1})),
        news_filter=_news_filter_config(raw.get("news_filter", {})),
        scoring=ScoringConfig(**raw.get("scoring", {"min_score_trend": 50.0, "min_score_reversal": 67.0})),
        signals_log_path=raw["logging"]["signals_path"],
        raw=raw,
    )
