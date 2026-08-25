"""Filtro de calendario economico: evita operar alrededor de eventos de alto impacto
(NFP, FOMC, CPI) donde el spread se dispara y el movimiento es impredecible por
noticias, no por el edge tecnico del modelo -- practica estandar en mesas
institucionales de trend-following/intradia.

Sin API externa de pago (decision del usuario, 2026-08-24: cero costo, cero cuenta
que crear/mantener):
- NFP sigue una regla recurrente y conocida (primer viernes de cada mes, 8:30 hora
  del Este de EEUU) -- se calcula sola, ajustada automaticamente a horario de
  verano/invierno via `zoneinfo` (8:30 ET son 12:30 UTC en verano/EDT, 13:30 UTC en
  invierno/EST).
- FOMC y CPI NO siguen una regla fija (fechas anunciadas mes a mes por la Fed/BLS,
  no "el segundo martes" ni nada calculable) -- se mantienen en
  `config.yaml: trading.news_calendar.manual_events`, una lista que hay que revisar
  y actualizar de vez en cuando (unas 8-12 fechas al año cada uno, mantenimiento
  minimo).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_US_EASTERN = ZoneInfo("America/New_York")


def _first_friday_utc(year: int, month: int, hour_et: int, minute_et: int) -> datetime:
    local = datetime(year, month, 1, hour_et, minute_et, tzinfo=_US_EASTERN)
    days_until_friday = (4 - local.weekday()) % 7  # Monday=0 ... Friday=4
    local += timedelta(days=days_until_friday)
    return local.astimezone(timezone.utc)


def _nfp_datetime_utc(reference_utc: datetime, time_et: str) -> datetime:
    hour, minute = (int(part) for part in time_et.split(":"))
    return _first_friday_utc(reference_utc.year, reference_utc.month, hour, minute)


def is_news_blackout(timestamp: datetime, config: dict | None) -> bool:
    """True si `timestamp` cae dentro de la ventana de bloqueo de algun evento.

    `config=None` significa "sin calendario configurado" -> nunca bloquea (distinto
    de pasar `{}`, que activa el calendario con los valores por defecto: NFP si,
    ventana de 15 min antes/despues).
    """
    if config is None:
        return False
    cfg = config
    if not cfg.get("enabled", True):
        return False

    before = timedelta(minutes=cfg.get("minutes_before", 15))
    after = timedelta(minutes=cfg.get("minutes_after", 15))
    ts = timestamp.astimezone(timezone.utc)

    nfp_cfg = cfg.get("nfp", {})
    if nfp_cfg.get("enabled", True):
        event = _nfp_datetime_utc(ts, nfp_cfg.get("time_et", "08:30"))
        if event - before <= ts <= event + after:
            return True

    for event in cfg.get("manual_events", []):
        event_dt = datetime.fromisoformat(f"{event['date']}T{event['time_utc']}:00+00:00")
        if event_dt - before <= ts <= event_dt + after:
            return True

    return False
