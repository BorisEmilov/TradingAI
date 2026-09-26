"""NUEVO -- Prompt 7. Piezas puras/testeables de la gestión de posiciones EN
VIVO, separadas del script de arranque (`scripts/run_mtf_pilot.py`) para
poder testearlas sin gateway real, mismo principio que el resto de
`mtf_strategies/` (aislado de `pipeline/` y del piloto RSI).

Decisión de diseño clave (Prompt 7, punto 2): SL y TP2 se colocan como
órdenes REALES en MT5 al abrir la posición (`gateway_client.open_position`
ya envía `sl`/`tp` directo al `TRADE_ACTION_DEAL` -- confirmado leyendo
`PythonGetaway/app/routers/trading.py`, sin necesidad de corregir nada). TP2
(no TP1) es el que se manda como `tp` real: MT5 solo soporta UN take-profit
por posición y lo ejecuta cerrando el volumen ENTERO al tocarlo -- si se
mandara TP1 ahí, el bróker cerraría el 100% al llegar a 1R en vez del 50%
parcial que pide la regla 14. Consecuencia práctica, y la razón de este
diseño:
  - PRE-parcial: la posición está protegida por SL real Y por TP2 real (si
    el proceso muere antes de TP1, el peor caso posible es que el bróker
    cierre todo en SL o en TP2 -- nunca queda sin protección).
  - Cuando el bot detecta que se tocó TP1 (chequeo propio, vela a vela --
    MT5 no sabe de TP1), hace un cierre PARCIAL manual (50%) y mueve el SL a
    breakeven con una orden real (`modify_sl`) -- desde ahí en adelante la
    posición vuelve a quedar 100% gestionada por el bróker (SL=breakeven
    real, TP=TP2 real, ambas órdenes ya estaban puestas) y el bot deja de
    necesitar tocarla: solo la vigila por si desaparece.
  - La salida temporal (6 velas / 90 min sin +0.5R) es la ÚNICA acción que
    el bot inicia además del parcial -- y por diseño de la regla 14 solo
    aplica ANTES del parcial, así que una vez el parcial ya se hizo, el bot
    no vuelve a intervenir nunca más sobre esa posición.
Esto significa que SL nunca deja de ser una orden real viva mientras la
posición existe -- el requisito de seguridad del Prompt 7 se cumple en cada
instante, no solo al abrir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trader.mtf_strategies.signal import MTFSignal


def signal_key(sig: MTFSignal) -> tuple:
    """Identidad real de un setup -- MISMA tupla de deduplicación que
    `scripts/scan_mtf_strategies.py` usó para el conteo de frecuencia
    (Paso 3/5/6): un setup sigue "vigente" (retraced_to_50pct se vuelve
    permanente) y se re-detecta en CADA poll subsiguiente hasta que sale de
    la ventana reciente -- sin esta dedup, el piloto re-intentaría abrir el
    mismo setup una y otra vez. Usar la misma clave que ya validó la
    frecuencia estimada (~225 operaciones/12 semanas) es necesario para que
    ese número siga siendo lo que este piloto reproduce en la práctica."""
    return (
        sig.symbol, sig.strategy, sig.setup, sig.direction,
        str(sig.sweep_timestamp), str(sig.mss_timestamp), str(sig.fvg_confirmed_at),
    )


def tp1_touched(direction: str, bar_high: float, bar_low: float, tp1: float) -> bool:
    return bar_high >= tp1 if direction == "long" else bar_low <= tp1


def round_down_to_step(volume: float, step: float) -> float:
    """Redondea HACIA ABAJO al step de volumen del símbolo -- nunca hacia
    arriba (un cierre parcial más grande de lo pedido dejaría MENOS
    corriendo de lo que la regla 14 pide, y un cierre más chico simplemente
    deja algo más corriendo, nunca arriesga de más). Un epsilon chico evita
    que el error de punto flotante (0.37/0.01 == 36.99999999999999) tire un
    valor exacto un escalón hacia abajo de lo que corresponde."""
    if step <= 0:
        return volume
    steps = int(volume / step + 1e-9)
    return round(steps * step, 8)


@dataclass(frozen=True)
class ExitDeal:
    volume: float
    price: float
    entry_name: str  # "IN" | "OUT" | "OUT_BY"
    reason_name: str  # "CLIENT" | "SL" | "TP" | "EXPERT" | ...


def realized_r_from_deals(direction: str, entry_price: float, initial_sl: float, exit_deals: list[ExitDeal]) -> float:
    """R realizado, ponderado por volumen, a partir de los deals de SALIDA
    reales de MT5 (uno si se cerró de una vez, dos si hubo parcial+resto).
    `entry_price`/`initial_sl` son el fill real de apertura y el SL
    ESTRUCTURAL original (no el de breakeven) -- R se mide siempre contra el
    riesgo realmente asumido al entrar, no contra el SL ya movido."""
    risk = abs(entry_price - initial_sl)
    if risk <= 0 or not exit_deals:
        return 0.0
    total_volume = sum(d.volume for d in exit_deals)
    if total_volume <= 0:
        return 0.0
    weighted = 0.0
    for d in exit_deals:
        moved = (d.price - entry_price) if direction == "long" else (entry_price - d.price)
        weighted += (moved / risk) * (d.volume / total_volume)
    return weighted


def exit_deals_from_history(deals: list[dict[str, Any]]) -> list[ExitDeal]:
    """Filtra los deals de un `position_history()` del gateway a solo los de
    SALIDA (`entry_name` OUT/OUT_BY -- IN es la entrada, no cuenta acá)."""
    return [
        ExitDeal(volume=float(d["volume"]), price=float(d["price"]),
                  entry_name=d.get("entry_name", ""), reason_name=d.get("reason_name", ""))
        for d in deals
        if d.get("entry_name") in ("OUT", "OUT_BY")
    ]


def concurrency_open_symbols(
    open_positions: dict[str, dict[str, Any]], pending_orders: dict[str, dict[str, Any]] | None = None
) -> set[str]:
    """Deriva qué símbolos tienen una posición abierta AHORA (o, desde la
    migración a orden límite real, una orden límite pendiente todavía sin
    llenar) directamente de los diccionarios trackeados -- se evita mantener
    un segundo objeto de estado (`PositionConcurrencyState`) en paralelo que
    podría desincronizarse; esta función lo reconstruye on-demand como única
    fuente de verdad. Una orden pendiente cuenta igual que una posición
    abierta para concurrencia (Prompt: migración a orden límite, punto 4):
    todavía no comprometió capital, pero ya reserva el símbolo."""
    symbols = {pos["symbol"] for pos in open_positions.values()}
    if pending_orders:
        symbols |= {pend["symbol"] for pend in pending_orders.values()}
    return symbols
