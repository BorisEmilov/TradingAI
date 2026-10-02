"""Prompt 7: ejecución en vivo, continua, de Estrategia 1 (Continuación) y
Estrategia 2 (Reversión) -- el comando de ARRANQUE (`python3
scripts/run_mtf_pilot.py`, corrido en background igual que el piloto RSI).
Aislado por completo de `pipeline/scoring.py` (producción) y de
`scripts/run_pilot_rsi_xpt.py` (otro piloto, otro símbolo, otro proceso,
otra sesión de gateway) -- mismo principio de aislamiento que el resto de
`mtf_strategies/`.

SEGURIDAD (confirmado antes de escribir este script, ver
`PythonGetaway/app/routers/trading.py`): `gateway_client.open_position()` ya
manda `sl`/`tp` directo al `TRADE_ACTION_DEAL` de MT5 -- son órdenes REALES
del bróker, no monitoreo del lado del cliente. Acá se manda `sl`=SL
estructural y `tp`=TP2 (nunca TP1: MT5 solo soporta un TP por posición y lo
ejecuta cerrando el 100%, así que TP1 -- que es un parcial del 50% -- tiene
que gestionarlo el bot). Consecuencia: la posición SIEMPRE tiene un SL real
vivo en el bróker, en cualquier momento de su vida (incluso si este proceso
muere) -- ver `mtf_strategies/live_state.py` para el detalle completo del
diseño.

RESILIENCIA: mismo patrón ya validado en el piloto RSI
(`run_pilot_rsi_xpt.py`) -- UN login al arrancar, sesión mantenida con
`client.refresh()` periódico, nunca re-login en loop, cada ciclo protegido
por un try/except que nunca tumba el proceso por un error transitorio.

ARRANQUE EN PRODUCCIÓN: unidad systemd --user `deploy/systemd/mtf-pilot.service`
(Restart=on-failure). Un lock (`logs/mtf_pilot.lock`) impide dos pilotos a la vez.
Códigos de salida: 0 parada pedida, 1 caída por error, 3 otra instancia viva.

PARADA: `scripts/stop_mtf_pilot.py` escribe un archivo "stop flag" que este
bucle chequea al principio de cada ciclo -- termina el ciclo en curso
(nunca corta a mitad) y sale. Esto SOLO deja de evaluar señales nuevas: las
posiciones ya abiertas no se tocan, siguen su curso protegidas por sus
órdenes reales de SL/TP en MT5 aunque el proceso esté detenido.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import SymbolCost, cost_in_r
from trader.config import load_config
from trader.events import TF_DURATION, closed_candles_as_of
from trader.gateway_client import PythonGetawayClient
from trader.mtf_strategies.analysis import build_analysis
from trader.mtf_strategies.bias import classify_htf_structure, entry_4h_condition_holds
from trader.mtf_strategies.conflict import resolve
from trader.mtf_strategies.continuation import evaluate_continuation
from trader.mtf_strategies.live_events import log_event, now_iso
from trader.mtf_strategies.live_state import (
    concurrency_open_symbols,
    exit_deals_from_history,
    realized_r_from_deals,
    round_down_to_step,
    signal_key,
    tp1_touched,
)
from trader.mtf_strategies.reversal import evaluate_reversal
from trader.risk.levels import compute_trade_levels, limit_entry_still_ahead
from trader.mtf_strategies.session_risk import (
    DailyLossState,
    PositionConcurrencyState,
    RISK_PCT_MIN,
    active_entry_session,
    can_open_new_trade,
    compute_position_size_lots,
    daily_loss_limit_reached,
    entry_session_end,
    record_daily_result,
)

# +7 pares verificados 2026-09-25 (integridad/costo/sizing, ver
# scripts/verify_and_fetch_new_fx_pairs.py) -- NO desplegado hasta revisar la
# re-medición de frecuencia (scripts/measure_frequency_10_symbols.py).
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY"]
POLL_SECONDS = 300  # 5 min -- compromiso pragmático entre reaccionar rápido
# (gestión de parcial/detección de cierre) y no recomputar
# los detectores de los 3 símbolos sin necesidad; NO validado/barrido como
# óptimo, igual que MAX_CANDLES_M15 en exits.py.
REFRESH_SECONDS = 1200  # misma cadencia que el piloto RSI
HEARTBEAT_SECONDS = 1800  # 30 min
MIN_NET_RR = 2.0  # R:R neto de costo real -- mismo umbral usado en el recálculo del Prompt 6
RISK_PCT = RISK_PCT_MIN  # 0.25% de equity -- el extremo conservador del rango [0.25%,0.50%] de la regla 14, elegido por defecto para el piloto

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
STATE_PATH = LOGS_DIR / "mtf_pilot_state.json"
JSON_EVENTS_PATH = LOGS_DIR / "mtf_pilot_events.jsonl"
HUMAN_EVENTS_PATH = LOGS_DIR / "mtf_pilot_events.txt"
STOP_FLAG_PATH = LOGS_DIR / "mtf_pilot_stop.flag"
LOCK_PATH = LOGS_DIR / "mtf_pilot.lock"

# Códigos de salida (los lee systemd: Restart=on-failure relanza solo != 0, y
# RestartPreventExitStatus=3 evita pelear con otra instancia viva).
EXIT_STOPPED = 0          # parada pedida: stop flag, SIGTERM (systemctl stop) o Ctrl-C
EXIT_CRASHED = 1          # caída por error -- hay que relanzar
EXIT_ALREADY_RUNNING = 3  # otro piloto tiene el lock -- no se arranca


class PilotTerminated(BaseException):
    """SIGTERM recibido. BaseException para que el `except Exception` del ciclo no la trague."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _raise_terminated(signum, frame) -> None:
    raise PilotTerminated(signum)


def _acquire_single_instance_lock():
    """flock no bloqueante sobre LOCK_PATH, retenido mientras viva el proceso
    (el kernel lo suelta solo si el proceso muere, incluso con SIGKILL).
    None si otro piloto ya lo tiene."""
    import fcntl
    fh = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def _log(symbol: str, strategy: str, kind: str, detail: str, desktop_title: str | None = None, **fields) -> None:
    log_event(JSON_EVENTS_PATH, HUMAN_EVENTS_PATH, symbol, strategy, kind, detail, desktop_title=desktop_title, **fields)


def _load_state() -> dict:
    import json
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        state.setdefault("pending_orders", {})  # migración: estado viejo no tenía órdenes límite
        return state
    return {"started_at": now_iso(), "open_positions": {}, "pending_orders": {}, "seen_signal_keys": [], "daily_loss": None}


def _save_state(state: dict) -> None:
    """Atómico: temporal en el mismo directorio + os.replace. Un proceso muerto a
    mitad de escritura deja el archivo anterior intacto, nunca uno truncado."""
    import json
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)


def _daily_loss_state(state: dict, ts: pd.Timestamp) -> DailyLossState:
    raw = state.get("daily_loss")
    if raw is None:
        ds = DailyLossState(date=ts.normalize())
    else:
        ds = DailyLossState(date=pd.Timestamp(raw["date"]), cumulative_r=raw["cumulative_r"], locked_out=raw["locked_out"])
    ds, _ = daily_loss_limit_reached(ds, ts)  # rueda al día nuevo si corresponde, no-op si no
    return ds


def _save_daily_loss_state(state: dict, ds: DailyLossState) -> None:
    state["daily_loss"] = {"date": str(ds.date), "cumulative_r": ds.cumulative_r, "locked_out": ds.locked_out}


def _resample_h4(h1_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        h1_raw.set_index("timestamp").resample("4h", origin="start_day")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    )


def main() -> int:
    LOGS_DIR.mkdir(exist_ok=True)
    lock = _acquire_single_instance_lock()
    if lock is None:
        print(f"otro piloto MTF ya está corriendo (lock {LOCK_PATH} tomado) -- no se arranca un segundo", file=sys.stderr)
        return EXIT_ALREADY_RUNNING
    signal.signal(signal.SIGTERM, _raise_terminated)
    if STOP_FLAG_PATH.exists():
        STOP_FLAG_PATH.unlink()  # una parada anterior no debe impedir un arranque nuevo

    client = None
    try:
        config = load_config()
        client = PythonGetawayClient(config.gateway)
        client.login()
        _log("ALL", "system", "pilot_started",
             f"Piloto MTF (Continuación+Reversión) arrancado -- símbolos={SYMBOLS}, poll={POLL_SECONDS}s, "
             f"risk_pct={RISK_PCT}, min_net_rr={MIN_NET_RR}",
             desktop_title="Piloto MTF arrancado")
        _run_loop(client, _load_state(), config)
        return EXIT_STOPPED
    except (PilotTerminated, KeyboardInterrupt) as exc:
        name = signal.Signals(exc.signum).name if isinstance(exc, PilotTerminated) else "SIGINT"
        _log("ALL", "system", "pilot_terminated_by_signal",
             f"{name} recibido -- parada SOLICITADA (systemctl stop / kill / Ctrl-C), código de salida {EXIT_STOPPED}. "
             "Las posiciones abiertas (si las hay) siguen protegidas por SL/TP reales en MT5.",
             desktop_title="Piloto MTF detenido por señal")
        return EXIT_STOPPED
    except Exception as exc:  # noqa: BLE001 -- toda caída queda registrada y sale con código de error
        _log("ALL", "system", "pilot_crashed",
             f"CAÍDA POR ERROR (no es una parada pedida): {exc!r} -- código de salida {EXIT_CRASHED}, "
             "systemd lo relanza (Restart=on-failure). Posiciones protegidas por SL/TP reales en MT5.",
             desktop_title="Piloto MTF CAÍDO", trace=traceback.format_exc()[-2000:])
        return EXIT_CRASHED
    finally:
        if client is not None:
            try:
                client.logout()
            except Exception:  # noqa: BLE001 -- nunca tapar el código de salida real
                pass
            _log("ALL", "system", "session_closed",
                 "Sesión de gateway cerrada. Las posiciones abiertas, si las hay, siguen protegidas por "
                 "sus órdenes reales de SL/TP en MT5 -- se resuelven solas aunque este proceso no esté corriendo.")
        lock.close()


def _run_loop(client: PythonGetawayClient, state: dict, config) -> None:
    """Vuelve solo por la stop flag. Un error en refresh o en un ciclo se registra y
    se reintenta; lo que escape de acá (ej. no poder guardar el estado) es una caída."""
    last_refresh = time.time()
    last_heartbeat = 0.0
    while True:
        if STOP_FLAG_PATH.exists():
            STOP_FLAG_PATH.unlink()
            _log("ALL", "system", "pilot_stopping",
                 "Comando de parada recibido -- se deja de evaluar señales nuevas. "
                 "Las posiciones abiertas (si las hay) NO se tocan, siguen protegidas por SL/TP reales en MT5.",
                 desktop_title="Piloto MTF detenido")
            return

        if time.time() - last_refresh > REFRESH_SECONDS:
            try:
                client.refresh()
                last_refresh = time.time()
            except Exception as exc:  # noqa: BLE001 -- gateway caído un rato no debe tumbar el piloto
                _log("ALL", "system", "refresh_failed",
                     f"no se pudo renovar la sesión del gateway: {exc} -- se reintenta el próximo ciclo (sin re-login)")

        try:
            _tick(client, state, config)
        except Exception as exc:  # noqa: BLE001 -- un ciclo malo nunca debe tumbar el piloto
            _log("ALL", "system", "tick_error", f"{exc}", trace=traceback.format_exc()[-2000:])

        _save_state(state)

        if time.time() - last_heartbeat > HEARTBEAT_SECONDS:
            daily = state.get("daily_loss") or {}
            _log("ALL", "system", "heartbeat",
                 f"vivo -- {len(state['open_positions'])} posiciones abiertas, "
                 f"{len(state['pending_orders'])} órdenes límite pendientes, "
                 f"día acumulado={daily.get('cumulative_r', 0.0):+.2f}R "
                 f"(bloqueado={daily.get('locked_out', False)})")
            last_heartbeat = time.time()

        time.sleep(POLL_SECONDS)


def _tick(client: PythonGetawayClient, state: dict, config) -> None:
    _reconcile_closed_positions(client, state)
    _reconcile_pending_orders(client, state)
    _detect_broker_changes(client, state)

    for symbol in SYMBOLS:
        h1_raw = client.candles(symbol, "H1", 1000, timeout=60)
        m15_raw = client.candles(symbol, "M15", 2000, timeout=60)
        d1_raw = client.candles(symbol, "D1", 500, timeout=60)
        if len(m15_raw) == 0 or len(h1_raw) == 0:
            _log(symbol, "system", "insufficient_data", f"sin velas suficientes (h1={len(h1_raw)}, m15={len(m15_raw)})")
            continue

        _manage_open_positions(client, state, symbol, m15_raw)

        h4_raw = _resample_h4(h1_raw)
        _manage_pending_orders(client, state, config, symbol, h4_raw, m15_raw)
        _evaluate_new_signals(client, state, config, symbol, h4_raw, h1_raw, m15_raw, d1_raw)


# -- gestión de posiciones ya abiertas ---------------------------------------

def _reconcile_closed_positions(client: PythonGetawayClient, state: dict) -> None:
    """Detecta posiciones que YA NO están en `client.positions()` -- se
    cerraron por el bróker (SL real, TP2 real, stop-out) o manualmente, sin
    que este bot lo haya iniciado. Única fuente de verdad para saber qué
    pasó: `position_history()`, que trae los deals reales con precio y
    motivo."""
    live_tickets = {str(p["ticket"]) for p in client.positions()}
    for ticket in list(state["open_positions"].keys()):
        if ticket in live_tickets:
            continue
        pos = state["open_positions"][ticket]
        try:
            hist = client.position_history(int(ticket))
            exits = exit_deals_from_history(hist.get("deals", []))
        except Exception as exc:  # noqa: BLE001 -- no perder el ciclo por un fallo de history
            _log(pos["symbol"], pos["strategy"], "position_history_fetch_failed",
                 f"ticket={ticket}: {exc} -- se reintentará el próximo ciclo")
            continue
        realized_r = realized_r_from_deals(pos["direction"], pos["entry_price"], pos["initial_sl"], exits)
        reasons = sorted({d.reason_name for d in exits if d.reason_name}) or ["desconocido"]
        _finalize_close(client, state, ticket, pos, realized_r, f"cerrada_por_broker:{'+'.join(reasons)}", hist=hist)


def _reconcile_pending_orders(client: PythonGetawayClient, state: dict) -> None:
    """Detecta órdenes límite que YA NO están en `client.pending_orders()` --
    o se llenaron (pasan a ser una posición real, mismo ticket -- confirmado
    contra la API real: `result["order"]` de `place_pending` es el ticket que
    después aparece en `/positions` si se llena) o se cancelaron/expiraron sin
    llenar (ya no están ni en `/orders` ni en `/positions`)."""
    if not state["pending_orders"]:
        return
    live_order_tickets = {str(o["ticket"]) for o in client.pending_orders()}
    live_positions = {str(p["ticket"]): p for p in client.positions()}
    for ticket in list(state["pending_orders"].keys()):
        if ticket in live_order_tickets:
            continue
        pend = state["pending_orders"][ticket]
        if ticket in live_positions:
            _promote_pending_to_position(client, state, ticket, pend, live_positions[ticket])
        else:
            _finalize_pending_gone(client, state, ticket, pend)


def _detect_broker_changes(client: PythonGetawayClient, state: dict) -> None:
    """Reporta TODO cambio en MT5 que el bot no hizo él mismo: SL/TP/volumen
    de posiciones y precio/SL/TP/volumen de órdenes pendientes modificados
    (ej. a mano desde MT5), y posiciones/órdenes en la cuenta que el bot no
    abrió. Solo informa -- no cambia la gestión del bot."""
    live_pos = {str(p["ticket"]): p for p in client.positions()}
    live_ord = {str(o["ticket"]): o for o in client.pending_orders()}
    tracked = [(t, p, live_pos, "position") for t, p in state["open_positions"].items()] + \
              [(t, p, live_ord, "pending_order") for t, p in state["pending_orders"].items()]
    for ticket, rec, live, kind in tracked:
        lv = live.get(ticket)
        if lv is None:
            continue
        now = {"sl": float(lv.get("sl") or 0.0), "tp": float(lv.get("tp") or 0.0),
               "volume": float(lv.get("volume") or lv.get("volume_current") or 0.0)}
        if kind == "pending_order":
            now["price"] = float(lv.get("price_open") or 0.0)
        prev = rec.get("broker")
        if prev is None or rec.pop("broker_resync", False):
            rec["broker"] = now
            continue
        changes = [f"{k}: {prev[k]} -> {now[k]}" for k in now if prev.get(k) != now[k]]
        if changes:
            rec["broker"] = now
            floating = f" | P&L flotante={float(lv.get('profit') or 0.0):+.2f} USD" if kind == "position" else ""
            _log(rec["symbol"], rec["strategy"], f"{kind}_modified_external",
                 f"{rec['direction'].upper()} {rec['symbol']} ticket={ticket} MODIFICADA en MT5 fuera del bot -- "
                 f"{'; '.join(changes)}{floating} (el bot no cambia su gestión por esto)",
                 desktop_title=f"Modificación en MT5: {rec['symbol']}", ticket=ticket, changes=changes)

    seen = state.setdefault("external_seen", [])
    for ticket, lv in [*live_pos.items(), *live_ord.items()]:
        if ticket in state["open_positions"] or ticket in state["pending_orders"] or ticket in seen:
            continue
        seen.append(ticket)
        what = "posición" if ticket in live_pos else "orden pendiente"
        _log(lv.get("symbol", "?"), "external", "external_trade_detected",
             f"{what} NO abierta por el bot en la cuenta: ticket={ticket} {lv.get('type_name', lv.get('type'))} "
             f"{lv.get('symbol')} vol={lv.get('volume', lv.get('volume_current'))} "
             f"precio={lv.get('price_open')} sl={lv.get('sl')} tp={lv.get('tp')} -- el bot no la gestiona",
             desktop_title=f"Operación externa en MT5: {lv.get('symbol')}", ticket=ticket)


def _fill_time_utc(client: PythonGetawayClient, server_epoch) -> pd.Timestamp:
    """Hora real del llenado en UTC desde el epoch de MT5 (hora de SERVIDOR).
    Si falta el dato o el offset, 'ahora' -- posterior al fill real, así que
    a lo sumo se ignora una vela de más (lado seguro: nunca un TP1 falso)."""
    now = pd.Timestamp.now(tz="UTC")
    if not server_epoch:
        return now
    try:
        return min(pd.Timestamp(int(server_epoch), unit="s", tz="UTC") - client.server_utc_offset(), now)
    except Exception:  # noqa: BLE001
        return now


def _open_position_record(pend: dict, fill_price: float, fill_time: pd.Timestamp) -> dict:
    risk_now = abs(fill_price - pend["sl"])
    tp1_actual = fill_price + risk_now if pend["direction"] == "long" else fill_price - risk_now
    return {
        "symbol": pend["symbol"], "strategy": pend["strategy"], "setup": pend["setup"], "direction": pend["direction"],
        "entry_price": fill_price, "initial_sl": pend["sl"], "current_sl": pend["sl"],
        "tp1": tp1_actual, "tp2": pend["tp2"],
        "volume_total": pend["volume"], "volume_remaining": pend["volume"],
        "opened_at": fill_time.isoformat(), "entry_time": fill_time.isoformat(), "fill_time": fill_time.isoformat(),
        "partial_taken": False, "last_bar_processed": None,
        "net_rr": pend["net_rr"], "signal_key": pend["signal_key"],
    }


def _promote_pending_to_position(client: PythonGetawayClient, state: dict, ticket: str, pend: dict, live_pos: dict) -> None:
    fill_price = float(live_pos.get("price_open", pend["entry_price"]))
    pos = _open_position_record(pend, fill_price, _fill_time_utc(client, live_pos.get("time")))
    tp1_actual = pos["tp1"]

    state["open_positions"][ticket] = pos
    state["pending_orders"].pop(ticket, None)
    _log(pend["symbol"], pend["strategy"], "pending_order_filled",
         f"{pend['direction'].upper()} {pend['symbol']} orden límite LLENADA @ {fill_price:.5f} "
         f"(precio límite={pend['entry_price']:.5f}) sl={pend['sl']:.5f} tp1={tp1_actual:.5f} tp2={pend['tp2']:.5f}",
         desktop_title=f"Orden límite llenada: {pend['symbol']} {pend['direction']}", ticket=ticket)


def _finalize_pending_gone(client: PythonGetawayClient, state: dict, ticket: str, pend: dict) -> None:
    """La orden ya no está en /orders ni en /positions. El estado de la orden en
    el historial del bróker decide: FILLED = se llenó y la posición YA cerró
    entre dos polls (su R cuenta para el corte diario); si no, nunca se llenó.
    Sin historial no se decide: se reintenta el próximo ciclo."""
    try:
        hist = client.order_history(int(ticket))
    except Exception as exc:  # noqa: BLE001 -- no perder el ciclo por un fallo de history
        _log(pend["symbol"], pend["strategy"], "pending_history_fetch_failed",
             f"ticket={ticket}: {exc} -- se reintentará el próximo ciclo")
        return
    orders = hist.get("orders", [])
    reason = orders[0].get("state_name", "desconocido") if orders else "desconocido"
    if reason == "FILLED":
        _finalize_filled_and_closed_between_polls(client, state, ticket, pend)
        return
    state["pending_orders"].pop(ticket, None)
    _log(pend["symbol"], pend["strategy"], "pending_order_expired_or_cancelled",
         f"{pend['direction'].upper()} {pend['symbol']} orden límite @ {pend['entry_price']:.5f} "
         f"ya no está activa (motivo={reason}) -- nunca se llenó, no se abrió posición",
         desktop_title=f"Orden límite expirada: {pend['symbol']}", ticket=ticket)


def _finalize_filled_and_closed_between_polls(client: PythonGetawayClient, state: dict, ticket: str, pend: dict) -> None:
    try:
        pos_hist = client.position_history(int(ticket))
    except Exception as exc:  # noqa: BLE001
        _log(pend["symbol"], pend["strategy"], "position_history_fetch_failed",
             f"ticket={ticket} (llenada y cerrada entre polls): {exc} -- se reintentará el próximo ciclo")
        return
    deals = pos_hist.get("deals", [])
    entry = next((d for d in deals if d.get("entry_name") == "IN"), {})
    pos = _open_position_record(pend, float(entry.get("price", pend["entry_price"])),
                                _fill_time_utc(client, entry.get("time")))
    state["pending_orders"].pop(ticket, None)
    _log(pend["symbol"], pend["strategy"], "pending_order_filled",
         f"{pend['direction'].upper()} {pend['symbol']} orden límite LLENADA @ {pos['entry_price']:.5f} "
         f"y la posición ya CERRÓ entre dos ciclos (estado FILLED en el historial del bróker)", ticket=ticket)
    exits = exit_deals_from_history(deals)
    realized_r = realized_r_from_deals(pos["direction"], pos["entry_price"], pos["initial_sl"], exits)
    reasons = sorted({d.reason_name for d in exits if d.reason_name}) or ["desconocido"]
    _finalize_close(client, state, ticket, pos, realized_r, f"cerrada_por_broker:{'+'.join(reasons)}", hist=pos_hist)


def _cancel_pending(client: PythonGetawayClient, state: dict, ticket: str, pend: dict, reason: str) -> None:
    result = client.cancel_pending(int(ticket))
    if not result.get("success"):
        _log(pend["symbol"], pend["strategy"], "pending_cancel_failed",
             f"ticket={ticket}: {result.get('retcode_name')} -- se reintentará el próximo ciclo")
        return
    state["pending_orders"].pop(ticket, None)
    _log(pend["symbol"], pend["strategy"], "pending_order_cancelled_proactively",
         f"{pend['direction'].upper()} {pend['symbol']} orden límite @ {pend['entry_price']:.5f} "
         f"cancelada antes de su vencimiento -- motivo={reason}",
         desktop_title=f"Orden límite cancelada: {pend['symbol']}", ticket=ticket)


def _manage_pending_orders(client: PythonGetawayClient, state: dict, config, symbol: str,
                            h4_raw: pd.DataFrame, m15_raw: pd.DataFrame) -> None:
    """Cancelación proactiva (Prompt: migración a orden límite, punto 3):
    una orden límite pendiente no compromete capital todavía, así que
    cancelarla es siempre seguro -- a diferencia de una posición ya abierta,
    que nunca se toca. Corre para TODOS los símbolos con orden pendiente, sin
    importar la ventana de sesión de entrada (esa ventana solo gatea señales
    NUEVAS, no la vigilancia de lo ya colocado)."""
    tickets = [t for t, p in state["pending_orders"].items() if p["symbol"] == symbol]
    if not tickets:
        return

    daily = _daily_loss_state(state, pd.Timestamp.now(tz="UTC"))
    current_price: float | None = None
    h4 = None

    for ticket in tickets:
        pend = state["pending_orders"].get(ticket)
        if pend is None:
            continue

        if daily.locked_out:
            _cancel_pending(client, state, ticket, pend, "corte_diario_-1.5R_activo_no_hay_entradas_nuevas")
            continue

        if current_price is None:
            current_price = client.last_price(symbol)
        levels_now = compute_trade_levels(pend["direction"], current_price, pend["sl"], pend["tp2"], min_rr=0.0)
        if levels_now is None:
            _cancel_pending(client, state, ticket, pend,
                             f"geometría_invalidada: precio actual={current_price:.5f} ya cruzó el sl "
                             f"({pend['sl']:.5f}) o ya superó el tp2 ({pend['tp2']:.5f}) sin haber llenado la orden")
            continue

        if h4 is None:
            latest_m15_close_ts = m15_raw["timestamp"].iloc[-1] + TF_DURATION["M15"]
            h4 = build_analysis(h4_raw, "H4", latest_m15_close_ts, config)
        current_bias = classify_htf_structure(h4.swings, float(h4.df["close"].iloc[-1]) if len(h4.df) else current_price)
        # misma condición que validó el gate de entrada de ESTE setup (bias.py)
        if not entry_4h_condition_holds(pend["strategy"], pend.get("setup"), pend["bias_direction4h"], current_bias):
            _cancel_pending(client, state, ticket, pend,
                             f"sesgo_4h_invalida_el_setup: {pend['strategy']}/{pend.get('setup')} "
                             f"dirección={pend['bias_direction4h']}, 4H ahora={current_bias}")


def _manage_open_positions(client: PythonGetawayClient, state: dict, symbol: str, m15_raw: pd.DataFrame) -> None:
    """Solo actúa sobre posiciones PRE-parcial: TP1 tocado (parcial 50% +
    breakeven real). La salida temporal (6 velas/90min sin +0.5R) se SACÓ el
    2026-09-25 (validación retroactiva, logs/temporal_exit_validation.json):
    la posición corre hasta SL o TP1+breakeven+TP2.
    Post-parcial la posición queda 100% gestionada por el bróker (SL
    breakeven real, TP2 real) -- este bot deja de necesitar tocarla."""
    last_bar = m15_raw.iloc[-1]
    bar_close_ts = m15_raw["timestamp"].iloc[-1] + TF_DURATION["M15"]
    bar_close_str = str(bar_close_ts)

    for ticket in [t for t, p in state["open_positions"].items() if p["symbol"] == symbol and p.get("breakeven_pending")]:
        _retry_breakeven(client, state["open_positions"][ticket], ticket)

    for ticket in [t for t, p in state["open_positions"].items() if p["symbol"] == symbol and not p["partial_taken"]]:
        pos = state["open_positions"].get(ticket)
        if pos is None or pos.get("last_bar_processed") == bar_close_str:
            continue
        # la vela donde (o antes de donde) se llenó la orden no es progreso de la
        # posición: su máximo/mínimo pudo ocurrir ANTES del fill (mismo criterio
        # que el simulador, que arranca en la vela siguiente a la de entrada)
        if pos.get("fill_time") and m15_raw["timestamp"].iloc[-1] < pd.Timestamp(pos["fill_time"]):
            pos["last_bar_processed"] = bar_close_str
            continue

        if tp1_touched(pos["direction"], float(last_bar["high"]), float(last_bar["low"]), pos["tp1"]):
            _take_partial(client, state, ticket)

        if ticket in state["open_positions"]:
            state["open_positions"][ticket]["last_bar_processed"] = bar_close_str


def _take_partial(client: PythonGetawayClient, state: dict, ticket: str) -> None:
    pos = state["open_positions"][ticket]
    info = client.symbol_info(pos["symbol"], timeout=30)
    half = round_down_to_step(pos["volume_total"] / 2.0, info["volume_step"])

    if half < info["volume_min"] or half <= 0:
        # El volumen no alcanza para partir 50/50 (posición ya en el lote
        # mínimo del bróker) -- fallback simple y documentado: cerrar el
        # 100% en TP1 en vez de dejar la posición en un estado ambiguo.
        result = client.close_position(int(ticket))
        if not result.get("success"):
            _log(pos["symbol"], pos["strategy"], "tp1_full_close_failed", f"ticket={ticket}: {result.get('retcode_name')}")
            return
        _finalize_close(client, state, ticket, pos, 1.0, "tp1_full_close_volume_too_small_to_split")
        _log(pos["symbol"], pos["strategy"], "tp1_hit_full_close",
             f"TP1 alcanzado (+1R) pero volumen ({pos['volume_total']}) < mínimo partible del bróker "
             f"({info['volume_min']}) -- se cerró el 100% en TP1 en vez de partir 50/50",
             desktop_title=f"TP1 alcanzado: {pos['symbol']} (cierre completo)")
        return

    result = client.close_partial(int(ticket), half)
    if not result.get("success"):
        _log(pos["symbol"], pos["strategy"], "tp1_partial_failed", f"ticket={ticket}: {result.get('retcode_name')} -- se reintentará el próximo ciclo")
        return
    # el parcial YA se ejecutó en el bróker: se registra antes de tocar el SL, así un
    # fallo de modify_sl nunca provoca un segundo parcial en el próximo ciclo
    pos["partial_taken"] = True
    pos["volume_remaining"] = round(pos["volume_total"] - half, 8)
    pos["broker_resync"] = True  # cambios hechos por el bot, no reportarlos como modificación externa
    pos["breakeven_pending"] = True
    sl_result = _retry_breakeven(client, pos, ticket)
    money = _money_from_history(client, ticket)
    _log(pos["symbol"], pos["strategy"], "tp1_partial_and_breakeven",
         f"ticket={ticket} | parcial: {_money_str(money)} (realizado hasta ahora, incluye comisión de entrada) | "
         f"TP1 alcanzado (+1R): cerrado parcial {half} lotes @ {result.get('price')} -- "
         f"SL movido a breakeven {pos['entry_price']:.5f} (motivo: TP1 alcanzado tras +1R de progreso, "
         f"orden real modify_sl={'OK' if sl_result.get('success') else sl_result.get('retcode_name')}), "
         f"resto ({pos['volume_remaining']} lotes) sigue corriendo hasta TP2 real={pos['tp2']:.5f}",
         desktop_title=f"TP1 alcanzado: {pos['symbol']} (50% cerrado, SL a breakeven)")


def _retry_breakeven(client: PythonGetawayClient, pos: dict, ticket: str) -> dict:
    """Mueve el SL real a breakeven. Si MT5 lo rechaza o el gateway falla, queda
    `breakeven_pending` y se reintenta cada ciclo hasta lograrlo (o hasta que la
    posición cierre) -- nunca se abandona en silencio."""
    try:
        result = client.modify_sl(int(ticket), sl=pos["entry_price"])
    except Exception as exc:  # noqa: BLE001
        result = {"success": False, "retcode_name": f"error gateway: {exc}"}
    if result.get("success"):
        pos["breakeven_pending"] = False
        pos["current_sl"] = pos["entry_price"]
        pos["broker_resync"] = True
    else:
        _log(pos["symbol"], pos["strategy"], "breakeven_sl_failed",
             f"ticket={ticket}: no se pudo mover el SL a breakeven {pos['entry_price']:.5f} "
             f"({result.get('retcode_name')}) -- el SL real sigue en {pos['current_sl']:.5f}; se reintenta el próximo ciclo",
             desktop_title=f"SL a breakeven FALLÓ: {pos['symbol']}", ticket=ticket)
    return result


def _money_from_history(client: PythonGetawayClient, ticket: str, hist: dict | None = None) -> dict | None:
    """P&L real en dinero (moneda de la cuenta) de una posición, desde sus deals
    reales en MT5: profit + comisión + swap + fee. None si no se pudo leer --
    nunca debe tumbar el cierre por un fallo de history."""
    try:
        hist = hist if hist is not None else client.position_history(int(ticket))
    except Exception:  # noqa: BLE001
        return None
    deals = hist.get("deals", [])
    parts = {k: round(sum(float(d.get(k) or 0.0) for d in deals), 2) for k in ("profit", "commission", "swap", "fee")}
    exits = [d for d in deals if d.get("entry_name") in ("OUT", "OUT_BY")]
    return {"net": round(sum(parts.values()), 2), **parts,
            "exit_prices": [float(d.get("price", 0.0)) for d in exits],
            "exit_reasons": sorted({d.get("reason_name", "") for d in exits if d.get("reason_name")})}


def _money_str(money: dict | None) -> str:
    if money is None:
        return "P&L en dinero: no disponible (fallo al leer el historial de MT5)"
    return (f"P&L NETO={money['net']:+.2f} USD (bruto={money['profit']:+.2f}, comisión={money['commission']:+.2f}, "
            f"swap={money['swap']:+.2f}, fee={money['fee']:+.2f})")


def _finalize_close(client: PythonGetawayClient, state: dict, ticket: str, pos: dict, realized_r: float,
                    reason_kind: str, hist: dict | None = None) -> None:
    state["open_positions"].pop(ticket, None)
    ts = pd.Timestamp.now(tz="UTC")
    daily = _daily_loss_state(state, ts)
    daily = record_daily_result(daily, ts, realized_r)
    _save_daily_loss_state(state, daily)

    money = _money_from_history(client, ticket, hist)
    try:
        balance = f"{float(client.account()['balance']):.2f} USD"
    except Exception:  # noqa: BLE001
        balance = "no disponible"
    duration = ts - pd.Timestamp(pos["opened_at"])
    exits = ", ".join(f"{p:.5f}" for p in money["exit_prices"]) if money and money["exit_prices"] else "n/d"
    _log(pos["symbol"], pos["strategy"], "position_closed",
         f"{pos['direction'].upper()} {pos['symbol']} {pos['strategy']}{'/' + pos['setup'] if pos.get('setup') else ''} "
         f"CERRADA -- ticket={ticket} motivo={reason_kind} | {_money_str(money)} | r={realized_r:+.2f}R | "
         f"entrada={pos['entry_price']:.5f} salida(s)={exits} lotes={pos['volume_total']} "
         f"sl_inicial={pos['initial_sl']:.5f} tp2={pos['tp2']:.5f} parcial_tomado={pos['partial_taken']} | "
         f"duración={str(duration).split('.')[0]} | día acumulado={daily.cumulative_r:+.2f}R | balance cuenta={balance}",
         desktop_title=f"Posición cerrada: {pos['symbol']} {money['net']:+.2f} USD" if money else f"Posición cerrada: {pos['symbol']}",
         ticket=ticket, reason=reason_kind, realized_r=realized_r, money=money)

    if daily.locked_out:
        _log("ALL", "system", "daily_loss_lockout",
             f"CORTE DIARIO ALCANZADO ({daily.cumulative_r:+.2f}R <= -1.5R) -- se BLOQUEAN entradas nuevas "
             "por el resto del día (hasta el próximo cambio de fecha UTC). Las posiciones ya abiertas, si "
             "quedan, NO se tocan -- siguen su curso normal con SL/TP reales en MT5.",
             desktop_title="Corte diario -1.5R activado")


# -- evaluación de señales nuevas ---------------------------------------------

def _evaluate_new_signals(client: PythonGetawayClient, state: dict, config, symbol: str,
                           h4_raw: pd.DataFrame, h1_raw: pd.DataFrame, m15_raw: pd.DataFrame, d1_raw: pd.DataFrame) -> None:
    latest_m15_close_ts = m15_raw["timestamp"].iloc[-1] + TF_DURATION["M15"]
    if active_entry_session(latest_m15_close_ts) is None:
        return

    h4 = build_analysis(h4_raw, "H4", latest_m15_close_ts, config)
    h1 = build_analysis(h1_raw, "H1", latest_m15_close_ts, config)
    m15 = build_analysis(m15_raw, "M15", latest_m15_close_ts, config)
    d1_as_of = closed_candles_as_of(d1_raw, "D1", latest_m15_close_ts)
    current_price = client.last_price(symbol)

    # require_retracement=False (Prompt: migración a orden límite real): el setup
    # se detecta apenas MSS+FVG están confirmados, SIN esperar a que el precio ya
    # haya vuelto al 50% -- la orden límite en sí es el mecanismo que espera el
    # retroceso, ya no el bot chequeando precio y mandando a mercado tarde. Ver
    # project_mtf_pending_limit_orders_2026-09-23 en memoria.
    cont = evaluate_continuation(symbol, h4, h1, m15, d1_as_of, config, latest_m15_close_ts,
                                  current_price, require_retracement=False)
    rev = evaluate_reversal(symbol, h4, h1, m15, d1_as_of, config, latest_m15_close_ts, require_retracement=False)
    survivors, conflicts = resolve(cont, rev)

    for c in conflicts:
        for s in (c.continuation_signal, c.reversal_signal):
            key = list(signal_key(s))
            if key not in state["seen_signal_keys"]:
                state["seen_signal_keys"].append(key)
        _log(symbol, "conflict", "conflict_discarded",
             f"Estrategia 1 y 2 generaron señal simultánea sobre {symbol} -- ninguna se ejecuta (sin prioridad fija)")

    for sig in sorted(survivors, key=lambda s: s.risk_reward, reverse=True):
        key = list(signal_key(sig))
        if key in state["seen_signal_keys"]:
            continue  # mismo setup ya visto en un poll anterior -- no reintentar (misma dedup que Paso 3/5/6)
        state["seen_signal_keys"].append(key)
        _attempt_execute(client, state, config, sig, symbol, m15_raw)


def _pending_expiration_epoch(client: PythonGetawayClient, session: str, generated_at: pd.Timestamp) -> int:
    """Cierre de la sesión de entrada (11:00 hora local, Londres o NY, DST-aware
    -- ver `session_risk.entry_session_end`) en vez de un offset fijo de
    minutos (Prompt: ventana de expiración = fin de sesión, 2026-09-23,
    comparación única y pre-registrada contra el offset fijo de 90 min usado
    antes). Un setup detectado cerca del cierre de sesión (ej. 10:55) recibe
    una ventana corta (5 min) a propósito -- es correcto, no se le pone un
    mínimo artificial. NO afecta la salida temporal post-entrada
    (`exits.MAX_MINUTES_M15`; esa regla se sacó del piloto el 2026-09-25).
    El campo `expiration` del gateway es epoch EN HORA DE SERVIDOR (confirmado
    contra la API real, `PythonGetaway/app/models.py::PlacePendingRequest`),
    así que se corrige con el mismo offset que ya usa el resto del cliente."""
    expiration_utc = entry_session_end(session, generated_at)
    return int((expiration_utc + client.server_utc_offset()).timestamp())


def _attempt_execute(client: PythonGetawayClient, state: dict, config, sig, symbol: str, m15_raw: pd.DataFrame) -> None:
    info = client.symbol_info(symbol, timeout=30)

    # spread real del momento (ticks de los últimos 15 min), NO el campo `spread`
    # de vela: ese es el mínimo de la vela y está en ~0 desde jul-2026 (verificado
    # 2026-09-25, scripts/verify_spread_source.py) -- con él el gate veía costo 0.
    spread_price = client.recent_avg_spread(symbol)
    cost = SymbolCost(symbol=symbol, point=info["point"], avg_spread_points=spread_price / info["point"],
                      avg_spread_price=spread_price)
    risk_price = abs(sig.entry - sig.sl)
    net_rr = sig.risk_reward - cost_in_r(cost, risk_price)
    if net_rr < MIN_NET_RR:
        _log(symbol, sig.strategy, "signal_discarded_cost",
             f"descartada: R:R teórico={sig.risk_reward:.2f}, neto de costo real de spread={net_rr:.2f} < {MIN_NET_RR}")
        return

    concurrency = PositionConcurrencyState(
        open_symbols=concurrency_open_symbols(state["open_positions"], state["pending_orders"])
    )
    ok, reason = can_open_new_trade(concurrency, symbol)
    if not ok:
        _log(symbol, sig.strategy, "signal_discarded_concurrency", f"descartada: {reason}")
        return

    daily = _daily_loss_state(state, sig.generated_at)
    _save_daily_loss_state(state, daily)
    if daily.locked_out:
        _log(symbol, sig.strategy, "signal_discarded_daily_loss",
             f"descartada: corte diario -1.5R activo (día acumulado={daily.cumulative_r:+.2f}R) -- no se abren entradas nuevas")
        return

    account = client.account()
    lots = compute_position_size_lots(
        equity=float(account["equity"]), risk_pct=RISK_PCT, entry=sig.entry, sl=sig.sl,
        trade_tick_value=info["trade_tick_value"], trade_tick_size=info["trade_tick_size"],
        volume_min=info["volume_min"], volume_max=info["volume_max"], volume_step=info["volume_step"],
    )
    if lots <= 0:
        _log(symbol, sig.strategy, "signal_discarded_zero_size",
             f"descartada: tamaño calculado redondea a 0 lotes (equity={account['equity']:.2f}, risk_pct={RISK_PCT})")
        return

    # Migración a orden límite real (Prompt 2026-09-23): la revalidación contra
    # precio de mercado del prompt anterior (ver project_mtf_rr_prevalidation_
    # 2026-09-22) queda REDUNDANTE y se elimina -- ya no se manda una orden a
    # MERCADO al precio que sea, se manda una orden LÍMITE al precio TEÓRICO
    # (sig.entry = 50% FVG) con SL/TP2 adjuntos. Si el precio no vuelve ahí, la
    # orden simplemente expira sin llenarse (nunca se ejecuta a peor R:R); si
    # vuelve, se llena EXACTAMENTE al precio que ya aprobó el gate de arriba
    # (o mejor, nunca peor -- garantía de una orden límite real). Expiración =
    # fin de la sesión de entrada, no un offset fijo (ver
    # `_pending_expiration_epoch`) -- guardia defensiva por si ya venció para
    # cuando se intenta colocar la orden (no debería pasar para una señal
    # recién generada, pero nunca se manda al bróker una expiración ya pasada).
    expiration_epoch = _pending_expiration_epoch(client, sig.session, sig.generated_at)
    if expiration_epoch <= int(pd.Timestamp.now(tz="UTC").timestamp()):
        _log(symbol, sig.strategy, "signal_discarded_pending_window_elapsed",
             f"descartada: la sesión de entrada ({sig.session}) ya cerró para cuando se intentó "
             f"colocar la orden (generated_at={sig.generated_at}) -- la ventana ya venció")
        return

    order_type = "BUY_LIMIT" if sig.direction == "long" else "SELL_LIMIT"
    bid, ask = client.bid_ask(symbol)
    exec_price = ask if sig.direction == "long" else bid  # el lado contra el que MT5 valida la orden límite
    if not limit_entry_still_ahead(sig.direction, exec_price, sig.entry, sig.tp2):
        _log(symbol, sig.strategy, "signal_discarded_price_past_entry",
             f"descartada antes de enviar: {order_type} con entrada teórica={sig.entry:.5f} pero el mercado "
             f"({'ask' if sig.direction == 'long' else 'bid'}={exec_price:.5f}) ya pasó la entrada (o el tp2={sig.tp2:.5f}) "
             f"-- MT5 la rechazaría con INVALID_PRICE; no se entra a mercado (regla de entrada sin cambios)")
        return
    comment = f"mtf_{sig.strategy[:4]}_{(sig.setup or 'x')[:10]}"
    result = client.place_pending(
        symbol, order_type, volume=lots, price=round(sig.entry, info["digits"]),
        sl=round(sig.sl, info["digits"]), tp=round(sig.tp2, info["digits"]),
        expiration=expiration_epoch, comment=comment,
    )
    if not result.get("success"):
        _log(symbol, sig.strategy, "pending_order_rejected", f"rechazada por el bróker: {result.get('retcode_name')}")
        return

    ticket = str(result["order"])
    bias4h = "bullish" if sig.direction == "long" else "bearish"
    state["pending_orders"][ticket] = {
        "symbol": symbol, "strategy": sig.strategy, "setup": sig.setup, "direction": sig.direction,
        "entry_price": sig.entry, "sl": sig.sl, "tp2": sig.tp2,
        "volume": lots, "placed_at": now_iso(), "expiration_epoch": expiration_epoch,
        "bias_direction4h": bias4h, "net_rr": net_rr, "signal_key": list(signal_key(sig)),
    }
    risk_usd = abs(sig.entry - sig.sl) / info["trade_tick_size"] * info["trade_tick_value"] * lots
    _log(symbol, sig.strategy, "pending_order_placed",
         f"{sig.direction.upper()} {symbol} {sig.strategy}{'/' + sig.setup if sig.setup else ''} ticket={ticket} "
         f"sesión={sig.session} orden límite ({order_type}) lots={lots} price={sig.entry:.5f} sl={sig.sl:.5f} "
         f"tp1={sig.tp1:.5f} tp2={sig.tp2:.5f} riesgo≈{risk_usd:.2f} USD ({RISK_PCT:.2%} equity) "
         f"spread_actual={cost.avg_spread_points:.1f}pts "
         f"rr_neto_costo={net_rr:.2f} expira={(pd.Timestamp.fromtimestamp(expiration_epoch, tz='UTC') - client.server_utc_offset()).isoformat()} (UTC)",
         desktop_title=f"Orden límite colocada: {symbol} {sig.direction}", ticket=ticket)


if __name__ == "__main__":
    sys.exit(main())
