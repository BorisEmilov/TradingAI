"""Migración a orden límite real (Prompt 2026-09-23, ver
project_mtf_pending_limit_orders_2026-09-23 en memoria): en vez de mandar una
orden a MERCADO (con el riesgo de slippage ya documentado en
project_mtf_rr_slippage_gap_2026-09-22), `_attempt_execute` ahora coloca una
orden LÍMITE real (BUY_LIMIT/SELL_LIMIT) en el 50% FVG, con SL/TP2 adjuntos.

Expiración (Prompt 2026-09-23, "ventana de expiración = fin de sesión"):
YA NO es un offset fijo de 90 min -- vence al cierre de la sesión de entrada
(11:00 hora local, Londres o NY, DST-aware) en la que se generó la señal
(`session_risk.entry_session_end`). NO confundir con la salida temporal
post-entrada (`exits.MAX_MINUTES_M15`), que se SACÓ del piloto el 2026-09-25.

Estos tests cubren el ciclo de vida completo: colocación (con el nuevo
cálculo de expiración), cancelación proactiva (setup invalidado / corte
diario), bloqueo de concurrencia, y la reconciliación de llenado/expiración
contra el gateway."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_mtf_pilot as pilot
from trader.mtf_strategies.session_risk import entry_session_end
from trader.mtf_strategies.signal import MTFSignal


class FakeClient:
    """Stub mínimo de PythonGetawayClient -- solo lo que el ciclo de vida de
    órdenes pendientes llama. `_pending`/`_positions` simulan `/orders` y
    `/positions` del gateway real; un ticket que no aparece en ninguna de las
    dos (el estado por defecto) representa una orden ya expirada/cancelada;
    `simulate_fill` lo mueve a `_positions` para representar un llenado."""

    def __init__(self, last_price: float = 1.10000, place_pending_result: dict | None = None,
                 server_offset_hours: float = 0.0, spread_price: float = 0.0, order_state: str = "EXPIRED"):
        self._last_price = last_price
        self._order_state = order_state
        self._spread_price = spread_price
        self._place_pending_result = place_pending_result or {"success": True, "order": 999}
        self._offset = pd.Timedelta(hours=server_offset_hours)
        self._pending: list[dict] = []
        self._positions: list[dict] = []
        self.place_pending_calls: list[dict] = []
        self.cancel_pending_calls: list[int] = []

    def symbol_info(self, symbol, timeout=30):
        return {
            "point": 0.00001, "digits": 5,
            "trade_tick_value": 1.0, "trade_tick_size": 0.00001,
            "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
        }

    def account(self):
        return {"equity": 100_000.0}

    def last_price(self, symbol):
        return self._last_price

    def bid_ask(self, symbol):
        return self._last_price - self._spread_price / 2, self._last_price + self._spread_price / 2

    def recent_avg_spread(self, symbol):
        return self._spread_price

    def server_utc_offset(self):
        return self._offset

    def place_pending(self, symbol, order_type, volume, price, sl, tp, expiration, comment=""):
        call = {"symbol": symbol, "order_type": order_type, "volume": volume, "price": price,
                "sl": sl, "tp": tp, "expiration": expiration, "comment": comment}
        self.place_pending_calls.append(call)
        result = dict(self._place_pending_result)
        if result.get("success"):
            self._pending.append({"ticket": result["order"], "symbol": symbol})
        return result

    def cancel_pending(self, ticket):
        self.cancel_pending_calls.append(ticket)
        self._pending = [o for o in self._pending if o["ticket"] != ticket]
        return {"success": True}

    def pending_orders(self, symbol=None):
        if symbol:
            return [o for o in self._pending if o["symbol"] == symbol]
        return list(self._pending)

    def positions(self):
        return list(self._positions)

    def order_history(self, ticket):
        return {"orders": [{"ticket": ticket, "state_name": self._order_state}]}

    def position_history(self, ticket):
        return {"deals": [
            {"entry_name": "IN", "price": 1.10000, "profit": 0.0, "commission": -3.5, "swap": 0.0, "fee": 0.0},
            {"entry_name": "OUT", "price": 1.10500, "profit": 500.0, "commission": -3.5, "swap": -1.2,
             "fee": 0.0, "reason_name": "TP"},
        ]}

    def close_position(self, ticket):
        self.close_calls = getattr(self, "close_calls", []) + [ticket]
        return {"success": True, "price": self._last_price}

    def close_partial(self, ticket, volume):
        self.partial_calls = getattr(self, "partial_calls", []) + [(ticket, volume)]
        return {"success": True, "price": self._last_price}

    def modify_sl(self, ticket, sl):
        self.modify_sl_calls = getattr(self, "modify_sl_calls", []) + [(ticket, sl)]
        return {"success": True}

    def simulate_fill(self, ticket: int, price_open: float, symbol: str = "EURUSD",
                      fill_time_utc: pd.Timestamp | None = None) -> None:
        self._pending = [o for o in self._pending if o["ticket"] != ticket]
        pos = {"ticket": ticket, "symbol": symbol, "price_open": price_open}
        if fill_time_utc is not None:  # MT5 da `time` en hora de SERVIDOR
            pos["time"] = int((fill_time_utc + self._offset).timestamp())
        self._positions.append(pos)


def _safe_generated_at(session: str = "london", minutes_before_session_end: float = 30.0) -> pd.Timestamp:
    """Un `generated_at` que SIEMPRE cae dentro de la ventana de `session`,
    sin importar la hora real a la que corra el test: se ancla al cierre de
    la sesión de MAÑANA (mismo motor que producción, `entry_session_end`) y
    retrocede `minutes_before_session_end` -- deja margen amplio (horas)
    antes de que esa ventana venza contra el reloj real, sin depender de en
    qué momento del día se ejecuten los tests."""
    tomorrow = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    session_end = entry_session_end(session, tomorrow)
    return session_end - pd.Timedelta(minutes=minutes_before_session_end)


def _make_signal(entry: float = 1.10000, sl: float = 1.09500, tp2: float = 1.11250,
                  direction: str = "long", session: str = "london",
                  generated_at: pd.Timestamp | None = None) -> MTFSignal:
    """risk=0.005, reward=0.0125 -> risk_reward=2.5."""
    ts = generated_at if generated_at is not None else _safe_generated_at(session)
    risk = abs(entry - sl)
    return MTFSignal(
        strategy="continuation", setup=None, symbol="EURUSD", direction=direction, session=session,
        entry=entry, sl=sl, tp1=entry + risk, tp2=tp2, risk_reward=abs(tp2 - entry) / risk,
        sweep_timestamp=ts, mss_timestamp=ts, fvg_confirmed_at=ts, generated_at=ts,
    )


def _make_state() -> dict:
    return {"open_positions": {}, "pending_orders": {}, "seen_signal_keys": [], "daily_loss": None}


def _make_pending(symbol="EURUSD", direction="long", sl=1.09500, tp2=1.11250,
                   entry_price=1.10000, bias="bullish", net_rr=2.5, volume=0.1) -> dict:
    return {
        "symbol": symbol, "strategy": "continuation", "setup": None, "direction": direction,
        "entry_price": entry_price, "sl": sl, "tp2": tp2, "volume": volume,
        "placed_at": pilot.now_iso(), "expiration_epoch": 9999999999,
        "bias_direction4h": bias, "net_rr": net_rr, "signal_key": ["k"],
    }


def _no_spread_candles() -> pd.DataFrame:
    return pd.DataFrame({"close": [1.1, 1.1]})  # sin columna "spread" -> costo=0, no interfiere


def _patch_log_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pilot, "JSON_EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(pilot, "HUMAN_EVENTS_PATH", tmp_path / "events.txt")


# -- 1. colocación con nivel/expiración correctos ----------------------------

def test_places_limit_order_at_fvg_level_with_correct_expiration(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    sig = _make_signal()
    client = FakeClient(last_price=1.10200, place_pending_result={"success": True, "order": 555})
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert len(client.place_pending_calls) == 1
    call = client.place_pending_calls[0]
    assert call["order_type"] == "BUY_LIMIT"
    assert call["price"] == pytest.approx(sig.entry)
    assert call["sl"] == pytest.approx(sig.sl)
    assert call["tp"] == pytest.approx(sig.tp2)
    expected_expiration = int(entry_session_end(sig.session, sig.generated_at).timestamp())
    assert call["expiration"] == expected_expiration

    assert "555" in state["pending_orders"]
    pend = state["pending_orders"]["555"]
    assert pend["symbol"] == "EURUSD" and pend["direction"] == "long"
    assert pend["bias_direction4h"] == "bullish"
    assert "pending_order_placed" in (tmp_path / "events.txt").read_text()


def test_places_sell_limit_for_short_signals(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    sig = _make_signal(entry=1.10000, sl=1.10500, tp2=1.08750, direction="short")
    client = FakeClient(last_price=1.09800, place_pending_result={"success": True, "order": 556})
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert client.place_pending_calls[0]["order_type"] == "SELL_LIMIT"
    assert state["pending_orders"]["556"]["bias_direction4h"] == "bearish"


def test_expiration_window_is_short_near_session_close(monkeypatch, tmp_path):
    """Instrucción 2 del prompt: un setup detectado 5 min antes del cierre de
    sesión (ej. 10:55) recibe una ventana corta de 5 min -- correcto tal
    cual, sin mínimo artificial."""
    _patch_log_paths(monkeypatch, tmp_path)
    generated_at = _safe_generated_at("london", minutes_before_session_end=5.0)
    sig = _make_signal(generated_at=generated_at)
    client = FakeClient(last_price=1.10200, place_pending_result={"success": True, "order": 557})
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    expiration = client.place_pending_calls[0]["expiration"]
    expected = int(entry_session_end("london", generated_at).timestamp())
    assert expiration == expected
    assert 0 < expiration - int(generated_at.timestamp()) <= 5 * 60 + 1  # ventana corta, ~5 min


def test_signal_discarded_when_session_already_closed(monkeypatch, tmp_path):
    """Guardia defensiva: si la sesión de entrada ya cerró para cuando se
    intenta colocar la orden (generated_at muy en el pasado), se descarta en
    vez de mandarle al bróker una expiración ya vencida."""
    _patch_log_paths(monkeypatch, tmp_path)
    stale_generated_at = entry_session_end("london", pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)) - pd.Timedelta(minutes=30)
    sig = _make_signal(generated_at=stale_generated_at)
    client = FakeClient()
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert client.place_pending_calls == []
    assert state["pending_orders"] == {}
    assert "signal_discarded_pending_window_elapsed" in (tmp_path / "events.txt").read_text()


# -- 2. cancelación proactiva por invalidación de setup ----------------------

def test_proactive_cancel_when_price_crosses_sl_before_fill(monkeypatch, tmp_path):
    """El precio se movió más allá del SL sin haber llegado a tocar el nivel
    de entrada -- la orden límite todavía no comprometió capital, cancelarla
    es seguro."""
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["777"] = _make_pending(sl=1.09500, tp2=1.11250)
    client = FakeClient(last_price=1.09000)  # por debajo del sl (long)

    pilot._manage_pending_orders(client, state, None, "EURUSD", pd.DataFrame(), _no_spread_candles())

    assert state["pending_orders"] == {}
    assert client.cancel_pending_calls == [777]
    assert "pending_order_cancelled_proactively" in (tmp_path / "events.txt").read_text()


# -- 3. cancelación por corte diario -----------------------------------------

def test_proactive_cancel_when_daily_loss_locked_out(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["888"] = _make_pending()
    state["daily_loss"] = {"date": str(pd.Timestamp.now(tz="UTC").normalize()), "cumulative_r": -2.0, "locked_out": True}
    client = FakeClient(last_price=1.10000)  # precio válido -- igual se cancela por el corte diario

    pilot._manage_pending_orders(client, state, None, "EURUSD", pd.DataFrame(), _no_spread_candles())

    assert state["pending_orders"] == {}
    assert client.cancel_pending_calls == [888]
    assert "corte_diario" in (tmp_path / "events.txt").read_text()


# -- 4. bloqueo de concurrencia mientras está pendiente ----------------------

def test_concurrency_blocks_new_signal_while_pending_order_open(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["111"] = _make_pending()  # ya hay una orden límite viva en EURUSD
    sig = _make_signal()  # nueva señal, mismo símbolo
    client = FakeClient()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert client.place_pending_calls == []
    assert set(state["pending_orders"].keys()) == {"111"}  # no se agregó una segunda
    assert "signal_discarded_concurrency" in (tmp_path / "events.txt").read_text()


# -- 5. transición correcta a posición abierta al llenarse -------------------

def test_reconcile_promotes_filled_pending_order_to_open_position(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["222"] = _make_pending(sl=1.09500, tp2=1.11250, entry_price=1.10000)
    client = FakeClient()
    client.simulate_fill(222, price_open=1.10000, symbol="EURUSD")  # se llenó exactamente al límite

    pilot._reconcile_pending_orders(client, state)

    assert state["pending_orders"] == {}
    assert "222" in state["open_positions"]
    pos = state["open_positions"]["222"]
    assert pos["entry_price"] == pytest.approx(1.10000)
    assert pos["initial_sl"] == pytest.approx(1.09500)  # SL intacto, el mismo que tenía la orden
    assert pos["tp2"] == pytest.approx(1.11250)          # TP2 intacto
    assert pos["tp1"] == pytest.approx(1.10500)           # 1R desde el fill real
    assert pos["partial_taken"] is False
    assert "pending_order_filled" in (tmp_path / "events.txt").read_text()


# -- 6. expiración limpia sin fill -------------------------------------------

def test_reconcile_finalizes_pending_order_that_expired_without_fill(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["333"] = _make_pending()
    client = FakeClient()  # ni en /orders ni en /positions -- expiró (o se canceló) sin llenar

    pilot._reconcile_pending_orders(client, state)

    assert state["pending_orders"] == {}
    assert state["open_positions"] == {}  # nunca se llenó, no se abrió posición
    log = (tmp_path / "events.txt").read_text()
    assert "pending_order_expired_or_cancelled" in log
    assert "EXPIRED" in log


def test_cost_gate_uses_live_tick_spread_not_candle_field(monkeypatch, tmp_path):
    # R:R teórico 2.5, riesgo 0.005. Velas sin spread (como el campo real desde
    # jul-2026) -- si el gate leyera las velas, costo 0 y colocaría. Con spread
    # real de ticks 0.003 -> costo 0.6R -> neto 1.9 < 2.0 -> descartada.
    _patch_log_paths(monkeypatch, tmp_path)
    sig = _make_signal()
    client = FakeClient(spread_price=0.003)
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert client.place_pending_calls == []
    assert "signal_discarded_cost" in (tmp_path / "events.txt").read_text()


# -- monitor: dinero al cerrar, modificaciones externas, operaciones ajenas ---

def _open_position_state(ticket="777"):
    state = _make_state()
    state["open_positions"][ticket] = {
        "symbol": "EURUSD", "strategy": "continuation", "setup": None, "direction": "long",
        "entry_price": 1.10000, "initial_sl": 1.09500, "current_sl": 1.09500, "tp1": 1.10500, "tp2": 1.11250,
        "volume_total": 1.0, "volume_remaining": 1.0, "opened_at": pilot.now_iso(), "entry_time": pilot.now_iso(),
        "partial_taken": False, "last_bar_processed": None, "net_rr": 2.5, "signal_key": ["k"],
    }
    return state


def test_close_reports_real_money_from_mt5_deals(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _open_position_state()
    pos = state["open_positions"]["777"]

    pilot._finalize_close(FakeClient(), state, "777", pos, 1.0, "cerrada_por_broker:TP")

    event = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()][-1]
    assert event["kind"] == "position_closed"
    assert event["money"]["net"] == pytest.approx(500.0 - 3.5 - 3.5 - 1.2)
    assert "+491.80 USD" in event["detail"] and "777" not in state["open_positions"]


def test_external_sl_change_is_reported_but_bot_changes_are_not(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _open_position_state()
    client = FakeClient()
    client._positions = [{"ticket": 777, "symbol": "EURUSD", "sl": 1.09500, "tp": 1.11250, "volume": 1.0, "profit": 12.0}]

    pilot._detect_broker_changes(client, state)  # línea base, sin evento
    state["open_positions"]["777"]["broker_resync"] = True  # el bot movió SL (parcial) -> sin evento
    client._positions[0].update(sl=1.10000, volume=0.5)
    pilot._detect_broker_changes(client, state)
    assert not (tmp_path / "events.txt").exists()

    client._positions[0]["tp"] = 1.12000  # cambio manual en MT5
    pilot._detect_broker_changes(client, state)
    text = (tmp_path / "events.txt").read_text()
    assert "position_modified_external" in text and "tp: 1.1125 -> 1.12" in text


def test_foreign_position_reported_once(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    client = FakeClient()
    client._positions = [{"ticket": 42, "symbol": "XPTUSD", "volume": 0.1, "price_open": 1000.0, "sl": 0, "tp": 0}]

    pilot._detect_broker_changes(client, state)
    pilot._detect_broker_changes(client, state)

    assert (tmp_path / "events.txt").read_text().count("external_trade_detected") == 1


# -- regresión bug 2026-09-25: cancelación 4H por estrategia -----------------
# Ciclo de gestión REAL (`_manage_pending_orders`), solo se fija el valor que
# devuelve el clasificador 4H. Precio 1.10200: dentro de la geometría (entre
# SL 1.09500 y TP2 1.11250), así que la ÚNICA razón posible de cancelación
# es el sesgo 4H.

def _run_manage_with_4h(monkeypatch, tmp_path, strategy, setup, structure_4h):
    _patch_log_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(pilot, "build_analysis", lambda *a, **k: SimpleNamespace(swings=[], df=pd.DataFrame({"close": [1.102]})))
    monkeypatch.setattr(pilot, "classify_htf_structure", lambda swings, close: structure_4h)
    client = FakeClient(last_price=1.10200)
    client._pending = [{"ticket": 900, "symbol": "EURUSD"}]
    state = _make_state()
    state["pending_orders"]["900"] = {**_make_pending(bias="bullish"), "strategy": strategy, "setup": setup}
    m15 = pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-25 07:00", tz="UTC")], "close": [1.102]})
    pilot._manage_pending_orders(client, state, None, "EURUSD", pd.DataFrame(), m15)
    return client, state


def test_reversal_a_survives_ambiguous_4h(monkeypatch, tmp_path):
    client, state = _run_manage_with_4h(monkeypatch, tmp_path, "reversal", "setup_a_asia_long", "none")
    assert client.cancel_pending_calls == [] and "900" in state["pending_orders"]


def test_reversal_a_cancelled_when_4h_turns_opposite(monkeypatch, tmp_path):
    client, state = _run_manage_with_4h(monkeypatch, tmp_path, "reversal", "setup_a_asia_long", "bearish")
    assert client.cancel_pending_calls == [900] and "900" not in state["pending_orders"]


@pytest.mark.parametrize("structure_4h", ["none", "bullish", "bearish"])
def test_reversal_b_never_cancelled_by_4h(monkeypatch, tmp_path, structure_4h):
    client, state = _run_manage_with_4h(monkeypatch, tmp_path, "reversal", "setup_b_pdl_long", structure_4h)
    assert client.cancel_pending_calls == [] and "900" in state["pending_orders"]


@pytest.mark.parametrize("structure_4h,cancelled", [("bullish", False), ("none", True), ("bearish", True)])
def test_continuation_4h_cancellation_unchanged(monkeypatch, tmp_path, structure_4h, cancelled):
    client, _ = _run_manage_with_4h(monkeypatch, tmp_path, "continuation", None, structure_4h)
    assert (client.cancel_pending_calls == [900]) is cancelled


# -- salida temporal SACADA (2026-09-25): la posición corre hasta SL o TP1+BE+TP2 --

def _m15_bar(high, low, bars_after_entry=12):
    ts = pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta(minutes=15 * bars_after_entry)
    return pd.DataFrame({"timestamp": [ts], "open": [1.1], "high": [high], "low": [low], "close": [1.1]})


def test_no_forced_close_after_90min_without_progress(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state, client = _open_position_state(), FakeClient()
    pilot._manage_open_positions(client, state, "EURUSD", _m15_bar(high=1.10100, low=1.09900))  # 3h, +0.2R
    assert not getattr(client, "close_calls", []) and "777" in state["open_positions"]
    assert state["open_positions"]["777"]["partial_taken"] is False


def test_tp1_partial_and_breakeven_still_applied(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state, client = _open_position_state(), FakeClient()
    pilot._manage_open_positions(client, state, "EURUSD", _m15_bar(high=1.10600, low=1.10000))  # toca TP1
    pos = state["open_positions"]["777"]
    assert client.partial_calls == [(777, 0.5)] and client.modify_sl_calls == [(777, 1.10000)]
    assert pos["partial_taken"] is True and pos["current_sl"] == pytest.approx(1.10000)


# -- chequeo de precio pre-envío (2026-09-29, GBPJPY INVALID_PRICE) ------------

def test_signal_discarded_when_market_already_past_limit_entry(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    # SELL_LIMIT en 1.10000 pero el bid ya está en 1.10050 (por encima de la entrada)
    sig = _make_signal(entry=1.10000, sl=1.10500, tp2=1.08750, direction="short")
    client = FakeClient(last_price=1.10050)
    state = _make_state()

    pilot._attempt_execute(client, state, None, sig, "EURUSD", _no_spread_candles())

    assert client.place_pending_calls == [] and state["pending_orders"] == {}
    text = (tmp_path / "events.txt").read_text()
    assert "signal_discarded_price_past_entry" in text and "1.10000" in text and "bid=1.10050" in text


def test_limit_entry_still_ahead_reuses_geometry():
    from trader.risk.levels import limit_entry_still_ahead
    assert limit_entry_still_ahead("short", 1.0990, 1.1000, 1.0900)       # debajo de la entrada: colocable
    assert not limit_entry_still_ahead("short", 1.1005, 1.1000, 1.0900)   # ya la cruzó
    assert not limit_entry_still_ahead("short", 1.0890, 1.1000, 1.0900)   # ya pasó TP2
    assert limit_entry_still_ahead("long", 1.1010, 1.1000, 1.1100)
    assert not limit_entry_still_ahead("long", 1.0995, 1.1000, 1.1100)


# -- bugs de ejecución (revisión 2026-10-01) ----------------------------------

def _bar_at(ts: str, high: float, low: float) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [pd.Timestamp(ts, tz="UTC")], "open": [1.1], "high": [high], "low": [low], "close": [1.1]})


def test_bug1_tp1_ignores_m15_bar_that_started_before_the_fill(monkeypatch, tmp_path):
    # BUY_LIMIT 1.10000 llenado a las 13:07:59 UTC; la vela 13:00-13:15 tuvo máximo 1.10600
    # (> TP1 1.10500) ANTES de bajar a llenar -- ese máximo no es progreso de la posición.
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["222"] = _make_pending(volume=1.0)
    client = FakeClient(server_offset_hours=3)
    client.simulate_fill(222, price_open=1.10000, fill_time_utc=pd.Timestamp("2026-10-01 13:07:59", tz="UTC"))
    pilot._reconcile_pending_orders(client, state)

    pilot._manage_open_positions(client, state, "EURUSD", _bar_at("2026-10-01 13:00", high=1.10600, low=1.09990))
    assert getattr(client, "partial_calls", []) == []
    assert state["open_positions"]["222"]["partial_taken"] is False

    pilot._manage_open_positions(client, state, "EURUSD", _bar_at("2026-10-01 13:15", high=1.10600, low=1.10100))
    assert client.partial_calls == [(222, 0.5)]  # la primera vela posterior al fill sí cuenta


@pytest.mark.parametrize("failure", ["exception", "rejected"])
def test_bug2_partial_never_repeated_and_breakeven_retried(monkeypatch, tmp_path, failure):
    import requests
    _patch_log_paths(monkeypatch, tmp_path)
    state, client = _open_position_state(), FakeClient()
    outcomes = iter([failure, "ok"])

    def modify_sl(ticket, sl):
        client.modify_sl_calls = getattr(client, "modify_sl_calls", []) + [(ticket, sl)]
        out = next(outcomes)
        if out == "exception":
            raise requests.HTTPError("502 Bad Gateway")
        return {"success": out == "ok", "retcode_name": "INVALID_STOPS" if out != "ok" else "DONE"}
    client.modify_sl = modify_sl

    try:  # en vivo la excepción la atrapa el try/except de _tick
        pilot._manage_open_positions(client, state, "EURUSD", _m15_bar(high=1.10600, low=1.10000, bars_after_entry=12))
    except Exception:  # noqa: BLE001
        pass
    pos = state["open_positions"]["777"]
    assert pos["partial_taken"] is True and client.partial_calls == [(777, 0.5)]
    assert pos["current_sl"] == pytest.approx(1.09500)  # el SL real sigue en el original

    pilot._manage_open_positions(client, state, "EURUSD", _m15_bar(high=1.10600, low=1.10000, bars_after_entry=13))
    assert client.partial_calls == [(777, 0.5)]  # NUNCA un segundo parcial
    assert client.modify_sl_calls == [(777, 1.10000), (777, 1.10000)]  # breakeven reintentado
    assert pos["current_sl"] == pytest.approx(1.10000) and not pos.get("breakeven_pending")
    assert "breakeven_sl_failed" in (tmp_path / "events.txt").read_text()


def test_bug3_fill_and_close_between_polls_counts_for_daily_loss(monkeypatch, tmp_path):
    _patch_log_paths(monkeypatch, tmp_path)
    state = _make_state()
    state["pending_orders"]["444"] = _make_pending(volume=1.0)  # entrada 1.10000, SL 1.09500
    client = FakeClient(order_state="FILLED")  # ni en /orders ni en /positions, pero el bróker la LLENÓ
    client.position_history = lambda ticket: {"deals": [
        {"entry_name": "IN", "price": 1.10000, "volume": 1.0, "time": 1790000000, "profit": 0.0},
        {"entry_name": "OUT", "price": 1.09500, "volume": 1.0, "profit": -500.0, "reason_name": "SL"},
    ]}

    pilot._reconcile_pending_orders(client, state)

    assert state["pending_orders"] == {} and state["open_positions"] == {}
    assert state["daily_loss"]["cumulative_r"] == pytest.approx(-1.0)
    log = (tmp_path / "events.txt").read_text()
    assert "position_closed" in log and "nunca se llenó" not in log


def _run_main(monkeypatch, tmp_path, client, tick=None, save_state=None):
    _patch_log_paths(monkeypatch, tmp_path)
    for name in ("STATE_PATH", "STOP_FLAG_PATH", "LOCK_PATH"):
        monkeypatch.setattr(pilot, name, tmp_path / name.lower(), raising=False)
    monkeypatch.setattr(pilot, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(pilot, "load_config", lambda: SimpleNamespace(gateway=None))
    monkeypatch.setattr(pilot, "PythonGetawayClient", lambda cfg: client)
    monkeypatch.setattr(pilot, "REFRESH_SECONDS", -1)  # refresca en cada ciclo
    monkeypatch.setattr(pilot.time, "sleep", lambda s: None)
    if hasattr(pilot, "signal"):
        monkeypatch.setattr(pilot.signal, "signal", lambda *a: None)  # no tocar el handler real de pytest
    ticks = []

    def default_tick(c, state, cfg):
        ticks.append(1)
        if len(ticks) == 2:
            pilot.STOP_FLAG_PATH.write_text("stop")
    monkeypatch.setattr(pilot, "_tick", tick or default_tick)
    if save_state:
        monkeypatch.setattr(pilot, "_save_state", save_state)
    return pilot.main(), ticks, (tmp_path / "events.txt").read_text()


class _SessionClient(FakeClient):
    def __init__(self, refresh_error=None):
        super().__init__()
        self.refresh_error, self.logins = refresh_error, 0

    def login(self):
        self.logins += 1

    def refresh(self):
        if self.refresh_error:
            raise self.refresh_error

    def logout(self):
        pass


def test_bug4_refresh_failure_does_not_kill_pilot(monkeypatch, tmp_path):
    import requests
    code, ticks, log = _run_main(monkeypatch, tmp_path, _SessionClient(requests.ConnectionError("gateway caído")))
    assert code == pilot.EXIT_STOPPED and len(ticks) == 2  # siguió operando hasta la parada pedida
    assert "refresh_failed" in log and "pilot_stopping" in log and "pilot_crashed" not in log


def test_bug4_crash_is_distinguished_from_requested_stop(monkeypatch, tmp_path):
    def broken_save(state):
        raise OSError("disco lleno")
    code, _, log = _run_main(monkeypatch, tmp_path, _SessionClient(), save_state=broken_save)
    assert code == pilot.EXIT_CRASHED and code != pilot.EXIT_STOPPED
    assert "pilot_crashed" in log and "pilot_stopping" not in log


def test_single_instance_lock_blocks_second_pilot(monkeypatch, tmp_path):
    import fcntl
    client = _SessionClient()
    lock_path = tmp_path / "lock_path"
    with open(lock_path, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # otro piloto vivo
        monkeypatch.setattr(pilot, "LOCK_PATH", lock_path)
        monkeypatch.setattr(pilot, "LOGS_DIR", tmp_path)
        assert pilot.main() == pilot.EXIT_ALREADY_RUNNING
    assert client.logins == 0  # ni siquiera tocó el gateway


def test_state_write_is_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "STATE_PATH", tmp_path / "state.json")
    pilot._save_state({"open_positions": {"1": {}}, "pending_orders": {}})
    real_dump = json.dump

    def dump_then_die(obj, f, **kw):
        f.write('{"open_positions": {"2"')  # proceso muerto a mitad de escritura
        raise KeyboardInterrupt
    monkeypatch.setattr(json, "dump", dump_then_die)
    with pytest.raises(KeyboardInterrupt):
        pilot._save_state({"open_positions": {"2": {}}, "pending_orders": {}})
    monkeypatch.setattr(json, "dump", real_dump)
    assert pilot._load_state()["open_positions"] == {"1": {}}  # el archivo anterior sigue intacto
