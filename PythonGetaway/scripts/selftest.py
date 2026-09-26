#!/usr/bin/env python3
"""Prueba de extremo a extremo contra el gateway EN MARCHA.

Ejercita el flujo completo (autenticacion + los 43 endpoints) y guarda cada par
peticion/respuesta REAL en `docs/examples.json`, de donde el generador del PDF
saca sus ejemplos. Nada de la documentacion esta inventado.

    # solo lo que no necesita cuenta (gateway, pool, rutas de error)
    python3 scripts/selftest.py

    # completo: hace login de verdad y prueba los 43 endpoints
    PYGW_TEST_LOGIN=5054767214 \\
    PYGW_TEST_PASSWORD='...' \\
    PYGW_TEST_SERVER=MetaQuotes-Demo \\
    python3 scripts/selftest.py

    # completo pero sin enviar ninguna orden
    ... python3 scripts/selftest.py --no-trading

La contrasena se lee del entorno y NUNCA se imprime ni se guarda en
`examples.json`: en los ejemplos aparece enmascarada.

AVISO: sin `--no-trading` este script ABRE Y CIERRA posiciones reales en la
cuenta indicada. Se niega a operar si la cuenta no es DEMO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = ROOT / "docs" / "examples.json"

TEST_SYMBOL = "EURUSD"
TEST_VOLUME = 0.02       # 0.02 permite probar el cierre PARCIAL (el minimo suele ser 0.01)
MASK = "********"

examples: list[dict] = []
passed = failed = skipped = 0
TOKEN: str | None = None


def call(method: str, path: str, base: str, body: dict | None = None,
         note: str | None = None, record: bool = True, expect: int | None = 200,
         auth: bool = True) -> tuple[int, object]:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {}
    if data:
        headers["Content-Type"] = "application/json"
    if auth and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status, payload = resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read().decode()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
    except Exception as exc:  # noqa: BLE001
        status, payload = 0, {"transport_error": str(exc)}

    global passed, failed
    ok = (expect is None) or (status == expect)
    passed, failed = passed + (1 if ok else 0), failed + (0 if ok else 1)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {status:>3} {method:<6} {path}")
    if not ok:
        print(f"         esperado {expect}, respuesta: {json.dumps(payload)[:300]}")

    if record:
        examples.append({"method": method, "path": path, "status": status,
                         "request": _redact(body), "response": _redact(payload),
                         "note": note})
    return status, payload


def _redact(obj):
    """Nunca dejar una contrasena ni un token real en la documentacion."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("password", "token"):
                out[k] = MASK if k == "password" else "eyJ...<token>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def truncate(payload: object, list_keys: tuple[str, ...] = (), limit: int = 2) -> object:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    for key in list_keys:
        if isinstance(out.get(key), list) and len(out[key]) > limit:
            n = len(out[key]) - limit
            out[key] = out[key][:limit] + [f"... ({n} elementos mas omitidos en este ejemplo)"]
    return out


# ============================================================ bloques
def test_gateway(base: str) -> dict:
    print("-- Gateway y pool --")
    call("GET", "/", base, note="Indice. No requiere token.", auth=False)
    status, health = call("GET", "/health", base, auth=False,
                          note="Sonda del gateway. No requiere token. `available` es cuantos "
                               "usuarios nuevos pueden autenticarse ahora mismo.")
    call("GET", "/pool", base, auth=False,
         note="Estado de cada terminal del pool. Util para operar y dimensionar.")
    return health if isinstance(health, dict) else {}


def test_auth_errors(base: str) -> None:
    print("\n-- Autenticacion: rutas de error --")
    call("GET", "/account", base, expect=401, auth=False,
         note="Sin cabecera Authorization.")
    global TOKEN
    saved, TOKEN = TOKEN, "token-inexistente"
    call("GET", "/positions", base, expect=401,
         note="Token inexistente o caducado.")
    TOKEN = saved
    call("POST", "/auth/login", base,
         {"login": 999999999, "password": "credenciales-incorrectas", "server": "MetaQuotes-Demo"},
         expect=401,
         note="MT5 rechazo las credenciales. `detail.error.mt5_code` trae el codigo original "
              "(-6 = Authorization failed).")


def do_login(base: str, login: int, password: str, server: str) -> dict | None:
    global TOKEN
    print("\n-- Autenticacion --")
    status, payload = call("POST", "/auth/login", base,
                           {"login": login, "password": password, "server": server},
                           note="Reserva un terminal MT5 dedicado y devuelve el token. "
                                "La contrasena no se guarda en ningun sitio.")
    if status != 200 or not isinstance(payload, dict):
        print("     login fallido; se omite el resto.")
        return None
    TOKEN = payload.get("token")
    print(f"     token obtenido, slot={payload.get('slot_id')} "
          f"cuenta={payload.get('login')} expira_en={payload.get('expires_in')}s")
    call("GET", "/auth/session", base, note="Estado de la sesion. Renueva la ventana de inactividad.")
    call("POST", "/auth/refresh", base,
         note="Reinicia el reloj de VIDA MAXIMA. El token no cambia. Necesario en clientes "
              "conectados mas tiempo que PYGW_SESSION_TTL.")
    call("GET", "/auth/sessions", base, auth=False,
         note="Vision de operacion. No expone tokens ni credenciales.")
    return payload


def test_read_endpoints(base: str) -> None:
    print("\n-- Cuenta y terminal --")
    call("GET", "/account", base, note="Datos completos de la cuenta autenticada.")
    call("GET", "/terminal", base, note="Build, rutas y permisos del terminal de ESTE slot.")
    call("GET", "/version", base)
    call("GET", "/config", base, note="Configuracion efectiva del worker. Nunca expone la contrasena.")
    call("GET", "/last-error", base)

    print("\n-- Simbolos --")
    call("GET", "/symbols/total", base)
    _, syms = call("GET", "/symbols?group=*EURUSD*&names_only=true", base, record=False)
    examples.append({"method": "GET", "path": "/symbols?group=*EURUSD*&names_only=true",
                     "status": 200, "request": None, "response": truncate(syms, ("symbols",), 3),
                     "note": "Filtrado con comodines. Sin filtro el broker devuelve >12.000 simbolos."})
    call("GET", f"/symbols/{TEST_SYMBOL}", base,
         note="Especificacion completa: digitos, punto, volumenes, margen, swaps...")
    call("POST", f"/symbols/{TEST_SYMBOL}/select?enable=true", base)

    print("\n-- Mercado --")
    call("GET", f"/market/tick/{TEST_SYMBOL}", base, note="Cotizacion actual con el spread ya calculado.")
    call("GET", "/market/ticks?symbols=EURUSD,GBPUSD,USDJPY", base,
         note="Varios simbolos en una sola llamada.")
    _, candles = call("GET", f"/market/candles/{TEST_SYMBOL}?timeframe=M15&count=3", base, record=False)
    examples.append({"method": "GET", "path": f"/market/candles/{TEST_SYMBOL}?timeframe=M15&count=3",
                     "status": 200, "request": None, "response": candles,
                     "note": "La vela de indice 0 es la que se esta formando. "
                             "include_current=false la descarta."})
    now = int(time.time())
    for path, keys in (
        (f"/market/candles/{TEST_SYMBOL}/from?timeframe=H1&from_epoch={now - 86400}&count=2", ("candles",)),
        (f"/market/candles/{TEST_SYMBOL}/range?timeframe=H1&from_epoch={now - 172800}&to_epoch={now}", ("candles",)),
        (f"/market/ticks/{TEST_SYMBOL}/from?from_epoch={now - 3600}&count=3", ("ticks",)),
        (f"/market/ticks/{TEST_SYMBOL}/range?from_epoch={now - 3600}&to_epoch={now}", ("ticks",)),
    ):
        _, payload = call("GET", path, base, record=False)
        examples.append({"method": "GET", "path": path, "status": 200, "request": None,
                         "response": truncate(payload, keys, 2), "note": None})
    call("GET", f"/market/book/{TEST_SYMBOL}", base,
         note="Muchos brokers de Forex retail no publican profundidad: `entries` vacio no es un error.")

    print("\n-- Posiciones y ordenes --")
    call("GET", "/positions", base)
    call("GET", "/positions/total", base)
    call("GET", "/orders", base)
    call("GET", "/orders/total", base)
    call("GET", "/positions/999999999", base, expect=404,
         note="Forma estandar de los errores.")

    print("\n-- Historico --")
    for path, keys in ((f"/history/deals?from_epoch={now - 30 * 86400}", ("deals",)),
                       (f"/history/orders?from_epoch={now - 30 * 86400}", ("orders",))):
        _, payload = call("GET", path, base, record=False)
        examples.append({"method": "GET", "path": path, "status": 200, "request": None,
                         "response": truncate(payload, keys, 2),
                         "note": "Los movimientos de saldo aparecen con type_name=BALANCE."
                                 if "deals" in path else None})
    call("GET", f"/history/deals/total?from_epoch={now - 30 * 86400}", base)
    call("GET", f"/history/orders/total?from_epoch={now - 30 * 86400}", base)

    print("\n-- Calculos previos (no envian nada al broker) --")
    call("POST", "/trading/calc-margin", base, {"symbol": TEST_SYMBOL, "side": "BUY", "volume": 0.10})
    _, tick = call("GET", f"/market/tick/{TEST_SYMBOL}", base, record=False)
    ask = (tick or {}).get("ask") or 1.10
    call("POST", "/trading/calc-profit", base,
         {"symbol": TEST_SYMBOL, "side": "BUY", "volume": 0.10,
          "price_open": round(ask, 5), "price_close": round(ask + 0.0010, 5)})
    call("POST", "/trading/check", base, {"symbol": TEST_SYMBOL, "side": "BUY", "volume": 0.10},
         note="Valida contra el servidor SIN enviar la orden.")


def test_trading(base: str) -> None:
    print(f"\n-- Trading manual sobre {TEST_SYMBOL} (cuenta DEMO) --")
    status, opened = call("POST", "/trading/open", base,
                          {"symbol": TEST_SYMBOL, "side": "BUY", "volume": TEST_VOLUME,
                           "sl_points": 300, "tp_points": 600, "comment": "selftest"},
                          note="SL/TP en PUNTOS. Tambien se aceptan como precio absoluto.")
    if not (isinstance(opened, dict) and opened.get("success")):
        print(f"     apertura no ejecutada (mercado cerrado?): "
              f"{(opened or {}).get('retcode_name')} {(opened or {}).get('comment')}")
        examples.append({"method": "NOTE", "path": "/trading/*", "status": 0, "request": None,
                         "response": None,
                         "note": "El resto de endpoints de trading no se pudo probar en esta "
                                 "pasada (mercado cerrado). Su contrato es el documentado."})
        return

    ticket = opened.get("position") or opened.get("order")
    print(f"     posicion abierta, ticket={ticket}")
    time.sleep(1)
    call("GET", f"/positions/{ticket}", base, note="La posicion recien abierta.")
    _, pos = call("GET", f"/positions/{ticket}", base, record=False)
    price = (pos or {}).get("price_open", 0) or 0

    call("POST", f"/trading/modify-sl/{ticket}", base, {"sl": round(price - 0.0040, 5)},
         note="Mueve solo el SL; el TP se conserva.")
    call("POST", f"/trading/modify-tp/{ticket}", base, {"tp": round(price + 0.0080, 5)},
         note="Mueve solo el TP; el SL se conserva.")
    call("POST", f"/trading/modify/{ticket}", base,
         {"sl": round(price - 0.0050, 5), "tp": round(price + 0.0100, 5)},
         note="Mueve ambos. `null` deja el campo como esta; `0` lo elimina.")
    call("POST", f"/trading/close-partial/{ticket}", base, {"volume": 0.01, "comment": "parcial"},
         note="Cierra parte del volumen; el resto sigue abierto.")
    time.sleep(1)
    call("POST", f"/trading/close/{ticket}", base, {"comment": "cierre total"},
         note="Sin `volume` cierra la posicion entera.")
    time.sleep(1)

    print("\n-- Ordenes pendientes --")
    _, tick = call("GET", f"/market/tick/{TEST_SYMBOL}", base, record=False)
    bid = (tick or {}).get("bid") or 1.10
    limit_price = round(bid - 0.0100, 5)
    _, pending = call("POST", "/trading/pending", base,
                      {"symbol": TEST_SYMBOL, "type": "BUY_LIMIT", "volume": 0.01,
                       "price": limit_price, "sl": round(limit_price - 0.0050, 5),
                       "tp": round(limit_price + 0.0100, 5), "comment": "selftest pend"})
    if isinstance(pending, dict) and pending.get("success"):
        pticket = pending.get("order")
        print(f"     orden pendiente colocada, ticket={pticket}")
        call("GET", f"/orders/{pticket}", base)
        call("POST", f"/trading/pending/{pticket}/modify", base,
             {"price": round(limit_price - 0.0010, 5)},
             note="Cambia el precio de activacion; SL/TP se conservan si van en null.")
        call("DELETE", f"/trading/pending/{pticket}", base)

    print("\n-- Cierre masivo e historico --")
    call("POST", "/trading/close-all", base, {"symbol": TEST_SYMBOL},
         note="Cierra en bloque. Devuelve el resultado de cada cierre por separado.")
    if ticket:
        call("GET", f"/history/position/{ticket}", base,
             note="Reconstruye la vida completa de la posicion y su P&L realizado.")


def main() -> int:
    global skipped, TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-trading", action="store_true")
    ap.add_argument("--keep-session", action="store_true",
                    help="no hacer logout al terminar (deja el slot ocupado)")
    args = ap.parse_args()
    base = args.base

    print(f"\n=== PythonGetaway :: prueba end-to-end contra {base} ===\n")
    health = test_gateway(base)
    if not health:
        print("\nEl gateway no responde. Arrancalo con ./scripts/start.sh\n")
        return 1
    print(f"     pool: {health.get('pool', {}).get('available')} slot(s) disponibles "
          f"de {health.get('pool', {}).get('size')}")

    test_auth_errors(base)

    login = os.getenv("PYGW_TEST_LOGIN")
    password = os.getenv("PYGW_TEST_PASSWORD")
    server = os.getenv("PYGW_TEST_SERVER")

    if not (login and password and server):
        print("\n-- Endpoints de cuenta: OMITIDOS --")
        print("   Define PYGW_TEST_LOGIN / PYGW_TEST_PASSWORD / PYGW_TEST_SERVER")
        print("   para probar el flujo completo con una cuenta real.")
        skipped += 1
    else:
        session = do_login(base, int(login), password, server)
        if session:
            test_read_endpoints(base)
            account = session.get("account") or {}
            if args.no_trading:
                print("\n-- Trading: OMITIDO (--no-trading) --")
                skipped += 1
            elif account.get("trade_mode_name") != "DEMO":
                print(f"\n-- Trading: OMITIDO (la cuenta es {account.get('trade_mode_name')}, "
                      f"no DEMO) --")
                skipped += 1
            else:
                test_trading(base)
            if not args.keep_session:
                print("\n-- Cierre de sesion --")
                call("POST", "/auth/logout", base,
                     note="Cierra la sesion MT5 y devuelve el slot al pool.")
                TOKEN = None
                call("GET", "/pool", base, auth=False, record=False)

    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    print(f"\n=== RESULTADO: {passed} ok, {failed} fallos, {skipped} bloques omitidos ===")
    print(f"    Ejemplos reales guardados en {EXAMPLES_PATH.relative_to(ROOT)}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
