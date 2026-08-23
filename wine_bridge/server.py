"""Servidor puente: corre bajo Wine (Python de Windows) y expone MetaTrader5 por HTTP.

En Linux no se puede importar el paquete `MetaTrader5` directamente porque envuelve
la DLL del terminal MT5, que es una aplicacion Windows. Este script corre dentro de
un prefijo de Wine junto al terminal, y expone lo que necesita `tradingai.mt5` (velas,
info de cuenta, envio de ordenes) como una API HTTP local en 127.0.0.1.

Deliberadamente usa solo la libreria estandar (http.server) para minimizar lo que hay
que instalar dentro de Wine. Unica dependencia externa: el paquete `MetaTrader5` (que
arrastra numpy). Ver wine_bridge/requirements-wine.txt para las versiones exactas
(numpy esta fijado a 1.26.4 por un bug de Wine 9.0 con numpy>=2.0: falta la funcion
ucrtbase.dll.crealf y el proceso aborta al importar numpy).

Uso (desde dentro del prefijo de Wine, con el Python de Windows):
    wine python.exe wine_bridge/server.py --terminal-path "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import MetaTrader5 as mt5

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


# Bits del filling_mode que devuelve symbol_info() -- que modos de llenado acepta
# el simbolo/broker. Los nombres SYMBOL_FILLING_FOK/IOC no estan expuestos en este
# binding de Python (solo los ORDER_FILLING_* para el request), asi que se usan los
# valores de bit documentados en MQL5 directamente.
_SYMBOL_FILLING_FOK_BIT = 1
_SYMBOL_FILLING_IOC_BIT = 2


def _order_filling_type(symbol: str) -> int:
    """Elige un modo de llenado que el simbolo realmente soporte.

    Probar ORDER_FILLING_IOC a ciegas falla con retcode 10030 "Unsupported filling
    mode" en simbolos/brokers que solo aceptan FOK (visto en vivo con EURUSD en la
    cuenta demo MetaQuotes-Demo) -- no es un valor fijo por simbolo, hay que
    consultarlo.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    if info.filling_mode & _SYMBOL_FILLING_IOC_BIT:
        return mt5.ORDER_FILLING_IOC
    if info.filling_mode & _SYMBOL_FILLING_FOK_BIT:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _build_order_request(body: dict) -> dict:
    order_type = mt5.ORDER_TYPE_BUY if body["type"] == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": body["symbol"],
        "volume": body["volume"],
        "type": order_type,
        "price": body["price"],
        "deviation": body.get("deviation", 10),
        "magic": body.get("magic", 0),
        "comment": body.get("comment", ""),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _order_filling_type(body["symbol"]),
    }
    if body.get("sl") is not None:
        request["sl"] = body["sl"]
    if body.get("tp") is not None:
        request["tp"] = body["tp"]
    return request


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (nombre exigido por BaseHTTPRequestHandler)
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/health":
                self._send_json({"status": "ok", "mt5_initialized": mt5.terminal_info() is not None})

            elif parsed.path == "/account":
                info = mt5.account_info()
                self._send_json(info._asdict() if info else None)

            elif parsed.path == "/positions/count":
                positions = mt5.positions_get()
                self._send_json({"count": len(positions) if positions is not None else 0})

            elif parsed.path == "/positions":
                positions = mt5.positions_get()
                self._send_json(
                    {
                        "positions": [
                            {
                                "symbol": p.symbol,
                                # POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1
                                "type": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                                "volume": p.volume,
                                "profit": p.profit,
                            }
                            for p in (positions or [])
                        ]
                    }
                )

            elif parsed.path == "/symbol_info":
                info = mt5.symbol_info(qs["symbol"][0])
                self._send_json(info._asdict() if info else None)

            elif parsed.path == "/symbol_tick":
                tick = mt5.symbol_info_tick(qs["symbol"][0])
                self._send_json(tick._asdict() if tick else None)

            elif parsed.path == "/candles":
                symbol = qs["symbol"][0]
                timeframe = qs["timeframe"][0]
                n = int(qs.get("n", ["500"])[0])
                mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
                if mt5_timeframe is None:
                    self._send_json({"error": f"timeframe desconocido: {timeframe}"}, status=400)
                    return

                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, n)
                if rates is None:
                    self._send_json({"error": str(mt5.last_error())}, status=502)
                    return

                candles = [
                    {
                        "time": int(r["time"]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r["tick_volume"]),
                    }
                    for r in rates
                ]
                self._send_json({"candles": candles})

            else:
                self._send_json({"error": "not found"}, status=404)

        except Exception as exc:  # noqa: BLE001 - se reporta al cliente, no debe tumbar el servidor
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        try:
            if parsed.path == "/order":
                result = mt5.order_send(_build_order_request(body))
                self._send_json(result._asdict() if result else {"error": str(mt5.last_error())})
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - firma exigida por la clase base
        pass  # silencia el log de acceso por defecto; usar --verbose si hace falta depurar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--terminal-path", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    args = parser.parse_args()

    kwargs: dict = {"path": args.terminal_path}
    if args.login and args.password and args.server:
        kwargs.update(login=args.login, password=args.password, server=args.server)

    if not mt5.initialize(**kwargs):
        raise SystemExit(f"No se pudo inicializar MT5: {mt5.last_error()}")

    print(f"MT5 bridge escuchando en http://{args.host}:{args.port}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
