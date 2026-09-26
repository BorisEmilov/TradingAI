"""Prompt "fix cancelación 4H por estrategia" (2026-09-25), punto 5:
verificación contra el gateway REAL de que una orden de reversión con 4H
ambiguo ("none") sobrevive un ciclo de gestión completo
(`run_mtf_pilot._manage_pending_orders`), donde antes se cancelaba.

- Orden sintética BUY_LIMIT 0.01 lotes ~3% por debajo del mercado (no se
  puede llenar), SL por debajo, TP2 por encima del precio actual (geometría
  válida -> la única causa posible de cancelación es el 4H), expira en 1h.
- Estado SEPARADO del piloto (dict en memoria), logs desviados a scratch.
- NO se llama logout(): el piloto en vivo comparte el mismo token
  (project_gateway_shared_session_token_2026-09-23).
- Al final se cancela la orden y se confirma CANCELED en el historial.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import run_mtf_pilot as pilot
from trader.config import load_config
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient
from trader.mtf_strategies.analysis import build_analysis
from trader.mtf_strategies.bias import classify_htf_structure, entry_4h_condition_holds

SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")


def main() -> None:
    pilot.JSON_EVENTS_PATH = SCRATCH / "verif_events.jsonl"
    pilot.HUMAN_EVENTS_PATH = SCRATCH / "verif_events.txt"
    config = load_config()
    client = PythonGetawayClient(config.gateway)
    client.login()  # reused: true con el piloto -- sin logout

    target = None
    for symbol in pilot.SYMBOLS:
        h1 = client.candles(symbol, "H1", 1000, timeout=60)
        m15 = client.candles(symbol, "M15", 2000, timeout=60)
        h4_raw = pilot._resample_h4(h1)
        ts = m15["timestamp"].iloc[-1] + TF_DURATION["M15"]
        h4 = build_analysis(h4_raw, "H4", ts, config)
        bias = classify_htf_structure(h4.swings, float(h4.df["close"].iloc[-1]))
        print(f"{symbol}: 4H={bias}")
        if bias == "none" and target is None:
            target = (symbol, h4_raw, m15)
    if target is None:
        raise SystemExit("ningún símbolo con 4H 'none' ahora -- no se puede verificar este caso")
    symbol, h4_raw, m15 = target

    info = client.symbol_info(symbol, timeout=30)
    price = client.last_price(symbol)
    d = info["digits"]
    entry, sl, tp2 = round(price * 0.97, d), round(price * 0.96, d), round(price * 1.03, d)
    expiration = int((pd.Timestamp.now(tz="UTC") + client.server_utc_offset() + pd.Timedelta(hours=1)).timestamp())
    res = client.place_pending(symbol, "BUY_LIMIT", volume=0.01, price=entry, sl=sl, tp=tp2,
                               expiration=expiration, comment="verif_4h_sintetica")
    assert res.get("success"), res
    ticket = str(res["order"])
    print(f"\ncolocada orden sintética {symbol} ticket={ticket} BUY_LIMIT 0.01 @ {entry} sl={sl} tp={tp2} (mercado {price})")

    try:
        pend = {"symbol": symbol, "strategy": "reversal", "setup": "setup_a_asia_long", "direction": "long",
                "entry_price": entry, "sl": sl, "tp2": tp2, "volume": 0.01, "placed_at": pilot.now_iso(),
                "expiration_epoch": expiration, "bias_direction4h": "bullish", "net_rr": 3.0, "signal_key": ["verif"]}
        state = {"open_positions": {}, "pending_orders": {ticket: pend}, "seen_signal_keys": [], "daily_loss": None}
        print(f"regla vieja (continuación aplicada a todo) habría cancelado: "
              f"{not entry_4h_condition_holds('continuation', None, 'bullish', 'none')}")

        pilot._manage_pending_orders(client, state, config, symbol, h4_raw, m15)
        pilot._reconcile_pending_orders(client, state)
        live = {str(o["ticket"]) for o in client.pending_orders()}
        survived = ticket in state["pending_orders"] and ticket in live
        print(f"tras ciclo de gestión real: en estado={ticket in state['pending_orders']} viva en MT5={ticket in live} "
              f"-> {'SOBREVIVIÓ (fix OK)' if survived else 'CANCELADA (FALLA)'}")
        events = pilot.HUMAN_EVENTS_PATH.read_text() if pilot.HUMAN_EVENTS_PATH.exists() else "(sin eventos)"
        print(f"eventos del ciclo: {events.strip()}")
    finally:
        c = client.cancel_pending(int(ticket))
        hist = client.order_history(int(ticket))
        states = [o.get("state_name") for o in hist.get("orders", [])]
        print(f"limpieza: cancel_pending success={c.get('success')} historial={states}")
    print("sin logout (sesión compartida con el piloto en vivo)")


if __name__ == "__main__":
    main()
