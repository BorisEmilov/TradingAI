"""Serie de evidencia LIMPIA: operaciones reales del piloto MTF, separada del backtest.

Fuente única: logs/mtf_pilot_events.jsonl (append-only, la escribe el piloto en cada
cierre). Este script solo la LEE y regenera logs/live_evidence_ledger.csv -- el CSV
nunca se edita a mano, así cualquier fila es trazable a su evento original.

Reglas para que siga siendo evidencia limpia:
- No se usa para ajustar parámetros/reglas. Si alguna operación termina influyendo
  en una decisión de diseño, se agrega su ticket a USED_IN_DESIGN (queda marcada y
  sale de la serie "limpia" desde ese momento).
- Cada cambio de reglas en vivo agrega una fila a SYSTEM_VERSIONS (fecha UTC de
  relanzamiento): las operaciones se comparan solo dentro de la misma versión.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOGS = Path(__file__).resolve().parent.parent / "logs"
SRC, OUT = LOGS / "mtf_pilot_events.jsonl", LOGS / "live_evidence_ledger.csv"

SYSTEM_VERSIONS = [  # (desde, versión) -- agregar una fila por cada cambio de reglas en vivo
    ("2026-09-21T00:00:00Z", "v1 orden a mercado + salida temporal 90m"),
    ("2026-09-23T00:00:00Z", "v2 orden límite 50% FVG + salida temporal 90m"),
    ("2026-09-25T19:54:00Z", "v3 orden límite, SIN salida temporal"),
    ("2026-09-29T06:49:00Z", "v4 v3 + chequeo de precio pre-envío"),
]
USED_IN_DESIGN = {  # ticket -> decisión que influyó
    "58570299249": "gap de R:R por slippage -> revalidación R:R y migración a orden límite (2026-09-22/23)",
}


def build() -> pd.DataFrame:
    ev = [json.loads(l) for l in SRC.read_text().splitlines() if '"position_closed"' in l]
    ev = [e for e in ev if e.get("kind") == "position_closed"]
    vers = pd.DataFrame(SYSTEM_VERSIONS, columns=["desde", "version"]).assign(desde=lambda d: pd.to_datetime(d["desde"]))
    rows = []
    for e in ev:
        ts = pd.Timestamp(e["ts"])
        money = e.get("money") or {}
        rows.append({
            "cerrada_utc": ts.isoformat(timespec="seconds"), "ticket": e["ticket"], "symbol": e["symbol"],
            "strategy": e["strategy"], "reason": e["reason"], "realized_r": round(e["realized_r"], 3),
            "pnl_usd": money.get("net"), "version_sistema": vers.loc[vers["desde"] <= ts, "version"].iloc[-1],
            "usada_en_diseno": USED_IN_DESIGN.get(e["ticket"], ""),
        })
    df = pd.DataFrame(rows)
    df["r_acumulado"] = df["realized_r"].cumsum().round(3)
    return df


def main() -> None:
    df = build()
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    clean = df[df["usada_en_diseno"] == ""]
    current = clean[clean["version_sistema"] == SYSTEM_VERSIONS[-1][1]]
    print(f"\ntodas: n={len(df)} suma={df['realized_r'].sum():+.2f}R | limpias (no usadas en diseño): n={len(clean)} "
          f"suma={clean['realized_r'].sum():+.2f}R | versión vigente ({SYSTEM_VERSIONS[-1][1]}): n={len(current)} "
          f"suma={current['realized_r'].sum():+.2f}R")
    print(f"guardado en {OUT}")


if __name__ == "__main__":
    main()
