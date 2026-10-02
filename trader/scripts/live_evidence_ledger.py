"""Serie de evidencia LIMPIA: operaciones reales del piloto MTF, separada del backtest.

Fuente única: logs/mtf_pilot_events.jsonl (append-only, la escribe el piloto en cada
evento). Este script solo la LEE y regenera logs/live_evidence_ledger.csv y
logs/live_pilot_status.json -- nunca se editan a mano, así cualquier fila es
trazable a su evento original.

Reglas para que siga siendo evidencia limpia:
- No se usa para ajustar parámetros/reglas. Si alguna operación termina influyendo
  en una decisión de diseño, se agrega su ticket a USED_IN_DESIGN.
- Cada cambio de reglas en vivo agrega una fila a SYSTEM_VERSIONS; un cambio que
  NO toca reglas (ej. fixes de ejecución) se marca también, pero su serie sigue en
  CURRENT_RULES.

Criterio de abandono pre-registrado (logs/live_pilot_prereg_2026-10-02.md), sobre la
serie de reglas vigentes: con n >= 20, R medio < -0.3R, o IC 90% por bloques diarios
enteramente < 0 -> parar el piloto. Si no se dispara, reevaluar en n = 50.
`--enforce` (lo llama el watchdog en cada cierre) escribe la stop flag si se dispara.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

LOGS = Path(__file__).resolve().parent.parent / "logs"
SRC, OUT = LOGS / "mtf_pilot_events.jsonl", LOGS / "live_evidence_ledger.csv"
STATUS = LOGS / "live_pilot_status.json"

SYSTEM_VERSIONS = [  # (desde, versión) -- agregar una fila por cada cambio en vivo
    ("2026-09-21T00:00:00Z", "v1 orden a mercado + salida temporal 90m"),
    ("2026-09-23T00:00:00Z", "v2 orden límite 50% FVG + salida temporal 90m"),
    ("2026-09-25T19:54:00Z", "v3 orden límite, SIN salida temporal"),
    ("2026-09-29T06:49:00Z", "v4 v3 + chequeo de precio pre-envío"),
    ("2026-10-02T07:01:56Z", "v5 v4 + fixes de ejecución y systemd (mismas reglas)"),
]
CURRENT_RULES = {"v4 v3 + chequeo de precio pre-envío", "v5 v4 + fixes de ejecución y systemd (mismas reglas)"}
USED_IN_DESIGN = {  # ticket -> decisión que influyó
    "58570299249": "gap de R:R por slippage -> revalidación R:R y migración a orden límite (2026-09-22/23)",
}
MIN_N, MAX_MEAN_R, TARGET_N = 20, -0.3, 50
_FIELDS = {"session": r"sesión=(\w+)", "rr_neto": r"rr_neto_costo=([\d.]+)", "entry": r"price=([\d.]+)",
           "sl": r" sl=([\d.]+)", "tp2": r"tp2=([\d.]+)"}


def _events(kind: str) -> list[dict]:
    return [e for e in (json.loads(l) for l in SRC.read_text().splitlines() if f'"{kind}"' in l) if e.get("kind") == kind]


def build() -> pd.DataFrame:
    placed = {e["ticket"]: e for e in _events("pending_order_placed")}
    vers = pd.DataFrame(SYSTEM_VERSIONS, columns=["desde", "version"]).assign(desde=lambda d: pd.to_datetime(d["desde"]))
    rows = []
    for e in _events("position_closed"):
        ts = pd.Timestamp(e["ts"])
        money = e.get("money") or {}
        p = placed.get(e["ticket"], {})
        f = {k: (m.group(1) if (m := re.search(rx, p.get("detail", ""))) else None) for k, rx in _FIELDS.items()}
        rr_bruto = (abs(float(f["tp2"]) - float(f["entry"])) / abs(float(f["entry"]) - float(f["sl"]))
                    if f["entry"] and f["sl"] and f["tp2"] else None)
        rows.append({
            "cerrada_utc": ts.isoformat(timespec="seconds"), "colocada_utc": p.get("ts", "")[:19],
            "ticket": e["ticket"], "symbol": e["symbol"], "strategy": e["strategy"],
            "direction": p.get("detail", "").split(" ")[0].lower() or None, "session": f["session"],
            "rr_objetivo_bruto": round(rr_bruto, 2) if rr_bruto else None,
            "rr_objetivo_neto": float(f["rr_neto"]) if f["rr_neto"] else None,
            "reason": e["reason"], "realized_r": round(e["realized_r"], 3), "pnl_usd": money.get("net"),
            # casi-no-llenado: penetración del precio más allá de la entrada < spread pesimista
            # del símbolo. Necesita las velas M15 del momento del fill; el piloto no las guarda y no
            # se hace login en paralelo a él (sesión compartida) -> pendiente hasta tener velas.
            "llenado_marginal": "pendiente_velas",
            "version_sistema": vers.loc[vers["desde"] <= ts, "version"].iloc[-1],
            "usada_en_diseno": USED_IN_DESIGN.get(e["ticket"], ""),
        })
    df = pd.DataFrame(rows)
    df["r_acumulado"] = df["realized_r"].cumsum().round(3)
    return df


def series(df: pd.DataFrame) -> pd.DataFrame:
    """Serie del criterio: reglas vigentes, no usada en diseño, en orden de cierre."""
    s = df[df["version_sistema"].isin(CURRENT_RULES) & (df["usada_en_diseno"] == "")].copy()
    day_src = s["colocada_utc"].where(s["colocada_utc"] != "", s["cerrada_utc"])
    s["day"] = pd.to_datetime(day_src, utc=True).dt.normalize()
    return s.rename(columns={"realized_r": "net_r"}).reset_index(drop=True)


def evaluate(s: pd.DataFrame, rng: np.random.Generator | None = None) -> dict:
    from honest_reevaluation import block_boot, ci90  # mismo bootstrap por bloques diarios
    n = len(s)
    out = {"n": n, "objetivo_n": TARGET_N, "r_medio": round(float(s["net_r"].mean()), 3) if n else None,
           "total_r": round(float(s["net_r"].sum()), 2), "win_rate": round(float((s["net_r"] > 0).mean()), 3) if n else None}
    if n >= 2:
        b = block_boot(s, lambda d: d["net_r"].mean(), rng or np.random.default_rng(0))
        out |= {"ci90_bloques": ci90(b), "P(r>0)": round(float((b > 0).mean()), 3)}
    trig = []
    if n >= MIN_N and out["r_medio"] < MAX_MEAN_R:
        trig.append(f"R medio {out['r_medio']:+.3f} < {MAX_MEAN_R} con n={n}")
    if n >= MIN_N and out.get("ci90_bloques") and out["ci90_bloques"][1] < 0:
        trig.append(f"IC90 {out['ci90_bloques']} enteramente < 0 con n={n}")
    out["abandono"] = trig
    out["reevaluar_n50"] = n >= TARGET_N and not trig
    return out


def by_symbol(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby("symbol")["net_r"]
    return pd.DataFrame({"n": g.size(), "r_acumulado": g.sum().round(2), "r_medio": g.mean().round(3)}) \
        .sort_values("r_acumulado", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enforce", action="store_true", help="si se dispara el abandono, escribir la stop flag")
    args = ap.parse_args()
    df = build()
    df.to_csv(OUT, index=False)
    s = series(df)
    ev, sym = evaluate(s), by_symbol(s)
    STATUS.write_text(json.dumps({"actualizado_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                  "criterio": ev, "por_simbolo": sym.reset_index().to_dict(orient="records"),
                                  "operaciones": s.drop(columns=["day"]).to_dict(orient="records")}, indent=2, default=str))
    cols = ["cerrada_utc", "symbol", "direction", "session", "rr_objetivo_neto", "net_r", "reason", "llenado_marginal"]
    print(f"SERIE REGLAS VIGENTES ({len(s)}/{TARGET_N}): " + ", ".join(f"{k}={v}" for k, v in ev.items() if k != "abandono"))
    print(s[cols].to_string(index=False))
    print("\nR acumulado por símbolo:\n" + sym.to_string())
    if ev["abandono"]:
        print("\nALERTA ABANDONO: criterio pre-registrado DISPARADO -> " + " | ".join(ev["abandono"]))
        if args.enforce:
            import stop_mtf_pilot
            stop_mtf_pilot.main()
            print("ALERTA ABANDONO: stop flag escrita, el piloto se detiene en su próximo ciclo")
    elif ev["reevaluar_n50"]:
        print(f"\nAVISO: n={len(s)} >= {TARGET_N} -- toca la reevaluación pre-registrada")


if __name__ == "__main__":
    main()
