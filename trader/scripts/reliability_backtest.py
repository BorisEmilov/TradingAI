"""Prompt "backtest de fiabilidad y expectancy (con incertidumbre honesta)" (2026-10-01).

1) Expectancy / win rate / profit factor de las 35 operaciones del sistema actual
   (sin salida temporal, simulador alineado, corte -1.5R) con IC95% bootstrap
   percentil (10k remuestreos, seed 0 -- mismo método que validate_temporal_exit).
   Desglose por estrategia y símbolo con n explícito.
2) Línea base aleatoria: mismas 66 "señales" pero con hora y dirección al azar.
   Cada señal aleatoria toma la geometría de una señal real del MISMO símbolo
   (distancia de la entrada límite a la apertura, riesgo, tp2, costo de spread),
   así el SL/TP/llenado tienen la misma escala. Pasa por el mismo `dl._simulate`
   (conflicto, concurrencia, corte diario, parcial+BE, expiración fin de sesión).
   Hora: una vela M15 al azar del mismo símbolo dentro de la ventana y dentro de
   una sesión de entrada (Londres/NY).
REUTILIZACIÓN: la misma ventana se usó para decidir sacar la salida temporal
(comparación única) -- se reporta, no se oculta.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import measure_frequency_10_symbols as ten
import measure_frequency_10_symbols_daily_loss as dl
from trader.mtf_strategies.session_risk import active_entry_session

OUT_PATH = dl.OUT_PATH.with_name("reliability_backtest.json")
BOOTSTRAP_N = 10_000
RANDOM_TRIALS = int(os.environ.get("RANDOM_TRIALS", 500))


def _pf(r: np.ndarray) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("inf")


def _summary(r: np.ndarray, rng: np.random.Generator) -> dict:
    out = {"n": int(len(r)), "expectancy_r": round(float(r.mean()), 3), "win_rate": round(float((r > 0).mean()), 3),
           "profit_factor": round(_pf(r), 2), "total_r": round(float(r.sum()), 2)}
    if len(r) >= 2:
        idx = rng.integers(0, len(r), size=(BOOTSTRAP_N, len(r)))
        b = r[idx]
        pos, neg = np.where(b > 0, b, 0).sum(1), -np.where(b < 0, b, 0).sum(1)
        pf = np.divide(pos, neg, out=np.full(BOOTSTRAP_N, np.inf), where=neg > 0)
        ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
        out |= {"exp_ci95": ci(b.mean(1)), "wr_ci95": ci((b > 0).mean(1)), "pf_ci95": ci(pf),
                "P(exp>0)": round(float((b.mean(1) > 0).mean()), 3)}
    return out


def _random_signals(cands: pd.DataFrame, m15: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, donor in cands.iterrows():
        sym = donor["symbol"]
        bars = _SESSION_BARS[sym]
        t = bars[rng.integers(len(bars))]
        df = m15[sym]
        s_d = 1 if donor["direction"] == "long" else -1
        d_open = float(df.loc[df["timestamp"] >= donor["generated_at"], "open"].iloc[0])
        ahead, risk, reward = (d_open - donor["entry"]) * s_d, (donor["entry"] - donor["sl"]) * s_d, (donor["tp2"] - donor["entry"]) * s_d
        direction = "long" if rng.random() < 0.5 else "short"
        s = 1 if direction == "long" else -1
        open_t = float(df.loc[df["timestamp"] >= t, "open"].iloc[0])
        entry = open_t - s * ahead
        rows.append({**donor.to_dict(), "direction": direction, "generated_at": t, "sweep_timestamp": t,
                     "mss_timestamp": t, "fvg_confirmed_at": t, "session": active_entry_session(t),
                     "entry": entry, "sl": entry - s * risk, "tp1": entry + s * risk, "tp2": entry + s * reward})
    return pd.DataFrame(rows)


_SESSION_BARS: dict[str, np.ndarray] = {}


def main() -> None:
    rng = np.random.default_rng(0)
    prev = json.loads(ten.OUT_PATH.read_text())
    cand = pd.DataFrame([r for k in ("executed", "expired_unfilled", "discarded_conflict", "discarded_concurrency")
                         for r in prev[k]]).drop(columns=["filled", "symbol_freed_at"], errors="ignore")
    cand["generated_at"] = pd.to_datetime(cand["generated_at"], utc=True)
    end, n_days = pd.Timestamp(prev["common_end"]), prev["n_days"]
    m15 = {s: ten._load_clipped(s, "M15", end) for s in prev["symbols"]}

    # -- 1) sistema real ------------------------------------------------------
    res = dl._simulate(cand, m15, apply_daily_loss=True, temporal_exit=False)
    ex = pd.DataFrame(res["executed"]).dropna(subset=["net_r"])
    assert len(res["executed"]) == 35, len(res["executed"])
    real = _summary(ex["net_r"].to_numpy(), rng)
    by = {"estrategia": {k: _summary(g["net_r"].to_numpy(), rng) for k, g in ex.groupby("strategy")},
          "simbolo": {k: _summary(g["net_r"].to_numpy(), rng) for k, g in ex.groupby("symbol")}}
    print(f"=== 1) backtest sistema actual: {len(ex)} operaciones resueltas de {len(res['executed'])} ejecutadas, "
          f"{n_days} días, 10 símbolos ===")
    print(pd.Series(real).to_string())
    for name, d in by.items():
        print(f"\n-- por {name} (n chico: IC muy anchos) --")
        print(pd.DataFrame(d).T.reindex(columns=["n", "expectancy_r", "exp_ci95", "win_rate", "profit_factor", "total_r"]).to_string())

    # -- 2) línea base aleatoria ----------------------------------------------
    t0, t1 = cand["generated_at"].min(), cand["generated_at"].max()
    for s, df in m15.items():
        ts = df.loc[(df["timestamp"] >= t0) & (df["timestamp"] <= t1), "timestamp"]
        _SESSION_BARS[s] = ts[ts.map(lambda x: active_entry_session(x) is not None)].to_numpy()
    _SESSION_BARS.update({s: pd.to_datetime(v, utc=True) for s, v in _SESSION_BARS.items()})
    base = []
    for i in range(RANDOM_TRIALS):
        sigs = _random_signals(cand, m15, rng)
        r = pd.Series([x["net_r"] for x in dl._simulate(sigs, m15, True, temporal_exit=False)["executed"]]).dropna()
        base.append({"n": len(r), "exp": r.mean() if len(r) else np.nan, "total": r.sum()})
        if (i + 1) % 50 == 0:
            print(f"  aleatorio {i + 1}/{RANDOM_TRIALS}", flush=True)
    b = pd.DataFrame(base).dropna()
    p_exp = float((b["exp"] >= real["expectancy_r"]).mean())
    p_tot = float((b["total"] >= real["total_r"]).mean())
    baseline = {"trials": len(b), "n_ejecutadas_media": round(b["n"].mean(), 1),
                "exp_media": round(b["exp"].mean(), 3), "exp_p5_p95": [round(b["exp"].quantile(q), 3) for q in (.05, .95)],
                "total_media": round(b["total"].mean(), 2),
                "p_valor_expectancy (P azar >= real)": round(p_exp, 3), "p_valor_total_r": round(p_tot, 3)}
    print("\n=== 2) línea base aleatoria (misma gestión y geometría, hora+dirección al azar) ===")
    print(pd.Series(baseline).to_string())

    OUT_PATH.write_text(json.dumps({"real": real, "breakdown": by, "random_baseline": baseline,
                                    "random_trials": b.to_dict(orient="list"),
                                    "nota_reutilizacion": "misma ventana usada para sacar la salida temporal (comparación única)"},
                                   indent=2, default=str))
    print(f"\nGuardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
