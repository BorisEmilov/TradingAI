"""prompt-analisis-edge-real.md, Parte 3: sobre el MISMO pool de candidatos
ICT de la Parte 1 (POI+confirmacion validos -- logs/edge_analysis_ict_candidates.jsonl,
generado por scripts/edge_analysis_mfe_mae.py), compara expectancy bajo
distintas variantes de gestion, manteniendo fija la seleccion de entrada:

  (a) actual (bruto): SL estructural + TP estructural (proximo nivel de
      liquidez) + parcial 50% + breakeven -- via el `simulate_trade` real de
      produccion, con un SymbolCost en CERO (comparacion bruta, sin costos,
      para que las 5 variantes sean comparables entre si en igualdad de
      condiciones).
  (a-neto) actual, CON costos reales (spread observado) -- unica variante
      con costos, incluida como referencia de "que entregaria produccion
      realmente", no para comparar contra las demas.
  (b) sin parcial/breakeven: mismo SL/TP estructural, corre hasta el final.
  (c/d/e) R:R fijo 1:1.5 / 1:2 / 1:3: mismo SL estructural, TP =
      entrada +- riesgo*multiplo (reemplaza el TP por nivel de liquidez),
      sin parcial/breakeven -- aisla si el problema es el NIVEL de TP.

Sin costos en (a)/(b)/(c)/(d)/(e): comparacion controlada de solo la
gestion, con el entry fijo -- costos reales solo se muestran en (a-neto)
como referencia aparte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from trader.backtest.costs import SymbolCost, estimate_symbol_cost
from trader.backtest.trade import simulate_trade
from trader.config import load_config
from trader.events import TF_DURATION
from trader.gateway_client import PythonGetawayClient
from trader.pipeline.confluence import ConfluenceCheck
from trader.pipeline.engine import TimeframeAnalysis
from trader.pipeline.experiment_ltf import LTF_15M, _next_liquidity_target, _structural_buffer, build_ltf_analysis
from trader.pipeline.scoring import ScoreBreakdown
from trader.risk.levels import TradeLevels
from trader.sessions import classify_session
from trader.signal import TradingSignal

D1_COUNT = 1500
M15_COUNT = 50000
MAX_HOLDING = pd.Timedelta(days=3)
_ZERO_COST = SymbolCost(symbol="", point=0.0001, avg_spread_points=0.0, avg_spread_price=0.0)
_ZERO_SCORE = ScoreBreakdown(bias=0.0, poi=0.0, confirmation=0.0, sweep=0.0, session=0.0, diversity=0.0)


def _load_ict_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            d["as_of"] = pd.Timestamp(d["as_of"])
            rows.append(d)
    return rows


def _simple_simulate(direction: str, entry: float, sl: float, tp: float, df: pd.DataFrame, entry_bar_idx: int) -> float | None:
    """No partial/BE, no cost -- SL-first-on-tie (same convention as
    simulate_trade), gross R only. Used for variants (b)-(e).
    """
    fill_idx = entry_bar_idx + 1
    if fill_idx >= len(df):
        return None
    fill_price = float(df["open"].iloc[fill_idx])
    risk = (fill_price - sl) if direction == "long" else (sl - fill_price)
    reward = (tp - fill_price) if direction == "long" else (fill_price - tp)
    if risk <= 0 or reward <= 0:
        return None
    entry_time = df["timestamp"].iloc[fill_idx]
    deadline = entry_time + MAX_HOLDING

    last_idx = fill_idx
    for k in range(fill_idx, len(df)):
        bar_open_ts = df["timestamp"].iloc[k]
        if bar_open_ts > deadline:
            break
        last_idx = k
        bar_high = float(df["high"].iloc[k])
        bar_low = float(df["low"].iloc[k])
        sl_hit = (bar_low <= sl) if direction == "long" else (bar_high >= sl)
        if sl_hit:
            return -1.0
        tp_hit = (bar_high >= tp) if direction == "long" else (bar_low <= tp)
        if tp_hit:
            return reward / risk

    exit_price = float(df["close"].iloc[last_idx])
    moved = (exit_price - fill_price) if direction == "long" else (fill_price - exit_price)
    return moved / risk


def main() -> None:
    config = load_config()
    logs_dir = Path(config.signals_log_path).parent
    ict_rows = _load_ict_candidates(logs_dir / "edge_analysis_ict_candidates.jsonl")
    print(f"Candidatos ICT cargados: {len(ict_rows)}")

    by_symbol: dict[str, list[dict]] = {}
    for r in ict_rows:
        by_symbol.setdefault(r["symbol"], []).append(r)

    client = PythonGetawayClient(config.gateway)
    client.login()

    results = {"a_bruto": [], "a_neto": [], "b_sin_parcial": [], "c_rr15": [], "d_rr20": [], "e_rr30": []}

    try:
        for symbol, rows in by_symbol.items():
            print(f"=== {symbol}: {len(rows)} candidatos ===")
            d1_df = client.candles(symbol, "D1", D1_COUNT, timeout=120)
            m15_df = client.candles(symbol, "M15", M15_COUNT, timeout=120)
            info = client.symbol_info(symbol, timeout=30)
            real_cost = estimate_symbol_cost(symbol, float(info["point"]), m15_df)
            d1_analysis = TimeframeAnalysis.from_candles(d1_df, "D1", config)
            ltf_analysis = build_ltf_analysis(m15_df, LTF_15M, config)
            m15_close_ts = m15_df["timestamp"] + TF_DURATION["M15"]
            idx_by_close_ts = {t: i for i, t in enumerate(m15_close_ts)}

            for r in rows:
                bar_idx = r["bar_idx"]
                direction = r["direction"]
                trade_direction = "long" if direction == "bullish" else "short"
                entry = r["entry"]
                as_of = r["as_of"]

                poi_candidates = [
                    z for z in (ltf_analysis.order_blocks + ltf_analysis.fvgs)
                    if z.direction == direction and z.confirmed_at is not None and z.confirmed_at <= as_of
                    and (z.broken_at is None or as_of < z.broken_at)
                    and z.overlaps(entry * (1 - config.poi.tolerance_pct / 100.0), entry * (1 + config.poi.tolerance_pct / 100.0))
                ]
                if not poi_candidates:
                    continue
                poi = max(poi_candidates, key=lambda z: z.timestamp)
                buffer = _structural_buffer(m15_df.iloc[: bar_idx + 1], LTF_15M.structural_buffer_lookback)
                sl = poi.price_low - buffer if trade_direction == "long" else poi.price_high + buffer
                risk = (entry - sl) if trade_direction == "long" else (sl - entry)
                if risk <= 0:
                    continue

                target = _next_liquidity_target(trade_direction, ltf_analysis, entry)

                session = classify_session(as_of, config.sessions)
                confirmation_stub = None

                # (a) actual management, structural TP -- needs a real TradingSignal for simulate_trade
                if target is not None:
                    tp_struct = target
                    reward_struct = (tp_struct - entry) if trade_direction == "long" else (entry - tp_struct)
                    if reward_struct > 0:
                        levels = TradeLevels(direction=trade_direction, entry=entry, sl=sl, tp=tp_struct, risk_reward=reward_struct / risk)
                        signal = TradingSignal(
                            symbol=symbol, direction=trade_direction, category="edge_analysis", bias_1d="up",
                            session=session, poi=poi, confirmation=poi, confluences=ConfluenceCheck(passed=True, families=set(), timeframes=set(), events=[]),
                            score=_ZERO_SCORE, levels=levels, partial_at_progress_pct=config.risk.partial_at_progress_pct, generated_at=as_of,
                        )
                        # d1_analysis (not ltf_analysis) is the HTF-invalidation reference --
                        # matches the D1->M15-unico architecture this pool was built from,
                        # where D1 replaces H1 as the only higher timeframe left.
                        trade_bruto = simulate_trade(signal, m15_df, bar_idx, d1_analysis, _ZERO_COST, "edge_analysis", timeframe="M15")
                        if trade_bruto is not None:
                            results["a_bruto"].append(trade_bruto.net_r)
                        trade_neto = simulate_trade(signal, m15_df, bar_idx, d1_analysis, real_cost, "edge_analysis", timeframe="M15")
                        if trade_neto is not None:
                            results["a_neto"].append(trade_neto.net_r)

                        r_simple = _simple_simulate(trade_direction, entry, sl, tp_struct, m15_df, bar_idx)
                        if r_simple is not None:
                            results["b_sin_parcial"].append(r_simple)

                # (c/d/e) fixed R:R, TP = entry +- risk*multiple
                for key, mult in (("c_rr15", 1.5), ("d_rr20", 2.0), ("e_rr30", 3.0)):
                    tp_fixed = entry + risk * mult if trade_direction == "long" else entry - risk * mult
                    r_fixed = _simple_simulate(trade_direction, entry, sl, tp_fixed, m15_df, bar_idx)
                    if r_fixed is not None:
                        results[key].append(r_fixed)
    finally:
        client.logout()

    print("\n=== Parte 3: expectancy por variante de gestion (mismo pool de entrada) ===\n")
    report_lines = ["# Parte 3 -- variantes de gestion sobre el mismo pool de candidatos ICT", ""]
    for key, label in [
        ("a_bruto", "(a) actual (SL+TP estructural, parcial+BE) -- BRUTO, sin costos"),
        ("a_neto", "(a-neto) actual, CON costos reales -- referencia de produccion, no comparar directo con el resto"),
        ("b_sin_parcial", "(b) sin parcial/BE, mismo SL/TP estructural, bruto"),
        ("c_rr15", "(c) R:R fijo 1:1.5, bruto"),
        ("d_rr20", "(d) R:R fijo 1:2.0, bruto"),
        ("e_rr30", "(e) R:R fijo 1:3.0, bruto"),
    ]:
        rs = results[key]
        if not rs:
            print(f"{label}: sin datos")
            report_lines.append(f"- {label}: sin datos")
            continue
        n = len(rs)
        win_rate = sum(1 for x in rs if x > 0) / n
        expectancy = sum(rs) / n
        wins = [x for x in rs if x > 0]
        losses = [x for x in rs if x <= 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        line = f"{label}: n={n} win_rate={win_rate:.1%} expectancy_r={expectancy:+.3f} pf={pf_str}"
        print(line)
        report_lines.append(f"- {line}")

    (Path(config.signals_log_path).parent / "edge_analysis_management_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nGuardado: {Path(config.signals_log_path).parent / 'edge_analysis_management_report.md'}")


if __name__ == "__main__":
    main()
