"""Estrategia 1 -- Continuación MTF SMC (4H/1H/15M). NUEVO -- orquesta piezas
existentes (swings/BOS-CHoCH/sweeps/FVG/OB vía `analysis.py`) y nuevas
(`bias.py`, `liquidity_levels.py`, `post_sweep_structure.py`, `targets.py`,
`gate.py`) siguiendo el algoritmo EXACTO de la regla 13 -- cada paso se
evalúa en orden, retorna en el primer NO, sin bypass posible (mismo principio
estructural que `pipeline/engine.py::_STAGE_ORDER`, reimplementado acá
porque este paquete no importa nada de `pipeline/`).

Decisión de diseño confirmada: 4H aquí es la ÚNICA fuente de sesgo (no hay
D1 en esta estrategia) -- distinto del gate D1/H4 descartado el 2026-09-16,
que usaba H4 para CONFIRMAR un sesgo ya establecido en D1. Ese hallazgo no
aplica directamente acá por construcción, no por descuido.

"Cualquier nivel de liquidez 1H que se ataque activa su propio setup,
evaluado de forma independiente" (decisión de diseño confirmada) --
`evaluate_continuation` devuelve una LISTA de señales candidatas (una por
cada nivel efectivamente barrido y confirmado hasta el final), no una sola.
Si más de una sobrevive el algoritmo completo al mismo tiempo, no hay una
regla explícita en la spec para desempatar -- se ordenan por mejor R:R como
default razonable, documentado acá, no en la spec original; el caller puede
ignorar el orden si prefiere otro criterio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.config import TraderConfig
from trader.events import TF_DURATION
from trader.mtf_strategies.analysis import SimpleAnalysis
from trader.mtf_strategies.bias import classify_htf_structure
from trader.mtf_strategies.exits import fvg_setup_expired
from trader.mtf_strategies.gate import evaluate_all_or_nothing
from trader.mtf_strategies.liquidity_levels import compute_asia_session_high_low, compute_liquidity_levels, detect_named_level_sweep
from trader.mtf_strategies.post_sweep_structure import first_post_sweep_mss
from trader.mtf_strategies.session_risk import active_entry_session
from trader.mtf_strategies.signal import MTFSignal, NoSignal
from trader.mtf_strategies.targets import select_tp2
from trader.risk.levels import compute_trade_levels

_DIR_MAP = {"bullish": "long", "bearish": "short"}

# Rule 3 candidates ("swing low 1H", "equal lows 1H", "demand OB 1H") mean
# CURRENT/nearby liquidity price is actually reacting to, not "any swing this
# symbol ever made" -- filtering h1.swings/equal_levels/order_blocks (which
# `analysis.py` returns as the full causal-to-date list) to a recent window
# is both a performance fix (thousands of stale candidates were each
# triggering their own sweep scan) and a correctness one: a 6-year-old swing
# low is not "1H liquidity" a real setup would reference. Same bound used for
# how far back a sweep itself is searched (rule 4 implies a fresh, current
# sweep, not one from years ago).
RECENT_H1_LOOKBACK_BARS = 96  # ~4 días -- 48 (~2 días) se probó primero y dio
# resultados DISTINTOS al calculo sin acotar para compute_asia_session_high_low
# (la ventana arrancaba a mitad de una sesión Asia real, truncando su high) --
# verificado con 96/120/168 contra el cálculo completo en 8 cutoffs
# aleatorios antes de fijar este valor, no elegido a ojo.


def _recent_h1_df(h1: SimpleAnalysis) -> pd.DataFrame:
    return h1.df.tail(RECENT_H1_LOOKBACK_BARS).reset_index(drop=True)


def _recent_cutoff(h1: SimpleAnalysis) -> pd.Timestamp:
    from trader.events import TF_DURATION
    if len(h1.df) == 0:
        return pd.Timestamp.min.tz_localize("UTC")
    last_close = h1.df["timestamp"].iloc[-1] + TF_DURATION["H1"]
    return last_close - RECENT_H1_LOOKBACK_BARS * TF_DURATION["H1"]


@dataclass(frozen=True)
class _CandidateLevel:
    name: str
    price: float


def _long_liquidity_candidates(h1: SimpleAnalysis, d1_df: pd.DataFrame, sessions_config) -> list[_CandidateLevel]:
    lv = compute_liquidity_levels(d1_df)
    # `.tail()` acá, no `h1.df` completo -- ver la misma nota de performance
    # en reversal.py (`.apply()` de classify_session sobre años de velas fue
    # el cuello de botella real medido en el Paso 3).
    asia_low = compute_asia_session_high_low(_recent_h1_df(h1), sessions_config)[1]
    recent = _recent_cutoff(h1)
    candidates = []
    if lv.pdl is not None:
        candidates.append(_CandidateLevel("PDL", lv.pdl))
    if asia_low is not None:
        candidates.append(_CandidateLevel("AsiaLow", asia_low))
    for e in h1.equal_levels:
        if e.kind == "equal_lows" and e.timestamp >= recent:
            candidates.append(_CandidateLevel("EqualLow_1H", e.price))
    for s in h1.swings:
        if s.kind == "swing_low" and s.timestamp >= recent:
            candidates.append(_CandidateLevel("SwingLow_1H", s.price))
    for ob in h1.order_blocks:
        if ob.direction == "bullish" and ob.timestamp >= recent:  # zona de demanda
            candidates.append(_CandidateLevel("DemandOB_1H", ob.price))
    for iob in h1.inverted_order_blocks:
        # decision de diseño confirmada: "zona de demanda/oferta 1H" = Order
        # Block/IOB -- un IOB alcista (antes oferta, ahora demanda tras
        # romperse) cuenta igual que un OB de demanda.
        if iob.direction == "bullish" and iob.timestamp >= recent:
            candidates.append(_CandidateLevel("DemandIOB_1H", iob.price))
    return candidates


def _short_liquidity_candidates(h1: SimpleAnalysis, d1_df: pd.DataFrame, sessions_config) -> list[_CandidateLevel]:
    lv = compute_liquidity_levels(d1_df)
    asia_high = compute_asia_session_high_low(_recent_h1_df(h1), sessions_config)[0]
    recent = _recent_cutoff(h1)
    candidates = []
    if lv.pdh is not None:
        candidates.append(_CandidateLevel("PDH", lv.pdh))
    if asia_high is not None:
        candidates.append(_CandidateLevel("AsiaHigh", asia_high))
    for e in h1.equal_levels:
        if e.kind == "equal_highs" and e.timestamp >= recent:
            candidates.append(_CandidateLevel("EqualHigh_1H", e.price))
    for s in h1.swings:
        if s.kind == "swing_high" and s.timestamp >= recent:
            candidates.append(_CandidateLevel("SwingHigh_1H", s.price))
    for ob in h1.order_blocks:
        if ob.direction == "bearish" and ob.timestamp >= recent:  # zona de oferta
            candidates.append(_CandidateLevel("SupplyOB_1H", ob.price))
    for iob in h1.inverted_order_blocks:
        if iob.direction == "bearish" and iob.timestamp >= recent:
            candidates.append(_CandidateLevel("SupplyIOB_1H", iob.price))
    return candidates


def _tp2_priority(direction: str, d1_df: pd.DataFrame, h4: SimpleAnalysis, h1: SimpleAnalysis) -> list[float | None]:
    """"Swing High 4H"/"máximo relevante 1H" (regla 10) se leen como el
    swing/extremo MÁS RECIENTE, no el máximo histórico de toda la serie --
    un máximo de 8 años de historia casi nunca sería un objetivo realista, y
    "relevante" en la spec apunta a lo cercano/actual, no a un récord
    histórico. Acotado a la misma ventana reciente que los candidatos de
    liquidez 1H (`RECENT_H1_LOOKBACK_BARS`), por consistencia y para no
    volver a escanear años de velas innecesariamente."""
    lv = compute_liquidity_levels(d1_df)
    recent = _recent_cutoff(h1)
    recent_h1_df = _recent_h1_df(h1)

    def _latest(events, kind) -> float | None:
        matches = [e for e in events if e.kind == kind]
        return max(matches, key=lambda e: e.timestamp).price if matches else None

    if direction == "long":
        swing_high_4h = _latest(h4.swings, "swing_high")
        equal_high = _latest([e for e in h1.equal_levels if e.timestamp >= recent], "equal_highs")
        max_1h = float(recent_h1_df["high"].max()) if len(recent_h1_df) else None
        return [lv.pdh, lv.pwh, swing_high_4h, equal_high, max_1h]
    swing_low_4h = _latest(h4.swings, "swing_low")
    equal_low = _latest([e for e in h1.equal_levels if e.timestamp >= recent], "equal_lows")
    min_1h = float(recent_h1_df["low"].min()) if len(recent_h1_df) else None
    return [lv.pdl, lv.pwl, swing_low_4h, equal_low, min_1h]


def evaluate_continuation(
    symbol: str,
    h4: SimpleAnalysis,
    h1: SimpleAnalysis,
    m15: SimpleAnalysis,
    d1_df_as_of: pd.DataFrame,
    config: TraderConfig,
    as_of: pd.Timestamp,
    current_price: float,
    require_retracement: bool = True,
) -> list[MTFSignal] | NoSignal:
    # paso 1: sesgo 4H
    bias = classify_htf_structure(h4.swings, float(h4.df["close"].iloc[-1]) if len(h4.df) else current_price)
    if bias == "none":
        return NoSignal(strategy="continuation", reason="4H_structure_not_established", stage="bias_4h")
    direction4h, trade_direction = bias, _DIR_MAP[bias]

    # paso 3: ventana de sesión de entrada
    session = active_entry_session(as_of)
    if session is None:
        return NoSignal(strategy="continuation", reason="outside_entry_session", stage="session")

    # paso 2 + 4: mapa de liquidez 1H candidato, evaluado nivel por nivel (independiente)
    candidates = (
        _long_liquidity_candidates(h1, d1_df_as_of, config.sessions)
        if trade_direction == "long"
        else _short_liquidity_candidates(h1, d1_df_as_of, config.sessions)
    )
    sweep_direction = "bullish" if trade_direction == "long" else "bearish"
    recent_h1_df = _recent_h1_df(h1)

    signals: list[MTFSignal] = []
    for cand in candidates:
        sweep = detect_named_level_sweep(recent_h1_df, "H1", cand.price, sweep_direction, config.liquidity.sweep_wick_min_pct)
        mss = None
        if sweep is not None:
            mss = first_post_sweep_mss(m15.df, m15.swings, sweep.timestamp, direction4h)

        fvg = None
        if mss is not None:
            same_direction_fvgs = [
                f for f in m15.fvgs
                if f.direction == direction4h and f.timestamp >= mss.timestamp
            ]
            fvg = min(same_direction_fvgs, key=lambda f: f.timestamp) if same_direction_fvgs else None

        retraced = False
        entry_price = None
        if fvg is not None:
            fifty_pct = (fvg.price_high + fvg.price_low) / 2.0
            touched_since = m15.df[m15.df["timestamp"] >= fvg.timestamp]
            if trade_direction == "long":
                retraced = bool((touched_since["low"] <= fifty_pct).any())
            else:
                retraced = bool((touched_since["high"] >= fifty_pct).any())
            entry_price = fifty_pct

        # require_retracement=False (piloto en vivo, orden límite real en el 50% FVG,
        # ver project_mtf_pending_limit_orders_2026-09-23): el disparo ya no es "el
        # precio YA volvió al 50%" sino "el setup está listo para que una orden límite
        # espere ahí" -- la condición de retraced se satisface por construcción, pero
        # la expiración por tiempo SIGUE contando estrictamente desde fvg_confirmed_at
        # (nunca se "blinda" por un retraced histórico como en el modo original).
        gate_retraced = retraced or not require_retracement
        retraced_for_expiry = retraced if require_retracement else False
        expired = fvg is not None and not retraced_for_expiry and fvg_setup_expired(fvg.timestamp, as_of, retraced_for_expiry)

        sl = None
        if sweep is not None:
            # regla 9: "el mínimo/máximo CREADO DURANTE EL SWEEP" -- la vela
            # exacta que barrió el nivel (sweep.timestamp es su close_ts, ver
            # detect_named_level_sweep), no una ventana alrededor de ella.
            sweep_candle = recent_h1_df[recent_h1_df["timestamp"] + TF_DURATION["H1"] == sweep.timestamp]
            if len(sweep_candle) == 1:
                sl = float(sweep_candle["low"].iloc[0]) if trade_direction == "long" else float(sweep_candle["high"].iloc[0])

        levels = None
        if entry_price is not None and sl is not None and gate_retraced and not expired:
            priority = _tp2_priority(trade_direction, d1_df_as_of, h4, h1)
            tp2 = select_tp2(trade_direction, priority, entry_price)
            if tp2 is not None:
                levels = compute_trade_levels(trade_direction, entry_price, sl, tp2, config.risk.min_risk_reward)

        # piso mínimo de riesgo: la regla 9 solo describe el caso "SL demasiado
        # GRANDE" (-> NO TRADE, nunca se acorta a mano) pero un SL demasiado
        # CHICO es igual de inválido -- el spread solo ya se lo comería. Esto
        # no está en la spec explícitamente, pero SÍ existe ya en el pipeline
        # de producción (`pipeline/engine.py::_minimum_risk_floor`,
        # `config.risk.min_risk_atr_multiple`) -- reutilizado acá, no
        # reinventado, encontrado necesario al ver un R:R de ~107 en el
        # smoke test (SL de ~1.85 pips) antes de dar esto por terminado.
        h1_atr_latest = float(h1.atr.iloc[-1]) if len(h1.atr) and not pd.isna(h1.atr.iloc[-1]) else None
        risk_distance = abs(entry_price - sl) if (entry_price is not None and sl is not None) else None
        risk_floor_ok = (
            risk_distance is not None and h1_atr_latest is not None
            and risk_distance >= config.risk.min_risk_atr_multiple * h1_atr_latest
        )

        gate = evaluate_all_or_nothing({
            "4h_bias": bias != "none",
            "in_entry_session": session is not None,
            "1h_sweep": sweep is not None,
            "15m_mss": mss is not None,
            "15m_fvg": fvg is not None,
            "fvg_not_expired": not expired,
            "retraced_to_50pct": gate_retraced,
            "sl_valid": sl is not None and entry_price is not None and (
                (trade_direction == "long" and sl < entry_price) or (trade_direction == "short" and sl > entry_price)
            ),
            "sl_above_minimum_risk_floor": risk_floor_ok,
            "rr_gate": levels is not None,
        })
        if not gate.passed:
            continue  # este nivel candidato no llegó al final -- se sigue con el siguiente, independiente

        risk_dist = abs(entry_price - sl)
        tp1 = entry_price + risk_dist if trade_direction == "long" else entry_price - risk_dist  # TP1 siempre a 1R exacto (regla 10)

        signals.append(MTFSignal(
            strategy="continuation", setup=None, symbol=symbol, direction=trade_direction, session=session,
            entry=entry_price, sl=sl, tp1=tp1, tp2=levels.tp, risk_reward=levels.risk_reward,
            sweep_timestamp=sweep.timestamp, mss_timestamp=mss.timestamp, fvg_confirmed_at=fvg.timestamp,
            generated_at=as_of, trace={"liquidity_level": cand.name, "liquidity_price": cand.price},
        ))

    if not signals:
        return NoSignal(strategy="continuation", reason="no_level_completed_full_sequence", stage="sequence")

    return sorted(signals, key=lambda s: s.risk_reward, reverse=True)
