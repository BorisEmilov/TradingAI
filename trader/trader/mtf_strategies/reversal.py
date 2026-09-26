"""Estrategia 2 -- Reversión MTF SMC (4H/1H/15M). NUEVO -- mismas piezas que
`continuation.py` (comparten `analysis.py`, `bias.py`, `liquidity_levels.py`,
`post_sweep_structure.py`, `targets.py`, `gate.py`, `exits.py`,
`session_risk.py`), filosofía de entrada distinta: precio EXTENDIDO a un
extremo estructural 4H -> sweep -> rechazo -> giro, nunca en mitad de rango
(regla 2).

Setup A (reversión del rango asiático, prioridad de testeo) y Setup B
(PDH/PDL) comparten el mismo motor de "extremo -> sweep -> MSS -> FVG ->
entrada" -- lo que cambia entre ellos es SOLO qué nivel HTF define el
extremo y qué condición extra exige el setup (A: Londres rompe el Asia
High/Low; B: ninguna condición extra, solo PDH/PDL con confluencia
opcional). `evaluate_reversal` evalúa ambos setups de forma independiente
sobre el mismo símbolo/momento y devuelve todas las señales que completen la
secuencia -- mismo criterio de "no preseleccionar" que Estrategia 1.
"""

from __future__ import annotations

import pandas as pd

from trader.config import TraderConfig
from trader.events import TF_DURATION
from trader.mtf_strategies.analysis import SimpleAnalysis
from trader.mtf_strategies.bias import classify_htf_structure, entry_4h_condition_holds
from trader.mtf_strategies.exits import fvg_setup_expired
from trader.mtf_strategies.extension import DEFAULT_EXTENSION_THRESHOLD, find_swing_origin, passes_extension_filter
from trader.mtf_strategies.gate import evaluate_all_or_nothing
from trader.mtf_strategies.liquidity_levels import compute_asia_session_high_low, compute_liquidity_levels, detect_named_level_sweep
from trader.mtf_strategies.post_sweep_structure import first_post_sweep_mss
from trader.mtf_strategies.session_risk import active_entry_session
from trader.mtf_strategies.signal import MTFSignal, NoSignal
from trader.mtf_strategies.targets import select_tp2
from trader.risk.levels import compute_trade_levels

_DIR_MAP = {"bullish": "long", "bearish": "short"}

# mismo razonamiento y mismo valor que continuation.py::RECENT_H1_LOOKBACK_BARS
# -- un sweep o un swing 1H de años atrás no es "liquidez actual", y escanear
# toda la historia en cada candidato era el cuello de botella real detectado
# al intentar correr el Paso 3 sobre una ventana de meses. 96 (no 48) porque
# 48 dio resultados distintos al cálculo sin acotar en compute_asia_session_
# high_low -- verificado contra 8 cutoffs aleatorios antes de fijar el valor.
RECENT_H1_LOOKBACK_BARS = 96


def _recent_h1_df(h1: SimpleAnalysis) -> pd.DataFrame:
    return h1.df.tail(RECENT_H1_LOOKBACK_BARS).reset_index(drop=True)


def _recent_cutoff(h1: SimpleAnalysis) -> pd.Timestamp:
    if len(h1.df) == 0:
        return pd.Timestamp.min.tz_localize("UTC")
    last_close = h1.df["timestamp"].iloc[-1] + TF_DURATION["H1"]
    return last_close - RECENT_H1_LOOKBACK_BARS * TF_DURATION["H1"]


def _latest_extremity_swing(h4: SimpleAnalysis, kind: str) -> "MarketEvent | None":  # noqa: F821 -- typing only
    candidates = [s for s in h4.swings if s.kind == kind]
    return max(candidates, key=lambda s: s.timestamp) if candidates else None


def _latest_by_kind(events, kind: str) -> float | None:
    matches = [e for e in events if e.kind == kind]
    return max(matches, key=lambda e: e.timestamp).price if matches else None


def _evaluate_one_setup(
    setup_name: str,
    symbol: str,
    direction4h: str,  # "bearish" (extremo superior, SHORT) | "bullish" (extremo inferior, LONG)
    htf_level_price: float,
    htf_level_name: str,
    h4: SimpleAnalysis,
    h1: SimpleAnalysis,
    m15: SimpleAnalysis,
    d1_df_as_of: pd.DataFrame,
    config: TraderConfig,
    as_of: pd.Timestamp,
    session: str,
    extension_threshold: float,
    asia_high: float | None,
    asia_low: float | None,
    require_retracement: bool = True,
) -> MTFSignal | NoSignal:
    trade_direction = _DIR_MAP[direction4h]
    sweep_direction = direction4h  # sweep de liquidez SUPERIOR (bearish) para SHORT, INFERIOR (bullish) para LONG

    # regla 2: nunca en mitad de rango -- filtro de extensión (regla 8, Estrategia 2)
    extremity_kind = "swing_high" if direction4h == "bearish" else "swing_low"
    extremity = _latest_extremity_swing(h4, extremity_kind)
    origin = find_swing_origin(h4.swings, extremity) if extremity is not None else None
    h4_atr_latest = float(h4.atr.iloc[-1]) if len(h4.atr) and not pd.isna(h4.atr.iloc[-1]) else None
    extension_ok = (
        extremity is not None and origin is not None and h4_atr_latest is not None
        and passes_extension_filter(extremity.price, origin.price, h4_atr_latest, extension_threshold)
    )

    recent_h1_df = _recent_h1_df(h1)
    sweep = detect_named_level_sweep(recent_h1_df, "H1", htf_level_price, sweep_direction, config.liquidity.sweep_wick_min_pct)
    # BUG REAL encontrado en la auditoría del Paso 3 (2026-09-20): esto decía
    # "bullish" para un SHORT (direction4h="bearish") -- invertido. La regla
    # 3 de Estrategia 2 es explícita para el caso SHORT: "MSS BAJISTA" y "FVG
    # BAJISTA" tras un sweep de liquidez superior -- MISMA dirección que
    # direction4h (el rechazo confirma la reversión hacia esa dirección), no
    # la opuesta. El síntoma que lo delató: en la auditoría visual, el 100%
    # de las señales de reversión mostraban una FVG "no genuina" (el gap no
    # coincidía con la dirección real del trade) -- porque se buscaba FVG en
    # la dirección MSS equivocada. continuation.py ya usaba `direction4h`
    # directo (correcto); esta inversión era específica de reversal.py.
    expected_mss_direction = direction4h
    mss = first_post_sweep_mss(m15.df, m15.swings, sweep.timestamp, expected_mss_direction) if sweep is not None else None

    fvg = None
    if mss is not None:
        same_dir_fvgs = [f for f in m15.fvgs if f.direction == expected_mss_direction and f.timestamp >= mss.timestamp]
        fvg = min(same_dir_fvgs, key=lambda f: f.timestamp) if same_dir_fvgs else None

    retraced = False
    entry_price = None
    if fvg is not None:
        fifty_pct = (fvg.price_high + fvg.price_low) / 2.0
        since = m15.df[m15.df["timestamp"] >= fvg.timestamp]
        retraced = bool((since["low"] <= fifty_pct).any()) if trade_direction == "long" else bool((since["high"] >= fifty_pct).any())
        entry_price = fifty_pct

    # require_retracement=False (piloto en vivo, orden límite real en el 50% FVG,
    # ver project_mtf_pending_limit_orders_2026-09-23 / mismo criterio que
    # continuation.py): la expiración por tiempo sigue contando estrictamente
    # desde fvg.timestamp, nunca se "blinda" por un retraced histórico.
    gate_retraced = retraced or not require_retracement
    retraced_for_expiry = retraced if require_retracement else False
    expired = fvg is not None and not retraced_for_expiry and fvg_setup_expired(fvg.timestamp, as_of, retraced_for_expiry)

    sl = None
    if sweep is not None:
        sweep_candle = recent_h1_df[recent_h1_df["timestamp"] + TF_DURATION["H1"] == sweep.timestamp]
        if len(sweep_candle) == 1:
            sl = float(sweep_candle["high"].iloc[0]) if trade_direction == "short" else float(sweep_candle["low"].iloc[0])

    h1_atr_latest = float(h1.atr.iloc[-1]) if len(h1.atr) and not pd.isna(h1.atr.iloc[-1]) else None
    risk_distance = abs(entry_price - sl) if (entry_price is not None and sl is not None) else None
    risk_floor_ok = risk_distance is not None and h1_atr_latest is not None and risk_distance >= config.risk.min_risk_atr_multiple * h1_atr_latest

    levels = None
    if entry_price is not None and sl is not None and gate_retraced and not expired and risk_floor_ok:
        lv = compute_liquidity_levels(d1_df_as_of)
        recent = _recent_cutoff(h1)
        recent_equal_levels = [e for e in h1.equal_levels if e.timestamp >= recent]
        recent_h1_swings = [s for s in h1.swings if s.timestamp >= recent]
        # "Swing High/Low 4H" y "1H" (regla 10): el MÁS RECIENTE, no el
        # extremo histórico de toda la serie -- mismo razonamiento que
        # continuation.py::_tp2_priority.
        if trade_direction == "short":  # TP2 hacia liquidez INFERIOR
            equal_low = _latest_by_kind(recent_equal_levels, "equal_lows")
            swing_low_4h = _latest_by_kind(h4.swings, "swing_low")
            swing_low_1h = _latest_by_kind(recent_h1_swings, "swing_low")
            priority = [asia_low, lv.pdl, equal_low, swing_low_1h, swing_low_4h]
        else:  # LONG, TP2 hacia liquidez SUPERIOR
            equal_high = _latest_by_kind(recent_equal_levels, "equal_highs")
            swing_high_4h = _latest_by_kind(h4.swings, "swing_high")
            swing_high_1h = _latest_by_kind(recent_h1_swings, "swing_high")
            priority = [asia_high, lv.pdh, equal_high, swing_high_1h, swing_high_4h]
        tp2 = select_tp2(trade_direction, priority, entry_price)
        if tp2 is not None:
            levels = compute_trade_levels(trade_direction, entry_price, sl, tp2, config.risk.min_risk_reward)

    gate = evaluate_all_or_nothing({
        "htf_extremity_with_extension": extension_ok,
        "1h_sweep": sweep is not None,
        "15m_mss_rejection": mss is not None,
        "15m_fvg": fvg is not None,
        "fvg_not_expired": not expired,
        "retraced_to_50pct": gate_retraced,
        "sl_above_minimum_risk_floor": risk_floor_ok,
        "rr_gate": levels is not None,
    })
    if not gate.passed:
        return NoSignal(strategy="reversal", reason=f"{setup_name}:{gate.failed_condition}", stage=setup_name)

    risk_dist = abs(entry_price - sl)
    tp1 = entry_price - risk_dist if trade_direction == "short" else entry_price + risk_dist

    return MTFSignal(
        strategy="reversal", setup=setup_name, symbol=symbol, direction=trade_direction, session=session,
        entry=entry_price, sl=sl, tp1=tp1, tp2=levels.tp, risk_reward=levels.risk_reward,
        sweep_timestamp=sweep.timestamp, mss_timestamp=mss.timestamp, fvg_confirmed_at=fvg.timestamp,
        generated_at=as_of, trace={"htf_level": htf_level_name, "htf_level_price": htf_level_price, "extension": extension_ok},
    )


def evaluate_reversal(
    symbol: str,
    h4: SimpleAnalysis,
    h1: SimpleAnalysis,
    m15: SimpleAnalysis,
    d1_df_as_of: pd.DataFrame,
    config: TraderConfig,
    as_of: pd.Timestamp,
    extension_threshold: float = DEFAULT_EXTENSION_THRESHOLD,
    require_retracement: bool = True,
) -> list[MTFSignal] | NoSignal:
    session = active_entry_session(as_of)
    if session is None:
        return NoSignal(strategy="reversal", reason="outside_entry_session", stage="session")

    h4_close = float(h4.df["close"].iloc[-1]) if len(h4.df) else None
    structure_4h = classify_htf_structure(h4.swings, h4_close) if h4_close is not None else "none"

    lv = compute_liquidity_levels(d1_df_as_of)
    # calculado UNA sola vez acá (no de nuevo por cada setup en
    # `_evaluate_one_setup`, y no sobre `h1.df` completo -- `.apply()` de
    # `classify_session` sobre años de velas H1 fue, medido, el cuello de
    # botella real al intentar escanear una ventana de meses en el Paso 3)
    asia_high, asia_low = compute_asia_session_high_low(_recent_h1_df(h1), config.sessions)

    attempts: list[tuple[str, str, float | None, str]] = []  # (setup_name, direction4h, level_price, level_name)

    # Setup B (PDH/PDL) -- sin condición de "4H no fuertemente bullish/bearish"
    if lv.pdh is not None:
        attempts.append(("setup_b_pdh_short", "bearish", lv.pdh, "PDH"))
    if lv.pdl is not None:
        attempts.append(("setup_b_pdl_long", "bullish", lv.pdl, "PDL"))

    # Setup A (reversión Asia) -- SHORT exige 4H "no fuertemente bullish"
    # (negación EXACTA del test de Estrategia 1, decisión de diseño
    # confirmada: bajista O "none" cuentan como "no fuertemente bullish").
    # LONG exige el espejo: "no fuertemente bearish".
    if asia_high is not None and entry_4h_condition_holds("reversal", "setup_a_asia_short", "bearish", structure_4h):
        attempts.append(("setup_a_asia_short", "bearish", asia_high, "AsiaHigh"))
    if asia_low is not None and entry_4h_condition_holds("reversal", "setup_a_asia_long", "bullish", structure_4h):
        attempts.append(("setup_a_asia_long", "bullish", asia_low, "AsiaLow"))

    if not attempts:
        return NoSignal(strategy="reversal", reason="no_htf_extremity_candidates", stage="liquidity_map")

    signals: list[MTFSignal] = []
    for setup_name, direction4h, level_price, level_name in attempts:
        result = _evaluate_one_setup(
            setup_name, symbol, direction4h, level_price, level_name,
            h4, h1, m15, d1_df_as_of, config, as_of, session, extension_threshold,
            asia_high, asia_low, require_retracement,
        )
        if isinstance(result, MTFSignal):
            signals.append(result)

    if not signals:
        return NoSignal(strategy="reversal", reason="no_setup_completed_full_sequence", stage="sequence")
    return sorted(signals, key=lambda s: s.risk_reward, reverse=True)
