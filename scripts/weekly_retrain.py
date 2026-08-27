"""Reentrenamiento semanal automatizado: para cada simbolo, refresca el historico de
MERCADO (no las operaciones propias del piloto -- ver nota abajo), valida con
walk-forward, y promueve el checkpoint nuevo a produccion SOLO si el resultado sigue
siendo positivo. Mismo criterio que se aplico manualmente durante todo el desarrollo
del piloto: nunca se reemplaza un checkpoint en produccion sin validar primero.

Por que NO reentrena sobre las operaciones del propio piloto: entrenar un modelo sobre
las decisiones que el MISMO modelo ya tomo introduce riesgo de sesgo de
retroalimentacion -- las operaciones que hizo no son una muestra representativa del
mercado, son una muestra filtrada por sus propios criterios (si tiene un error
sistematico, reforzarlo con sus propios datos lo empeora en vez de corregirlo). Este
script siempre reentrena sobre historico de mercado fresco descargado de MT5, igual
que el entrenamiento original.

Criterio de promocion (ver `_should_promote`): al menos la mitad de los folds del
walk-forward positivos Y pnl medio > 0. Si no se cumple, se mantiene el checkpoint
actual y se reporta -- no se sube un modelo peor solo por ser mas reciente.

Uso:
    python scripts/weekly_retrain.py
    python scripts/weekly_retrain.py --symbols EURUSD GBPUSD
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402

from loguru import logger  # noqa: E402

RESULTS_LOG = PROJECT_ROOT / "data" / "models" / "weekly_retrain_history.json"
_FOLD_LINE = re.compile(r"BACKTEST.*?win_rate=([\d.]+)%, pnl=(-?[\d.]+)%")
_RISK_LINE = re.compile(r"riesgo: sharpe=(-?[\d.]+), sortino=(-?[\d.]+), max_drawdown=([\d.]+)%")
_EXPECTANCY_LINE = re.compile(r"expectancy_r=(-?[\d.]+), profit_factor=(-?[\d.]+|inf)")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)


def _walk_forward_folds(symbol: str, confidence_threshold: float, thermal_args: list[str]) -> list[dict]:
    result = _run(
        [
            sys.executable, "scripts/baseline_gbm.py",
            "--symbol", symbol,
            "--backtest-step", "20",
            "--confidence-threshold", str(confidence_threshold),
            *thermal_args,
        ]
    )
    output = result.stdout + result.stderr
    fold_stats = [{"win_rate": float(w), "pnl": float(p)} for w, p in _FOLD_LINE.findall(output)]
    risk_stats = _RISK_LINE.findall(output)
    for fold, (sharpe, sortino, max_dd) in zip(fold_stats, risk_stats):
        fold["sharpe"] = float(sharpe)
        fold["sortino"] = float(sortino)
        fold["max_drawdown_pct"] = float(max_dd)

    expectancy_stats = _EXPECTANCY_LINE.findall(output)
    for fold, (expectancy_r, profit_factor) in zip(fold_stats, expectancy_stats):
        fold["expectancy_r"] = float(expectancy_r)
        fold["profit_factor"] = float(profit_factor)  # float("inf") parsea bien "inf"
    return fold_stats


def _should_promote(folds: list[dict]) -> bool:
    if not folds:
        return False
    positive = sum(1 for f in folds if f["pnl"] > 0)
    avg_pnl = sum(f["pnl"] for f in folds) / len(folds)
    return positive >= len(folds) / 2 and avg_pnl > 0


def retrain_symbol(symbol: str, confidence_threshold: float, thermal_args: list[str]) -> dict:
    logger.info(f"[{symbol}] Walk-forward de validacion")
    folds = _walk_forward_folds(symbol, confidence_threshold, thermal_args)
    promote = _should_promote(folds)

    if not folds:
        logger.error(f"[{symbol}] No se pudo leer el resultado del walk-forward, no se promueve")
        return {"symbol": symbol, "error": "walk_forward_failed"}

    avg_pnl = sum(f["pnl"] for f in folds) / len(folds)
    positive = sum(1 for f in folds if f["pnl"] > 0)
    logger.info(f"[{symbol}] {positive}/{len(folds)} folds positivos, pnl medio={avg_pnl:.2f}%")

    if promote:
        logger.info(f"[{symbol}] PROMOVIDO: reentrenando checkpoint de produccion")
        train = _run([sys.executable, "scripts/train_gbm.py", "--symbol", symbol])
        if train.returncode != 0:
            logger.error(f"[{symbol}] Fallo al reentrenar el checkpoint final:\n{train.stderr}")
            return {"symbol": symbol, "folds": folds, "promoted": False, "error": "train_failed"}
    else:
        logger.warning(f"[{symbol}] NO promovido (walk-forward no supero el criterio) — se mantiene el checkpoint actual")

    return {"symbol": symbol, "folds": folds, "promoted": promote, "avg_pnl": avg_pnl, "positive_folds": positive}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None, help="Por defecto, los de config.yaml")
    # Passthrough a baseline_gbm.py -- ver tradingai.utils.thermal. Los defaults de
    # abajo son los "seguros" (1 hilo, pausa a 80C) para la corrida automatica de los
    # sabados; para un reentreno manual con vigilancia humana y refrigeracion extra
    # se pueden relajar (ver conversacion 2026-08-25: "--max-threads 999 --max-temp-c
    # 999 --fold-cooldown-seconds 0" en el walk-forward de esa sesion).
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--max-temp-c", type=float, default=80.0)
    parser.add_argument("--fold-cooldown-seconds", type=float, default=20.0)
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="No descargar historico (usar el ya presente en data/raw/) -- para cuando se quiere "
             "apagar el bridge/Wine ANTES de entrenar (ver conversacion 2026-08-25/26: entrenar sin "
             "limite de hilos con el bridge todavia vivo puede tumbar la terminal MT5).",
    )
    args = parser.parse_args()

    config = get_settings()
    symbols = args.symbols or config["trading"]["symbols"]
    confidence_threshold = config["model"]["outputs"].get("confidence_threshold", 0.75)
    thermal_args = [
        "--max-threads", str(args.max_threads),
        "--max-temp-c", str(args.max_temp_c),
        "--fold-cooldown-seconds", str(args.fold_cooldown_seconds),
    ]

    # Descarga UNA sola vez, para todos los simbolos, antes de entrenar nada -- no
    # intercalada por simbolo. `baseline_gbm.py`/`train_gbm.py` solo leen CSVs locales
    # (data/raw/), no necesitan el bridge/MT5 en absoluto. Intercalar fetch+train por
    # simbolo (como se hacia antes) deja el entrenamiento pesado corriendo mientras el
    # bridge sigue vivo -- y un entrenamiento sin limite de hilos puede saturar tanto
    # la CPU que tira la terminal MT5 (visto en vivo el 2026-08-25), rompiendo en
    # cascada la descarga de TODOS los simbolos siguientes. Separar las fases evita
    # ese riesgo por completo: la parte que necesita al bridge termina en segundos,
    # mucho antes de que el entrenamiento pesado siquiera empiece.
    if args.skip_fetch:
        logger.info(f"--skip-fetch: usando el historico ya presente en data/raw/ para {len(symbols)} simbolos")
    else:
        logger.info(f"Descargando historico fresco para {len(symbols)} simbolos (requiere el bridge MT5 vivo)")
        fetch = _run([sys.executable, "scripts/fetch_historical_data.py", "--symbols", *symbols])
        if fetch.returncode != 0:
            logger.error(f"Fallo la descarga de historico (¿bridge MT5 caido?), se aborta sin entrenar nada:\n{fetch.stderr}")
            return

    run_date = datetime.now(timezone.utc).isoformat()
    results = [retrain_symbol(symbol, confidence_threshold, thermal_args) for symbol in symbols]

    history = json.loads(RESULTS_LOG.read_text()) if RESULTS_LOG.exists() else {}
    history[run_date] = results
    RESULTS_LOG.write_text(json.dumps(history, indent=2))

    n_promoted = sum(1 for r in results if r.get("promoted"))
    logger.info(f"=== Reentrenamiento semanal completo: {n_promoted}/{len(symbols)} simbolos promovidos ===")
    for r in results:
        status = "PROMOVIDO" if r.get("promoted") else r.get("error", "sin cambios")
        logger.info(f"  {r['symbol']}: {status}")


if __name__ == "__main__":
    main()
