"""Descarga historico real via el bridge MT5 para las 4 temporalidades de la estrategia
(D1 bias -> H1 -> M15 entrada -> M5 entrada puntual) y lo guarda como CSV en data/raw/.

Requiere el bridge corriendo: ./scripts/start_mt5_bridge.sh --port 18812

Uso:
    python scripts/fetch_historical_data.py --symbols EURUSD GBPUSD XAUUSD
    python scripts/fetch_historical_data.py --symbols EURUSD --n-d1 3000 --n-h1 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from tradingai.mt5.connector import MT5Connector  # noqa: E402
from tradingai.utils.logging import setup_logging  # noqa: E402

from loguru import logger  # noqa: E402

# Cuantas velas pedir por temporalidad, como techo a intentar. El servidor demo
# (MetaQuotes-Demo) da 502 no solo por pedir demasiado en general (~90-99k es el techo
# absoluto medido el 2026-08-22), sino tambien cuando un simbolo concreto no tiene tanta
# profundidad historica como otro (p.ej. XAUUSD/USDJPY fallan en D1=5000 mientras
# EURUSD no) — varia por instrumento, no es un limite fijo. Por eso el fetch reintenta
# con menos velas en vez de fallar directo (ver _fetch_with_retry).
DEFAULT_N_CANDLES = {"D1": 5000, "H1": 90000, "M15": 90000, "M5": 90000}

# Al fallar, reintentar con la mitad de velas, hasta este piso.
MIN_N_CANDLES = 500


def _fetch_with_retry(connector: MT5Connector, symbol: str, timeframe: str, n: int):
    while n >= MIN_N_CANDLES:
        try:
            return connector.get_candles(symbol, timeframe, n_candles=n)
        except Exception as exc:
            logger.warning(f"{symbol} {timeframe}: fallo con {n} velas ({exc}); probando con {n // 2}")
            n //= 2
    raise RuntimeError(f"{symbol} {timeframe}: no se pudo descargar ni con {MIN_N_CANDLES} velas")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None, help="Por defecto, los de config.yaml")
    parser.add_argument("--n-d1", type=int, default=DEFAULT_N_CANDLES["D1"])
    parser.add_argument("--n-h1", type=int, default=DEFAULT_N_CANDLES["H1"])
    parser.add_argument("--n-m15", type=int, default=DEFAULT_N_CANDLES["M15"])
    parser.add_argument("--n-m5", type=int, default=DEFAULT_N_CANDLES["M5"])
    args = parser.parse_args()

    config = get_settings()
    secrets = config["secrets"]
    setup_logging(secrets.log_level, config["paths"]["logs_dir"])

    symbols = args.symbols or config["trading"]["symbols"]
    n_candles = {"D1": args.n_d1, "H1": args.n_h1, "M15": args.n_m15, "M5": args.n_m5}

    out_dir = Path(config["paths"]["raw_data"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timeout generoso: la primera peticion a un rango historico profundo (sobre todo
    # D1) puede obligar al terminal a descargar datos antiguos del servidor del broker.
    connector = MT5Connector(base_url=secrets.mt5_bridge_url, timeout=90.0)
    with connector:
        for symbol in symbols:
            for timeframe, n in n_candles.items():
                try:
                    df = _fetch_with_retry(connector, symbol, timeframe, n)
                except Exception:
                    logger.exception(f"Fallo descargando {symbol} {timeframe}")
                    continue

                out_path = out_dir / f"{symbol}_{timeframe}.csv"
                df.to_csv(out_path, index=False)
                span = f"{df['timestamp'].min()} -> {df['timestamp'].max()}" if len(df) else "sin datos"
                logger.info(f"{symbol} {timeframe}: {len(df)} velas ({span}) -> {out_path}")


if __name__ == "__main__":
    main()
