# TradingAI

Proyecto dividido en dos mitades desacopladas, unidas por un contrato de datos simple (`TradingSignal`):

1. **`tradingai.ai`** — IA entrenable de analisis tecnico. A partir de velas OHLCV calcula
   features de Smart Money Concepts (order blocks, liquidez, premium/discount), Fair Value
   Gaps, perfil de volumen (acumulacion/distribucion) e indicadores clasicos, y entrena un
   modelo multi-tarea (Transformer o LSTM) que predice **direccion, entrada, take-profit y
   stop-loss**.
2. **`tradingai.mt5`** — Conexion a MetaTrader 5: descarga de velas, gestion de riesgo,
   sizing de posiciones y envio de ordenes.

`tradingai.core` orquesta ambas mitades (`TradingPipeline`) sin que ninguna conozca los
detalles internos de la otra.

## Estructura

```
config/                     # config.yaml (parametros) + settings.py (.env -> Secrets)
src/tradingai/
  ai/
    data/
      loader.py             # carga CSV / guarda parquet procesado
      preprocessor.py       # normalizacion + construccion de secuencias
      features/
        market_structure.py # swing highs/lows, BOS/CHoCH
        smc.py               # order blocks, liquidity pools, premium/discount
        gaps.py               # Fair Value Gaps
        volume_profile.py    # POC, value area, acumulacion/distribucion
        indicators.py        # EMA, RSI, ATR, MACD, Bollinger
        pipeline.py           # combina todo lo anterior segun config.yaml
    models/
      architectures/         # encoders: transformer.py, lstm.py
      base.py                 # MultiHeadTradingModel (encoder + 3 cabezas)
      direction_model.py      # cabeza: direccion (long/short/neutral)
      entry_model.py          # cabeza: precio de entrada
      tp_sl_model.py           # cabeza: take-profit / stop-loss
    training/
      dataset.py              # etiquetado (triple barrera) + Dataset de secuencias
      trainer.py               # bucle de entrenamiento, loss multi-tarea
    inference/
      predictor.py             # velas -> features -> modelo -> TradingSignal
    evaluation/
      backtester.py            # backtest simplificado sobre historico
  mt5/
    connector.py               # cliente HTTP del bridge MT5 (ver wine_bridge/)
    data_feed.py                # espera cierre de nueva vela
    account.py                  # balance/equity de la cuenta
    risk_manager.py             # aprueba senales, calcula tamano de posicion
    order_executor.py           # envia ordenes via el bridge
  core/
    signal.py                   # TradingSignal: contrato entre IA y MT5
    pipeline.py                  # orquesta: datos -> prediccion -> riesgo -> orden
  utils/                         # logging, timeframes
scripts/
  train.py                      # entrena el modelo desde un CSV
  backtest.py                    # backtest de un checkpoint
  run_live.py                    # bucle en vivo conectado al bridge MT5
  start_mt5_bridge.sh            # lanza wine_bridge/server.py dentro de Wine
wine_bridge/
  server.py                      # corre bajo Wine: MetaTrader5 -> HTTP local
  requirements-wine.txt          # deps del Python de Windows dentro de Wine
tests/                           # pytest, con datos sinteticos (no requieren MT5)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # MT5_BRIDGE_URL por defecto ya apunta a localhost:18812
```

Ajusta parametros de features/modelo/riesgo en `config/config.yaml`.

## MT5 en Linux (resuelto: bridge sobre Wine)

El paquete oficial `MetaTrader5` **solo funciona sobre un terminal MT5 real**, que es una
aplicacion Windows — no hay build nativa para Linux. Este proyecto usa Wine en la misma
maquina: el terminal MT5 y un Python de Windows corren dentro de un prefijo de Wine
dedicado, y exponen todo por HTTP en `localhost` para que el resto del proyecto (que corre
en Linux nativo, con torch, etc.) lo consuma como un servicio normal.

```
Linux nativo (.venv)                    Wine (WINEPREFIX=~/.wine-mt5)
┌─────────────────────────┐             ┌──────────────────────────────┐
│ tradingai.* (torch,      │  HTTP       │ wine_bridge/server.py         │
│ features, training...)  │◄───────────►│  -> MetaTrader5 (pip)          │
│ MT5Connector (cliente)  │  :18812     │  -> terminal64.exe (MT5 real)  │
└─────────────────────────┘             └──────────────────────────────┘
```

Setup ya realizado en esta maquina (dejar documentado para reproducirlo en otra):

1. `sudo apt install -y wine` + soporte 32-bit (`wine32:i386`).
2. Prefijo dedicado: `WINEARCH=win64 WINEPREFIX=~/.wine-mt5 wineboot --init`.
3. Instalar el terminal MT5 dentro del prefijo (instalador oficial de MetaQuotes, via `wine`).
4. Instalar Python de Windows dentro del mismo prefijo, y ahi:
   `pip install -r wine_bridge/requirements-wine.txt`
   (numpy fijado a 1.26.4: Wine 9.0 no implementa `ucrtbase.dll.crealf`, necesaria por
   numpy>=2.0, y el import aborta).

Para arrancar el bridge (con el terminal ya logueado en una cuenta, demo o real):

```bash
./scripts/start_mt5_bridge.sh --port 18812
```

El script usa `script` internamente para darle una pseudo-tty a `python.exe` — sin eso,
Wine 9.0 falla con `init_sys_streams: ... Invalid handle` en cuanto el proceso no tiene una
consola real (p.ej. lanzado en background o desde un servicio).

Con el bridge corriendo, `MT5Connector` (lado Linux) es solo un cliente HTTP — no importa
`MetaTrader5` ni sabe nada de Wine:

```python
from tradingai.mt5.connector import MT5Connector
c = MT5Connector()  # usa MT5_BRIDGE_URL del .env, por defecto localhost:18812
c.connect()
df = c.get_candles("EURUSD", "M15", n_candles=500)
```

## Flujo de trabajo

```bash
# 1. Entrenar con historico (CSV con columnas timestamp,open,high,low,close,volume)
python scripts/train.py --csv data/raw/EURUSD_M15.csv --symbol EURUSD

# 2. Backtest simplificado del checkpoint
python scripts/backtest.py --csv data/raw/EURUSD_M15.csv --checkpoint data/models/EURUSD_transformer.pt --symbol EURUSD

# 3. Bucle en vivo (requiere el bridge corriendo, ver seccion anterior)
./scripts/start_mt5_bridge.sh --port 18812 &
python scripts/run_live.py --checkpoint data/models/EURUSD_transformer.pt --symbol EURUSD
```

## Tests

```bash
pytest
```

Los tests usan velas sinteticas (`tests/conftest.py`) y no requieren conexion a MT5.

## Estado actual / siguientes pasos

Arquitectura base + conexion MT5 ya funcionando de extremo a extremo (velas y cuenta reales
via el bridge sobre Wine, cuenta demo). Pendiente: conseguir historico real para entrenar
(datos sinteticos por ahora), labeling multi-temporalidad (D1 bias -> H1 -> M15/M5 entrada,
ver notas del proyecto), backtest con costes realistas, y mas reglas de riesgo (drawdown
diario, correlacion entre simbolos) antes de un piloto en demo.
