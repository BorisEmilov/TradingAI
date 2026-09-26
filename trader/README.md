# trader

Multi-timeframe ICT/SMC day-trading agent, implementing
`../prompt-estrategia-trading-agente.md`: D1 bias -> H4 confirmation -> H1
POI -> M15 confirmation, minimum 3 confluences across 2+ timeframes, minimum
1:2 R:R, structural (never fixed-pip) SL/TP, 50% partial + breakeven. Built
phase-by-phase per `../prompt-implementacion-agente-trading.md`.

## Status by phase

- **Fase 1 (conexion y datos): DONE, validated against real data.**
  `gateway_client.py` talks to `../PythonGetaway` (already-built MT5 HTTP
  gateway) instead of reimplementing an MT5 bridge. Credentials live in
  `trader/.env` (gitignored, loaded via `python-dotenv`, never in source).
  `scripts/smoke_test_connection.py` ran successfully against the live
  MetaQuotes-Demo account (2026-09-15), printing real EURUSD H1 candles with
  broker-server-time correctly corrected to UTC.

  **Real infra issue found and fixed along the way**: the account originally
  given (Dukascopy-demo-mt5-1) hit a consistent `IPC timeout` (mt5_code
  -10005) from `mt5.initialize()`, reproduced even with a bare no-credentials
  attach — isolating it to something about that specific broker/server
  combination, not a Wine version or Dukascopy credentials issue per se
  (Wine went 9.0 -> 11.16 Staging -> 11.0 stable along the way, all three
  gave the identical failure). Switching to a MetaQuotes-Demo account
  connected immediately on the same Wine 11.0 stable + same prefix, isolating
  the problem to Dukascopy-demo-mt5-1 specifically (untested further — not
  investigated *why* that broker fails, since MetaQuotes-Demo unblocks
  everything downstream).

- **Fase 2 (detectores deterministas): DONE, validated against real data.**
  `scripts/visualize_detectors.py` plots every detector's output (OB/IOB,
  FVG/IFVG, BOS/CHoCH, sweeps, equal highs/lows, turtle soup, sharp turns,
  S/R, Elliott context) over a real candlestick chart using the exact same
  `TimeframeAnalysis` code the pipeline runs. Run against 500 real EURUSD H1
  and 500 real EURUSD M15 candles (2026-09-15), charts sent to the user for
  visual review — no code issues found running against real data beyond what
  the 55 synthetic-data tests already cover.

- **Fase 3 (pipeline multi-timeframe): DONE, validated against real data.**
  `scripts/analyze_once.py` ran the full D1->H4->H1->M15 pipeline against
  live data for all 3 configured symbols (2026-09-15): EURUSD progressed all
  the way to the risk stage before a legitimate R:R rejection (proves every
  earlier stage — bias, session, POI, confirmation, confluence — works
  end-to-end on real data), GBPUSD/USDJPY were correctly rejected early at
  H4_confirmation (H4 bias contradicting D1, exactly the spec's invalidation
  rule). `logs/signals.jsonl` has the structured log with stage+reason for
  every outcome, the Fase 3 deliverable.

- **Fases 4-7 (backtesting, walk-forward, paper trading, ML): not started.**
  Fase 4 (backtesting engine over 2-5 years of real history) is the natural
  next step now that Fases 1-3 are confirmed live.

## Not wired up yet, deliberately

Nothing in this repo calls
`open_position`/`close_position`/`modify_sl` automatically. The gateway
client exposes those methods, but turning `TradingSignal`s into live orders
and running a continuous loop is a separate step that should follow the
project's established practice of validating on a clean backtest and a
forward demo before trusting a strategy with real (even demo) capital — see
the parent project's memory of the September pivot for why that discipline
matters here specifically.

## Layout

```
trader/
  config.py, config.yaml      # typed config, loaded from YAML + env (gateway creds)
  events.py                   # MarketEvent model + causal timeframe-alignment helpers
  sessions.py                 # Asia/London/NY session classification, DST-aware
  gateway_client.py           # HTTP client for ../PythonGetaway (candles, ticks, trading)
  detectors/
    structure.py               # swings, BOS/CHoCH
    order_blocks.py             # OB (scans back from a confirmed displacement break)
    fvg.py                       # Fair Value Gap (confirmed at C3's close)
    liquidity.py                 # sweeps, equal highs/lows, turtle soup, sharp turns
    support_resistance.py        # classic S/R, confirmed at the Nth touch
    elliott.py                   # causal ZigZag + Fibonacci-ratio impulse context
    common.py                    # zone mitigation/inversion (OB->IOB, FVG->IFVG)
    indicators.py                # causal ATR
  risk/
    levels.py                   # SL/TP from structure, R:R gate
    management.py                # partial + breakeven + invalidation state machine
  pipeline/
    confluence.py                # confluence family/timeframe counting
    engine.py                    # MultiTimeframePipeline: enforces the D1->H4->H1->M15 order
  signal.py                    # TradingSignal + the report format from the spec
  logging_.py                  # structured JSONL log of every signal / no-signal
scripts/
  smoke_test_connection.py     # Fase 1 deliverable: prints last N candles, read-only
  visualize_detectors.py       # Fase 2 deliverable: detector zones plotted on a real chart
  analyze_once.py              # full pipeline pass per configured symbol, read-only
tests/                        # 55 tests, synthetic candles hand-verified by calculation
```

## Design notes worth knowing before touching this

- **Timestamp identity, never row position.** Every `MarketEvent` is keyed by
  `timestamp`. Nothing here passes a row index computed against one
  DataFrame into a function operating on a different one — that exact
  pattern caused a real, costly look-ahead bug in this project's earlier ML
  research line (see the parent project's memory).
- **Causal by construction, not by convention.** `closed_candles_as_of()`
  strips any candle not yet fully closed as of `as_of` before it reaches a
  detector. Swings confirm `right` bars after the pivot. FVGs confirm at C3's
  close. S/R confirms at the Nth touch. Every detector has a test asserting
  the *confirmation timestamp*, not just the final event set.
- **Broker time vs. UTC.** PythonGetaway deliberately does not convert MT5
  timestamps (`time` is broker-server wall clock, commonly UTC+2/3, not
  UTC — see its own README/converters.py). `gateway_client.server_utc_offset()`
  measures the live offset from a real tick and corrects every candle before
  it reaches the session classifier, which is DST-aware and needs true UTC.
- **numpy bool arithmetic.** `elliott.py` sums rule flags as plain Python
  `bool`/`int`, not raw numpy booleans — `np.bool_ + np.bool_` is a logical
  OR, not addition, and silently produced wrong confidence scores in an
  earlier sibling implementation.

## Running

```bash
cd trader
./.venv/bin/pip install -r requirements.txt   # already done if .venv exists
./.venv/bin/pytest tests/ -q

# credentials in trader/.env (gitignored). The gateway itself
# (../PythonGetaway/scripts/start.sh) is NOT part of this folder and hasn't
# been started from here -- ask before starting it, it logs into a real
# demo brokerage account and spins up Wine/MT5 processes.
./.venv/bin/python scripts/smoke_test_connection.py
./.venv/bin/python scripts/visualize_detectors.py --symbol EURUSD --timeframe H1
./.venv/bin/python scripts/analyze_once.py
```
