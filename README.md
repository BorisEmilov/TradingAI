# TradingAI

Automated trading agent for Forex and metals on MetaTrader 5 (**demo** account), plus a
users/plans backend intended to offer it as a service.

> Status as of **2026-09-26**: research and forward-testing phase on demo.
> **No strategy has a demonstrated statistical edge yet.** Everything is shut down.

---

## Components

```
TradingAI/
├── PythonGetaway/   HTTP gateway (FastAPI) over MT5 running under Wine
├── trader/          The agent: detectors, strategies, backtests, live pilots (Python)
├── api_backend/     Users, plans and auth API (.NET 10, SQL Server, Redis)
└── client_react/    Frontend (empty, not started)
```

| Folder | What it does | Details |
|---|---|---|
| `PythonGetaway/` | Exposes MT5 over REST: account, symbols, candles, ticks, positions, orders and trading (open, close, partial close, move SL/TP, pending orders). Per-user login with token, pool of MT5 terminals. Blocks real accounts unless `PYGW_ALLOW_REAL=1`. | [`PythonGetaway/README.md`](PythonGetaway/README.md) |
| `trader/` | Reads data from the gateway, detects SMC/ICT structure (swings, BOS/CHoCH, order blocks, FVG, liquidity sweeps, sessions), generates signals, backtests and runs pilots that trade the demo account. | [`trader/README.md`](trader/README.md) |
| `api_backend/` | JWT register/login with refresh, email verification and password change via link (MailKit), profiles, subscription plans with automatic expiration, roles (`FREE_USER`, `PLUS_USER`, `GOLDEN_USER`, `ADMIN`, `SUPER_ADMIN`), per-endpoint/per-role rate limiting. | `api_backend/src/ApiBackend.Api/` |
| `client_react/` | Future frontend. | — |

## How it works

```
 MT5 (Wine) ──► PythonGetaway (HTTP :8000) ──► trader/  ──► orders on the demo account
                                                  │
                                                  └─► logs/ (events, trades, reports)

 client_react ──► api_backend (users, plans, auth)      [not connected to the trader yet]
```

1. `PythonGetaway/scripts/start.sh` starts MT5 under Wine and the API.
2. The pilot (`trader/scripts/run_mtf_pilot.py`) performs **a single login** and keeps it alive
   with `refresh`; it never re-logs in a loop (the gateway shares the session between processes
   of the same account: a logout from any of them kills all of them).
3. Each cycle it analyzes 4H (bias) → 1H (liquidity/zones) → 15M (entry), places **limit
   orders** with structural SL/TP, minimum 2:1 R:R, daily loss cutoff at -1.5R, and manages
   partials/breakeven.
4. Every event is logged in `trader/logs/` (`mtf_pilot_events.jsonl`, state in `mtf_pilot_state.json`).
5. It stops cleanly with `trader/scripts/stop_mtf_pilot.py` (stop flag) and `PythonGetaway/scripts/stop.sh`.

## Current phase

**Active strategy:** MTF SMC in `trader/trader/mtf_strategies/` — two strategies,
*Continuation* and *Reversal* (liquidity sweep → MSS → entry at FVG/OB), 10 FX pairs
(EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, EURGBP, GBPJPY).

- Mechanics implemented and validated: 309 tests, visual audit, costs using real tick spread.
- Measured frequency: ~0.38 trades/day (~2.6/week) across 10 pairs.
- Demo pilot relaunched and **stopped by user decision on 2026-09-25 21:00 UTC**,
  with no open positions or orders.
- Pending: relaunch to accumulate a live sample; revisit the time-based exit (removed on
  inconclusive evidence) if live results diverge.

**Already ruled out (with evidence):** ICT scoring and 4-layer systems (worse than
random), GBM/ML on ICT features, 60 quantitative tests (momentum/mean reversion/seasonality,
0 survive multiple-comparison correction) and a classic TA battery (did not replicate on TEST).

**RSI XPTUSD pilot** (RSI-14 30/70 on platinum, pre-registered rules in
`trader/logs/pilot_rsi_xpt_prereg_2026-09-20.md`): the process stopped logging events on
2026-09-20 14:06 UTC and is not running. Relaunch manually to continue.

**Backend:** in development (auth, plans, email, rate limiting); does not expose trader data yet.

## Project rules

- Demo account only. Nothing touches real money.
- Nothing is considered an edge without out-of-sample validation (TRAIN/VALIDATION/TEST, random
  baseline, multiple-comparison correction).
- Secrets stay out of the repo: `trader/.env`, the backend's `appsettings.json` (see `.gitignore`).
