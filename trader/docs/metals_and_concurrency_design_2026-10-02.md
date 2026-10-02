# BORRADOR — Más frecuencia: metales y concurrencia 2 por símbolo

**Estado:** EN ESPERA. Solo preparación; nada de esto corre ni se construye todavía.
**Fecha:** 2026-10-02.

**Condición para activarlo:** el piloto demo llega a **n ≥ 50** en la serie de reglas vigentes sin disparar el criterio de abandono (`logs/live_pilot_prereg_2026-10-02.md`), **y** la reevaluación en n=50 muestra edge. Después de eso, el usuario da luz verde explícita escribiendo `logs/metals_greenlight.txt`.

**Si el piloto dispara el abandono, este plan se CANCELA, no se pospone.** No tiene sentido multiplicar la exposición sobre una base sin edge.

**Por qué espera:**
- La prueba fuera de muestra pre-registrada dio +0.08R pesimista, con IC 90% [−0.254, +0.430]: inconcluso y cerca de cero.
- El resultado se concentró en 2 de 10 símbolos.
- Sumar símbolos o concurrencia hoy multiplicaría la exposición, no el edge.

---

## Parte A — Metales (XAUUSD, XAGUSD)

### Qué quedó preparado (sin ejecutar)

`scripts/metals_pipeline.py`, con 4 subcomandos:
- `gate`: solo informa el estado del bloqueo.
- `fetch`: baja los datos.
- `spread`: mide el spread real de ticks.
- `backtest`: corre el backtest.

Todos salvo `gate` se niegan a correr sin las dos condiciones. Está testeado en `tests/test_metals_pipeline_gate.py`; con `n=2/50` sale con código 2 sin hacer login.

| Paso | Qué hace | Reutiliza |
|---|---|---|
| `fetch` | D1, H1 y M15 vía gateway, **un login y sin logout** (la sesión es compartida con el piloto). Verifica que no haya velas M15 en sábado, que es la huella del bug de offset. Guarda los CSV viejos como `*.pre_offset_fix.csv`. | el mismo chequeo que `refetch_10_symbols.py` |
| `spread` | Spread real de ticks en 2 ventanas → `logs/metals_spread.json` | `verify_spread_source._compare` (**sin** su logout) |
| `backtest` | El mismo pipeline que la prueba fuera de muestra de FX, con símbolos = metales. La ventana se pasa por argumentos obligatorios, que salen del pre-registro de metales. | `oos_prereg_2026_10_02.candidates/evaluate`, `honest_reevaluation` |

### Problemas conocidos que hay que resolver ANTES de correr (todos verificables, ninguno resuelto)

1. **Datos de metales inválidos hoy.** Los CSV de XAU y XAG en `data/quant_battery/` se bajaron el 2026-09-20 con el bug de offset de fin de semana, y además no hay H1. `fetch` los reemplaza.
2. **Sizing sospechoso de XAUUSD.**
   - **El dato:** `symbol_info_full.json` dice `trade_tick_value=0.1` con `tick_size=0.01` y `contract_size=100`. Con 100 oz por lote, un movimiento de 0.01 debería valer **1.0 USD**, no 0.1.
   - **El riesgo:** si el piloto usara 0.1, calcularía 10 veces más lotes de los debidos.
   - **Qué hacer:** verificarlo contra `/symbols/XAUUSD` en vivo y contra una operación demo mínima antes de cualquier uso en vivo. XAGUSD (5000 oz, 0.001 → 5.0 USD esperado, reporta 0.5) tiene la misma sospecha.
3. **Spread pesimista.**
   - **El problema:** `honest_reevaluation._spread_price` solo conoce FX: lee `spread_source_verification.json` y adivina el point por el sufijo JPY.
   - **Cómo lo resuelve el pipeline:** lo reemplaza en memoria con el point real y el spread medido en `metals_spread.json`, sin modificar el archivo compartido.
   - **Escala del spread:** el spread nominal del bróker es de 60 pts en XAU (0.60 USD) y 19 pts en XAG. Es mucho más alto que en FX en términos de R, y el gate de R:R neto ≥ 2 va a descartar más.
4. **Sesiones.** Se usan las mismas ventanas de Londres y Nueva York (08–11 hora local). Los metales tienen una pausa diaria del bróker (~22–23 UTC) que no cae dentro de esas ventanas. No hay cambio de regla.
5. **Correlación.** XAU y XAG están muy correlacionados entre sí y con el USD (EURUSD, GBPUSD). Sumarlos agrega menos diversificación de la que sugiere "+2 símbolos". Hay que medir la correlación de los R diarios con los FX en el backtest.
6. **Historia del usuario con el oro.** XAUUSD se sacó de un piloto anterior por decisión del usuario el 2026-08-25. Era otro linaje (GBM), pero hay que reconfirmar explícitamente antes de incluirlo.

### Ventana fuera de muestra para metales (se fija en su pre-registro, no ahora)

- **Disponible:** las reglas MTF nunca se evaluaron en metales, así que cualquier período sirve como fuera de muestra respecto de estas reglas.
- **Ya usado:** la ventana FX de enero a junio de 2026 está quemada solo para FX.
- **Contaminación indirecta a declarar:** las baterías cuantitativa y de TA clásica usaron XAG y XPT en D1/M15, y hubo un piloto de RSI en XPTUSD.
- **Criterio:** mismo formato y umbrales A/B/C que la prueba fuera de muestra de FX, escrito antes de bajar datos.

---

## Parte B — Concurrencia 2 por símbolo y conflicto E1/E2 (solo diseño)

### Cómo funciona hoy (código actual, sin tocar)

- **Conflicto** (`trader/mtf_strategies/conflict.py:19`, `resolve`): si continuación y reversión dan señal sobre el mismo símbolo en el mismo momento, **se descartan las dos**, sin prioridad. En la ventana de diseño eso costó 7 de 69 candidatos.
- **Concurrencia** (`trader/mtf_strategies/session_risk.py:84`, `can_open_new_trade`): como máximo 1 posición u orden pendiente por símbolo. En el piloto lo arma `live_state.concurrency_open_symbols` (`run_mtf_pilot.py:741-746`).
- **Backtest:** el mismo par de reglas está en `measure_frequency_10_symbols_daily_loss._simulate` (`open_until`, `conflicted`).

### Hecho que habilita el diseño

La cuenta es **RETAIL_HEDGING** (`margin_mode=2`). Dos posiciones del mismo símbolo conviven con tickets independientes, cada una con su propio SL y TP reales, y no se netean.

### Propuesta (a revisar antes de construir)

| Regla | Propuesta | Motivo |
|---|---|---|
| R1. Máximo por símbolo | 2 entre posiciones y órdenes pendientes, **como máximo 1 por estrategia** | Que la segunda plaza sea para la otra lógica, no para piramidar la misma |
| R2. Conflicto, misma dirección | Se ejecuta **una sola**: la de mayor R:R neto. Empate → ninguna. | Dos órdenes en la misma dirección y el mismo momento duplican el riesgo de una misma idea |
| R3. Conflicto, direcciones opuestas | **Se descartan las dos** (igual que hoy) | Señales contradictorias; hedgear el mismo símbolo solo paga spread |
| R4. Segunda entrada con la primera abierta | Solo si es de la **otra** estrategia y en la **misma** dirección, o si la primera ya está en breakeven (riesgo libre) | Limita el riesgo abierto por símbolo a 2 × 0.25% en el peor caso |
| R5. Tope de cartera | Riesgo abierto total ≤ 1.0% del equity (4 operaciones a 0.25%), contando las pendientes | Con 10–12 símbolos y concurrencia 2, el corte diario de −1.5R por sí solo no acota el peor día |
| R6. Corte diario | Sin cambios (−1.5R); cancela todas las pendientes | — |
| R7. Gestión | Sin cambios: cada posición tiene su TP1 parcial, breakeven y TP2. El piloto ya identifica cada una por ticket. | — |

### Qué tendría que cambiar en el código (cuando haya luz verde)

1. `conflict.resolve`: agregar R2 y R3 en lugar de "descartar todo".
2. `session_risk`: `PositionConcurrencyState` pasa de `set` de símbolos a un conteo por (símbolo, estrategia, dirección), y `can_open_new_trade` aplica R1, R4 y R5.
3. `live_state.concurrency_open_symbols` pasa a devolver ese conteo.
4. `_simulate` del backtest: el mismo cambio, para que el backtest siga representando el vivo.
5. **Tests:** cada regla R1–R5 con un caso que hoy pasa y otro que hoy falla, en el piloto y en el backtest.

### Cómo se validaría

- **Ganancia esperada:** pequeña. En la ventana de diseño, concurrencia más conflicto recuperaban unas +4–7 operaciones en 84 días (+0.08 por día). No es la palanca que lleva a 2 por día.
- **Prueba:** fuera de muestra pre-registrada, comparando la concurrencia 1 actual contra la propuesta R1–R7 una sola vez, sin iterar.
- **Ventana:** FX **anterior a 2026** (2024-10 a 2025-12; hay M15 desde 2024-09-19). Nunca la tocó el linaje MTF, pero sí el linaje "reformed", y hay que declararlo.
- **Criterio:** que la variante no empeore el R medio por operación (diferencia por bloques diarios con IC 90% que incluya ≥ 0) y que el drawdown máximo no suba más de un 25%.

---

## Lo que NO se hizo (a propósito)

- No se descargaron datos.
- No se corrió ningún backtest con metales.
- No se tocó el código de concurrencia ni el de conflicto, ni del piloto ni del backtest.
- El piloto en vivo no se tocó.
