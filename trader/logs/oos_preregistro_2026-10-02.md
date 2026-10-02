# Prueba fuera de muestra pre-registrada — sistema MTF tal como corre hoy

**Fecha del pre-registro:** 2026-10-02 (hora de Sofía), escrito ANTES de evaluar un solo dato del período OOS.
**Uso único:** una vez corrida, la ventana OOS queda quemada. Ningún resultado de acá puede usarse para ajustar reglas y volver a llamarse "fuera de muestra".
**Regla de edición:** la Parte 1 no se edita después de ver resultados. Los resultados van en la Parte 2. Cualquier desvío se anota en la Parte 2 como desvío, nunca reescribiendo la Parte 1.

---

## PARTE 1 — PRE-REGISTRO

### 0. Versión congelada

- **Repo:** `/home/borislav/Desktop/TradingAI`, rama `fix/mtf-pilot-execution-bugs`.
- **Commit base de las reglas:** `35976b2b2bc06267f43036b6258287c5e5058609`. El commit que agrega este documento y el harness se anota en la Parte 2; no cambia ninguna regla.
- **Archivos congelados (sha256, primeros 16 caracteres):**

| Archivo | Rol | sha256 |
|---|---|---|
| `scripts/oos_prereg_2026_10_02.py` | harness de esta prueba (ventanas, orquestación) | `41f6efd718eba06d` |
| `scripts/honest_reevaluation.py` | escenarios pesimistas y bootstrap por bloques | `3850b3cf2244d374` |
| `scripts/measure_frequency_10_symbols_daily_loss.py` | `_simulate`: conflicto, concurrencia, corte diario | `7b73d96b372a2f24` |
| `scripts/measure_frequency_10_symbols.py` | `_scan_one`, `_load_clipped` | `aad8024d905fb10b` |
| `scripts/measure_pending_order_frequency_session_end.py` | `_scan`, `MIN_NET_RR` | `c02faf0c8acc0416` |
| `trader/mtf_strategies/trade_simulation.py` | motor de llenado y salida | `8d9816178551ea57` |
| `trader/mtf_strategies/continuation.py` | Estrategia 1 | `4b2d3bdef0710d77` |
| `trader/mtf_strategies/reversal.py` | Estrategia 2 | `6d20d8d6f8702dc8` |
| `trader/mtf_strategies/session_risk.py` | sesiones, corte diario, concurrencia | `6b43a7e2467179f9` |
| `trader/mtf_strategies/targets.py` | TP2 por prioridad | `38a4791625f2f6b7` |
| `trader/mtf_strategies/conflict.py` | conflicto E1/E2 | `c8ed3aed6c6067b7` |
| `trader/mtf_strategies/analysis.py` | swings, FVG y MSS causales (`as_of`) | `7e50a5a20046032a` |
| `trader/risk/levels.py` | geometría de niveles y chequeo de precio | `54c6166db104a370` |
| `trader/backtest/costs.py` | costo de spread en R | `034ee801ba4ee5f1` |
| `config.yaml` | parámetros de detectores | `8c4a422bb26d23cf` |

- **Datos:** 30 CSV (10 símbolos × M15/H1/D1) en `data/quant_battery/`, más `symbol_info_full.json` y `logs/spread_source_verification.json`. Los sha256 completos están en `logs/oos_data_hashes_2026-10-02.txt`. Son los CSV bajados el 2026-09-25 19:44, ya con el fix de offset de fin de semana.

### a) Reglas exactas a evaluar (referencias al código, no resumen)

Pipeline del backtest, el mismo que produjo los +0.474R y los +0.330R pesimistas:

1. **Universo:** `scripts/run_mtf_pilot.py:83` → `SYMBOLS` (10 pares), repetido en `scripts/oos_prereg_2026_10_02.py:39`.
2. **Ventanas de entrada:** `trader/mtf_strategies/session_risk.py:42-43`, Londres y NY de 08:00 a 11:00 hora local, DST-aware. `active_entry_session` en `session_risk.py:49`.
3. **Detección de señales (causal, vela a vela):** `scripts/measure_pending_order_frequency_session_end.py:75` (`_scan`), que llama a:
   - `evaluate_continuation(..., require_retracement=False)`, `trader/mtf_strategies/continuation.py:160`;
   - `evaluate_reversal(..., require_retracement=False)`, `trader/mtf_strategies/reversal.py:192`;
   - niveles y R:R: `trader/mtf_strategies/targets.py:34` (`compute_levels_with_priority_tp2`) → `trader/risk/levels.py:20` (`compute_trade_levels`);
   - parámetros de los detectores: `config.yaml`, según el hash de arriba.
4. **Deduplicado** por (símbolo, estrategia, setup, dirección, sweep, MSS, FVG): `scripts/oos_prereg_2026_10_02.py:55-60`, idéntico a `measure_frequency_10_symbols.py:66-71`.
5. **Gate de R:R neto de costo ≥ 2.0:** `MIN_NET_RR` en `measure_pending_order_frequency_session_end.py:43` (igual a `run_mtf_pilot.py:90`). El costo sale de `trader/backtest/costs.py:35,48`: spread medio del campo de las velas M15 hasta el fin del escaneo.
6. **Simulación de cartera:** `scripts/measure_frequency_10_symbols_daily_loss.py:46` (`_simulate`, `apply_daily_loss=True`, `temporal_exit=False`):
   - conflicto E1/E2 en el mismo símbolo y momento → ambas se descartan;
   - concurrencia de 1 por símbolo, contando también la orden pendiente;
   - corte diario de −1.5R (`session_risk.py:45`), que bloquea entradas nuevas y cancela las pendientes.
7. **Orden y salida:** `trader/mtf_strategies/trade_simulation.py:137` (`simulate_pending_order_outcome`) y `:53` (`_simulate_close_from_entry`):
   - **Orden:** límite en la entrada (50% del FVG), colocada en `generated_at` solo si el precio no la cruzó (`levels.py:47`), y vence al fin de la sesión (`session_risk.py:62`).
   - **Gestión de la posición:** 50% en TP1 (+1R) con SL a breakeven y el resto a TP2. **Sin salida temporal.**
   - **Convenciones de OHLC:** si una vela toca SL y TP, se asume SL primero; la vela de entrada no cuenta.
8. **R neto** = R bruto − costo en R (`_simulate`, `cost_r = risk_reward − net_rr`).

**Diferencias conocidas entre el vivo y el backtest** (se declaran, no se corrigen):
- La cancelación por sesgo 4H de órdenes pendientes no está modelada.
- El gate de costo usa el spread del campo de vela (el mínimo de la vela), no los ticks en vivo.
- El chequeo de precio usa la apertura de la vela, no el bid/ask en vivo.

### b) Lo que NO se incluye

- **El filtro R:R neto ≥ 3 NO se aplica.** Se evalúa el sistema de producción actual (gate R:R neto ≥ 2.0), sin el hallazgo del reanálisis.
- Ningún parámetro nuevo: ni símbolos, ni sesiones, ni umbrales, ni gestión.

### Ventana fuera de muestra

- **Señales evaluadas:** `generated_at` dentro de **[2026-01-01 00:00 UTC, 2026-06-19 23:59:59 UTC]** (~170 días).
- **Por qué termina el 19 de junio:** ningún análisis del linaje MTF tocó ese tramo. Las fechas más tempranas en todos los logs MTF son: `mtf_scan_result.json` 2026-06-20; `mtf_pending_order_frequency*.json` 2026-06-20; `mtf_frequency_10_symbols.json` 2026-06-26. Desde el 2026-06-20 los datos ya se vieron.
- **Escaneo:** los datos se recortan al 2026-06-19 23:59:59 (causal), con un período de 6 meses y filtro de inicio el 2026-01-01.
- **Resolución:** para resolver operaciones que sigan abiertas al cierre se usan velas hasta el 2026-06-29. Solo es mecánica, no hay decisión de diseño.

**Contaminación indirecta, que no se puede eliminar y se declara:**
- **Linaje anterior:** el pipeline "reformed" (`logs/reformed_trades.jsonl`, de 2024-09 a 2026-09) evaluó conceptos ICT relacionados (barridas de liquidez, FVG) sobre este período, con reglas distintas.
- **Baterías cuantitativas y de TA clásica** (otras familias de estrategias) usaron estos datos.
- **Efecto:** las reglas MTF no se ajustaron contra este período, pero el investigador no llega "ciego" a esta familia de conceptos en estos meses.

### c) Umbrales de decisión, fijados ahora

- **Métrica primaria:** R medio por operación en el **escenario pesimista completo** (SL en la vela de entrada y lado ask, con el mayor promedio de ticks por símbolo), con **IC 90% por bloques diarios**.
- **Métrica secundaria:** el mismo cálculo con el motor actual. Solo se reporta, no decide.
- **Mínimo de muestra:** si el escenario pesimista tiene **n < 30** operaciones, el resultado se declara **INSUFICIENTE** y no se concluye nada.
- **Clasificación**, mutuamente excluyente, en este orden:

| Resultado | Condición (escenario pesimista) | Lectura |
|---|---|---|
| **A — SE REPLICA** | límite inferior del IC 90% > 0 | El edge aparece fuera de muestra. Si el límite superior es menor que +0.33R, se anota "positivo pero menor que lo estimado". |
| **B — NO SE REPLICA** | R medio ≤ 0, **o** (límite inferior ≤ 0 **y** límite superior < +0.33R) | Inconsistente con el +0.33R de diseño. El edge medido era sobreajuste o ruido. |
| **C — INCONCLUSO** | límite inferior ≤ 0, R medio > 0 y límite superior ≥ +0.33R | Compatible tanto con 0 como con +0.33R. Se necesitan más datos (piloto en vivo). |

- **Potencia, declarada antes:** con n ≈ 55–65 y una desviación de ~1.37R por operación, el IC 90% mide ±0.29R. Si el valor real fuera +0.33R, la probabilidad de caer en A es solo ~55%, así que un C no es evidencia en contra.

### d) Métricas que se reportan

Para cada escenario (actual, solo SL en la vela de entrada, solo lado ask, pesimista completo):
- R medio por operación;
- IC 90% por bloques diarios: 10 000 remuestreos de días de `generated_at` con reemplazo, percentiles 5 y 95, seed 0 (`honest_reevaluation.py:102,113`);
- n de operaciones, operaciones por día, días con operaciones;
- R total;
- probabilidad de R > 0.

Además, el embudo: setups únicos y candidatos con R:R neto ≥ 2.

**Spread pesimista:** el MAYOR `tick_spread_mean` por símbolo entre las dos ventanas de `logs/spread_source_verification.json` (`honest_reevaluation.py:45`). Es el mismo valor que en el reanálisis.

**Exploratorio, solo si se reporta:** el subconjunto R:R neto ≥ 3, etiquetado **"exploratorio, no confirmatorio"**.

### Validación del harness, hecha sin mirar la ventana OOS

`--window insample` sobre la ventana de diseño, ya vista. Reproduce exactamente los resultados del reanálisis:
- 71 setups únicos y 69 candidatos;
- motor actual: n=34, +0.474R, IC 90% [+0.136, +0.848];
- pesimista: n=31, +0.330R, IC 90% [−0.071, +0.779].

El log está en `logs/oos_harness_insample_check.log`. Esto confirma que el harness corre el mismo pipeline.

### Punto pendiente de confirmación: origen de los datos

El pedido dice "descargar enero–junio vía el gateway". **Los datos ya están en disco**: son los CSV del 2026-09-25 con el fix de offset, y sus hashes están arriba.

- **Propuesta:** usar esos CSV y no volver a bajar nada.
- **Motivo:** el piloto en vivo usa la misma cuenta y el gateway comparte la sesión entre logins. Un login extra en paralelo (y sobre todo un logout) puede tumbar la sesión del piloto; ya pasó 2 veces.
- **Alternativa, si preferís bajar de nuevo:** un único login, sin logout al terminar, y comparar los hashes contra estos CSV antes de correr.

---

## PARTE 2 — RESULTADOS

**Corrida:** 2026-10-02, entre 11:45 y 12:30 aprox. (hora de Sofía), una sola vez.
- **Confirmación previa del usuario:** pre-registro aprobado y uso de los CSV que ya estaban en disco, sin descarga nueva.
- **Verificación antes de correr:** los 32 sha256 de datos y el del harness (`41f6efd718eba06d`) coincidían con la Parte 1, y no había cambios sin commitear en el código.
- **Comando:** `scripts/oos_prereg_2026_10_02.py --window oos`. Resultado crudo en `logs/oos_result_oos_2026-10-02.json`, log en `logs/oos_run_2026-10-02.log`.

### Resultados — pre-registro y resultado lado a lado

| | Ventana de diseño (jun–sep 2026, in-sample) | **OOS (6 ene – 18 jun 2026)** |
|---|---|---|
| Setups únicos → candidatos con R:R neto ≥ 2 | 71 → 69 | 118 → 112 |
| **Motor actual**: n / ops por día | 34 / 0.374 | 53 / 0.327 |
| R medio | +0.474R | **+0.074R** |
| IC 90% por bloques diarios | [+0.136, +0.848] | [−0.228, +0.402] |
| Probabilidad de R > 0 | 0.99 | 0.65 |
| Solo SL en la vela de entrada | +0.474R | +0.074R, IC [−0.225, +0.404] |
| Solo lado ask | +0.330R | +0.080R, IC [−0.247, +0.431] (n=50) |
| **Pesimista completo (métrica primaria)**: n | 31 | **50** |
| R medio | +0.330R | **+0.080R** |
| IC 90% por bloques diarios | [−0.071, +0.779] | **[−0.254, +0.430]** |
| Probabilidad de R > 0 | 0.91 | 0.64 |
| R total | +10.23R | +4.02R |

### Clasificación según la regla pre-registrada (escenario pesimista)

- n = 50 ≥ 30, así que la muestra es suficiente.
- Límite inferior del IC 90% = −0.254 ≤ 0, así que **no es A**.
- R medio = +0.080 > 0 y límite superior = +0.430 ≥ +0.33, así que **no es B**.
- **→ RESULTADO C — INCONCLUSO.** El dato es compatible tanto con 0 como con +0.33R.

### Lectura (no cambia la clasificación)

- **El edge no apareció fuera de muestra con la fuerza del diseño.**
  - La estimación puntual cae de +0.33R a **+0.08R** en el pesimista, y de +0.47R a +0.07R con el motor actual.
  - El win rate baja del ~65% al ~50%.
  - El límite superior (+0.43R) apenas incluye el valor de diseño. El resultado está mucho más cerca de "sin edge" que de "edge confirmado".
- **Lo que sí se replicó:** la frecuencia (0.31–0.33 operaciones por día, frente a 0.34–0.37) y el bajo impacto del SL en la vela de entrada.
- **Concentración:** EURUSD (+6.32R) y GBPUSD (+5.09R) sostienen casi todo el total; 6 de los 10 símbolos dan negativo. Es un descriptivo, no un hallazgo.
- **Por estrategia y sesión** (descriptivo, motor actual): continuación +0.17R (n=23), reversión 0.00R (n=30); Londres +0.15R (n=29), Nueva York −0.02R (n=24).

### Exploratorio, NO confirmatorio: subconjunto R:R neto ≥ 3

El filtro salió de la ventana de diseño, así que esto no es una prueba limpia de esa hipótesis.

| | n | R del subconjunto | R del resto | Diferencia, IC 90% |
|---|---|---|---|---|
| Motor actual | 20 / 33 | +0.156 | +0.025 | [−0.449, +0.792] |
| Pesimista | 18 / 32 | +0.126 | +0.055 | [−0.552, +0.797] |

En diseño el subconjunto daba +1.07R frente a +0.11R del resto. **Fuera de muestra la ventaja desaparece:** +0.13R frente a +0.06R, una diferencia indistinguible de 0. El hallazgo del reanálisis no se sostiene. Archivo: `logs/oos_exploratory_rr3_2026-10-02.json`.

### Desvíos e incidencias

- **Ningún bug técnico del motor durante la corrida.** No hubo operaciones sin resolver y las señales caen dentro de la ventana (primera 2026-01-06, última 2026-06-18).
- **Cosmético:** el log de la corrida muestra la línea de progreso de solo 6 de los 10 símbolos, por el buffer de stdout de los procesos worker. Los 10 símbolos tienen operaciones en el resultado, así que no afecta los números.
- **Ninguna regla ni parámetro se tocó.** La ventana OOS queda quemada.
