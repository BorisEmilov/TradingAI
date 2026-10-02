# Piloto demo hasta ~50 operaciones: criterio de abandono pre-registrado

**Fecha:** 2026-10-02 (hora de Sofía), fijado ANTES de ver más operaciones en vivo.
**Alcance:** monitoreo y reporte. Ninguna regla de entrada o salida cambia, no se aplica el filtro R:R ≥ 3 y no se reinicia el piloto salvo que se dispare el abandono o haya un fallo técnico real.

## Serie que se evalúa

- **Fuente:** `logs/mtf_pilot_events.jsonl` (append-only), procesado por `scripts/live_evidence_ledger.py`, que genera `logs/live_evidence_ledger.csv` y `logs/live_pilot_status.json`.
- **Serie del criterio:** las operaciones CERRADAS bajo las reglas vigentes:
  - `v4` desde 2026-09-29 06:49 UTC: chequeo de precio antes de enviar;
  - `v5` desde 2026-10-02 07:01 UTC: v4 más los fixes de ejecución y systemd, **mismas reglas**.
- **Excluidas:** las operaciones marcadas como usadas en diseño (EURUSD 22-sep, v1) y las de versiones con otras reglas (v1–v3).
- **Punto de partida:** n = 2 (AUDUSD −1R el 30-sep y NZDUSD −1R el 1-oct). Se incluyen porque corrieron con las mismas reglas de entrada y salida, y porque ninguno de los bugs corregidos en v5 las afectó: ambas cerraron por SL, sin parcial. Incluirlas hace más probable disparar el abandono, no menos.

## Criterio de abandono (se evalúa en cada cierre)

| # | Condición | Acción |
|---|---|---|
| 1 | n ≥ 20 y R medio < −0.3R | Detener el piloto (stop flag) y reportar de inmediato |
| 2 | n ≥ 20 y el límite superior del IC 90% por bloques diarios < 0 | Igual |
| — | Ninguna se dispara | Seguir hasta n = 50 y reevaluar con el mismo formato que el reanálisis y la prueba fuera de muestra: R medio, IC 90% por bloques diarios, probabilidad de R > 0 y win rate |

- **IC 90%:** mismo método que `honest_reevaluation.py`. Se remuestrean días (de colocación de la orden) con reemplazo, 10 000 veces, percentiles 5 y 95, seed 0.
- **Automático:** el watchdog (`scripts/mtf_monitor.sh`, servicio `mtf-monitor`) corre `live_evidence_ledger.py --enforce` en cada `position_closed`. Si se dispara una condición, escribe la stop flag: el piloto termina su ciclo y sale con código 0, y las posiciones abiertas siguen con su SL y TP reales. Además emite "ALERTA ABANDONO" en `logs/mtf_monitor.log`.
- **Lógica testeada** en `tests/test_live_evidence_ledger.py`.

## Registro por operación, desde la operación 1

En cada cierre se registra:
- símbolo, dirección y estrategia;
- sesión (Londres o Nueva York);
- R:R objetivo bruto y neto de costo (a TP2);
- resultado en R y en USD, y motivo del cierre;
- versión del sistema;
- **R acumulado por símbolo**, que se muestra en cada actualización. Es solo descriptivo: no se actúa sobre la concentración por símbolo sin validarla aparte.

**Llenado marginal ("casi no se llena")** — definición, igual que el escenario pesimista:
- **Qué cuenta:** la penetración del precio más allá de la entrada límite, desde la colocación hasta el llenado, en velas M15 bid. Para un BUY_LIMIT, `entrada − mínimo`; para un SELL_LIMIT, `máximo − entrada`.
- **Cuándo es marginal:** si esa penetración es menor que el spread pesimista del símbolo, el mayor promedio de ticks de `logs/spread_source_verification.json`.
- **Limitación conocida:** el piloto no guarda las velas del momento del llenado, y no se hace un login en paralelo para bajarlas, porque la sesión es compartida con el piloto. Queda como `pendiente_velas` hasta poder calcularlo (ver la Parte 2).

## Lo que NO dispara nada

Ni la concentración por símbolo, ni el resultado por sesión, ni el de llenados marginales: se reportan y no se actúa sobre ellos.

---

## PARTE 2 — Seguimiento

*Se agrega abajo, sin editar lo anterior.*
