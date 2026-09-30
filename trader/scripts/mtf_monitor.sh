#!/bin/bash
# Monitor del piloto MTF: cada línea de stdout = un aviso.
# - Eventos MT5 (colocación/llenado/cierre con USD/parcial/modificaciones/externos/errores).
# - Watchdog cada 60s: piloto muerto -> ALERTA + relanzamiento automático (salvo parada
#   intencional: último evento de sistema = session_closed/pilot_stopping); gateway caído
#   -> ALERTA (no se relanza solo: arrancar MT5/Wine es pesado); heartbeat viejo -> ALERTA.
# ponytail: máx 3 relanzamientos por hora, después solo alerta (evita bucle de crash).
cd /home/borislav/Desktop/TradingAI/trader || exit 1
LOG=logs/mtf_pilot_events.txt
KINDS='pilot_started|pending_order_placed|pending_order_filled|pending_order_expired_or_cancelled|pending_order_cancelled_proactively|pending_order_rejected|position_closed|tp1_partial_and_breakeven|tp1_hit_full_close|_modified_external|external_trade_detected|daily_loss_lockout|_failed|tick_error|pilot_stopping|session_closed'
PILOT_RE='[.]venv/bin/python -u scripts/run_mtf_pilot.py'

tail -F -n0 "$LOG" 2>/dev/null | grep -E --line-buffered "$KINDS" &

restarts=()
state=""
alert() { [ "$state" != "$1" ] && echo "ALERTA WATCHDOG: $2"; state=$1; }

while true; do
  sleep 60
  if ! curl -s -m 10 localhost:8000/health | grep -q '"status":"ok"'; then
    alert gw "gateway (puerto 8000) no responde -- MT5 sin conexión, el piloto no puede operar"
    continue
  fi
  if ! pgrep -f "$PILOT_RE" >/dev/null; then
    if grep -E "ALL +system" "$LOG" | tail -1 | grep -qE "session_closed|pilot_stopping"; then
      alert stopped "piloto detenido intencionalmente (parada limpia) -- no se relanza"
      continue
    fi
    now=$(date +%s); recent=()
    for t in "${restarts[@]}"; do [ $((now - t)) -lt 3600 ] && recent+=("$t"); done
    restarts=("${recent[@]}")
    if [ ${#restarts[@]} -ge 3 ]; then
      alert crashloop "piloto murió 3+ veces en 1h -- relanzamiento automático suspendido, revisar logs/mtf_pilot_stdout.log"
      continue
    fi
    echo "ALERTA WATCHDOG: el piloto murió sin parada limpia -- últimas líneas: $(tail -3 logs/mtf_pilot_stdout.log | tr '\n' ' ') -- relanzando"
    nohup .venv/bin/python -u scripts/run_mtf_pilot.py >> logs/mtf_pilot_stdout.log 2>&1 < /dev/null & disown
    restarts+=("$now"); state=""
    continue
  fi
  last=$(grep "heartbeat" "$LOG" | tail -1 | sed -E 's/^\[([^]]+)\].*/\1/')
  age=$(( $(date +%s) - $(date -d "$last" +%s 2>/dev/null || echo 0) ))
  if [ "$age" -gt 2700 ]; then
    alert stale "sin heartbeat hace $((age/60)) min (proceso vivo pero posiblemente colgado)"
  else
    state=""
  fi
done
