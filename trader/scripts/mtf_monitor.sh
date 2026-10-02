#!/bin/bash
# Monitor del piloto MTF: cada línea de stdout = un aviso.
# - Eventos MT5 (colocación/llenado/cierre con USD/parcial/modificaciones/externos/errores).
# - Watchdog cada 60s. Los relanzamientos del piloto los hace systemd
#   (mtf-pilot.service, Restart=on-failure, máx 3/h); acá solo se distingue y se avisa:
#   código de salida 0 = parada pedida (stop flag / systemctl stop), != 0 = caída por error,
#   unidad "failed" = systemd dejó de relanzar (crashloop). Gateway caído y heartbeat viejo -> ALERTA.
# - En cada position_closed: scripts/live_evidence_ledger.py --enforce (R por símbolo + criterio de abandono).
cd /home/borislav/Desktop/TradingAI/trader || exit 1
LOG=logs/mtf_pilot_events.txt
KINDS='pilot_started|pending_order_placed|pending_order_filled|pending_order_expired_or_cancelled|pending_order_cancelled_proactively|pending_order_rejected|position_closed|tp1_partial_and_breakeven|tp1_hit_full_close|_modified_external|external_trade_detected|daily_loss_lockout|_failed|tick_error|pilot_stopping|pilot_crashed|pilot_terminated_by_signal|session_closed'
UNIT=mtf-pilot.service

# classify_pilot <ActiveState> <SubState> <ExecMainStatus>  -> running|stopped|already_running|crashloop|crashed
classify_pilot() {
  [ "$2" = auto-restart ] && { echo crashed; return; }  # caído, esperando RestartSec
  case "$1" in
    active|activating|reloading|deactivating) echo running ;;
    failed) echo crashloop ;;
    *) case "$3" in 0) echo stopped ;; 3) echo already_running ;; *) echo crashed ;; esac ;;
  esac
}
[ -n "$MTF_MONITOR_LIB" ] && return 0  # tests: solo cargar la función

# en cada cierre: ledger + criterio de abandono pre-registrado (--enforce para el piloto si se dispara)
on_event() {
  while IFS= read -r line; do
    echo "$line"
    [[ $line == *position_closed* ]] && OMP_NUM_THREADS=1 .venv/bin/python scripts/live_evidence_ledger.py --enforce 2>&1
  done
}
tail -F -n0 "$LOG" 2>/dev/null > >(grep -E --line-buffered "$KINDS" | on_event) &
TAIL_PID=$!  # pid real del tail (con una tubería `|` el $! sería el grep y el tail quedaría huérfano)
trap 'echo "WATCHDOG: señal de terminación recibida ($(TZ=Europe/Sofia date +%F_%T) hora Sofía) -- watchdog saliendo, el piloto queda SIN vigilancia"; kill $TAIL_PID 2>/dev/null; pkill -P $$ 2>/dev/null; exit 0' TERM INT HUP

state=""
restarts=$(systemctl --user show "$UNIT" -p NRestarts --value 2>/dev/null || echo 0)
alert() { [ "$state" != "$1" ] && echo "ALERTA WATCHDOG: $2"; state=$1; }

while true; do
  sleep 60 & wait $!  # wait: la señal se atiende al instante, no al terminar el sleep
  if ! curl -s -m 10 localhost:8000/health | grep -q '"status":"ok"'; then
    alert gw "gateway (puerto 8000) no responde -- MT5 sin conexión, el piloto no puede operar"
    continue
  fi
  props=$(systemctl --user show "$UNIT" -p ActiveState -p SubState -p ExecMainStatus -p NRestarts 2>/dev/null)
  get() { sed -n "s/^$1=//p" <<<"$props"; }
  active=$(get ActiveState); sub=$(get SubState); status=$(get ExecMainStatus); n=$(get NRestarts)
  if [ "${n:-0}" -gt "${restarts:-0}" ]; then
    echo "ALERTA WATCHDOG: systemd relanzó el piloto tras una caída (reinicio #$n desde que arrancó la unidad)"
  fi
  restarts=${n:-0}
  case $(classify_pilot "$active" "$sub" "$status") in
    stopped) alert stopped "piloto detenido intencionalmente (salida 0: parada pedida) -- no se relanza"; continue ;;
    already_running) alert dup "el piloto no arrancó: otra instancia tenía el lock (salida 3)"; continue ;;
    crashloop) alert crashloop "piloto cayó 3+ veces en 1h, systemd dejó de relanzarlo -- revisar logs/mtf_pilot_stdout.log y 'systemctl --user status $UNIT'"; continue ;;
    crashed) alert crashed "piloto CAÍDO POR ERROR (salida $status) -- systemd lo relanza en 30s; últimas líneas: $(tail -3 logs/mtf_pilot_stdout.log | tr '\n' ' ')"; continue ;;
  esac
  last=$(grep "heartbeat" "$LOG" | tail -1 | sed -E 's/^\[([^]]+)\].*/\1/')
  age=$(( $(date +%s) - $(date -d "$last" +%s 2>/dev/null || echo 0) ))
  if [ "$age" -gt 2700 ]; then
    alert stale "sin heartbeat hace $((age/60)) min (proceso vivo pero posiblemente colgado)"
  else
    state=""
  fi
done
