#!/usr/bin/env bash
# Arranca el piloto completo: bridge MT5 (Wine) + terminal + un proceso run_live.py
# por cada simbolo listado en config.yaml (trading.symbols) que tenga checkpoint
# entrenado. Seguro de correr varias veces: si el bridge ya esta arriba, no lo
# relanza (evita duplicar terminales MT5, ver memoria del proyecto 2026-08-24).
#
# Uso: ./scripts/start_all.sh
set -eu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs/live

if curl -s http://127.0.0.1:18812/health 2>/dev/null | grep -q '"status": "ok"'; then
    echo "El bridge MT5 ya esta corriendo, no se relanza."
else
    echo "Arrancando el bridge MT5 (Wine)..."
    nohup ./scripts/start_mt5_bridge.sh --port 18812 > logs/bridge.log 2>&1 &
    disown

    echo -n "Esperando a que el bridge responda"
    ready=""
    for _ in $(seq 1 30); do
        if curl -s http://127.0.0.1:18812/health 2>/dev/null | grep -q '"status": "ok"'; then
            ready=1
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    if [ -z "$ready" ]; then
        echo "ERROR: el bridge no respondio tras 60s. Revisa logs/bridge.log." >&2
        exit 1
    fi
    echo "Bridge listo."
fi

SYMBOLS=$(.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from config.settings import get_settings
print(' '.join(get_settings()['trading']['symbols']))
")

echo "Lanzando run_live.py para: $SYMBOLS"
for sym in $SYMBOLS; do
    if pgrep -f "run_live.py --checkpoint data/models/${sym}_gbm.joblib" > /dev/null 2>&1; then
        echo "  [$sym] ya esta corriendo, se salta."
        continue
    fi
    checkpoint="data/models/${sym}_gbm.joblib"
    if [ ! -f "$checkpoint" ]; then
        echo "  [$sym] SALTADO: no existe $checkpoint (falta entrenar)."
        continue
    fi
    nohup .venv/bin/python scripts/run_live.py --checkpoint "$checkpoint" --symbol "$sym" \
        > "logs/live/${sym}.log" 2>&1 &
    disown
    echo "  [$sym] lanzado (PID $!)"
done

sleep 3
n_running=$(pgrep -f "scripts/run_live.py" 2>/dev/null | wc -l)
echo ""
echo "Piloto arrancado: $n_running procesos run_live.py corriendo."
echo "Logs en logs/live/<SIMBOLO>.log — tabla de operaciones en logs/live/trades.csv"
echo "Para pararlo todo: ./scripts/stop_all.sh"
