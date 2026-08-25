#!/usr/bin/env bash
# Para el piloto completo: los procesos run_live.py, el bridge MT5 y el terminal.
# Las posiciones abiertas NO se tocan -- su SL/TP vive en el servidor del broker y
# sigue activo aunque el bridge/terminal/PC esten apagados; solo se deja de generar
# señales nuevas y de recibir avisos hasta la proxima vez que se arranque todo con
# scripts/start_all.sh.
#
# Uso: ./scripts/stop_all.sh
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Posiciones abiertas ahora mismo (se quedaran abiertas, gestionadas por el broker):"
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from tradingai.mt5.connector import MT5Connector
try:
    with MT5Connector() as conn:
        positions = conn.get_open_positions()
        if not positions:
            print('  (ninguna)')
        for p in positions:
            print(f\"  {p['symbol']} {p['type']} {p['volume']} lotes, profit=\${p['profit']}\")
except Exception as exc:
    print(f'  (no se pudo consultar el bridge: {exc})')
" 2>/dev/null || echo "  (bridge no disponible, no se pudo consultar)"

echo ""
echo "Deteniendo procesos run_live.py..."
pkill -9 -f "scripts/run_live.py" 2>/dev/null || true
sleep 1

echo "Deteniendo el bridge MT5 y el terminal..."
pkill -9 -f "wine_bridge/server.py" 2>/dev/null || true
pkill -9 -f "terminal64.exe" 2>/dev/null || true
sleep 2
pkill -9 -f "start_mt5_bridge.sh" 2>/dev/null || true
pkill -9 -f wineserver64 2>/dev/null || true

sleep 1
echo ""
echo "Todo detenido. Las posiciones abiertas (si las hay) siguen gestionadas por el"
echo "broker via su SL/TP -- no dependen de que esta maquina este encendida."
echo "Para volver a arrancar: ./scripts/start_all.sh"
