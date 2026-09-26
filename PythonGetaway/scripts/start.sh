#!/usr/bin/env bash
# =============================================================================
# PythonGetaway -- ARRANQUE COMPLETO EN 1 COMANDO
#
#     ./scripts/start.sh
#
# Levanta el gateway multiusuario. El propio gateway provisiona y arranca el
# pool: un terminal MT5 + un proceso worker por cada slot (usuario concurrente).
#
# Opciones:
#     --port 9000        puerto publico (por defecto 8000)
#     --host 0.0.0.0     escuchar en toda la red (por defecto solo local)
#     --pool 5           numero de usuarios concurrentes (por defecto 3)
#     --allow-real       permitir operar en cuentas REALES (por defecto NO)
#     --foreground       no daemonizar (Ctrl+C para parar)
#
# AVISO: con --host 0.0.0.0 y sin TLS delante, las credenciales MT5 de tus
# usuarios viajan en claro. Pon un reverse proxy con HTTPS.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export WINEARCH="${WINEARCH:-win64}"
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-mt5}"
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-mscoree=;mshtml=}"
export WINEDEBUG="${WINEDEBUG:--all}"

VENV_PY="$ROOT/.venv-gateway/bin/python"
PYW="${PYGW_WINE_PYTHON:-$WINEPREFIX/drive_c/users/$USER/AppData/Local/Programs/Python/Python311/python.exe}"
TEMPLATE="${PYGW_MT5_TEMPLATE:-$WINEPREFIX/drive_c/Program Files/MetaTrader 5}"

export PYGW_HOST="${PYGW_HOST:-127.0.0.1}"
export PYGW_PORT="${PYGW_PORT:-8000}"
export PYGW_POOL_SIZE="${PYGW_POOL_SIZE:-3}"
FOREGROUND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       export PYGW_PORT="$2"; shift 2 ;;
        --host)       export PYGW_HOST="$2"; shift 2 ;;
        --pool)       export PYGW_POOL_SIZE="$2"; shift 2 ;;
        --allow-real) export PYGW_ALLOW_REAL=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Opcion desconocida: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$ROOT/run"; LOG_DIR="$ROOT/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
PID_FILE="$RUN_DIR/gateway.pid"
GW_LOG="$LOG_DIR/gateway.log"

say()  { printf '  %s\n' "$*"; }
fail() { printf '\n  ERROR: %s\n\n' "$*" >&2; exit 1; }

echo
echo "=============================================="
echo " PythonGetaway - arrancando (multiusuario)"
echo "=============================================="

# --- Comprobaciones previas --------------------------------------------------
command -v wine >/dev/null 2>&1 || fail "Wine no esta instalado o no esta en el PATH."
[[ -x "$VENV_PY" ]]  || fail "Falta el venv del gateway. Crealo con:
      python3 -m venv .venv-gateway && ./.venv-gateway/bin/pip install fastapi 'uvicorn[standard]' httpx"
[[ -f "$PYW" ]]      || fail "No se encontro el Python de Windows en: $PYW"
[[ -f "$TEMPLATE/terminal64.exe" ]] || fail "No se encontro la plantilla de MT5 en: $TEMPLATE"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    say "Ya hay un gateway corriendo (PID $(cat "$PID_FILE"))."
    say "Para reiniciarlo:  ./scripts/stop.sh && ./scripts/start.sh"
    exit 0
fi
rm -f "$PID_FILE"

say "Pool: $PYGW_POOL_SIZE slot(s) = hasta $PYGW_POOL_SIZE usuarios concurrentes."
say "Cada slot arranca su propio terminal MT5; la primera vez ademas lo clona."
say "Esto tarda un poco. Paciencia."
echo

# --- Arranque ----------------------------------------------------------------
if [[ "$FOREGROUND" == "1" ]]; then
    exec "$VENV_PY" -m uvicorn gateway.main:app \
        --host "$PYGW_HOST" --port "$PYGW_PORT" --workers 1 --log-level info
fi

nohup "$VENV_PY" -m uvicorn gateway.main:app \
    --host "$PYGW_HOST" --port "$PYGW_PORT" --workers 1 --log-level info \
    >"$GW_LOG" 2>&1 &
echo $! >"$PID_FILE"

# --- Esperar a que el pool este listo ----------------------------------------
HEALTH="http://$PYGW_HOST:$PYGW_PORT/health"
DEADLINE=$(( $(date +%s) + 400 ))
while [[ $(date +%s) -lt $DEADLINE ]]; do
    sleep 2
    if BODY=$(curl -fsS --max-time 5 "$HEALTH" 2>/dev/null); then
        AVAIL=$(printf '%s' "$BODY" | sed -n 's/.*"available"[: ]*\([0-9]*\).*/\1/p')
        if [[ -n "$AVAIL" && "$AVAIL" -gt 0 ]]; then
            echo
            say "LISTO."
            echo
            printf '  API .............. http://%s:%s\n' "$PYGW_HOST" "$PYGW_PORT"
            printf '  Swagger UI ....... http://%s:%s/docs\n' "$PYGW_HOST" "$PYGW_PORT"
            printf '  Esquema OpenAPI .. http://%s:%s/openapi.json\n' "$PYGW_HOST" "$PYGW_PORT"
            printf '  Estado del pool .. http://%s:%s/pool\n' "$PYGW_HOST" "$PYGW_PORT"
            printf '  Logs ............. %s  (y logs/slot-*.log)\n' "$GW_LOG"
            printf '  Parar todo ....... ./scripts/stop.sh\n'
            echo
            printf '  Slots disponibles: %s de %s\n' "$AVAIL" "$PYGW_POOL_SIZE"
            echo
            say "Primer paso desde .NET:"
            printf '    POST http://%s:%s/auth/login  {"login":..., "password":"...", "server":"..."}\n' \
                   "$PYGW_HOST" "$PYGW_PORT"
            echo
            exit 0
        fi
    fi
    kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || break
done

echo
tail -40 "$GW_LOG" >&2 || true
fail "El pool no llego a tener ningun slot disponible. Log completo en $GW_LOG"
