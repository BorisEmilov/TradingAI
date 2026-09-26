#!/usr/bin/env bash
# =============================================================================
# PythonGetaway -- PARADA COMPLETA EN 1 COMANDO
#
#     ./scripts/stop.sh
#
# Para el gateway (que a su vez cierra su pool) y despues barre cualquier
# terminal o worker que hubiera quedado suelto.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-mt5}"
RUN_DIR="$ROOT/run"
PID_FILE="$RUN_DIR/gateway.pid"

say() { printf '  %s\n' "$*"; }

# -----------------------------------------------------------------------------
# `pkill -f patron` mata cualquier proceso cuya LINEA DE COMANDOS contenga el
# patron -- incluida la shell que ejecuta este script (ya provoco un suicidio
# del script en pruebas) o un editor con el fichero abierto. Aqui se filtra por
# el ejecutable real y se excluye el propio arbol de procesos.
# -----------------------------------------------------------------------------
matching_pids() {
    local pattern="$1" pid comm
    for pid in $(pgrep -f -- "$pattern" 2>/dev/null); do
        [[ "$pid" == "$$" || "$pid" == "${PPID:-0}" ]] && continue
        comm="$(cat "/proc/$pid/comm" 2>/dev/null)" || continue
        case "$comm" in
            bash|sh|zsh|dash|pgrep|pkill|grep|tail|less|vi|vim|nano|nvim) continue ;;
        esac
        echo "$pid"
    done
}

kill_pids() {  # kill_pids "descripcion" pid...
    local label="$1"; shift
    [[ $# -eq 0 ]] && return 1
    local pids=("$@")
    say "$label: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null
    for _ in $(seq 1 20); do
        sleep 1
        local alive=0
        for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [[ "$alive" == "0" ]] && return 0
    done
    say "  no respondieron al TERM, forzando."
    kill -KILL "${pids[@]}" 2>/dev/null
    return 0
}

echo
echo "=============================================="
echo " PythonGetaway - parando"
echo "=============================================="

# --- 1. Gateway (cierra el pool de forma ordenada) ---------------------------
if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" 2>/dev/null)"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill_pids "Parando el gateway (pidfile)" "$PID"
        say "Esperando a que el pool se cierre..."
        sleep 3
    fi
    rm -f "$PID_FILE"
else
    say "Sin pidfile del gateway."
fi

mapfile -t gw < <(matching_pids "uvicorn gateway.main:app")
[[ ${#gw[@]} -gt 0 ]] && kill_pids "Gateways residuales" "${gw[@]}"

# --- 2. Workers --------------------------------------------------------------
mapfile -t workers < <(matching_pids "uvicorn app.main:app")
if [[ ${#workers[@]} -gt 0 ]]; then
    kill_pids "Cerrando workers" "${workers[@]}"
else
    say "No quedaban workers."
fi

# --- 3. Terminales MT5 (los de los slots y el de la plantilla) ---------------
mapfile -t terms < <(matching_pids "terminal64.exe")
if [[ ${#terms[@]} -gt 0 ]]; then
    kill_pids "Cerrando terminales MT5" "${terms[@]}"
else
    say "No quedaban terminales MT5."
fi

command -v wineserver >/dev/null 2>&1 && wineserver -k 2>/dev/null

# --- 4. Comprobacion final ---------------------------------------------------
sleep 1
echo
say "Comprobacion final:"
printf '    gateway ....... %s proceso(s)\n' "$(matching_pids 'uvicorn gateway.main:app' | wc -l)"
printf '    workers ....... %s proceso(s)\n' "$(matching_pids 'uvicorn app.main:app' | wc -l)"
printf '    terminales .... %s proceso(s)\n' "$(matching_pids 'terminal64.exe' | wc -l)"
if curl -fsS --max-time 2 "http://${PYGW_HOST:-127.0.0.1}:${PYGW_PORT:-8000}/health" >/dev/null 2>&1; then
    printf '    API ........... TODAVIA RESPONDE (revisar)\n'
else
    printf '    API ........... no responde (correcto)\n'
fi
echo
say "Las instancias MT5 clonadas se conservan en:"
printf '      %s\n' "$WINEPREFIX/drive_c/mt5-instances"
say "Se reutilizan en el siguiente arranque (borrarlas solo obliga a re-clonarlas)."
echo
