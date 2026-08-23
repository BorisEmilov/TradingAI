#!/usr/bin/env bash
# Lanza el bridge MT5 (wine_bridge/server.py) dentro del prefijo de Wine dedicado.
# Requiere: wine, y el prefijo $HOME/.wine-mt5 con MT5 + Python de Windows + MetaTrader5
# ya instalados (ver README > "MT5 en Linux").
set -euo pipefail

export WINEARCH=win64
export WINEPREFIX="${WINEPREFIX:-$HOME/.wine-mt5}"
export WINEDLLOVERRIDES="mscoree=;mshtml="

PYW="$WINEPREFIX/drive_c/users/$USER/AppData/Local/Programs/Python/Python311/python.exe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$PYW" ]; then
    echo "No se encontro el Python de Windows en: $PYW" >&2
    echo "Revisa el prefijo de Wine o ajusta esta ruta." >&2
    exit 1
fi

# El python.exe de Wine falla ("init_sys_streams ... Invalid handle") si no tiene
# una consola/tty real (p.ej. lanzado en background, por systemd o sin terminal
# interactiva). 'script' le da una pseudo-tty y evita el problema en cualquier caso.
cmd="wine $(printf '%q ' "$PYW" "$SCRIPT_DIR/../wine_bridge/server.py" "$@")"
exec script -qec "$cmd" /dev/null
