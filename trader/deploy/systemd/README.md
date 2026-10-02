# Piloto MTF como servicios systemd --user

Corren fuera de cualquier terminal/sesión de Claude Code, así que cerrar una sesión ya no los mata.

```bash
mkdir -p ~/.config/systemd/user
ln -sf "$PWD"/deploy/systemd/mtf-pilot.service "$PWD"/deploy/systemd/mtf-monitor.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"   # que sigan vivos sin sesión de login abierta (una sola vez)
```

Con el gateway ya arriba (`PythonGetaway/scripts/start.sh`):

| Acción | Comando |
|---|---|
| Arrancar | `systemctl --user start mtf-pilot mtf-monitor` |
| Parar limpio | `.venv/bin/python scripts/stop_mtf_pilot.py` (sale con 0, no se relanza) o `systemctl --user stop mtf-pilot` |
| Parar watchdog | `systemctl --user stop mtf-monitor` |
| Estado | `systemctl --user status mtf-pilot mtf-monitor` |
| Tras un crashloop (unidad `failed`) | `systemctl --user reset-failed mtf-pilot && systemctl --user start mtf-pilot` |

Códigos de salida del piloto: 0 parada pedida, 1 caída por error (systemd relanza, máx 3/h), 3 ya había otra instancia (lock).
