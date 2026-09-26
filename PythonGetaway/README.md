# PythonGetaway — Gateway HTTP multiusuario sobre MetaTrader 5

API REST que expone **todo lo que MT5 puede dar** (cuenta, símbolos, velas, ticks, profundidad,
posiciones, órdenes, histórico) y **trading manual** (abrir, cerrar, cierre parcial, mover SL/TP,
órdenes pendientes), con **autenticación por usuario**: cada cliente aporta sus propias
credenciales MT5 y recibe un token.

**Documentación completa con ejemplos reales: [`docs/PythonGetaway_API.pdf`](docs/PythonGetaway_API.pdf)**

---

## Los dos comandos

```bash
cd /home/borislav/Desktop/TradingAI/PythonGetaway

./scripts/start.sh              # gateway + pool completo (3 usuarios concurrentes)
./scripts/start.sh --pool 5     # 5 usuarios concurrentes
./scripts/stop.sh               # cierra todo
```

`start.sh` no devuelve el control hasta que hay al menos un slot disponible. El primer arranque
tarda más: clona la instalación de MT5 una vez por slot.

| Recurso | URL |
|---|---|
| API | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Esquema OpenAPI | `http://127.0.0.1:8000/openapi.json` |
| Estado del pool | `http://127.0.0.1:8000/pool` |
| Logs | `logs/gateway.log`, `logs/slot-N-worker.log` |

---

## El flujo, en tres pasos

```bash
# 1. autenticarse con las credenciales MT5 del usuario
curl -X POST http://127.0.0.1:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"login":5054767214,"password":"...","server":"MetaQuotes-Demo"}'
# -> {"token":"...", "slot_id":"slot-1", "expires_in":3600, "account":{...}}

# 2. todo lo demás con el token
curl http://127.0.0.1:8000/account -H "Authorization: Bearer <token>"

# 3. si la sesion dura mas de 1 h, renovarla (el token NO cambia)
curl -X POST http://127.0.0.1:8000/auth/refresh -H "Authorization: Bearer <token>"

# 4. liberar el terminal
curl -X POST http://127.0.0.1:8000/auth/logout -H "Authorization: Bearer <token>"
```

### Caducidad de la sesion

| Reloj | Defecto | Se renueva |
|---|---|---|
| Inactividad (`PYGW_SESSION_IDLE`) | 15 min | **Sola**, con cualquier llamada autenticada |
| Vida maxima (`PYGW_SESSION_TTL`) | 1 h | Con `POST /auth/refresh` |

Si no quieres tope absoluto, arranca con `PYGW_SESSION_TTL=0`: la sesion vivira mientras el
cliente siga trabajando y `/auth/refresh` pasa a ser opcional.

---

## Por qué esta arquitectura

> La librería `MetaTrader5` de Python es un **singleton de proceso**: `mt5.login()` cambia la cuenta
> de *todo el proceso y de su terminal*. Si varios usuarios compartieran proceso, el login de uno
> cambiaría la sesión de otro y un "cerrar posición" podría ejecutarse **contra la cuenta
> equivocada**. Verificado: dos procesos atados al mismo terminal comparten cuenta.

Por eso cada sesión recibe **su propio terminal MT5 y su propio proceso**:

```
Cliente .NET (A)          Cliente .NET (B)
      │                        │
      │  HTTP + Authorization: Bearer <token>
      ▼                        ▼
┌─────────────────────────────────────────┐
│  GATEWAY (Python de Linux, async)       │  /auth/*, sesiones, pool
└─────────────────────────────────────────┘
      │  reenvío interno por localhost
      ▼                        ▼
 WORKER slot-1 (Wine)     WORKER slot-2 (Wine)
      │                        │
 terminal MT5 propio      terminal MT5 propio
 cuenta de A              cuenta de B
```

El **gateway** corre en Python de Linux: solo enruta, no necesita MT5 y gana concurrencia real.
Los **workers** corren bajo Wine, que es donde vive la DLL del terminal.

### Coste por usuario concurrente

| Recurso | Coste | Comentario |
|---|---|---|
| RAM | ~400 MB | ~265 MB terminal + ~135 MB worker |
| Disco | ~300 MB | Clon de la instalación + caché de histórico |
| Arranque | 10-20 s | Al calentar el pool, **no** dentro del login |
| Login | 1-3 s | El terminal ya está caliente |

Dimensiona `--pool` con la RAM disponible. Para más usuarios de los que quepan en una máquina,
replica el gateway completo detrás de un balanceador: cada instancia gestiona su propio pool.

---

## Seguridad

**Las credenciales no se guardan en ningún sitio.** Viajan una vez del cliente al worker, autentican
el terminal y se descartan. No van a disco, ni a los logs, ni quedan en memoria del gateway. Si el
gateway se reinicia, los usuarios vuelven a autenticarse.

> **TLS es obligatorio si sales de localhost.** Con `--host 0.0.0.0` y sin HTTPS delante, las
> credenciales MT5 de tus usuarios viajan **en claro**. Pon un reverse proxy (nginx, Caddy,
> Traefik) con certificado. El gateway no termina TLS a propósito: eso es del proxy.

Otras protecciones:

- **Guardarraíl de cuenta real**: por defecto el trading devuelve **403** si la cuenta no es DEMO.
  Se levanta con `--allow-real` / `PYGW_ALLOW_REAL=1`. La lectura nunca se bloquea.
- **Límite de intentos**: 5 fallos por cuenta en 5 minutos → 429.
- **Reciclado al cerrar sesión**: el terminal se reinicia para que no quede ninguna cuenta
  conectada (`PYGW_RECYCLE_ON_LOGOUT=1`).
- **Verificación de cuenta tras el login**: si el terminal no queda autenticado exactamente con la
  cuenta pedida, la sesión se rechaza. Sin esto se podrían servir los datos de quien estuviera
  logueado antes en ese slot.
- **`/internal/*` nunca se expone**: el proxy lo bloquea y los workers exigen un secreto compartido.
- **La plantilla se clona sin `accounts.dat`**: ese fichero lleva credenciales guardadas y no debe
  acabar en el slot de otro usuario.

---

## Probarlo

```bash
# solo lo que no necesita cuenta (gateway, pool, rutas de error)
python3 scripts/selftest.py

# completo: login real + los 41 endpoints de cuenta
PYGW_TEST_LOGIN=5054767214 PYGW_TEST_PASSWORD='...' PYGW_TEST_SERVER=MetaQuotes-Demo \
  python3 scripts/selftest.py

# completo pero sin enviar ninguna orden
... python3 scripts/selftest.py --no-trading
```

La contraseña se lee del entorno y **nunca** se imprime ni se guarda en `docs/examples.json`
(aparece enmascarada). Sin `--no-trading` abre y cierra posiciones reales; se niega si la cuenta
no es DEMO.

Regenerar el PDF (requiere el gateway arrancado):

```bash
python3 docs/build_pdf.py
```

---

## Configuración

| Variable | Defecto | Para qué |
|---|---|---|
| `PYGW_HOST` / `PYGW_PORT` | `127.0.0.1` / `8000` | Interfaz y puerto públicos |
| `PYGW_POOL_SIZE` | `3` | Usuarios concurrentes |
| `PYGW_SESSION_TTL` | `3600` | Vida máxima del token (s). `0` = sin tope; solo caduca por inactividad |
| `PYGW_SESSION_IDLE` | `900` | Caducidad por inactividad (s) — libera el slot sola |
| `PYGW_RECYCLE_ON_LOGOUT` | `1` | Reiniciar el terminal al cerrar sesión |
| `PYGW_ALLOW_REAL` | `0` | Permitir operar en cuentas REALES |
| `PYGW_MAX_LOGIN_ATTEMPTS` | `5` | Intentos fallidos antes de 429 |
| `PYGW_MAGIC` | `20260910` | Magic number por defecto |
| `PYGW_DEVIATION` | `20` | Slippage máximo por defecto (puntos) |

---

## Lo que hay que saber antes de escribir el cliente

**Un HTTP 200 no significa que la orden se ejecutara.** Significa que MT5 recibió la petición y
respondió. Comprueba siempre `success` / `retcode_name`.

**Tiempos.** MT5 devuelve epochs con el reloj del **servidor del broker**, no UTC. Cada campo sale
como `time` (epoch) y `time_iso` (ISO-8601 **sin** `Z`). En .NET deserialízalo como `DateTime` con
`DateTimeKind.Unspecified`.

**Errores**: todos comparten la forma
`{"error": {"code": ..., "message": ..., "mt5_code": ..., "detail": ...}}`.

**Volumen**: se ajusta al `volume_step` **siempre hacia abajo**. Por debajo de `volume_min` → 400.

**SL/TP**: precio absoluto. `0` los elimina; `null` los deja como estaban.

### Generar el cliente .NET

```bash
nswag openapi2csclient /input:http://127.0.0.1:8000/openapi.json \
                       /classname:Mt5Client /namespace:PythonGetaway /output:Mt5Client.cs
```

El esquema ya declara `bearerAuth`, así que el cliente generado incluye la autenticación.

---

## Estructura

```
PythonGetaway/
├── gateway/                 GATEWAY — Python de Linux (.venv-gateway)
│   ├── config.py            configuración
│   ├── pool.py              pool de terminales: provisión, arranque, reciclado
│   ├── sessions.py          tokens en memoria, TTL, barrido de caducadas
│   ├── auth.py              /auth/login · /auth/logout · /auth/session
│   ├── proxy.py             reenvío transparente al worker de cada sesión
│   ├── errors.py            formato único de error
│   └── main.py              app + OpenAPI compuesto
├── app/                     WORKER — Python de Windows bajo Wine
│   ├── mt5_session.py       conexión atada a SU terminal, login/logout
│   ├── converters.py        MT5 → JSON, enumeraciones, tiempos
│   ├── routers/             account · symbols · market · positions · orders ·
│   │                        history · trading · internal
│   └── main.py
├── scripts/
│   ├── start.sh             arranque completo (1 comando)
│   ├── stop.sh              parada completa (1 comando)
│   └── selftest.py          prueba end-to-end + captura de ejemplos reales
├── docs/
│   ├── build_pdf.py         generador de la documentación
│   ├── examples.json        ejemplos reales capturados
│   └── PythonGetaway_API.pdf
├── requirements-wine.txt    dependencias del Python de Wine (workers)
├── requirements-gateway.txt dependencias del Python de Linux (gateway)
├── logs/                    gateway.log, slot-N-worker.log, slot-N-terminal.log
└── run/                     pidfiles
```

Las instancias MT5 clonadas viven fuera del proyecto, en
`~/.wine-mt5/drive_c/mt5-instances/slot-N`, y se reutilizan entre arranques.
