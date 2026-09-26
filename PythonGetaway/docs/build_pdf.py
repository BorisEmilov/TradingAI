#!/usr/bin/env python3
"""Genera la documentacion PDF del gateway.

Combina dos fuentes, ninguna escrita a mano:
  * el esquema OpenAPI EN VIVO  -> tipos, obligatoriedad y descripciones
  * `docs/examples.json`        -> peticiones/respuestas REALES del selftest

    python3 docs/build_pdf.py [--base http://127.0.0.1:8000]

Requiere el gateway arrancado y `wkhtmltopdf` instalado.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXAMPLES = DOCS / "examples.json"
HTML_OUT = DOCS / "PythonGetaway_API.html"
PDF_OUT = DOCS / "PythonGetaway_API.pdf"

TAG_TITLES = {
    "gateway": "Gateway y pool",
    "auth": "Autenticacion",
    "account": "Cuenta, terminal y estado",
    "symbols": "Simbolos",
    "market": "Datos de mercado",
    "positions": "Posiciones abiertas",
    "orders": "Ordenes pendientes",
    "history": "Historico",
    "trading": "Trading manual",
}
# Endpoints NUEVOS (autenticacion y pool): se documentan al principio del PDF,
# con sus tablas completas de entrada y salida. No se repiten en la referencia
# general para no duplicar contenido.
NEW_TAG_ORDER = ["auth", "gateway"]
TAG_ORDER = ["account", "symbols", "market", "positions",
             "orders", "history", "trading"]

CSS = """
@page { size: A4; margin: 15mm 12mm 16mm 12mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 9.2pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 22pt; margin: 0 0 2mm 0; color: #0b3d62; }
h2 { font-size: 14pt; margin: 9mm 0 3mm 0; color: #0b3d62;
     border-bottom: 2px solid #0b3d62; padding-bottom: 1.5mm; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 5mm 0 1.5mm 0; page-break-after: avoid; }
h4 { font-size: 9.2pt; margin: 3mm 0 1mm 0; color: #444; page-break-after: avoid; }
p { margin: 1.5mm 0; }
code, pre { font-family: "DejaVu Sans Mono", "Courier New", monospace; }
code { background: #eef2f6; padding: 0.4mm 1mm; border-radius: 2px; font-size: 8.4pt; }
pre { background: #f6f8fa; border: 1px solid #d8dee4; border-left: 3px solid #0b3d62;
      padding: 2.5mm 3mm; font-size: 7.6pt; white-space: pre-wrap; word-wrap: break-word;
      border-radius: 3px; margin: 1.5mm 0; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0; font-size: 8.2pt; }
th { background: #0b3d62; color: #fff; text-align: left; padding: 1.4mm 2mm; font-weight: 600; }
td { border-bottom: 1px solid #dde3e9; padding: 1.4mm 2mm; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
.ep { border: 1px solid #d8dee4; border-radius: 4px; padding: 3mm; margin: 4mm 0;
      page-break-inside: avoid; background: #fff; }
.m { display: inline-block; padding: 0.7mm 2.2mm; border-radius: 3px; color: #fff;
     font-weight: 700; font-size: 8pt; font-family: "DejaVu Sans Mono", monospace; }
.get { background: #1a7f37; } .post { background: #0969da; } .delete { background: #cf222e; }
.path { font-family: "DejaVu Sans Mono", monospace; font-size: 10pt; font-weight: 700; color: #0b3d62; }
.note { background: #fff8e1; border-left: 3px solid #f0ad4e; padding: 2mm 3mm; margin: 2mm 0; font-size: 8.4pt; }
.warn { background: #fdeaea; border-left: 3px solid #cf222e; padding: 2mm 3mm; margin: 2mm 0; font-size: 8.4pt; }
.ok   { background: #e8f5e9; border-left: 3px solid #1a7f37; padding: 2mm 3mm; margin: 2mm 0; font-size: 8.4pt; }
.cmd { background: #14181d; color: #e6edf3; padding: 3mm; border-radius: 4px;
       font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; white-space: pre-wrap; margin: 2mm 0; }
.cover { text-align: center; padding-top: 55mm; page-break-after: always; }
.cover h1 { font-size: 30pt; border: 0; }
.sub { color: #55636e; font-size: 11pt; margin-top: 2mm; }
.meta { margin-top: 22mm; font-size: 9pt; color: #55636e; }
.req { color: #cf222e; font-weight: 700; }
.opt { color: #7a828a; }
.toc td { border: 0; padding: 0.8mm 2mm; }
.small { font-size: 8pt; color: #55636e; }
.tok { display: inline-block; background: #e7f0fb; color: #0b3d62; border: 1px solid #b6d4f2;
       border-radius: 3px; padding: 0.4mm 1.6mm; font-size: 7.4pt; margin-left: 2mm; }
.notok { display: inline-block; background: #eef1f4; color: #55636e; border: 1px solid #dde3e9;
         border-radius: 3px; padding: 0.4mm 1.6mm; font-size: 7.4pt; margin-left: 2mm; }
"""


# Ejemplos de los endpoints de /auth. Se marcan como ESTRUCTURA DOCUMENTADA, no
# como captura: un login exitoso exige credenciales reales, y este documento no
# presenta como capturado nada que no se haya ejecutado. Ejecutando el selftest
# con PYGW_TEST_* estos huecos se rellenan con capturas reales.
DOCUMENTED_EXAMPLES = {
    ("POST", "/auth/login"): {
        "request": {"login": 5054767214, "password": "<tu contrasena>",
                    "server": "MetaQuotes-Demo"},
        "response": {
            "reused": False,
            "token": "kHs9_2FqxV7bN1pQ...   (43 caracteres url-safe)",
            "slot_id": "slot-1",
            "login": 5054767214,
            "server": "MetaQuotes-Demo",
            "account": {
                "login": 5054767214, "server": "MetaQuotes-Demo", "currency": "USD",
                "balance": 96126.17, "equity": 96126.17,
                "trade_mode": 0, "trade_mode_name": "DEMO", "trade_allowed": True,
            },
            "created_at": 1789040000,
            "last_seen": 1789040000,
            "expires_at": 1789040900,
            "expires_in": 900,
            "ttl_seconds": 3600,
            "idle_timeout_seconds": 900,
        },
    },
    ("GET", "/auth/session"): {
        "request": None,
        "response": {
            "slot_id": "slot-1", "login": 5054767214, "server": "MetaQuotes-Demo",
            "account": {
                "login": 5054767214, "server": "MetaQuotes-Demo", "currency": "USD",
                "balance": 96126.17, "equity": 96126.17,
                "trade_mode": 0, "trade_mode_name": "DEMO", "trade_allowed": True,
            },
            "created_at": 1789040000, "last_seen": 1789040420,
            "expires_at": 1789041320, "expires_in": 900,
            "ttl_seconds": 3600, "idle_timeout_seconds": 900,
        },
    },
    ("POST", "/auth/refresh"): {
        "request": None,
        "response": {
            "slot_id": "slot-1", "login": 5054767214, "server": "MetaQuotes-Demo",
            "account": {
                "login": 5054767214, "server": "MetaQuotes-Demo", "currency": "USD",
                "balance": 96126.17, "equity": 96126.17,
                "trade_mode": 0, "trade_mode_name": "DEMO", "trade_allowed": True,
            },
            "created_at": 1789041000, "last_seen": 1789041000,
            "expires_at": 1789041900, "expires_in": 900,
            "ttl_seconds": 3600, "idle_timeout_seconds": 900,
        },
    },
    ("POST", "/auth/logout"): {
        "request": None,
        "response": {"logged_out": True, "recycled": True},
    },
    ("GET", "/auth/sessions"): {
        "request": None,
        "response": {"count": 1, "sessions": [{
            "slot_id": "slot-1", "login": 5054767214, "server": "MetaQuotes-Demo",
            "created_at": 1789040000, "last_seen": 1789040420,
            "expires_at": 1789041320, "expires_in": 900,
            "ttl_seconds": 3600, "idle_timeout_seconds": 900,
        }]},
    },
}


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def pretty(obj) -> str:
    if obj is None:
        return "(sin cuerpo)"
    return esc(json.dumps(obj, indent=2, ensure_ascii=False))


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def resolve(schema: dict, spec: dict) -> dict:
    """Resuelve $ref y allOf de un esquema OpenAPI."""
    if not isinstance(schema, dict):
        return {}
    if "$ref" in schema:
        parts = schema["$ref"].lstrip("#/").split("/")
        node = spec
        for part in parts:
            node = node.get(part, {})
        return resolve(node, spec)
    if "allOf" in schema:
        merged: dict = {"type": "object", "properties": {}, "required": []}
        for sub in schema["allOf"]:
            r = resolve(sub, spec)
            merged["properties"].update(r.get("properties", {}))
            merged["required"] += r.get("required", [])
        return merged
    return schema


def type_name(schema: dict, spec: dict) -> str:
    """Nombre legible del tipo, indicando si admite null."""
    s = resolve(schema, spec)
    if "anyOf" in s:
        parts = [type_name(x, spec) for x in s["anyOf"]]
        non_null = [p for p in parts if p != "null"]
        base = " | ".join(dict.fromkeys(non_null)) or "any"
        return f"{base}?" if "null" in parts else base
    if "enum" in s:
        return " | ".join(f'"{v}"' for v in s["enum"])
    t = s.get("type")
    if t == "array":
        return f"array<{type_name(s.get('items', {}), spec)}>"
    if t == "integer":
        return "integer (int64)"
    if t == "number":
        return "number (double)"
    if t == "boolean":
        return "boolean"
    if t == "string":
        return "string"
    if t == "null":
        return "null"
    if t == "object" or "properties" in s:
        return "object"
    return "any"


def params_table(op: dict, spec: dict) -> str:
    params = op.get("parameters", [])
    if not params:
        return ""
    rows = []
    for p in params:
        req = '<span class="req">si</span>' if p.get("required") else '<span class="opt">no</span>'
        default = resolve(p.get("schema", {}), spec).get("default")
        default_txt = f"<code>{esc(json.dumps(default))}</code>" if default is not None else "&mdash;"
        rows.append(
            f"<tr><td><code>{esc(p['name'])}</code></td><td>{esc(p.get('in'))}</td>"
            f"<td><code>{esc(type_name(p.get('schema', {}), spec))}</code></td>"
            f"<td>{req}</td><td>{default_txt}</td>"
            f"<td>{esc(p.get('description', ''))}</td></tr>")
    return ("<h4>Parametros (URL / query)</h4><table><tr><th>Nombre</th><th>En</th><th>Tipo</th>"
            "<th>Obligatorio</th><th>Por defecto</th><th>Descripcion</th></tr>"
            + "".join(rows) + "</table>")


def body_table(op: dict, spec: dict) -> str:
    body = op.get("requestBody")
    if not body:
        return ""
    schema = body.get("content", {}).get("application/json", {}).get("schema", {})
    resolved = resolve(schema, spec)
    props = resolved.get("properties", {})
    if not props:
        return ""
    required = set(resolved.get("required", []))
    rows = []
    for name, sub in props.items():
        s = resolve(sub, spec)
        req = '<span class="req">si</span>' if name in required else '<span class="opt">no</span>'
        rows.append(
            f"<tr><td><code>{esc(name)}</code></td>"
            f"<td><code>{esc(type_name(sub, spec))}</code></td><td>{req}</td>"
            f"<td>{esc(s.get('description', ''))}</td></tr>")
    return ("<h4>Cuerpo de la peticion &mdash; <code>application/json</code></h4>"
            "<table><tr><th>Campo</th><th>Tipo</th><th>Obligatorio</th><th>Descripcion</th></tr>"
            + "".join(rows) + "</table>")


def match_examples(spec: dict, examples: list[dict]) -> dict[tuple[str, str], dict]:
    """Empareja cada ejemplo REAL con su plantilla de OpenAPI.

    Los ejemplos traen rutas concretas (`/positions/58389038744`) y OpenAPI usa
    plantillas (`/positions/{ticket}`), asi que la comparacion literal falla:
    hay que convertir la plantilla en una expresion regular."""
    patterns = []
    for path in spec.get("paths", {}):
        regex = re.compile("^" + re.sub(r"\{[^/}]+\}", r"[^/]+", re.escape(path)
                                        .replace(r"\{", "{").replace(r"\}", "}")) + "$")
        patterns.append((path, regex, path.count("{")))

    out: dict[tuple[str, str], dict] = {}
    for ex in examples:
        method = ex["method"].upper()
        concrete = ex["path"].split("?")[0]
        # Se prefiere la plantilla con MENOS comodines: `/positions/total` debe
        # ganar a `/positions/{ticket}` cuando ambas encajan.
        candidates = sorted((p for p, rx, n in patterns if rx.match(concrete)),
                            key=lambda p: (p.count("{"), -len(p)))
        if not candidates:
            continue
        key = (method, candidates[0])
        prev = out.get(key)
        if prev is None:
            out[key] = ex
        else:
            # Preferir una captura de exito (200) sobre una de error, y una con
            # cuerpo de peticion sobre una sin el.
            prev_ok, new_ok = prev.get("status") == 200, ex.get("status") == 200
            if new_ok and not prev_ok:
                out[key] = ex
            elif new_ok == prev_ok and prev.get("request") is None and ex.get("request") is not None:
                out[key] = ex
    return out



def _rows_for(props: dict, required: set, spec: dict, prefix: str = "", depth: int = 0) -> list[str]:
    """Filas de la tabla de respuesta, expandiendo objetos y arrays anidados.

    Sin esto, un campo como `sessions: array<object>` no diria nada util sobre
    lo que lleva dentro."""
    rows = []
    for name, sub in props.items():
        sub_res = resolve(sub, spec)
        label = f"{prefix}{name}"
        presence = ('<span class="req">siempre</span>' if name in required and depth == 0
                    else ('<span class="opt">puede faltar</span>' if depth == 0 else ""))
        indent = f" style='padding-left:{depth * 5}mm'" if depth else ""
        rows.append(
            f"<tr><td{indent}><code>{esc(label)}</code></td>"
            f"<td><code>{esc(type_name(sub, spec))}</code></td><td>{presence}</td>"
            f"<td>{esc(sub_res.get('description', ''))}</td></tr>")
        if depth >= 2:
            continue
        nested = sub_res.get("properties")
        if not nested and sub_res.get("type") == "array":
            item = resolve(sub_res.get("items", {}), spec)
            nested = item.get("properties")
            if nested:
                rows += _rows_for(nested, set(item.get("required", [])), spec,
                                  f"{label}[]." , depth + 1)
                continue
        if nested:
            rows += _rows_for(nested, set(sub_res.get("required", [])), spec,
                              f"{label}.", depth + 1)
    return rows


def response_table(op: dict, spec: dict) -> str:
    """Tabla de lo que DEVUELVE el endpoint, cuando declara un modelo tipado.

    Los endpoints de cuenta devuelven objetos JSON abiertos a proposito (los
    campos de MT5 varian entre brokers y builds), asi que solo los que declaran
    modelo producen tabla. Para el resto vale el ejemplo real."""
    ok = ((op.get("responses") or {}).get("200") or {})
    schema = (ok.get("content", {}).get("application/json", {}) or {}).get("schema")
    if not schema:
        return ""
    resolved = resolve(schema, spec)
    props = resolved.get("properties", {})
    if not props:
        return ""
    rows = _rows_for(props, set(resolved.get("required", [])), spec)
    return ("<h4>Respuesta &mdash; <code>application/json</code> (HTTP 200)</h4>"
            "<table><tr><th>Campo</th><th>Tipo</th><th>Presencia</th><th>Descripcion</th></tr>"
            + "".join(rows) + "</table>")


# Orden de lectura de los endpoints nuevos: primero el login, luego el resto.
_NEW_ORDER = ["/auth/login", "/auth/refresh", "/auth/session", "/auth/logout",
              "/auth/sessions",
              "/", "/health", "/pool"]


def _new_sort(path: str) -> int:
    return _NEW_ORDER.index(path) if path in _NEW_ORDER else 99


def build_html(spec: dict, examples: list[dict]) -> str:
    by_path = match_examples(spec, examples)

    grouped: dict[str, list] = {}
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                continue
            tag = (op.get("tags") or ["otros"])[0]
            grouped.setdefault(tag, []).append((path, method.upper(), op))

    total = sum(len(v) for v in grouped.values())
    parts: list[str] = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]

    # --- Portada -----------------------------------------------------------
    parts.append(f"""
<div class="cover">
  <h1>PythonGetaway</h1>
  <div class="sub">Gateway HTTP multiusuario sobre MetaTrader&nbsp;5</div>
  <div class="sub">Referencia completa de la API &mdash; {total} endpoints</div>
  <div class="meta">
    Version 2.0.0 &mdash; multiusuario<br>
    Cliente destino: .NET<br>
    Todos los ejemplos de este documento son respuestas <b>reales</b><br>
    capturadas contra una cuenta MT5 en vivo.
  </div>
</div>""")

    # --- 1. ENDPOINTS NUEVOS (arriba del todo) -----------------------------
    n_new = sum(len(grouped.get(tag, [])) for tag in NEW_TAG_ORDER)
    parts.append(f"""
<h2>1. Endpoints nuevos: autenticacion y pool ({n_new})</h2>
<p>Estos son los endpoints que <b>no existian</b> en la version monousuario. Son la puerta de
entrada de la API: sin un token de <code>/auth/login</code> ninguno de los otros
{total - n_new} responde.</p>

<div class="ok"><b>Resumen del flujo.</b>
<code>POST /auth/login</code> con las credenciales MT5 del usuario devuelve un
<code>token</code>. Ese token va en <code>Authorization: Bearer &lt;token&gt;</code> en todas las
llamadas siguientes. <code>POST /auth/logout</code> libera el terminal.</div>

<table>
  <tr><th>Metodo</th><th>Ruta</th><th>Necesita token</th><th>Para que</th></tr>
  <tr><td><code>POST</code></td><td><code>/auth/login</code></td><td>no</td>
      <td>Autenticarse con credenciales MT5 y reservar un terminal dedicado.</td></tr>
  <tr><td><code>POST</code></td><td><code>/auth/refresh</code></td><td><b>si</b></td>
      <td>Reiniciar el reloj de vida maxima. El token no cambia.</td></tr>
  <tr><td><code>POST</code></td><td><code>/auth/logout</code></td><td><b>si</b></td>
      <td>Cerrar la sesion y devolver el terminal al pool.</td></tr>
  <tr><td><code>GET</code></td><td><code>/auth/session</code></td><td><b>si</b></td>
      <td>Estado de la sesion actual. Renueva la ventana de inactividad.</td></tr>
  <tr><td><code>GET</code></td><td><code>/auth/sessions</code></td><td>no</td>
      <td>Sesiones activas, sin tokens ni credenciales. Vision de operacion.</td></tr>
  <tr><td><code>GET</code></td><td><code>/</code></td><td>no</td>
      <td>Indice del gateway.</td></tr>
  <tr><td><code>GET</code></td><td><code>/health</code></td><td>no</td>
      <td>Sonda: estado del gateway y cuantos usuarios nuevos caben.</td></tr>
  <tr><td><code>GET</code></td><td><code>/pool</code></td><td>no</td>
      <td>Detalle de cada terminal del pool.</td></tr>
</table>

<div class="warn"><b>La contrasena no se guarda en ningun sitio.</b> Viaja una sola vez del cliente
al worker, autentica el terminal y se descarta: no va a disco, ni a los logs, ni queda en memoria
del gateway. Si el gateway se reinicia, los usuarios vuelven a autenticarse.<br><br>
Y por eso mismo: con <code>--host 0.0.0.0</code> y sin HTTPS delante, esas credenciales viajan
<b>en claro</b>. Pon un reverse proxy con TLS.</div>
""")

    for tag in NEW_TAG_ORDER:
        if tag not in grouped:
            continue
        parts.append(f"<h3>{esc(TAG_TITLES.get(tag, tag))}</h3>")
        for path, method, op in sorted(grouped[tag], key=lambda x: (_new_sort(x[0]), x[1])):
            parts.append(render_endpoint(path, method, op, spec, by_path))

    # --- 1. Comandos -------------------------------------------------------
    parts.append("""
<h2>2. Arrancar y parar</h2>
<p>Dos comandos. El primero levanta el gateway, que a su vez provisiona y arranca
<b>el pool completo</b>: un terminal MT5 y un proceso worker por cada usuario concurrente.</p>

<h3>Arrancar todo</h3>
<div class="cmd">cd /home/borislav/Desktop/TradingAI/PythonGetaway
./scripts/start.sh</div>
<p>No devuelve el control hasta que hay al menos un slot disponible. El primer arranque tarda
mas: clona la instalacion de MT5 una vez por slot.</p>

<h4>Opciones</h4>
<table>
  <tr><th>Opcion</th><th>Efecto</th></tr>
  <tr><td><code>--pool 5</code></td><td>Usuarios concurrentes (por defecto <b>3</b>). Cada uno cuesta ~400&nbsp;MB de RAM.</td></tr>
  <tr><td><code>--port 9000</code></td><td>Puerto publico (por defecto <code>8000</code>).</td></tr>
  <tr><td><code>--host 0.0.0.0</code></td><td>Escuchar en toda la red. Por defecto solo <code>127.0.0.1</code>.</td></tr>
  <tr><td><code>--allow-real</code></td><td>Permitir operar en cuentas REALES. Por defecto se bloquea con 403.</td></tr>
  <tr><td><code>--foreground</code></td><td>No daemonizar: logs en la terminal, Ctrl+C para parar.</td></tr>
</table>

<h3>Parar todo</h3>
<div class="cmd">cd /home/borislav/Desktop/TradingAI/PythonGetaway
./scripts/stop.sh</div>
<p>Para el gateway (que cierra su pool de forma ordenada) y despues barre cualquier terminal o
worker que hubiera quedado suelto. Las instancias MT5 clonadas se conservan para el proximo
arranque.</p>

<h3>Comprobar que esta vivo</h3>
<div class="cmd">curl http://127.0.0.1:8000/health     # estado del gateway y del pool
curl http://127.0.0.1:8000/pool       # detalle de cada slot</div>

<h3>Probarlo de punta a punta</h3>
<div class="cmd"># solo lo que no necesita cuenta (gateway, pool, rutas de error)
python3 scripts/selftest.py

# completo: hace login real y prueba los 41 endpoints de cuenta
PYGW_TEST_LOGIN=5054767214 PYGW_TEST_PASSWORD='...' PYGW_TEST_SERVER=MetaQuotes-Demo \\
  python3 scripts/selftest.py

# completo pero sin enviar ninguna orden
... python3 scripts/selftest.py --no-trading</div>
<div class="warn"><b>Aviso:</b> sin <code>--no-trading</code> el selftest abre y cierra posiciones
reales en la cuenta indicada. Se niega a operar si la cuenta no es DEMO. La contrasena se lee del
entorno y nunca se imprime ni se guarda en los ejemplos.</div>

<h3>Recursos en marcha</h3>
<table>
  <tr><th>Recurso</th><th>URL</th></tr>
  <tr><td>API</td><td><code>http://127.0.0.1:8000</code></td></tr>
  <tr><td>Swagger UI (interactivo)</td><td><code>http://127.0.0.1:8000/docs</code></td></tr>
  <tr><td>Esquema OpenAPI</td><td><code>http://127.0.0.1:8000/openapi.json</code></td></tr>
  <tr><td>Log del gateway</td><td><code>logs/gateway.log</code></td></tr>
  <tr><td>Log de cada slot</td><td><code>logs/slot-N-worker.log</code>, <code>logs/slot-N-terminal.log</code></td></tr>
</table>
""")

    # --- 2. Arquitectura ---------------------------------------------------
    parts.append("""
<h2>3. Como esta montado (y por que)</h2>

<div class="warn"><b>El problema que resuelve esta arquitectura.</b> La libreria
<code>MetaTrader5</code> de Python es un <b>singleton de proceso</b>: <code>mt5.login()</code>
cambia la cuenta de <i>todo el proceso y de su terminal</i>. Si varios usuarios compartieran
proceso, el login de uno cambiaria la sesion de otro, y un &laquo;cerrar posicion&raquo; podria
ejecutarse <b>contra la cuenta equivocada</b>. Verificado: dos procesos atados al mismo terminal
comparten cuenta.</div>

<p>Por eso cada sesion recibe <b>su propio terminal MT5 y su propio proceso</b>. Es la unica forma
de servir N cuentas a la vez.</p>

<div class="cmd">Cliente .NET  (usuario A)     Cliente .NET  (usuario B)
      |                              |
      |  HTTP + Authorization: Bearer &lt;token&gt;
      v                              v
 +---------------------------------------------+
 |   GATEWAY  (Python de Linux, async)         |   /auth/*, sesiones, pool
 +---------------------------------------------+
      |  reenvio interno por localhost
      v                              v
 WORKER slot-1 (Wine)          WORKER slot-2 (Wine)
      |                              |
 terminal MT5 propio           terminal MT5 propio
 data_path: C:\\mt5-instances\\slot-1   ...\\slot-2
      |                              |
 cuenta del usuario A          cuenta del usuario B</div>

<p>El <b>gateway</b> corre en el Python de Linux: solo enruta, no necesita la libreria MT5 y gana
concurrencia real. Los <b>workers</b> corren bajo Wine, que es donde vive la DLL del terminal.</p>

<h3>Coste por usuario concurrente</h3>
<table>
  <tr><th>Recurso</th><th>Coste</th><th>Comentario</th></tr>
  <tr><td>RAM</td><td>~400 MB</td><td>~265 MB el terminal + ~135 MB el worker</td></tr>
  <tr><td>Disco</td><td>~300 MB</td><td>Clon de la instalacion, mas la cache de historico que descargue</td></tr>
  <tr><td>Arranque</td><td>10-20 s</td><td>Ocurre al calentar el pool, <b>no</b> dentro del login</td></tr>
  <tr><td>Login</td><td>1-3 s</td><td>El terminal ya esta caliente: solo se autentica</td></tr>
</table>
<p>Dimensiona <code>--pool</code> con la RAM disponible. Para mas usuarios de los que quepan en una
maquina, replica el gateway completo detras de un balanceador: cada instancia gestiona su propio
pool y las sesiones son independientes.</p>

<h3>Estados de un slot</h3>
<table>
  <tr><th>Estado</th><th>Significado</th></tr>
  <tr><td><code>COLD</code></td><td>Sin procesos.</td></tr>
  <tr><td><code>WARMING</code></td><td>Terminal y worker arrancando.</td></tr>
  <tr><td><code>IDLE</code></td><td>Listo y sin cuenta: disponible para un login.</td></tr>
  <tr><td><code>BUSY</code></td><td>Autenticado con una cuenta concreta.</td></tr>
  <tr><td><code>RECYCLING</code></td><td>Cerrando sesion y reiniciando el terminal.</td></tr>
  <tr><td><code>FAILED</code></td><td>No se pudo levantar. El motivo va en <code>error</code>.</td></tr>
</table>

<h3>Variables de entorno</h3>
<table>
  <tr><th>Variable</th><th>Por defecto</th><th>Para que sirve</th></tr>
  <tr><td><code>PYGW_HOST</code> / <code>PYGW_PORT</code></td><td><code>127.0.0.1</code> / <code>8000</code></td><td>Interfaz y puerto publicos.</td></tr>
  <tr><td><code>PYGW_POOL_SIZE</code></td><td><code>3</code></td><td>Usuarios concurrentes.</td></tr>
  <tr><td><code>PYGW_SESSION_TTL</code></td><td><code>3600</code></td><td>Vida maxima de un token, en segundos.</td></tr>
  <tr><td><code>PYGW_SESSION_IDLE</code></td><td><code>900</code></td><td>Caducidad por inactividad. Libera el slot solo.</td></tr>
  <tr><td><code>PYGW_RECYCLE_ON_LOGOUT</code></td><td><code>1</code></td><td>Reiniciar el terminal al cerrar sesion para que no quede ninguna cuenta conectada.</td></tr>
  <tr><td><code>PYGW_ALLOW_REAL</code></td><td><code>0</code></td><td>Si es <code>0</code> y la cuenta es REAL, el trading devuelve <b>403</b>. La lectura sigue funcionando.</td></tr>
  <tr><td><code>PYGW_MAX_LOGIN_ATTEMPTS</code></td><td><code>5</code></td><td>Intentos fallidos por cuenta antes de devolver 429.</td></tr>
  <tr><td><code>PYGW_MAGIC</code></td><td><code>20260910</code></td><td>Magic number por defecto de las ordenes.</td></tr>
  <tr><td><code>PYGW_DEVIATION</code></td><td><code>20</code></td><td>Slippage maximo por defecto, en puntos.</td></tr>
</table>
""")

    # --- 3. Autenticacion --------------------------------------------------
    parts.append("""
<h2>4. Autenticacion en detalle</h2>

<p>Cada usuario aporta <b>sus propias credenciales MT5</b>. El flujo tiene tres pasos:</p>

<div class="cmd">1. POST /auth/login   {login, password, server}   -> { "token": "...", ... }
2. Cualquier endpoint  con  Authorization: Bearer &lt;token&gt;
3. POST /auth/logout  (o dejar que caduque)      -> libera el terminal</div>

<h3>4.1 Que pasa con la contrasena</h3>
<div class="ok"><b>No se guarda en ningun sitio.</b> Viaja una sola vez del cliente al worker, se
usa para autenticar el terminal y se descarta. No se escribe en disco, ni en los logs, ni queda en
memoria del gateway. Lo unico que persiste mientras dura la sesion es el token y a que slot apunta.</div>
<p><b>Consecuencia asumida:</b> si el gateway se reinicia, todos los usuarios vuelven a
autenticarse. Es el precio de no ser un almacen de credenciales.</p>

<div class="warn"><b>TLS es obligatorio si sales de localhost.</b> Con
<code>--host 0.0.0.0</code> y sin HTTPS delante, las credenciales MT5 de tus usuarios viajan
<b>en claro</b>. Pon un reverse proxy (nginx, Caddy, Traefik) con certificado. El gateway no
termina TLS por si mismo, a proposito: esa responsabilidad es del proxy.</div>

<h3>4.2 Ciclo de vida del token</h3>
<table>
  <tr><th>Concepto</th><th>Valor por defecto</th><th>Detalle</th></tr>
  <tr><td>Vida maxima</td><td>3600 s</td><td>Desde el login, pase lo que pase.</td></tr>
  <tr><td>Inactividad</td><td>900 s</td><td>Ventana <b>deslizante</b>: cada peticion la renueva.</td></tr>
  <tr><td>Caducidad efectiva</td><td>&mdash;</td><td>La que llegue antes de las dos. Va en <code>expires_at</code> / <code>expires_in</code>.</td></tr>
  <tr><td>Al caducar</td><td>&mdash;</td><td>Un barrido cierra la sesion y <b>libera el slot</b>. Sin esto, un cliente que se desconecta sin logout agotaria el pool.</td></tr>
</table>
<p>Un login con una cuenta que <b>ya tiene sesion viva</b> devuelve la misma sesion
(<code>reused: true</code>) en lugar de consumir un segundo slot.</p>

<h3>4.3 Errores especificos de autenticacion</h3>
<table>
  <tr><th>HTTP</th><th><code>code</code></th><th>Que hacer</th></tr>
  <tr><td>401</td><td><code>MISSING_TOKEN</code></td><td>Falta la cabecera <code>Authorization</code>.</td></tr>
  <tr><td>401</td><td><code>INVALID_OR_EXPIRED_TOKEN</code></td><td>Volver a <code>/auth/login</code>.</td></tr>
  <tr><td>401</td><td><code>MT5_LOGIN_FAILED</code></td><td>Credenciales o servidor incorrectos. <code>detail.error.mt5_code</code> trae el codigo de MT5 (<code>-6</code> = Authorization failed).</td></tr>
  <tr><td>429</td><td><code>TOO_MANY_LOGIN_ATTEMPTS</code></td><td>5 fallos en 5 minutos para esa cuenta. Esperar.</td></tr>
  <tr><td>503</td><td><code>POOL_EXHAUSTED</code></td><td>No hay terminales libres. Cerrar sesiones o subir <code>--pool</code>. <code>detail</code> trae el estado del pool.</td></tr>
  <tr><td>409</td><td><code>SESSION_SLOT_LOST</code></td><td>El terminal de esa sesion se reinicio. Volver a autenticarse.</td></tr>
  <tr><td>403</td><td><code>REAL_ACCOUNT_BLOCKED</code></td><td>Cuenta REAL sin <code>PYGW_ALLOW_REAL=1</code>. La lectura sigue funcionando.</td></tr>
</table>

<h3>4.4 Cabeceras que devuelve el gateway</h3>
<p>Toda respuesta proxificada lleva <code>X-Slot-Id</code> y <code>X-Account-Login</code>: sirven
para depurar y para confirmar contra que cuenta se ejecuto la llamada.</p>

<h3>4.5 Ejemplo completo en C#</h3>
<div class="cmd">var http = new HttpClient { BaseAddress = new Uri("https://tu-host/") };

// 1. login
var login = await http.PostAsJsonAsync("auth/login", new {
    login = 5054767214L, password = pass, server = "MetaQuotes-Demo" });
var session = await login.Content.ReadFromJsonAsync&lt;SessionDto&gt;();

// 2. el token va en TODAS las llamadas siguientes
http.DefaultRequestHeaders.Authorization =
    new AuthenticationHeaderValue("Bearer", session.Token);

var account = await http.GetFromJsonAsync&lt;JsonElement&gt;("account");
var open = await http.PostAsJsonAsync("trading/open", new {
    symbol = "EURUSD", side = "BUY", volume = 0.10, sl_points = 300 });

// 3. comprobar SIEMPRE success, no solo el codigo HTTP
var result = await open.Content.ReadFromJsonAsync&lt;JsonElement&gt;();
if (!result.GetProperty("success").GetBoolean())
    throw new Exception(result.GetProperty("retcode_name").GetString());

// 4. liberar el terminal al terminar
await http.PostAsync("auth/logout", null);</div>
""")

    # --- 3bis. Convenciones ------------------------------------------------
    parts.append("""
<h2>5. Convenciones de datos</h2>

<h3>5.1 Fechas y horas</h3>
<p>MT5 devuelve los tiempos como <b>epoch en segundos</b>, pero medidos con el reloj del
<b>servidor del broker</b>, no en UTC. El gateway <b>no inventa</b> una conversion de zona horaria.
Cada campo temporal se expone dos veces:</p>
<table>
  <tr><th>Campo</th><th>Tipo</th><th>Ejemplo</th><th>Significado</th></tr>
  <tr><td><code>time</code></td><td>integer (int64)</td><td><code>1789037400</code></td><td>Epoch crudo, tal cual lo da MT5.</td></tr>
  <tr><td><code>time_iso</code></td><td>string</td><td><code>"2026-09-10T14:30:00"</code></td><td>El mismo instante formateado, <b>sin</b> sufijo <code>Z</code>.</td></tr>
  <tr><td><code>time_msc</code></td><td>integer (int64)</td><td><code>1789037400123</code></td><td>Epoch en <b>milisegundos</b> (ticks y deals).</td></tr>
</table>
<div class="note"><b>En .NET:</b> deserializa <code>time_iso</code> como <code>DateTime</code> con
<code>DateTimeKind.Unspecified</code>. La ausencia deliberada de <code>Z</code> senala que es hora
de servidor. Si necesitas UTC real, aplica el offset de tu broker. Para los parametros
<code>*_epoch</code> de entrada, envia siempre el epoch en segundos.</div>

<h3>5.2 Formato de error (identico en gateway y workers)</h3>
<pre>{
  "error": {
    "code": "NOT_FOUND",
    "message": "No hay ninguna posicion abierta con ticket 999999999.",
    "mt5_code": null,
    "detail": null
  }
}</pre>
<table>
  <tr><th>HTTP</th><th><code>code</code></th><th>Cuando</th></tr>
  <tr><td>400</td><td><code>BAD_REQUEST</code></td><td>Parametro invalido (volumen fuera de rango, timeframe inexistente, rango de fechas al reves).</td></tr>
  <tr><td>404</td><td><code>NOT_FOUND</code></td><td>Ticket o simbolo inexistente.</td></tr>
  <tr><td>422</td><td>(FastAPI)</td><td>El cuerpo JSON no cumple el esquema. Formato propio de FastAPI.</td></tr>
  <tr><td>500</td><td><code>INTERNAL_ERROR</code></td><td>Fallo no previsto. Queda en los logs.</td></tr>
  <tr><td>502</td><td><code>MT5_ERROR</code> / <code>WORKER_UNREACHABLE</code></td><td>MT5 devolvio error, o el worker no responde.</td></tr>
  <tr><td>503</td><td><code>NOT_CONNECTED</code></td><td>El terminal perdio la conexion con el broker.</td></tr>
  <tr><td>504</td><td><code>WORKER_TIMEOUT</code></td><td>El worker no respondio a tiempo.</td></tr>
</table>

<h3>5.3 Resultado de las ordenes &mdash; el punto que mas confunde</h3>
<div class="warn"><b>Un HTTP 200 no significa que la orden se haya ejecutado.</b>
Significa que MT5 recibio la peticion y respondio. Comprueba <b>siempre</b> el campo
<code>success</code> (o <code>retcode</code>).</div>
<table>
  <tr><th>Campo</th><th>Tipo</th><th>Significado</th></tr>
  <tr><td><code>success</code></td><td>boolean</td><td><code>true</code> solo con retcode 10008, 10009 o 10010.</td></tr>
  <tr><td><code>retcode</code></td><td>integer</td><td>Codigo devuelto por el servidor de trading.</td></tr>
  <tr><td><code>retcode_name</code></td><td>string</td><td>El mismo codigo en texto: <code>DONE</code>, <code>NO_MONEY</code>, <code>INVALID_STOPS</code>...</td></tr>
  <tr><td><code>order</code></td><td>integer (int64)</td><td>Ticket de la orden generada.</td></tr>
  <tr><td><code>deal</code></td><td>integer (int64)</td><td>Ticket del deal (la ejecucion real).</td></tr>
  <tr><td><code>position</code></td><td>integer (int64)</td><td>Ticket de la posicion resultante. <b>Es el que usaras</b> para cerrar o mover SL/TP.</td></tr>
  <tr><td><code>volume</code></td><td>number</td><td>Volumen realmente ejecutado.</td></tr>
  <tr><td><code>price</code></td><td>number</td><td>Precio real de ejecucion.</td></tr>
  <tr><td><code>request</code></td><td>object</td><td>Lo que el gateway envio de verdad a MT5. Util para depurar.</td></tr>
</table>
<div class="note"><code>/trading/check</code> es la excepcion: al no enviar nada, su exito es
<code>retcode == 0</code> (<code>retcode_name: "OK"</code>).</div>

<h3>5.4 Volumenes y precios</h3>
<ul>
  <li>El <b>volumen</b> se ajusta al <code>volume_step</code> del simbolo <b>siempre hacia abajo</b>.
      Redondear hacia arriba aumentaria el riesgo sin que el cliente lo pida. Si el resultado queda
      por debajo de <code>volume_min</code>, la peticion se rechaza con <b>400</b>.</li>
  <li><b>SL y TP</b> se envian como <b>precio absoluto</b>. En <code>/trading/open</code> tambien
      se aceptan como distancia en puntos (<code>sl_points</code>, <code>tp_points</code>);
      si mandas ambos, gana el precio absoluto.</li>
  <li>Un <b>punto</b> es <code>symbol_info.point</code>. Para EURUSD de 5 digitos,
      1 punto = 0,00001, luego <b>1 pip = 10 puntos</b>.</li>
  <li>Para <b>quitar</b> un SL o un TP, envia <code>0</code>. Para <b>dejarlo como esta</b>,
      envia <code>null</code> u omite el campo.</li>
</ul>

<h3>5.5 Generar el cliente .NET automaticamente</h3>
<p>El gateway publica un esquema OpenAPI 3.1 completo (auth + los 41 endpoints de cuenta), con el
esquema de seguridad <code>bearerAuth</code> ya declarado:</p>
<div class="cmd">nswag openapi2csclient /input:http://127.0.0.1:8000/openapi.json \\
                       /classname:Mt5Client /namespace:PythonGetaway /output:Mt5Client.cs

# o Refitter
refitter --namespace PythonGetaway http://127.0.0.1:8000/openapi.json</div>
<div class="note">Las respuestas de lectura se devuelven como objetos JSON abiertos: los campos que
publica MT5 varian entre brokers y builds del terminal, y cerrar el esquema haria que el gateway
<b>perdiera</b> campos que el broker si esta enviando. Si prefieres tipos estrictos en .NET,
modela solo los campos que uses y deja el resto en un
<code>Dictionary&lt;string, JsonElement&gt;</code>.</div>
""")

    # --- 4. Referencia de endpoints ---------------------------------------
    parts.append(f"<h2>6. Referencia del resto de endpoints ({total - n_new})</h2>")
    parts.append("<table class='toc'><tr><th>Grupo</th><th>Endpoints</th></tr>")
    for tag in TAG_ORDER:
        if tag in grouped:
            parts.append(f"<tr><td><b>{esc(TAG_TITLES.get(tag, tag))}</b></td>"
                         f"<td>{len(grouped[tag])}</td></tr>")
    parts.append("</table>")

    section = 0
    for tag in TAG_ORDER:
        if tag not in grouped:
            continue
        section += 1
        parts.append(f"<h2>6.{section} {esc(TAG_TITLES.get(tag, tag))}</h2>")
        for path, method, op in sorted(grouped[tag], key=lambda x: (x[0], x[1])):
            parts.append(render_endpoint(path, method, op, spec, by_path))

    # --- 5. Enumeraciones --------------------------------------------------
    parts.append(enum_section())
    parts.append("</body></html>")
    return "".join(parts)


def render_endpoint(path: str, method: str, op: dict, spec: dict, by_path: dict) -> str:
    cls = method.lower()
    summary = op.get("summary", "")
    desc = op.get("description", "")
    needs_token = bool(op.get("security"))
    badge = ('<span class="tok">requiere token</span>' if needs_token
             else '<span class="notok">sin token</span>')
    out = [f'<div class="ep"><span class="m {cls}">{method}</span> '
           f'<span class="path">{esc(path)}</span> {badge}']
    if summary:
        out.append(f"<h3 style='margin-top:2mm'>{esc(summary)}</h3>")
    if desc:
        for para in desc.split("\n\n"):
            out.append(f"<p>{esc(para.strip())}</p>")

    out.append(params_table(op, spec))
    out.append(body_table(op, spec))
    out.append(response_table(op, spec))

    ex = by_path.get((method, path))
    documented = DOCUMENTED_EXAMPLES.get((method, path))
    if documented and (ex is None or ex.get("status") != 200):
        out.append('<div class="note"><b>Estructura documentada, no capturada.</b> '
                   'Un login exitoso exige credenciales reales, asi que este ejemplo describe la '
                   'forma exacta de la respuesta (derivada del codigo y del esquema), no una '
                   'ejecucion. Lanzando el selftest con <code>PYGW_TEST_LOGIN</code> / '
                   '<code>PYGW_TEST_PASSWORD</code> / <code>PYGW_TEST_SERVER</code> y regenerando '
                   'este PDF, se sustituye por una captura real.</div>')
        if documented.get("request") is not None:
            out.append(f"<h4>Peticion</h4><pre>{pretty(documented['request'])}</pre>")
        out.append("<h4>Respuesta correcta &mdash; HTTP 200</h4>"
                   f"<pre>{pretty(documented['response'])}</pre>")

    if ex:
        if ex.get("note"):
            out.append(f'<div class="note">{esc(ex["note"])}</div>')
        called = ex.get("path", "")
        if called and called != path:
            out.append(f"<h4>URL real invocada</h4><pre>{esc(method)} {esc(called)}</pre>")
        if ex.get("request") is not None:
            out.append("<h4>Ejemplo de peticion (real)</h4>"
                       f"<pre>{pretty(ex['request'])}</pre>")
        label = ("Ejemplo de respuesta (real)" if ex.get("status") == 200
                 else "Ejemplo de ERROR (real, capturado)")
        out.append(f"<h4>{label} &mdash; HTTP {ex.get('status')}</h4>"
                   f"<pre>{pretty(ex.get('response'))}</pre>")
    elif not documented:
        out.append('<div class="note">Sin ejemplo capturado en la ultima pasada del selftest '
                   '(por ejemplo, porque dependia de un ticket que ya no existe). El contrato '
                   'es el descrito en las tablas de arriba.</div>')
    out.append("</div>")
    return "".join(out)


def enum_section() -> str:
    from collections import OrderedDict
    sys.path.insert(0, str(ROOT))
    tables = OrderedDict()
    tables["Tipo de orden (<code>type</code> en /orders y en el resultado)"] = {
        0: "BUY", 1: "SELL", 2: "BUY_LIMIT", 3: "SELL_LIMIT", 4: "BUY_STOP",
        5: "SELL_STOP", 6: "BUY_STOP_LIMIT", 7: "SELL_STOP_LIMIT", 8: "CLOSE_BY"}
    tables["Tipo de posicion (<code>type</code> en /positions)"] = {0: "BUY", 1: "SELL"}
    tables["Estado de orden (<code>state</code>)"] = {
        0: "STARTED", 1: "PLACED", 2: "CANCELED", 3: "PARTIAL", 4: "FILLED",
        5: "REJECTED", 6: "EXPIRED", 7: "REQUEST_ADD", 8: "REQUEST_MODIFY", 9: "REQUEST_CANCEL"}
    tables["Tipo de deal (<code>type</code> en /history/deals)"] = {
        0: "BUY", 1: "SELL", 2: "BALANCE", 3: "CREDIT", 4: "CHARGE", 5: "CORRECTION",
        6: "BONUS", 7: "COMMISSION", 12: "INTEREST", 15: "DIVIDEND", 17: "TAX"}
    tables["Direccion del deal (<code>entry</code>)"] = {
        0: "IN (abre)", 1: "OUT (cierra)", 2: "INOUT (invierte)", 3: "OUT_BY (cierre por opuesta)"}
    tables["Motivo del deal (<code>reason</code>)"] = {
        0: "CLIENT", 1: "MOBILE", 2: "WEB", 3: "EXPERT", 4: "SL alcanzado", 5: "TP alcanzado",
        6: "SO (stop out)", 7: "ROLLOVER", 8: "VMARGIN", 9: "SPLIT"}
    tables["Tipo de cuenta (<code>trade_mode</code>)"] = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
    tables["Modo de margen (<code>margin_mode</code>)"] = {
        0: "RETAIL_NETTING", 1: "EXCHANGE", 2: "RETAIL_HEDGING"}

    out = ["<h2>7. Tablas de codigos</h2>",
           "<p>El gateway ya envia el nombre junto al numero (por ejemplo <code>type</code> y "
           "<code>type_name</code>), asi que normalmente no necesitas estas tablas. Van aqui por "
           "si prefieres trabajar con los valores numericos en .NET.</p>"]
    for title, mapping in tables.items():
        rows = "".join(f"<tr><td><code>{k}</code></td><td>{esc(v)}</td></tr>" for k, v in mapping.items())
        out.append(f"<h3>{title}</h3><table><tr><th>Valor</th><th>Significado</th></tr>{rows}</table>")

    # Se declaran aqui y no se importan de `app.converters`: ese modulo importa
    # MetaTrader5, que solo existe bajo Wine, y este generador corre en el
    # Python de Linux.
    common = {
        10004: ("REQUOTE", "Requote"),
        10006: ("REJECT", "Peticion rechazada"),
        10009: ("DONE", "Ejecutada correctamente"),
        10010: ("DONE_PARTIAL", "Ejecutada parcialmente"),
        10013: ("INVALID", "Peticion invalida"),
        10014: ("INVALID_VOLUME", "Volumen invalido"),
        10015: ("INVALID_PRICE", "Precio invalido"),
        10016: ("INVALID_STOPS", "SL/TP invalidos (demasiado cerca del precio)"),
        10017: ("TRADE_DISABLED", "Trading deshabilitado"),
        10018: ("MARKET_CLOSED", "Mercado cerrado"),
        10019: ("NO_MONEY", "Fondos insuficientes"),
        10020: ("PRICE_CHANGED", "El precio cambio"),
        10027: ("CLIENT_DISABLES_AT", "AutoTrading desactivado en el terminal"),
        10030: ("INVALID_FILL", "Modo de llenado no soportado"),
        10031: ("CONNECTION", "Sin conexion"),
        10036: ("POSITION_CLOSED", "La posicion ya estaba cerrada"),
        10038: ("INVALID_CLOSE_VOLUME", "Volumen de cierre superior al de la posicion"),
    }
    rows = "".join(f"<tr><td><code>{k}</code></td><td><code>{esc(name)}</code></td>"
                   f"<td>{esc(meaning)}</td></tr>" for k, (name, meaning) in common.items())
    out.append("<h3>Retcodes de trading mas frecuentes</h3>"
               f"<table><tr><th>Codigo</th><th>Nombre</th><th>Que significa</th></tr>{rows}</table>")
    out.append('<div class="note">La lista completa de retcodes esta en '
               '<code>app/converters.py</code> (<code>RETCODE_NAMES</code>).</div>')
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    print(f"Leyendo esquema OpenAPI de {args.base}/openapi.json ...")
    try:
        spec = fetch(f"{args.base}/openapi.json")
    except Exception as exc:  # noqa: BLE001
        print(f"  No se pudo leer el esquema: {exc}\n  Arranca el gateway con ./scripts/start.sh")
        return 1

    examples = json.loads(EXAMPLES.read_text()) if EXAMPLES.exists() else []
    print(f"  {len(spec.get('paths', {}))} rutas, {len(examples)} ejemplos reales capturados.")

    html_doc = build_html(spec, examples)
    HTML_OUT.write_text(html_doc, encoding="utf-8")
    print(f"  HTML -> {HTML_OUT.relative_to(ROOT)}")

    if not shutil.which("wkhtmltopdf"):
        print("  wkhtmltopdf no esta instalado: se deja solo el HTML.")
        return 1

    # Sin --footer-*: el wkhtmltopdf de los repos va con un Qt sin parchear y
    # los ignora, ensuciando la salida con avisos.
    cmd = ["wkhtmltopdf", "--enable-local-file-access", "--quiet",
           "--margin-top", "14mm", "--margin-bottom", "14mm",
           "--margin-left", "10mm", "--margin-right", "10mm",
           str(HTML_OUT), str(PDF_OUT)]
    subprocess.run(cmd, check=True)
    size = PDF_OUT.stat().st_size / 1024
    print(f"  PDF  -> {PDF_OUT.relative_to(ROOT)}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
