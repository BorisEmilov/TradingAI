"""Carga de configuracion: variables sensibles desde .env, resto desde config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = Path(__file__).resolve().parent / "config.yaml"


class Secrets(BaseSettings):
    """Valores sensibles / por entorno, leidos desde .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Credenciales del terminal MT5: las usa wine_bridge/server.py (dentro de Wine),
    # no el lado Linux. Aqui solo vive la URL para hablar con ese bridge por HTTP.
    mt5_bridge_url: str = "http://127.0.0.1:18812"

    trading_mode: str = Field(default="paper")
    risk_per_trade_pct: float = Field(default=1.0)
    max_open_positions: int = Field(default=3)

    log_level: str = Field(default="INFO")


def load_yaml_config(path: Path = CONFIG_YAML_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> dict:
    """Punto unico de acceso a la configuracion combinada (yaml + secrets)."""
    secrets = Secrets()
    yaml_config = load_yaml_config()
    return {"secrets": secrets, **yaml_config}


if __name__ == "__main__":
    import json

    cfg = get_settings()
    print(json.dumps({k: v for k, v in cfg.items() if k != "secrets"}, indent=2))
    print("secrets loaded:", cfg["secrets"].model_dump())
