"""Informacion de cuenta MT5 (balance, equity, margen) usada por el gestor de riesgo."""

from __future__ import annotations

from dataclasses import dataclass

from tradingai.mt5.connector import MT5Connector


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin_free: float
    currency: str
    leverage: int


def get_account_info(connector: MT5Connector) -> AccountInfo:
    info = connector.get_account_info()
    return AccountInfo(
        balance=info["balance"],
        equity=info["equity"],
        margin_free=info["margin_free"],
        currency=info["currency"],
        leverage=info["leverage"],
    )
