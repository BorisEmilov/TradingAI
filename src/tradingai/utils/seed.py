"""Semilla aleatoria fija para que los entrenamientos sean reproducibles.

Sin esto, cada corrida parte de pesos iniciales distintos y el orden de barajado
del DataLoader tambien cambia, haciendo imposible distinguir "mejor modelo" de
"mejor tirada de dados" al comparar resultados entre corridas (ver walk-forward
del 2026-08-22: varianza enorme entre folds, en parte por esto).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
