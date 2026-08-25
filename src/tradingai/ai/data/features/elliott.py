"""Ondas de Elliott con confirmaciones Fibonacci: aproximacion algoritmica basada en
reglas de ratio, NO un contador de ondas definitivo.

Aviso honesto: el conteo de ondas de Elliott "real" es notoriamente subjetivo incluso
entre analistas profesionales (el mismo tramo de precio admite mas de un conteo valido).
Lo que se implementa aqui es la parte objetivamente verificable: un ZigZag por
desviacion porcentual (pivotes confirmados solo cuando el precio se mueve
`deviation_pct` desde el extremo, sin look-ahead) mas las validaciones de ratio
Fibonacci clasicas entre los ultimos 5 tramos (ondas 1-2-3-4-5):
  - onda 2 retrocede entre 38.2%-88.6% de la onda 1
  - onda 3 no es mas corta que la onda 1 (simplificacion de "onda 3 nunca es la mas
    corta entre 1, 3 y 5")
  - onda 4 no se solapa con el territorio de precio de la onda 1
  - onda 4 retrocede entre 14.6%-61.8% de la onda 3

`elliott_impulse_confidence` resume cuantas de esas 4 reglas se cumplen (0.0-1.0) para
el ultimo conteo tentativo de 5 ondas -> el modelo puede usarlo como confluencia sin
necesitar interpretar cada ratio por separado.

Los valores se mantienen constantes (forward-filled) entre una confirmacion de pivote
y la siguiente, igual que `trend` en market_structure.py: representan "el conteo
vigente hasta ahora", no una prediccion de eventos futuros -> causal por construccion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_elliott_features(df: pd.DataFrame, deviation_pct: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    high, low, close = out["high"].to_numpy(), out["low"].to_numpy(), out["close"].to_numpy()

    wave2_retrace = np.zeros(n)
    wave3_ext = np.zeros(n)
    wave4_retrace = np.zeros(n)
    wave4_overlap = np.zeros(n, dtype=bool)
    wave5_ratio = np.zeros(n)
    impulse_confidence = np.zeros(n)
    direction_arr = np.zeros(n, dtype=int)

    # Se siembra con el precio de arranque como "P0" (punto 0 del conteo, no es en si
    # una reversion) para que un impulso de 5 ondas (P0..P5) produzca 6 pivotes, no 5.
    pivots: list[float] = [float(close[0])]
    running_max, running_max_idx = close[0], 0
    running_min, running_min_idx = close[0], 0
    trend = 0  # 0=indefinido, 1=buscando reversion desde un maximo, -1=desde un minimo

    cur_wave2 = cur_wave3 = cur_wave4 = cur_wave5 = cur_conf = 0.0
    cur_overlap = False
    cur_dir = 0

    for i in range(1, n):
        if high[i] > running_max:
            running_max, running_max_idx = high[i], i
        if low[i] < running_min:
            running_min, running_min_idx = low[i], i

        down_dev = (running_max - low[i]) / running_max * 100 if running_max > 0 else 0.0
        up_dev = (high[i] - running_min) / running_min * 100 if running_min > 0 else 0.0

        confirmed = False
        if trend == 0:
            # Bootstrap: ambos extremos parten del mismo precio semilla, asi que un
            # pequeno ruido bidireccional en las primeras velas podria disparar las dos
            # ramas casi a la vez. Se resuelve por magnitud (cual desviacion es mayor),
            # no por orden de chequeo, para no confirmar un pivote espurio en la semilla.
            if down_dev >= deviation_pct and down_dev >= up_dev:
                pivots.append(float(running_max))
                trend = -1
                running_min, running_min_idx = low[i], i
                running_max, running_max_idx = high[i], i
                confirmed = True
            elif up_dev >= deviation_pct and up_dev > down_dev:
                pivots.append(float(running_min))
                trend = 1
                running_max, running_max_idx = high[i], i
                running_min, running_min_idx = low[i], i
                confirmed = True
        elif trend > 0 and down_dev >= deviation_pct:
            pivots.append(float(running_max))
            trend = -1
            running_min, running_min_idx = low[i], i
            running_max, running_max_idx = high[i], i
            confirmed = True
        elif trend < 0 and up_dev >= deviation_pct:
            pivots.append(float(running_min))
            trend = 1
            running_max, running_max_idx = high[i], i
            running_min, running_min_idx = low[i], i
            confirmed = True

        if confirmed and len(pivots) >= 6:
            p1, p2, p3, p4, p5, p6 = pivots[-6:]
            leg1, leg2, leg3, leg4, leg5 = abs(p2 - p1), abs(p3 - p2), abs(p4 - p3), abs(p5 - p4), abs(p6 - p5)

            cur_wave2 = leg2 / leg1 if leg1 > 0 else 0.0
            cur_wave3 = leg3 / leg1 if leg1 > 0 else 0.0
            cur_wave4 = leg4 / leg3 if leg3 > 0 else 0.0
            cur_wave5 = leg5 / leg1 if leg1 > 0 else 0.0
            cur_dir = 1 if p2 > p1 else -1
            cur_overlap = (p5 < p2) if cur_dir == 1 else (p5 > p2)

            rule_a = 0.382 <= cur_wave2 <= 0.886
            rule_b = cur_wave3 >= 1.0
            rule_c = not cur_overlap
            rule_d = 0.146 <= cur_wave4 <= 0.618
            cur_conf = (rule_a + rule_b + rule_c + rule_d) / 4.0

        wave2_retrace[i] = cur_wave2
        wave3_ext[i] = cur_wave3
        wave4_retrace[i] = cur_wave4
        wave4_overlap[i] = cur_overlap
        wave5_ratio[i] = cur_wave5
        impulse_confidence[i] = cur_conf
        direction_arr[i] = cur_dir

    out["elliott_wave2_retrace_pct"] = wave2_retrace
    out["elliott_wave3_extension_ratio"] = wave3_ext
    out["elliott_wave4_retrace_pct"] = wave4_retrace
    out["elliott_wave4_overlap"] = wave4_overlap
    out["elliott_wave5_ratio"] = wave5_ratio
    out["elliott_impulse_confidence"] = impulse_confidence
    out["elliott_direction"] = direction_arr
    return out
