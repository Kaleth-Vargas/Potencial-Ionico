"""Verificación con SIMION real de los candidatos de control del gemelo v2
(Matérn 1.5 + consigna anclada al punto real más cercano al objetivo).

Corridas de verificación — no cuentan contra el presupuesto de entrenamiento.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from real_data import split_train_test, ELECTRODE_IDS
from gp_model import BeamlineTwin
from control import find_optimal_voltages, find_inverse_setpoint
import simion_interface

SEED = 42
TARGET = 35.0
_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"

X_tr, y_tr, _, _ = split_train_test(n_train=100, seed=SEED)
c1 = np.load(_DATA_DIR / "gradient_probe_cache.npz")
X = np.vstack([X_tr, c1["X_probe"]])
y = np.concatenate([y_tr, c1["y_probe"]])

twin = BeamlineTwin(seed=SEED).fit(X, y)
print("Gemelo v2 (Matern 1.5) ajustado con 116 puntos.\n")

# Direccion
best_v, best_pred = find_optimal_voltages(twin, n_restarts=25, seed=SEED)
print(f"DIRECCION  — predicho {best_pred:.1f}")
mean, std = simion_interface.evaluate_one_real_averaged(best_v, n_repeats=5)
print(f"             real {mean:.1f} +- {std:.1f}   (C = {mean/72:.3f} del optimo 72)")
print(f"             V = {np.round(best_v, 2).tolist()}\n")

# Consigna inversa, dos radios
for radius in (5.0, 10.0):
    inv_v, inv_pred = find_inverse_setpoint(twin, target=TARGET, n_restarts=25,
                                            seed=SEED + 1, trust_radius=radius)
    mean, std = simion_interface.evaluate_one_real_averaged(inv_v, n_repeats=5)
    print(f"CONSIGNA r={radius:.0f}V — predicho {inv_pred:.1f}, real {mean:.1f} +- {std:.1f}, "
          f"error real = {abs(mean - TARGET):.1f} iones")
    print(f"             V = {np.round(inv_v, 2).tolist()}\n")
