"""Chequeo rápido (sin SIMION): ¿mejora el gradiente si el gemelo se
reentrena con las sondas reales de eps=10V ya pagadas?

Se evalúa contra el gradiente real de eps=5V como HELD-OUT (esas sondas no
entran al entrenamiento en la variante honesta), para no ser circular.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from real_data import split_train_test
from gp_model import BeamlineTwin
from control import best_known_real_point
from gradient_eps_study import twin_gradient_central, real_gradient_from_probes, cosine

_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"

X_tr, y_tr, _, _ = split_train_test(n_train=100, seed=42)
c1 = np.load(_DATA_DIR / "gradient_probe_cache.npz")
c2 = dict(np.load(_DATA_DIR / "gradient_eps_study_cache.npz"))
anchor = best_known_real_point()

grad_real_5 = real_gradient_from_probes(c2["y_eps5"], eps=5.0)

combos = {
    "base (100+16 eps1)": [(c1["X_probe"], c1["y_probe"])],
    "+eps10 (held-out eps5)": [(c1["X_probe"], c1["y_probe"]),
                               (c2["X_eps10"], c2["y_eps10"])],
    "+eps5+eps10 (circular)": [(c1["X_probe"], c1["y_probe"]),
                               (c2["X_eps10"], c2["y_eps10"]),
                               (c2["X_eps5"], c2["y_eps5"])],
}
for name, extras in combos.items():
    X = np.vstack([X_tr] + [e[0] for e in extras])
    y = np.concatenate([y_tr] + [e[1] for e in extras])
    twin = BeamlineTwin(seed=42).fit(X, y)
    g5 = twin_gradient_central(twin, anchor, eps=5.0)
    print(f"{name:26s} n={len(y):3d}  cos(vs real eps=5) = {cosine(g5, grad_real_5):+.3f}")
