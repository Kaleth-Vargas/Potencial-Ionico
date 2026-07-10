"""Experimento: ¿mejora el regresor de magnitud con kernel Matern en vez de RBF?

Mismo entrenamiento del modelo entregado (116 puntos), mismo test real (394).
Sin llamadas nuevas a SIMION.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

import gp_model
from gp_model import BeamlineTwin, N_AUG_DIMS
from real_data import split_train_test

SEED = 42
_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"

X_tr, y_tr, X_te, y_te = split_train_test(n_train=100, seed=SEED)
c1 = np.load(_DATA_DIR / "gradient_probe_cache.npz")
X = np.vstack([X_tr, c1["X_probe"]])
y = np.concatenate([y_tr, c1["y_probe"]])

mask = y_te > 0


def make_kernel(kind):
    if kind == "rbf":
        base = RBF(length_scale=np.ones(N_AUG_DIMS) * 100.0,
                   length_scale_bounds=(1e-2, 1e5))
    else:
        nu = {"matern15": 1.5, "matern25": 2.5}[kind]
        base = Matern(length_scale=np.ones(N_AUG_DIMS) * 100.0,
                      length_scale_bounds=(1e-2, 1e5), nu=nu)
    return (ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * base
            + WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 100.0)))


for kind in ["rbf", "matern15", "matern25"]:
    twin = BeamlineTwin(seed=SEED)
    twin.magnitude_model.gp.kernel = make_kernel(kind)
    twin.fit(X, y)
    mu, _ = twin.predict_combined(X_te)
    r2 = twin.score_r2(X_te[mask], y_te[mask])
    mae = np.mean(np.abs(y_te[mask] - mu[mask]))
    print(f"{kind:9s}: R2 informativa = {r2:+.3f}   MAE = {mae:.2f} iones")
