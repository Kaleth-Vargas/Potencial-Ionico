"""Experimento: ¿mejora el gemelo si se reentrena con TODOS los puntos
reales ya pagados (100 base + 16 eps1 + 16 eps5 + 16 eps10 = 148)?

Compara contra el modelo entregado (116 puntos) en el mismo test real de
394 puntos. Sin llamadas nuevas a SIMION.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from real_data import split_train_test, MAX_IONS
from gp_model import BeamlineTwin

SEED = 42
_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"

X_tr, y_tr, X_te, y_te = split_train_test(n_train=100, seed=SEED)
c1 = np.load(_DATA_DIR / "gradient_probe_cache.npz")
c2 = dict(np.load(_DATA_DIR / "gradient_eps_study_cache.npz"))

variants = {
    "entregado (116)": [(c1["X_probe"], c1["y_probe"])],
    "completo (148)": [(c1["X_probe"], c1["y_probe"]),
                       (c2["X_eps5"], c2["y_eps5"]),
                       (c2["X_eps10"], c2["y_eps10"])],
    "+eps5 (132)": [(c1["X_probe"], c1["y_probe"]),
                    (c2["X_eps5"], c2["y_eps5"])],
    "+eps10 (132)": [(c1["X_probe"], c1["y_probe"]),
                     (c2["X_eps10"], c2["y_eps10"])],
    "solo eps5+eps10 (132)": [(c2["X_eps5"], c2["y_eps5"]),
                              (c2["X_eps10"], c2["y_eps10"])],
}

mask = y_te > 0
true_sig = (y_te > 0).astype(int)

for name, extras in variants.items():
    X = np.vstack([X_tr] + [e[0] for e in extras])
    y = np.concatenate([y_tr] + [e[1] for e in extras])
    twin = BeamlineTwin(seed=SEED).fit(X, y)

    mu, std = twin.predict_combined(X_te)
    p = twin.predict_proba_signal(X_te)
    pred_sig = (p >= 0.5).astype(int)

    r2_sig = twin.score_r2(X_te[mask], y_te[mask])
    mae_sig = np.mean(np.abs(y_te[mask] - mu[mask]))
    admis = np.all((mu >= 0) & (mu <= MAX_IONS))

    print(f"\n{name}: n_train={len(y)}")
    print(f"  R2 region informativa : {r2_sig:+.3f}")
    print(f"  MAE region informativa: {mae_sig:.2f} iones")
    print(f"  clasificador acc/prec/rec: {accuracy_score(true_sig, pred_sig):.3f} / "
          f"{precision_score(true_sig, pred_sig):.3f} / {recall_score(true_sig, pred_sig):.3f}")
    print(f"  admisible: {admis}")
