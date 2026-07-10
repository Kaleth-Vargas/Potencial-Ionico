"""
real_data.py — Carga y partición de los datos REALES de SIMION (no sintéticos).

Fuente: data/beamline_results.csv, generado por simulation/optimizer.py
(Optuna + SIMION real). De 824 corridas, 330 fallaron (value == -1, error
de subproceso) y quedan 494 válidas.

Presupuesto de entrenamiento = 100 (asignado por el instructor). Como ya
existen 494 puntos reales válidos, la estrategia es:

  1. Separar los 494 en "señal" (value > 0, 130 puntos) y "cero" (value == 0,
     364 puntos) — la región informativa y la zona muerta.
  2. Elegir 100 puntos por farthest-point sampling, estratificado: una cuota
     fija de cada grupo, para que el pico de transmisión quede bien cubierto
     aunque sea la minoría de los datos (ver advertencia del PDF sobre la
     "región escasa").
  3. El resto (~394 puntos) queda como conjunto de test real y gratuito
     (no consume presupuesto porque ya estaba hecho).
"""

import pathlib

import numpy as np
import pandas as pd

ELECTRODE_IDS = [3, 6, 9, 10, 11, 12, 15, 18]
MAX_IONS = 500
V_LOW, V_HIGH = -1000.0, 1000.0

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = _REPO_ROOT / "data" / "beamline_results.csv"

N_TRAIN = 100
N_TRAIN_SIGNAL = 45   # cuota de puntos con señal en el training set
N_TRAIN_ZERO   = N_TRAIN - N_TRAIN_SIGNAL


def load_raw(csv_path: pathlib.Path = CSV_PATH):
    """
    Lee beamline_results.csv y devuelve (X, y) solo con las filas válidas
    (state == COMPLETE y value != -1, es decir, sin errores de subproceso).

    Retorna
    -------
    X : array (n, 8) en el orden ELECTRODE_IDS = [3,6,9,10,11,12,15,18]
    y : array (n,) transmisión real (iones en el detector)
    """
    df = pd.read_csv(csv_path)
    df = df[(df["state"] == "COMPLETE") & (df["value"] != -1.0)].copy()

    cols = [f"params_V{e}" for e in ELECTRODE_IDS]
    X = df[cols].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    return X, y


def _farthest_point_sample(X: np.ndarray, n_pick: int, seed: int) -> np.ndarray:
    """
    Muestreo de máxima distancia mínima (farthest-point sampling).

    Empieza en un punto aleatorio y en cada paso agrega el punto candidato
    cuya distancia al punto ya elegido MAS CERCANO es la MAS GRANDE posible
    (el punto "más solo"). Da una cobertura pareja del espacio en vez de
    puntos apiñados.

    Retorna los índices (dentro de X) de los n_pick puntos elegidos.
    """
    n = X.shape[0]
    n_pick = min(n_pick, n)
    rng = np.random.RandomState(seed)

    chosen = [int(rng.randint(n))]
    # min_dist[i] = distancia de X[i] al punto elegido mas cercano hasta ahora
    min_dist = np.linalg.norm(X - X[chosen[0]], axis=1)

    while len(chosen) < n_pick:
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        new_dist = np.linalg.norm(X - X[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    return np.array(chosen)


def split_train_test(
    n_train: int = N_TRAIN,
    n_train_signal: int = N_TRAIN_SIGNAL,
    seed: int = 42,
    csv_path: pathlib.Path = CSV_PATH,
):
    """
    Carga los datos reales y separa 100 puntos de entrenamiento (diseñados
    por farthest-point sampling, estratificado señal/cero) y el resto como
    test real.

    Retorna
    -------
    X_train, y_train : (n_train, 8) y (n_train,)
    X_test,  y_test   : (~394, 8) y (~394,)
    """
    X, y = load_raw(csv_path)

    signal_mask = y > 0
    X_sig, y_sig = X[signal_mask], y[signal_mask]
    X_zero, y_zero = X[~signal_mask], y[~signal_mask]

    n_train_zero = n_train - n_train_signal

    idx_sig  = _farthest_point_sample(X_sig,  n_train_signal, seed=seed)
    idx_zero = _farthest_point_sample(X_zero, n_train_zero,   seed=seed + 1)

    X_train = np.vstack([X_sig[idx_sig],  X_zero[idx_zero]])
    y_train = np.concatenate([y_sig[idx_sig], y_zero[idx_zero]])

    # El resto (lo NO elegido) es el conjunto de test real
    mask_sig_test  = np.ones(len(X_sig),  dtype=bool)
    mask_sig_test[idx_sig] = False
    mask_zero_test = np.ones(len(X_zero), dtype=bool)
    mask_zero_test[idx_zero] = False

    X_test = np.vstack([X_sig[mask_sig_test], X_zero[mask_zero_test]])
    y_test = np.concatenate([y_sig[mask_sig_test], y_zero[mask_zero_test]])

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = split_train_test()

    print(f"Train: {len(y_train)} puntos "
          f"({np.sum(y_train > 0)} con señal, {np.sum(y_train == 0)} en cero)")
    print(f"Test : {len(y_test)} puntos "
          f"({np.sum(y_test > 0)} con señal, {np.sum(y_test == 0)} en cero)")
    print(f"Max señal train: {y_train.max():.0f}   Max señal test: {y_test.max():.0f}")

    # Chequeo de cobertura: distancia promedio de cada punto de train a su
    # vecino de train mas cercano (mientras mas grande, mas parejo el cubrimiento)
    from scipy.spatial.distance import pdist
    d = pdist(X_train)
    print(f"Distancia entre puntos de train: media={d.mean():.1f}, "
          f"minima={d.min():.1f} (voltajes en rango [{V_LOW},{V_HIGH}])")
