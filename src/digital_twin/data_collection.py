"""
data_collection.py — Recolección de datos (Lección 1 y 6).

Ya no genera datos sintéticos. La estrategia de recolección real es:

  1. Partir de los 494 puntos reales válidos que ya generó optimizer.py
     (Optuna + SIMION real) en data/beamline_results.csv.
  2. Elegir 100 de ellos por farthest-point sampling estratificado
     (señal / cero) como presupuesto de entrenamiento declarado — ver
     real_data.py.
  3. (Opcional) Si el grupo decide gastar MÁS presupuesto real, expandir_activo()
     corre SIMION real (simion_interface.evaluate_batch_real) en los puntos
     donde el gemelo tiene mayor incertidumbre. Esto SÍ ejecuta SIMION, así
     que llámenlo solo cuando decidan gastar ese presupuesto adicional.
"""

import numpy as np

from real_data import split_train_test, V_LOW, V_HIGH, ELECTRODE_IDS

N_CANDIDATES = 8_000


def load_training_set(n_train: int = 100, seed: int = 42):
    """Wrapper de real_data.split_train_test(): 100 train reales / resto test real."""
    return split_train_test(n_train=n_train, seed=seed)


def select_high_variance(twin_model, n_new: int, n_candidates: int = N_CANDIDATES,
                          seed: int = 42) -> np.ndarray:
    """
    Elige los n_new puntos donde el hurdle model tiene mayor incertidumbre
    combinada (predict_combined), de un pool aleatorio denso de candidatos.

    Retorna array (n_new, 8).
    """
    rng = np.random.RandomState(seed)
    candidates = rng.uniform(V_LOW, V_HIGH, size=(n_candidates, len(ELECTRODE_IDS)))
    _, std = twin_model.predict_combined(candidates)
    top_idx = np.argsort(std)[-n_new:]
    return candidates[top_idx]


def expandir_activo(twin_model, n_new: int, seed: int = 42) -> tuple:
    """
    OPCIONAL — expande el presupuesto de entrenamiento corriendo SIMION REAL
    en los puntos de mayor incertidumbre. Esto SÍ ejecuta simion.exe n_new
    veces; solo llamar si el grupo decide gastar más presupuesto del
    declarado (100), y anotarlo en el informe.

    Retorna
    -------
    X_new : array (n_new, 8)
    y_new : array (n_new,) — resultados reales de SIMION.
    """
    import simion_interface
    X_new = select_high_variance(twin_model, n_new, seed=seed)
    y_new = simion_interface.evaluate_batch_real(X_new)
    return X_new, y_new
