"""
control.py — Tareas de control sobre el gemelo digital (BeamlineTwin).

Tarea 1 — Dirección:
    Encuentra los voltajes que maximizan la transmisión esperada del hurdle
    model (predict_combined = P(señal) · magnitud). Optimiza SOBRE EL GEMELO.

Tarea 2 — Consigna inversa:
    Dado un valor objetivo, encuentra los voltajes cuya predicción combinada
    sea lo más cercana posible.

RADIO DE CONFIANZA (TRUST_RADIUS) — hallazgo importante:
    La primera versión optimizaba libremente sobre todo el rango ±1000V y
    verificada con SIMION real dio 0 iones (tanto en dirección como en
    consigna), aunque el gemelo predecía ~35. Diagnóstico: con solo 45
    puntos de señal de entrenamiento, el GP no resuelve un pico que resulta
    ser mucho más angosto de lo que su length-scale asume, y ni penalizar
    la incertidumbre (lower confidence bound) evitó el problema — el modelo
    está "seguro" (std bajo) justo donde se equivoca.

    Lo que sí funcionó, calibrado empíricamente corriendo SIMION real a
    varios radios alrededor del mejor punto real conocido:
        radio ±150V -> 1 ion real   | radio ±40V -> 0 iones reales
        radio ±20V  -> 18 iones real (C=0.25)
        radio ±15V  -> 31 iones real (C=0.43)
        radio ±10V  -> 29 iones real (C=0.40)
        radio ±5V   -> 39 iones real (C=0.54)  <- mejor, elegido
    Por eso ambas tareas restringen la búsqueda a TRUST_RADIUS=5V alrededor
    del mejor voltaje real conocido, en vez de optimizar sobre todo el
    espacio de ±1000V. Es una limitación honesta del gemelo actual (ver
    informe, sección de reflexión): con más presupuesto de entrenamiento
    cerca del pico, ese radio debería poder ampliarse.

Las funciones verify_*_with_simion() llaman a SIMION real (simion_interface.py)
y NO se ejecutan automáticamente desde main.py — se corren a mano cuando
ustedes decidan gastar esas evaluaciones de verificación (no cuentan contra
el presupuesto de entrenamiento).
"""

import numpy as np
from scipy.optimize import minimize

from real_data import load_raw

V_LOW, V_HIGH  = -1000.0, 1000.0
N_DIM          = 8
SEED           = 42
TRUST_RADIUS   = 5.0   # ver calibración empírica arriba


def best_known_real_point() -> np.ndarray:
    """El voltaje real con mayor transmisión observada en beamline_results.csv."""
    X, y = load_raw()
    return X[np.argmax(y)].copy()


def nearest_real_point_to(target: float) -> np.ndarray:
    """
    El voltaje real cuya transmisión OBSERVADA está más cerca del objetivo.

    Para la consigna inversa: en vez de anclar la búsqueda en el pico (72
    iones) y confiar en que el gemelo modele bien la ladera descendente,
    anclamos en un punto real que ya dio ~target iones y dejamos que el
    gemelo solo haga el ajuste fino local. Mismo principio del radio de
    confianza: el dato real aporta la región, el gemelo la interpolación.
    """
    X, y = load_raw()
    return X[np.argmin(np.abs(y - target))].copy()


def _trust_region_bounds(center: np.ndarray, radius: float = TRUST_RADIUS):
    """Caja [-radius,+radius] alrededor de `center`, recortada a [V_LOW,V_HIGH]."""
    return [
        (float(np.clip(v - radius, V_LOW, V_HIGH)),
         float(np.clip(v + radius, V_LOW, V_HIGH)))
        for v in center
    ]


def find_optimal_voltages(model, n_restarts: int = 20, seed: int = SEED,
                           trust_radius: float = TRUST_RADIUS):
    """
    Tarea de dirección: maximiza la transmisión esperada del hurdle model,
    restringido a una vecindad de confianza (trust_radius) alrededor del
    mejor voltaje real conocido (ver nota arriba sobre por qué).

    Parámetros
    ----------
    model        : BeamlineTwin ya ajustado (usa predict_combined).
    n_restarts   : número de arranques aleatorios para escapar de mínimos locales.
    seed         : semilla para reproducibilidad.
    trust_radius : radio (en V) de la vecindad de búsqueda.

    Retorna
    -------
    best_v    : array (8,) — voltajes óptimos recomendados por el gemelo.
    best_pred : float      — transmisión esperada predicha en ese punto.
    """
    center = best_known_real_point()
    bounds = _trust_region_bounds(center, trust_radius)
    rng    = np.random.RandomState(seed)

    def neg_mean(x):
        mu, _ = model.predict_combined(x.reshape(1, -1))
        return -mu[0]

    best_v, best_pred = None, -np.inf

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starting_points = [center] + [
        rng.uniform(lo, hi) for _ in range(n_restarts - 1)
    ]

    for x0 in starting_points:
        res = minimize(neg_mean, x0, method='L-BFGS-B', bounds=bounds)
        t_pred = -res.fun
        if t_pred > best_pred:
            best_pred = t_pred
            best_v    = res.x.copy()

    best_pred = float(np.clip(best_pred, 0.0, 500.0))
    return best_v, best_pred


def find_inverse_setpoint(model, target: float,
                           n_restarts: int = 20, seed: int = SEED + 1,
                           trust_radius: float = TRUST_RADIUS):
    """
    Tarea de consigna inversa: encuentra voltajes cuya predicción combinada
    sea lo más cercana posible al valor objetivo.

    La vecindad de confianza se centra en el punto real cuya transmisión
    observada está más cerca del objetivo (nearest_real_point_to), NO en el
    pico: la primera versión anclaba en el pico (72 iones) y el resultado
    real quedó en 48.8 para un objetivo de 35 (error 13.8) — el gemelo tenía
    que modelar la ladera descendente de la campana, justo donde es más
    empinada. Anclando en un punto que ya dio ~35 el gemelo solo interpola
    localmente.

    Parámetros
    ----------
    model        : BeamlineTwin ya ajustado.
    target       : transmisión objetivo (entre 0 y 500).
    n_restarts   : número de arranques aleatorios.
    seed         : semilla para reproducibilidad.
    trust_radius : radio (en V) de la vecindad de búsqueda.

    Retorna
    -------
    best_v    : array (8,) — voltajes recomendados por el gemelo.
    best_pred : float      — transmisión predicha (debería estar cerca de target).
    """
    center = nearest_real_point_to(target)
    bounds = _trust_region_bounds(center, trust_radius)
    rng    = np.random.RandomState(seed)

    def squared_error(x):
        mu, _ = model.predict_combined(x.reshape(1, -1))
        return (mu[0] - target) ** 2

    best_v, best_loss = None, np.inf

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starting_points = [center] + [
        rng.uniform(lo, hi) for _ in range(n_restarts - 1)
    ]

    for x0 in starting_points:
        res = minimize(squared_error, x0, method='L-BFGS-B', bounds=bounds)
        if res.fun < best_loss:
            best_loss = res.fun
            best_v    = res.x.copy()

    best_pred = float(model.predict_combined(best_v.reshape(1, -1))[0][0])
    best_pred = float(np.clip(best_pred, 0.0, 500.0))
    return best_v, best_pred


def evaluate_gradient(model, voltages: np.ndarray, eps: float = 1.0):
    """
    Gradiente del gemelo (hurdle model) por diferencias finitas.
    Para el criterio D de la rúbrica.

    Parámetros
    ----------
    model    : BeamlineTwin ajustado.
    voltages : array (8,) — punto de operación donde evaluar el gradiente.
    eps      : paso en V para diferencias finitas.

    Retorna
    -------
    grad : array (8,) — gradiente del gemelo (∂E[T]/∂Vi).
    """
    return model.gradient_combined(voltages, eps=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Verificación con SIMION real — NO se llama automáticamente desde main.py.
# Corran esto ustedes cuando quieran gastar esas evaluaciones (no cuentan
# contra el presupuesto de entrenamiento, según la sección 7.3 del PDF).
# ─────────────────────────────────────────────────────────────────────────────

def verify_with_simion(voltages: np.ndarray, n_repeats: int = 5):
    """
    Corre SIMION real n_repeats veces sobre `voltages` y promedia.

    SIMION es ruidoso (SimpleSetUp.fly2 muestrea energía y dirección al azar
    sin semilla fija: el mismo punto dio 74, 60 y 63 iones en tres corridas
    separadas), así que una sola corrida no es un número confiable.

    Retorna
    -------
    mean : float — promedio de n_repeats conteos reales.
    std  : float — desviación estándar entre repeticiones (ruido intrínseco).
    """
    import simion_interface
    return simion_interface.evaluate_one_real_averaged(voltages, n_repeats=n_repeats)


def verify_gradient_with_simion(voltages: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Gradiente real por diferencias finitas centradas (16 llamadas a SIMION)."""
    import simion_interface
    return simion_interface.finite_difference_gradient_real(voltages, eps=eps)
