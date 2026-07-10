"""
PROTOTIPO — NO forma parte del gemelo entregado.

simulator.py — Simulador sintético del beamline (R^8 -> R), usado solo para
prototipar la arquitectura antes de tener datos reales de SIMION. El gemelo
final (src/digital_twin/gp_model.py + real_data.py) se entrena y valida con
datos reales de data/beamline_results.csv, no con este archivo.
Se conserva aquí solo como referencia de la etapa de diseño inicial.

Imita el comportamiento real de SIMION:
- Zona muerta amplia: la mayoría de combinaciones de voltaje dan transmisión ~ 0.
- Un pico de transmisión alta cerca del punto de operación conocido.

La función evaluate() es la interfaz que luego se reemplaza por llamadas
reales a SIMION (fastadj + fly).
"""

import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────────
MAX_IONS    = 500
ELECTRODE_IDS = [3, 6, 9, 10, 11, 12, 15, 18]
V_LOW, V_HIGH = -1000.0, 1000.0
N_DIM       = len(ELECTRODE_IDS)

# Punto de operación real encontrado con SIMION (~47 hits).
SWEET_SPOT = np.array([-533.92, -861.2, 621.11, 48.47,
                        122.0, -491.87, -872.14, 186.12])

# Tolerancia por electrodo: cuánto se puede alejar antes de perder señal.
_SCALES = np.array([150., 100., 120., 80., 100., 120., 100., 80.])

# Pico secundario débil para que el espacio no sea trivialmente unimodal.
_SWEET2    = SWEET_SPOT * np.array([-0.5, -0.6, 0.8, -0.3, 0.7, -0.5, -0.4, 0.5])
_SCALES2   = _SCALES * 1.8
_PEAK2_AMP = 0.12 * MAX_IONS


# ── Funciones privadas ────────────────────────────────────────────────────────

def _gaussian_peak(v, center, scales, amplitude):
    delta = (v - center) / scales
    return amplitude * np.exp(-0.5 * np.sum(delta**2))


# ── Interfaz pública ──────────────────────────────────────────────────────────

def evaluate(voltages, noise_std: float = 2.0) -> float:
    """
    Evalúa la transmisión sintética para un vector de 8 voltajes.

    Parámetros
    ----------
    voltages  : array-like de forma (8,) con los voltajes en V.
    noise_std : desviación estándar del ruido gaussiano aditivo.

    Retorna
    -------
    float : número de iones en el detector (entero redondeado, en [0, 500]).
    """
    v = np.asarray(voltages, dtype=float)

    # Semilla reproducible basada en el vector de voltajes
    seed = int(abs(np.sum(v * 1000))) % (2**31)
    rng  = np.random.RandomState(seed)

    # Señal: pico principal + pico secundario
    signal = (_gaussian_peak(v, SWEET_SPOT, _SCALES, 47.0) +
              _gaussian_peak(v, _SWEET2, _SCALES2, _PEAK2_AMP))

    # Ruido aditivo
    if noise_std > 0:
        signal += rng.normal(0.0, noise_std)

    signal = max(0.0, signal)

    # Zona muerta: menos de 1.5 iones → cero (SIMION no reporta fracciones)
    if signal < 1.5:
        signal = 0.0

    return float(np.clip(round(signal), 0, MAX_IONS))


def evaluate_batch(voltages_2d, noise_std: float = 2.0) -> np.ndarray:
    """
    Evalúa múltiples vectores de voltaje.

    Parámetros
    ----------
    voltages_2d : array de forma (n, 8).
    noise_std   : ruido gaussiano aditivo.

    Retorna
    -------
    np.ndarray de forma (n,) con las transmisiones.
    """
    voltages_2d = np.atleast_2d(voltages_2d)
    return np.array([evaluate(v, noise_std) for v in voltages_2d])
