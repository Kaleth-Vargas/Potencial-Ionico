"""
PROTOTYPE ONLY — not part of the delivered digital twin.

Synthetic beamline simulator: R^8 -> R (transmission 0-500). Used only to
prototype the pipeline before real SIMION data was available. IMPORTANT:
the antisymmetric bender assumption below (V9=-V10, V11=-V12) was tested
against the real 494-point dataset and REJECTED (see conversation/report) —
a real quadrupole bender needs a net dipole component to bend the beam, so
it cannot be purely antisymmetric. Kept here only as design-stage reference.

Mimics realistic SIMION behavior:
- Large dead zones where most voltage combos yield zero transmission.
- A narrow peak region where ions are focused onto the detector.
- Physics-motivated penalties for bender asymmetry, einzel lenses, and deflectors.
"""

import numpy as np

SEED = 42
MAX_IONS = 500

ELECTRODE_IDS = [3, 6, 9, 10, 11, 12, 15, 18]
V_LOW, V_HIGH = -1000.0, 1000.0

# "Sweet spot" voltages (invented but physically plausible).
SWEET_SPOT = np.array([120.0, -250.0, 340.0, -340.0, 340.0, -340.0, 60.0, -80.0])

# Per-electrode length-scales: how tolerant each electrode is.
_LENGTH_SCALES = np.array([600.0, 500.0, 400.0, 400.0, 400.0, 400.0, 500.0, 500.0])


def _bender_penalty(v):
    """Penaliza si los polos opuestos no son antisimétricos.
    Ideal: V9 = -V10 y V11 = -V12."""
    asym_1 = (v[2] + v[3]) / 400.0
    asym_2 = (v[4] + v[5]) / 400.0
    return np.exp(-0.5 * (asym_1**2 + asym_2**2))


def _einzel_factor(v):
    """Transmisión cae si los electrodos de las lentes están cerca de 0V."""
    focus_1 = 1.0 - np.exp(-0.5 * (v[0] / 50.0)**2)
    focus_2 = 1.0 - np.exp(-0.5 * (v[1] / 50.0)**2)
    return focus_1 * focus_2


def _deflector_factor(v):
    """Las deflectoras solo funcionan bien con voltajes pequeños."""
    d1 = (v[6] / 500.0)**2
    d2 = (v[7] / 500.0)**2
    return np.exp(-0.5 * (d1 + d2))


def synthetic_transmission(voltages: np.ndarray, noise_std: float = 3.0) -> float:
    """Evaluate one voltage vector (shape (8,)) and return ion count 0-500."""
    rng = np.random.RandomState(
        int(abs(np.sum(voltages * 1000))) % (2**31)
    )

    v = np.asarray(voltages, dtype=float)
    delta = (v - SWEET_SPOT) / _LENGTH_SCALES

    # Primary peak: full Gaussian in 8-D.
    r2 = np.sum(delta ** 2)
    primary = MAX_IONS * np.exp(-0.5 * r2)

    # Secondary weaker peak offset from sweet spot.
    sweet2 = SWEET_SPOT * -0.6
    delta2 = (v - sweet2) / (_LENGTH_SCALES * 1.5)
    secondary = 0.15 * MAX_IONS * np.exp(-0.5 * np.sum(delta2 ** 2))

    signal = (primary + secondary) * _bender_penalty(v) * _einzel_factor(v) * _deflector_factor(v)

    if noise_std > 0:
        signal += rng.normal(0, noise_std)

    # Zona muerta: menos de 2 iones = cero (SIMION no da fracciones)
    signal = np.where(signal < 2.0, 0.0, signal)
    signal = np.round(signal)

    return float(np.clip(signal, 0.0, MAX_IONS))


def evaluate_batch(voltages_2d: np.ndarray, noise_std: float = 3.0) -> np.ndarray:
    """Evaluate multiple voltage vectors. Input shape (n, 8), output shape (n,)."""
    voltages_2d = np.atleast_2d(voltages_2d)
    return np.array([synthetic_transmission(v, noise_std) for v in voltages_2d])
