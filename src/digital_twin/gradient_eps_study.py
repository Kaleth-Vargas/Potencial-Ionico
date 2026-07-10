"""
gradient_eps_study.py — Mejora del criterio D: gradiente con ε más grande.

Hallazgo previo (informe, sección Diferenciabilidad): con ε=1V la similitud
coseno entre el gradiente del gemelo y el de SIMION real era -0.049.

Diagnóstico: SIMION es ruidoso (std ≈ 6-10 iones por corrida; promediando 5
corridas queda ≈ std/√5 ≈ 3-4 iones). Con ε=1V el gradiente por diferencias
finitas centradas divide ese ruido entre 2ε=2V → ruido de ±2-3 iones/V,
del mismo orden que el gradiente verdadero. La comparación era ruido puro.

Solución: repetir el chequeo con ε=5V y ε=10V. La señal (Δy entre x+ε y x-ε)
crece con ε mientras el ruido queda igual, así que la relación señal/ruido
mejora ~ε veces. El gradiente del gemelo se evalúa con el MISMO ε (secante
central sobre el gemelo), para comparar la misma cantidad en ambos lados.

Los resultados de SIMION real se congelan en gradient_eps_study_cache.npz
la primera vez (mismo patrón que gradient_probe_cache.npz) para que el
estudio sea reproducible. Correr con --refresh-cache para regenerarlos.
"""

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from real_data import split_train_test, ELECTRODE_IDS
from gp_model import BeamlineTwin
from control import best_known_real_point

SEED = 42
np.random.seed(SEED)

EPS_VALUES = [5.0, 10.0]
N_REPEATS = 5

_DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
CACHE_PATH = _DATA_DIR / "gradient_eps_study_cache.npz"


def twin_gradient_central(model, x: np.ndarray, eps: float) -> np.ndarray:
    """Gradiente del gemelo por diferencias finitas CENTRADAS con paso eps
    (misma fórmula que el gradiente real, para comparar la misma cantidad)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros(len(x))
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        mu_p, _ = model.predict_combined(x_plus.reshape(1, -1))
        mu_m, _ = model.predict_combined(x_minus.reshape(1, -1))
        grad[i] = (mu_p[0] - mu_m[0]) / (2 * eps)
    return grad


def real_gradient_from_probes(y_probe: np.ndarray, eps: float) -> np.ndarray:
    """Gradiente real por diferencias centradas a partir de los 16 conteos
    promediados (orden: [x0+eps, x0-eps, x1+eps, x1-eps, ...])."""
    n = len(y_probe) // 2
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = (y_probe[2 * i] - y_probe[2 * i + 1]) / (2 * eps)
    return grad


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("  Estudio de gradiente vs eps (criterio D)")
    print("=" * 64)

    # ── Gemelo idéntico al de main.py (100 train + 16 densificación) ─────────
    X_train, y_train, _, _ = split_train_test(n_train=100, seed=SEED)
    probe_cache = np.load(_DATA_DIR / "gradient_probe_cache.npz")
    X_probe1, y_probe1 = probe_cache["X_probe"], probe_cache["y_probe"]
    X_train = np.vstack([X_train, X_probe1])
    y_train = np.concatenate([y_train, y_probe1])

    twin = BeamlineTwin(seed=SEED)
    twin.fit(X_train, y_train)
    anchor = best_known_real_point()
    print(f"  Gemelo ajustado ({len(y_train)} puntos). Ancla: mejor punto real conocido.")

    # ── Referencia eps=1 (del caché ya existente) ─────────────────────────────
    grad_real_1 = real_gradient_from_probes(y_probe1, eps=1.0)
    grad_twin_1 = twin_gradient_central(twin, anchor, eps=1.0)
    print(f"\n  eps= 1V (referencia, caché previo): cos = {cosine(grad_twin_1, grad_real_1):+.3f}")

    # ── SIMION real con eps grandes (cacheado) ────────────────────────────────
    if "--refresh-cache" in sys.argv and CACHE_PATH.exists():
        CACHE_PATH.unlink()

    if CACHE_PATH.exists():
        cached = dict(np.load(CACHE_PATH))
        print(f"  (cargado de cache: {CACHE_PATH.name})")
    else:
        import simion_interface
        cached = {}
        for eps in EPS_VALUES:
            print(f"\n  Consultando SIMION real con eps={eps:.0f}V "
                  f"(16 puntos x {N_REPEATS} corridas = {16*N_REPEATS} llamadas)...")
            X_p, y_p = simion_interface.finite_difference_probe_points(
                anchor, eps=eps, n_repeats=N_REPEATS
            )
            cached[f"X_eps{eps:.0f}"] = X_p
            cached[f"y_eps{eps:.0f}"] = y_p
        np.savez(CACHE_PATH, **cached)
        print(f"  (guardado en cache: {CACHE_PATH.name})")

    # ── Comparación ───────────────────────────────────────────────────────────
    print("\n  Electrodos:", [f"V{e}" for e in ELECTRODE_IDS])
    results = {}
    for eps in EPS_VALUES:
        y_p = cached[f"y_eps{eps:.0f}"]
        grad_real = real_gradient_from_probes(y_p, eps=eps)
        grad_twin = twin_gradient_central(twin, anchor, eps=eps)
        cos = cosine(grad_twin, grad_real)
        results[eps] = (grad_twin, grad_real, cos)
        print(f"\n  eps={eps:.0f}V:")
        print(f"    gemelo : {np.array2string(grad_twin, precision=3, suppress_small=True)}")
        print(f"    real   : {np.array2string(grad_real, precision=3, suppress_small=True)}")
        print(f"    similitud coseno = {cos:+.3f}")

    print("\n" + "=" * 64)
    print("  Resumen (criterio D)")
    print("=" * 64)
    print(f"    eps= 1V : cos = {cosine(grad_twin_1, grad_real_1):+.3f}  (dominado por ruido)")
    for eps in EPS_VALUES:
        print(f"    eps={eps:2.0f}V : cos = {results[eps][2]:+.3f}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
