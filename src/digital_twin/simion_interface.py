"""
simion_interface.py — Conexión a SIMION real (Lección 6).

Adaptado de simulation/optimizer.py (misma configuración, mismos
comandos fastadj/fly). NADA de este módulo se ejecuta automáticamente al
correr main.py — son utilidades que ustedes invocan cuando quieran:

  - evaluate_batch_real(X)               -> recolección activa adicional,
                                             o verificar candidatos de control.
  - finite_difference_gradient_real(x)   -> chequeo de diferenciabilidad
                                             (criterio D) contra SIMION real.

Requiere que ya hayan corrido el Paso 0 (refine) como en optimizer.py, es
decir que electrode_.PA0 exista.
"""

import pathlib
import re
import subprocess

import numpy as np

# ── Configuración (igual que simulation/optimizer.py) ────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SIMULATION_DIR = _REPO_ROOT / "simulation"

SIMION_INSTALL_DIR = _SIMULATION_DIR
SIMION_EXE = SIMION_INSTALL_DIR / "simion.exe"
IOB_FILE   = _SIMULATION_DIR / "SimpleSetUp.iob"
PA0_FILE   = _SIMULATION_DIR / "electrode_.PA0"

ELECTRODE_IDS = [3, 6, 9, 10, 11, 12, 15, 18]
V_LOW, V_HIGH = -1000.0, 1000.0
MAX_IONS = 500

FIXED = {
    1: 500.0, 19: -2000.0, 2: 0.0,
    4: 0.0, 5: 0.0, 7: 0.0, 8: 0.0,
    13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0,
}

DETECTOR_REGION = {"x": (70, 82), "y": (70, 83), "z": (403, 407)}

FLY_COMMAND = (
    f'"{SIMION_EXE}" --nogui fly --recording-output=out.txt '
    f'--retain-trajectories=0 --restore-potential=0 --programs=0 '
    f'"{IOB_FILE}"'
)


def check_setup() -> None:
    """Verifica que SIMION y los archivos de la simulación existan."""
    missing = [p for p in (SIMION_INSTALL_DIR, IOB_FILE, PA0_FILE) if not p.exists()]
    if missing:
        raise SystemExit(
            "Faltan archivos de SIMION: " + ", ".join(str(p) for p in missing) +
            "\nCorran el Paso 0 (refine) desde simulation/ primero."
        )


def _run_simion(command: str) -> str:
    result = subprocess.run(
        command, cwd=str(SIMION_INSTALL_DIR), shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    return result.stdout


def _apply_voltages(chosen: dict) -> None:
    all_volts = {**FIXED, **chosen}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    command = f'"{SIMION_EXE}" --nogui fastadj "{PA0_FILE}" {settings}'
    _run_simion(command)


def _get_positions(simion_output: str) -> np.ndarray:
    pattern = r"xyz\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)mm"
    matches = re.findall(pattern, simion_output)
    return np.array(matches, dtype=float)


def _count_hits(positions: np.ndarray) -> int:
    if positions.shape[0] == 0:
        return 0
    x_min, x_max = DETECTOR_REGION["x"]
    y_min, y_max = DETECTOR_REGION["y"]
    z_min, z_max = DETECTOR_REGION["z"]
    in_box = (
        (positions[:, 0] > x_min) & (positions[:, 0] < x_max) &
        (positions[:, 1] > y_min) & (positions[:, 1] < y_max) &
        (positions[:, 2] > z_min) & (positions[:, 2] < z_max)
    )
    return int(in_box.sum())


def evaluate_one_real(voltages: np.ndarray) -> int:
    """
    Corre UNA evaluación real de SIMION: fastadj + fly + conteo de hits.

    Parámetros
    ----------
    voltages : array (8,) en el orden ELECTRODE_IDS = [3,6,9,10,11,12,15,18].

    Retorna
    -------
    int : iones que llegaron al detector (0-500).
    """
    chosen = {eid: float(v) for eid, v in zip(ELECTRODE_IDS, voltages)}
    try:
        _apply_voltages(chosen)
        output = _run_simion(FLY_COMMAND)
    except subprocess.CalledProcessError:
        return 0
    positions = _get_positions(output)
    return _count_hits(positions)


def evaluate_batch_real(X: np.ndarray) -> np.ndarray:
    """
    Corre SIMION real para cada fila de X (una llamada de subproceso por fila).
    Usar con cuidado del presupuesto: cada llamada es una evaluación real.

    Parámetros
    ----------
    X : array (n, 8) en el orden ELECTRODE_IDS.

    Retorna
    -------
    np.ndarray (n,) con los conteos reales.
    """
    X = np.atleast_2d(X)
    return np.array([evaluate_one_real(row) for row in X])


def evaluate_one_real_averaged(voltages: np.ndarray, n_repeats: int = 5):
    """
    Promedia n_repeats corridas reales de SIMION en el MISMO punto.

    Hallazgo importante: SimpleSetUp.fly2 lanza los 500 iones con energía
    cinética (gaussian_distribution) y dirección (cone_direction_distribution)
    muestreadas al azar, y SIMION no expone una bandera de semilla fija desde
    --nogui fly (solo simion.seed() desde un programa Lua del workbench, más
    invasivo de configurar). Por eso el MISMO punto exacto da resultados
    distintos en corridas separadas (confirmado empíricamente: 74, 60, 63
    iones para el mismo voltaje). Promediar varias corridas da una estimación
    mucho más confiable del valor esperado real.

    Parámetros
    ----------
    voltages   : array (8,) en el orden ELECTRODE_IDS.
    n_repeats  : cuántas corridas reales promediar.

    Retorna
    -------
    mean : float — promedio de los n_repeats conteos reales.
    std  : float — desviación estándar entre repeticiones (ruido intrínseco).
    """
    samples = np.array([evaluate_one_real(voltages) for _ in range(n_repeats)])
    return float(samples.mean()), float(samples.std())


def finite_difference_probe_points(x: np.ndarray, eps: float = 1.0, n_repeats: int = 5):
    """
    Genera y evalúa con SIMION real los 16 puntos vecinos (x_i +- eps por
    cada uno de los 8 electrodos) usados para el chequeo de gradiente.
    Se expone por separado de finite_difference_gradient_real() para poder
    REUTILIZAR estos 16 pares (voltaje, conteo real) como datos de
    entrenamiento adicionales (densificación local), no solo para el
    gradiente.

    Cada punto se promedia sobre n_repeats corridas reales (ver
    evaluate_one_real_averaged): SIMION es ruidoso (fuente de iones con
    energía/dirección aleatorias sin semilla fija), así que un solo conteo
    no es confiable — esto son n_repeats * 16 llamadas reales a SIMION.

    Parámetros
    ----------
    x         : array (8,) — punto de operación (centro de las diferencias).
    eps       : paso en voltios.
    n_repeats : repeticiones reales a promediar por punto.

    Retorna
    -------
    X_probe : array (16, 8) — los puntos x_i+eps y x_i-eps para cada electrodo.
    y_probe : array (16,)   — su conteo real PROMEDIADO de SIMION.
    """
    x = np.asarray(x, dtype=float)
    X_probe = []
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i]  += eps
        x_minus[i] -= eps
        X_probe.append(x_plus)
        X_probe.append(x_minus)
    X_probe = np.array(X_probe)
    y_probe = np.array([
        evaluate_one_real_averaged(row, n_repeats=n_repeats)[0] for row in X_probe
    ])
    return X_probe, y_probe


def finite_difference_gradient_real(x: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Gradiente real por diferencias finitas centradas, corriendo SIMION de
    verdad: 2 evaluaciones por electrodo (16 llamadas para 8 electrodos).
    Para comparar contra BeamlineTwin.gradient_combined() (criterio D).

    Parámetros
    ----------
    x   : array (8,) — punto de operación.
    eps : paso en voltios.

    Retorna
    -------
    grad : array (8,) — ∂(iones)/∂V_i estimado con SIMION real.
    """
    x = np.asarray(x, dtype=float)
    X_probe, y_probe = finite_difference_probe_points(x, eps=eps)
    grad = np.zeros(len(x))
    for i in range(len(x)):
        y_plus, y_minus = y_probe[2 * i], y_probe[2 * i + 1]
        grad[i] = (y_plus - y_minus) / (2 * eps)
    return grad


if __name__ == "__main__":
    # Ejemplo de uso manual (NO se ejecuta al importar el modulo ni desde main.py).
    check_setup()
    print("SIMION listo. Ejemplo: evaluar el mejor punto real conocido (72 iones)")
    best_known = np.array([-987.86, -319.59, 866.75, -64.07, -100.71, -908.34, -992.66, 197.35])
    print("Voltajes:", dict(zip(ELECTRODE_IDS, best_known)))
    print("Corriendo SIMION real...")
    result = evaluate_one_real(best_known)
    print(f"Resultado real: {result} iones")
