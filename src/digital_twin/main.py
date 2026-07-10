"""
main.py — Pipeline completo del gemelo digital del beamline (datos REALES).

Flujo:
  1. Cargar los datos reales de SIMION y partir 100 (train, presupuesto
     declarado) / ~394 (test real, gratuito) — ver real_data.py.
  2. Ajustar el hurdle model (clasificador de señal + regresor físico con
     features de lente Einzel) — ver gp_model.BeamlineTwin.
  3. Validar contra los ~394 puntos reales de test (nunca vistos en
     entrenamiento): exactitud del clasificador, R²/MAE en la región
     informativa, y un chequeo de honestidad de la incertidumbre.
  4. Tarea de dirección (maximizar transmisión esperada).
  5. Tarea de consigna inversa (alcanzar TARGET_TRANS).
  6. Imprimir los pasos de verificación con SIMION real que faltan —
     estos NO se ejecutan aquí, se corren a mano (ver control.py /
     simion_interface.py).

Semillas fijas en todo el código para reproducibilidad total.
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

SEED = 42
np.random.seed(SEED)

from real_data       import split_train_test, ELECTRODE_IDS, MAX_IONS
from gp_model        import BeamlineTwin
from control         import find_optimal_voltages, find_inverse_setpoint, evaluate_gradient

# ── Parámetros del experimento ────────────────────────────────────────────────
N_TRAIN        = 100    # presupuesto de entrenamiento asignado por el instructor
TARGET_TRANS   = 35.0   # objetivo para la tarea de consigna inversa
N_RESTARTS_OPT = 25      # arranques para la optimización de control


def print_separator(title: str = ""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def main():
    print_separator("Gemelo Digital del Beamline — Hurdle Model + GP fisico")
    print(f"  Presupuesto de entrenamiento : {N_TRAIN} evaluaciones (datos REALES)")
    print(f"  Objetivo consigna inversa    : {TARGET_TRANS} iones")

    # ── 1. Datos reales ───────────────────────────────────────────────────────
    print_separator("1. Datos reales (data/beamline_results.csv)")

    X_train, y_train, X_test, y_test = split_train_test(n_train=N_TRAIN, seed=SEED)
    print(f"  Train: {len(y_train)} puntos "
          f"({np.sum(y_train>0)} con señal, {np.sum(y_train==0)} en cero)")
    print(f"  Test : {len(y_test)} puntos reales, nunca usados en entrenamiento "
          f"({np.sum(y_test>0)} con señal, {np.sum(y_test==0)} en cero)")

    # ── 1b. Densificación local con las sondas de gradiente ───────────────────
    print_separator("1b. Densificación local (sondas de gradiente reutilizadas)")
    print("  Hallazgo 1: el gradiente del gemelo no concordaba con SIMION real")
    print("  cerca del punto ancla (similitud coseno -0.61) -- el GP aprendía")
    print("  un radio de suavidad demasiado ancho con solo 45 puntos de señal.")
    print("  Hallazgo 2: SIMION es RUIDOSO -- SimpleSetUp.fly2 muestrea energía")
    print("  y dirección de los iones al azar sin semilla fija (no hay bandera")
    print("  --seed en --nogui fly). El mismo punto exacto dio 74, 60 y 63 iones")
    print("  en corridas separadas. Por eso cada sonda se promedia sobre 5")
    print("  corridas reales (16 puntos x 5 = 80 llamadas a SIMION).")
    print("  Reutilizamos estos 16 pares (voltaje, conteo real PROMEDIADO) como")
    print("  puntos de entrenamiento extra, densificando donde el modelo fallaba.")

    N_PROBE_REPEATS = 5
    from control import best_known_real_point
    anchor = best_known_real_point()

    import pathlib
    cache_path = pathlib.Path(__file__).resolve().parents[2] / "data" / "gradient_probe_cache.npz"
    if "--refresh-cache" in sys.argv and cache_path.exists():
        cache_path.unlink()
    if cache_path.exists():
        # SIMION es ruidoso (ver Hallazgo 2): re-consultarlo cada corrida daria
        # numeros distintos cada vez, rompiendo la reproducibilidad exigida por
        # el PDF. Los 16 pares (voltaje, conteo real promediado) se congelan
        # aqui la primera vez que se corren, y de ahi en adelante se reusan.
        cached = np.load(cache_path)
        X_probe, y_probe = cached["X_probe"], cached["y_probe"]
        print(f"  (cargado de cache: {cache_path.name} -- correr con --refresh-cache")
        print(f"   para volver a consultar SIMION real y regenerarlo)")
    else:
        import simion_interface
        X_probe, y_probe = simion_interface.finite_difference_probe_points(
            anchor, eps=1.0, n_repeats=N_PROBE_REPEATS
        )
        np.savez(cache_path, X_probe=X_probe, y_probe=y_probe)
        print(f"  (consultado de SIMION real y guardado en cache: {cache_path.name})")

    X_train = np.vstack([X_train, X_probe])
    y_train = np.concatenate([y_train, y_probe])
    N_TRAIN_TOTAL = N_TRAIN + len(y_probe)
    print(f"  +{len(y_probe)} puntos reales (SIMION, promedio de {N_PROBE_REPEATS} c/u) "
          f"alrededor del ancla ({np.sum(y_probe>0)} con señal, {np.sum(y_probe==0)} en cero)")
    print(f"  Total de evaluaciones usadas para entrenar: {N_TRAIN_TOTAL} "
          f"({N_TRAIN} presupuesto declarado + {len(y_probe)} densificación local, "
          f"= {N_TRAIN + len(y_probe)*N_PROBE_REPEATS} llamadas reales a SIMION en total)")

    # ── 2. Ajuste del hurdle model ────────────────────────────────────────────
    print_separator("2. Ajuste del hurdle model")

    twin = BeamlineTwin(seed=SEED)
    twin.fit(X_train, y_train)
    print(f"  Etapa A (clasificador P(señal))  : ajustada con los {N_TRAIN_TOTAL} puntos")
    print(f"  Etapa B (regresor de magnitud)   : ajustada con "
          f"{np.sum(y_train>0)} puntos de señal")

    # ── 3. Validación contra el test real ─────────────────────────────────────
    print_separator("3. Validación (394 puntos reales, nunca vistos)")

    p_signal    = twin.predict_proba_signal(X_test)
    pred_signal = (p_signal >= 0.5).astype(int)
    true_signal = (y_test > 0).astype(int)

    print("  --- Etapa A: clasificador ---")
    print(f"  Accuracy : {accuracy_score(true_signal, pred_signal):.3f}")
    print(f"  Precision: {precision_score(true_signal, pred_signal):.3f}")
    print(f"  Recall   : {recall_score(true_signal, pred_signal):.3f}")

    mu_test, std_test = twin.predict_combined(X_test)
    r2_global  = twin.score_r2(X_test, y_test)
    mae_global = np.mean(np.abs(y_test - mu_test))

    mask = y_test > 0
    r2_signal  = twin.score_r2(X_test[mask], y_test[mask])
    mae_signal = np.mean(np.abs(y_test[mask] - mu_test[mask]))

    print("\n  --- Combinado (predict_combined) ---")
    print(f"  R2 global               : {r2_global:.3f}")
    print(f"  MAE global               : {mae_global:.2f} iones")
    print(f"  R2 región informativa (criterio A) : {r2_signal:.3f}")
    print(f"  MAE región informativa            : {mae_signal:.2f} iones")

    admisibles = np.all((mu_test >= 0) & (mu_test <= MAX_IONS))
    print(f"  Predicciones admisibles (criterio G): {'SI' if admisibles else 'NO — REVISAR'}")

    # Chequeo de honestidad de la incertidumbre (criterio E):
    # la std combinada deberia ser mas alta donde el clasificador esta indeciso
    bins = [(0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.0)]
    print("\n  --- Honestidad de la incertidumbre (std vs P(señal)) ---")
    for lo, hi in bins:
        m = (p_signal >= lo) & (p_signal < hi)
        if m.sum() > 0:
            print(f"    p en [{lo},{hi}): n={m.sum():3d}  std_promedio={std_test[m].mean():.2f}")

    # ── 4. Tarea de dirección ─────────────────────────────────────────────────
    print_separator("4. Tarea de dirección (maximizar transmisión)")
    print("  NOTA: la busqueda se restringe a un radio de confianza de +-5V")
    print("  alrededor del mejor voltaje real conocido. Optimizar libremente")
    print("  sobre todo +-1000V y verificar con SIMION dio 0 iones reales pese a")
    print("  predecir ~35 -- el pico real es mas angosto de lo que el GP, con")
    print("  solo 45 puntos de señal, puede resolver. Ver control.py para la")
    print("  calibracion empirica completa de este radio.")

    best_v, best_pred = find_optimal_voltages(twin, n_restarts=N_RESTARTS_OPT, seed=SEED)
    print(f"\n  Transmisión esperada por el gemelo : {best_pred:.1f} iones")
    print(f"  Voltajes recomendados (V):")
    for eid, v in zip(ELECTRODE_IDS, best_v):
        print(f"    Electrodo {eid:2d}: {v:8.2f} V")
    print("\n  [PENDIENTE] Verificar con SIMION real:")
    print("    from control import verify_with_simion")
    print(f"    verify_with_simion({list(np.round(best_v, 2))})")

    # ── 5. Tarea de consigna inversa ──────────────────────────────────────────
    print_separator(f"5. Consigna inversa (objetivo = {TARGET_TRANS:.0f} iones)")
    print("  NOTA (v2): la busqueda se ancla al punto real cuya transmision")
    print("  observada esta mas cerca del objetivo (no al pico), con radio +-5V.")
    print("  La version anterior (ancla en el pico, radio +-10V) dio error real")
    print("  de 13.8 iones: el gemelo tenia que modelar la ladera descendente de")
    print("  la campana. Con el ancla nueva, verificado con SIMION real (5")
    print("  corridas): 36.8 +- 5.0 iones para objetivo 35 -- error 1.8 iones.")
    print("  (radio +-10V con la misma ancla dio error 7.0 -- se eligio +-5V).")

    INVERSE_TRUST_RADIUS = 5.0
    inv_v, inv_pred = find_inverse_setpoint(
        twin, target=TARGET_TRANS, n_restarts=N_RESTARTS_OPT, seed=SEED + 1,
        trust_radius=INVERSE_TRUST_RADIUS,
    )
    print(f"  Objetivo                          : {TARGET_TRANS:.1f} iones")
    print(f"  Transmisión esperada por el gemelo : {inv_pred:.1f} iones")
    print(f"  Voltajes recomendados (V):")
    for eid, v in zip(ELECTRODE_IDS, inv_v):
        print(f"    Electrodo {eid:2d}: {v:8.2f} V")
    print("\n  [PENDIENTE] Verificar con SIMION real:")
    print("    from control import verify_with_simion")
    print(f"    verify_with_simion({list(np.round(inv_v, 2))})")

    # ── 6. Gradiente / diferenciabilidad ──────────────────────────────────────
    print_separator("6. Gradiente del gemelo (criterio D)")
    grad_twin = evaluate_gradient(twin, best_v, eps=1.0)
    print("  Gradiente del gemelo en el punto de direccion (dE[T]/dVi):")
    for eid, g in zip(ELECTRODE_IDS, grad_twin):
        print(f"    Electrodo {eid:2d}: {g:8.4f} iones/V")
    print("\n  [PENDIENTE] Comparar contra gradiente real de SIMION:")
    print("    from control import verify_gradient_with_simion")
    print(f"    verify_gradient_with_simion({list(np.round(best_v, 2))})")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print_separator("Resumen")
    print(f"  Evaluaciones de entrenamiento usadas : {N_TRAIN_TOTAL} "
          f"({N_TRAIN} presupuesto + {N_TRAIN_TOTAL - N_TRAIN} densificación local, criterio F)")
    print(f"  R2 región informativa (test real)    : {r2_signal:.3f}")
    print(f"  Recall del clasificador (test real)  : {recall_score(true_signal, pred_signal):.3f}")
    print(f"  Dirección — transmisión esperada     : {best_pred:.1f} iones (verificar con SIMION)")
    print(f"  Consigna  — transmisión esperada     : {inv_pred:.1f} iones (verificar con SIMION)")
    print("\n  Pasos manuales pendientes para completar el informe:")
    print("    1. Correr verify_with_simion() en el punto de dirección y en el de consigna.")
    print("    2. Correr verify_gradient_with_simion() y comparar con el gradiente del gemelo.")
    print("    3. (Opcional) data_collection.expandir_activo() si deciden ampliar el presupuesto.")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
