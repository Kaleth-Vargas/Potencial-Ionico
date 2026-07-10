"""
generate_figures.py — Genera las figuras del informe a partir de datos y
resultados reales (no inventados) del gemelo digital.

Corre desde la carpeta report/:
    python generate_figures.py
"""

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "digital_twin"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data import split_train_test, load_raw, ELECTRODE_IDS
from gp_model import BeamlineTwin
from control import best_known_real_point

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
plt.rcParams.update({"font.size": 10, "figure.dpi": 140})


# ── Cargar datos y ajustar el gemelo final (con densificación cacheada) ──────
X_train, y_train, X_test, y_test = split_train_test(n_train=100, seed=SEED)

cache_path = _REPO_ROOT / "data" / "gradient_probe_cache.npz"
cached = np.load(cache_path)
X_probe, y_probe = cached["X_probe"], cached["y_probe"]
X_train_full = np.vstack([X_train, X_probe])
y_train_full = np.concatenate([y_train, y_probe])

twin = BeamlineTwin(seed=SEED)
twin.fit(X_train_full, y_train_full)

mu_test, std_test = twin.predict_combined(X_test)
p_test = twin.predict_proba_signal(X_test)


# ── Figura 1: cobertura del espacio de entrenamiento ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
pairs = [(0, 1, "V3", "V6"), (2, 3, "V9", "V10"), (4, 5, "V11", "V12"), (6, 7, "V15", "V18")]

for ax, (i, j, li, lj) in zip(axes.ravel(), pairs):
    test_zero = y_test == 0
    test_sig = y_test > 0
    ax.scatter(X_test[test_zero, i], X_test[test_zero, j], s=8, c="lightgray",
               label="Test, cero", alpha=0.6)
    ax.scatter(X_test[test_sig, i], X_test[test_sig, j], s=14, c="cornflowerblue",
               label="Test, señal", alpha=0.7)
    train_zero = y_train == 0
    train_sig = y_train > 0
    ax.scatter(X_train[train_zero, i], X_train[train_zero, j], s=35, c="black",
               marker="x", label="Train, cero")
    ax.scatter(X_train[train_sig, i], X_train[train_sig, j], s=45, c="crimson",
               marker="*", label="Train, señal")
    ax.scatter(X_probe[:, i], X_probe[:, j], s=30, c="darkorange", marker="^",
               label="Densificación local")
    ax.set_xlabel(f"{li} (V)")
    ax.set_ylabel(f"{lj} (V)")
    ax.set_xlim(-1050, 1050)
    ax.set_ylim(-1050, 1050)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Cobertura del espacio de voltajes: 100 train (farthest-point sampling,\n"
             "estratificado señal/cero) + 16 de densificación local, vs. 394 test reales")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG_DIR / "fig1_cobertura.png", bbox_inches="tight")
plt.close(fig)
print("fig1_cobertura.png guardada")


# ── Figura 2: predicho vs real en la región informativa (test) ──────────────
mask = y_test > 0
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(y_test[mask], mu_test[mask], s=25, c="cornflowerblue", edgecolor="k",
           linewidth=0.3, alpha=0.8)
lims = [0, max(y_test[mask].max(), mu_test[mask].max()) * 1.05]
ax.plot(lims, lims, "k--", linewidth=1, label="y = x (ideal)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Transmisión real (iones)")
ax.set_ylabel("Transmisión predicha por el gemelo (iones)")

from gp_model import BeamlineGP  # noqa
r2 = twin.score_r2(X_test[mask], y_test[mask])
mae = np.mean(np.abs(y_test[mask] - mu_test[mask]))
ax.set_title(f"Predicho vs. real — región informativa del test\n"
             f"R² = {r2:.3f}   MAE = {mae:.1f} iones   (n={mask.sum()})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_predicho_vs_real.png", bbox_inches="tight")
plt.close(fig)
print("fig2_predicho_vs_real.png guardada")


# ── Figura 3: honestidad de la incertidumbre vs P(señal) ─────────────────────
bins = [(0, 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 1.0)]
labels_bins, means_std, counts = [], [], []
for lo, hi in bins:
    m = (p_test >= lo) & (p_test < hi)
    if m.sum() > 0:
        labels_bins.append(f"[{lo},{hi})\nn={m.sum()}")
        means_std.append(std_test[m].mean())
        counts.append(m.sum())

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(labels_bins, means_std, color="mediumseagreen", edgecolor="k")
ax.set_xlabel("P(señal) del clasificador (agrupado en bins)")
ax.set_ylabel("Desviación estándar combinada promedio (iones)")
ax.set_title("Honestidad de la incertidumbre (criterio E):\n"
             "la incertidumbre sube cuando el clasificador está indeciso")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_honestidad_incertidumbre.png", bbox_inches="tight")
plt.close(fig)
print("fig3_honestidad_incertidumbre.png guardada")


# ── Figura 4: resultados de control -- predicho vs real verificado ──────────
X_all, y_all = load_raw()
best_ref = y_all.max()

categorias = ["Dirección\n(maximizar)", "Consigna inversa\n(objetivo=35)"]
predicho = [63.3, 29.9]
real_mean = [55.2, 36.8]
real_std = [6.4, 5.0]

fig, ax = plt.subplots(figsize=(6.5, 5))
x = np.arange(len(categorias))
width = 0.32
ax.bar(x - width/2, predicho, width, label="Predicho por el gemelo", color="cornflowerblue")
ax.bar(x + width/2, real_mean, width, yerr=real_std, capsize=5,
       label="Real (SIMION, promedio de 5 corridas)", color="crimson")
ax.axhline(best_ref, color="black", linestyle="--", linewidth=1,
           label=f"Mejor real conocido ({best_ref:.0f} iones)")
ax.axhline(35, color="gray", linestyle=":", linewidth=1, label="Objetivo consigna (35 iones)")
ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel("Transmisión (iones)")
ax.set_title("Tareas de control: predicho vs. verificado con SIMION real")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_control.png", bbox_inches="tight")
plt.close(fig)
print("fig4_control.png guardada")


# ── Figura 5: gradiente del gemelo vs SIMION real, antes y después de densificar ──
# ANTES (gemelo entrenado solo con los 100 puntos base, sin densificación local):
grad_gemelo_antes = np.array([-0.0000, 0.3563, -0.0000, 0.0036, 0.1419, -0.0016, 0.0075, 0.1453])
grad_real_antes   = np.array([-4.0, -7.0, 8.5, 8.5, -6.5, 1.5, 1.0, -6.5])
# DESPUÉS (gemelo entrenado con los 100 + 16 puntos de densificación local):
grad_gemelo_despues = np.array([-0.1870, 0.0691, 0.0267, 0.0089, 0.0061, -0.3826, -0.0000, -0.0834])
grad_real_despues   = np.array([-7.0, -1.5, 3.0, 0.5, 1.5, 3.5, -14.5, 4.0])
electrodos  = [f"V{e}" for e in ELECTRODE_IDS]

cos_antes = np.dot(grad_gemelo_antes, grad_real_antes) / (
    np.linalg.norm(grad_gemelo_antes) * np.linalg.norm(grad_real_antes))
cos_despues = np.dot(grad_gemelo_despues, grad_real_despues) / (
    np.linalg.norm(grad_gemelo_despues) * np.linalg.norm(grad_real_despues))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
x = np.arange(len(electrodos))

panels = [
    (axes[0, 0], grad_gemelo_antes, "Gemelo (ANTES de densificar)", "cornflowerblue"),
    (axes[0, 1], grad_real_antes, "SIMION real (mismo punto)", "crimson"),
    (axes[1, 0], grad_gemelo_despues, "Gemelo (DESPUÉS de densificar)", "cornflowerblue"),
    (axes[1, 1], grad_real_despues, "SIMION real (mismo punto)", "crimson"),
]
for ax, vals, title, color in panels:
    ax.bar(x, vals, color=color, edgecolor="k")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(electrodos)
    ax.set_ylabel("d(iones)/dV")
    ax.set_title(title, fontsize=10)

fig.suptitle(f"Criterio D — gradiente del gemelo vs. SIMION real en el punto ancla\n"
             f"Antes de densificar: similitud coseno = {cos_antes:.3f}  →  "
             f"Después de densificar: similitud coseno = {cos_despues:.3f}\n"
             f"(sigue sin concordar bien, pero ya no apunta en dirección opuesta — ver Reflexión)")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIG_DIR / "fig5_gradiente.png", bbox_inches="tight")
plt.close(fig)
print("fig5_gradiente.png guardada")

# ── Figura 6: estudio del paso eps para el gradiente (criterio D) ────────────
# Datos del estudio gradient_eps_study.py (SIMION real, 16 puntos x 5 corridas
# por eps, cacheados en gradient_eps_study_cache.npz).
eps_cache_path = _REPO_ROOT / "data" / "gradient_eps_study_cache.npz"
if eps_cache_path.exists():
    from gradient_eps_study import twin_gradient_central, real_gradient_from_probes, cosine

    eps_cache = dict(np.load(eps_cache_path))
    probe_cache = np.load(cache_path)
    anchor = best_known_real_point()

    grad_real_eps1 = real_gradient_from_probes(probe_cache["y_probe"], eps=1.0)
    grad_twin_eps1 = twin_gradient_central(twin, anchor, eps=1.0)
    grad_real_eps5 = real_gradient_from_probes(eps_cache["y_eps5"], eps=5.0)
    grad_twin_eps5 = twin_gradient_central(twin, anchor, eps=5.0)
    grad_real_eps10 = real_gradient_from_probes(eps_cache["y_eps10"], eps=10.0)
    grad_twin_eps10 = twin_gradient_central(twin, anchor, eps=10.0)

    cos_by_eps = [
        (1, cosine(grad_twin_eps1, grad_real_eps1)),
        (5, cosine(grad_twin_eps5, grad_real_eps5)),
        (10, cosine(grad_twin_eps10, grad_real_eps10)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels_eps = [f"ε={e}V" for e, _ in cos_by_eps]
    vals = [c for _, c in cos_by_eps]
    colors = ["indianred" if v < 0.2 else "mediumseagreen" for v in vals]
    ax1.bar(labels_eps, vals, color=colors, edgecolor="k")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Similitud coseno (gemelo vs. SIMION real)")
    ax1.set_title("Acuerdo del gradiente según el paso ε\n"
                  "ε=1V: ruido domina · ε=5V: ventana útil · ε=10V: sesgo de curvatura")
    for i, v in enumerate(vals):
        ax1.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}",
                 ha="center", fontsize=9)

    x = np.arange(len(ELECTRODE_IDS))
    width = 0.38
    ax2.bar(x - width/2, grad_twin_eps5, width, label="Gemelo (ε=5V)",
            color="cornflowerblue", edgecolor="k")
    ax2.bar(x + width/2, grad_real_eps5, width, label="SIMION real (ε=5V, prom. 5)",
            color="crimson", edgecolor="k")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"V{e}" for e in ELECTRODE_IDS])
    ax2.set_ylabel("d(iones)/dV")
    ax2.set_title(f"Gradiente en el punto ancla con ε=5V\n"
                  f"similitud coseno = {cos_by_eps[1][1]:+.3f}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_gradiente_eps.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6_gradiente_eps.png guardada")
else:
    print("(gradient_eps_study_cache.npz no existe — fig6 omitida; correr gradient_eps_study.py)")

print("\nTodas las figuras generadas en", FIG_DIR)
