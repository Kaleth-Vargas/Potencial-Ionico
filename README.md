# Gemelo digital del beamline — Taller MIT-UNAL

Gemelo digital de un beamline de iones simulado en SIMION 8.1: un *hurdle
model* (clasificador de señal + proceso gaussiano físico) que aprende a
predecir la transmisión de iones a partir de un presupuesto fijo de 100
evaluaciones reales, y que resuelve dos tareas de control (maximizar
transmisión y alcanzar una consigna objetivo) sin necesitar SIMION en el
momento de predecir.

Ver [`report/reporte_tecnico.pdf`](report/reporte_tecnico.pdf) para la
explicación completa del proyecto (arquitectura, metodología, resultados y
limitaciones), y [`report/informe.pdf`](report/informe.pdf) para el informe
original entregado en el taller (las 7 secciones pedidas por la rúbrica).

## Estructura del repositorio

```
.
├── src/digital_twin/       El gemelo digital: modelo, control y experimentos.
├── data/                   Datos reales de SIMION y cachés de reproducibilidad.
├── simulation/             Simulación SIMION (geometría, .iob) y el optimizador Optuna.
├── report/                 Informes (LaTeX + PDF) y generador de figuras.
├── requirements.txt
└── .gitignore
```

### `src/digital_twin/`

| Archivo | Rol |
|---|---|
| `real_data.py` | Carga y partición de los datos reales (train/test). |
| `gp_model.py` | Hurdle model: `GaussianProcessClassifier` + GP físico. |
| `data_collection.py` | Utilidades de recolección (incluye `expandir_activo`, opcional). |
| `control.py` | Tareas de dirección y consigna inversa. |
| `simion_interface.py` | Conexión a SIMION real (evaluación y gradiente). |
| `main.py` | Pipeline completo, de punta a punta. |
| `kernel_experiment.py` | Comparación de kernels del regresor sobre el test real: Matérn 1.5 (R²=0.237, elegido) vs Matérn 2.5 (0.232) vs RBF (0.203). |
| `retrain_experiment.py` | Muestra que agregar las 32 sondas extra al entrenamiento degrada el R² global — por eso el modelo entregado usa 116 puntos. |
| `gradient_eps_study.py` | Estudio del paso ε para el chequeo de gradiente (criterio D): con ε=1V la comparación era ruido puro; con ε=5V la similitud coseno sube a +0.31. |
| `gradient_retrain_check.py` | Chequeo held-out: reentrenar con las sondas de ε=10V sube el acuerdo del gradiente a +0.47 (no incorporado al modelo entregado). |
| `verify_v2.py` | Verificación real de las tareas de control del modelo final (gasta corridas de SIMION; resultados en el informe: dirección 55.2±6.4, consigna 36.8±5.0). |
| `prototype_synthetic/` | Simuladores sintéticos usados solo para prototipar la arquitectura antes de tener datos reales — no forman parte del gemelo entregado, se conservan como referencia. |

### `data/`

- `beamline_results.csv` — 824 evaluaciones reales generadas por
  `simulation/optimizer.py` (Optuna + SIMION) antes de construir el gemelo.
  330 fallaron por interferencia de OneDrive con el archivo `electrode_.PA0`
  (ver informe, sección 2) — **no sincronizar esta carpeta con
  OneDrive/Google Drive/etc. al correr más evaluaciones.**
- `beamline_study.db` — base de datos de Optuna del estudio anterior.
- `gradient_probe_cache.npz` — 16 puntos de densificación local, ya
  congelados (SIMION es ruidoso, ver informe sección 7): promediados sobre 5
  corridas reales para que los resultados sean reproducibles.
- `gradient_eps_study_cache.npz` — datos reales del estudio de ε, congelados
  (160 corridas reales de SIMION, 16 puntos × 5 corridas por cada ε de
  {5, 10} V).

### `simulation/`

Archivos de la simulación SIMION + el optimizador Optuna que generó los
datos reales. **No incluye** `electrode_.PA0` ni `electrode_.PA1`...`PA19`
(arreglos de campo, pesados, cientos de MB cada uno) — se regeneran
localmente, es el mismo Paso 0 del enunciado del reto.

## Cómo reproducir

1. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Copiar SIMION 8.1 (`simion.exe`) a `simulation/`, o ajustar
   `SIMION_INSTALL_DIR` en `src/digital_twin/simion_interface.py` y
   `simulation/optimizer.py` si vive en otra ruta.
3. Generar los arreglos de campo (una sola vez, tarda unos minutos):
   ```
   cd simulation
   simion --nogui refine "electrode_.PA#"
   ```
   Esto crea `electrode_.PA0` y `electrode_.PA1`...`PA19` localmente.
4. Correr el gemelo:
   ```
   cd src/digital_twin
   python main.py
   ```
   Con las semillas fijas y el caché incluido, debe dar exactamente los
   mismos números reportados en el informe (R² región informativa = 0.237,
   etc.).
5. (Opcional) Verificación real de las tareas de control — gasta
   evaluaciones reales de SIMION, no se ejecuta automáticamente desde
   `main.py`:
   ```
   python -c "from control import verify_with_simion; print(verify_with_simion([...]))"
   ```

## Informes

- [`report/reporte_tecnico.tex`](report/reporte_tecnico.tex) — reporte
  técnico completo del proyecto (nuevo, explica todo de punta a punta).
- [`report/informe.tex`](report/informe.tex) — informe original entregado en
  el taller (7 secciones de la rúbrica).
- `report/generate_figures.py` — regenera las figuras de ambos informes
  desde los datos y cachés reales (sin llamadas nuevas a SIMION).
