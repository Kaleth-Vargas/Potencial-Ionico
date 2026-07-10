# CLAUDE.md

Este archivo le da guía a Claude Code (claude.ai/code) para trabajar en este repositorio.

## Qué es esto

Gemelo digital de un beamline de iones SIMION (hackathon MIT-UNAL). Una
pila de 8 electrodos dirige 500 iones Si+ hacia un detector; `src/digital_twin/`
construye un modelo surrogate (hurdle GP) a partir de un presupuesto fijo de
evaluaciones reales de SIMION, lo valida, y lo usa para resolver dos tareas
de control (maximizar transmisión, alcanzar una transmisión objetivo) — todo
sin necesitar SIMION en el momento de predecir.

Estructura del repositorio (ver también `README.md`):
- `simulation/` — la simulación cruda de SIMION 8.1 (archivos de
  geometría de electrodos `.stl`/`.PA*`, `SimpleSetUp.iob`/`.fly2`) más
  `optimizer.py`, el script de Optuna+SIMION que originalmente generó
  `data/beamline_results.csv` (824 corridas, 494 válidas tras fallos de
  subproceso).
- `data/` — los datos reales (`beamline_results.csv`, `beamline_study.db`) y
  los cachés de reproducibilidad (`gradient_probe_cache.npz`,
  `gradient_eps_study_cache.npz`).
- `src/digital_twin/` — el entregable real: el modelo surrogate y la
  lógica de control. Aquí es donde ocurre casi todo el trabajo.
- `report/` — los informes en LaTeX/PDF (`informe.tex` de la rúbrica del
  taller, `reporte_tecnico.tex` con la explicación completa del proyecto) y
  `generate_figures.py`.

## Comandos

```bash
pip install -r requirements.txt

# Correr el pipeline completo (carga datos reales, ajusta el modelo, valida,
# resuelve ambas tareas de control, imprime el chequeo de gradiente).
# Determinístico — semillas fijas + datos de sondas cacheados reproducen
# exactamente los números del informe.
cd src/digital_twin
python main.py

# Forzar volver a consultar SIMION para los puntos de densificación del
# gradiente en vez de usar el .npz cacheado (solo tiene sentido si SIMION
# y los archivos de campo refinados están presentes):
python main.py --refresh-cache

# Chequeos independientes (cada uno tiene un bloque __main__):
python real_data.py           # inspeccionar la partición train/test
python simion_interface.py    # una llamada real a SIMION sobre el mejor punto conocido
```

No hay suite de tests, linter, ni paso de build — este es un código de
investigación/informe. La "corrección" se verifica corriendo `main.py` y
comparando contra los números en `report/informe.pdf` y `report/reporte_tecnico.pdf`.

Para efectivamente correr SIMION (opcional, no requerido para `main.py`):
1. Copiar SIMION 8.1 (`simion.exe`) a `simulation/`, o ajustar
   `SIMION_INSTALL_DIR` en `src/digital_twin/simion_interface.py` y
   `simulation/optimizer.py`.
2. Resolver el campo una sola vez: `cd simulation && simion --nogui refine "electrode_.PA#"`
   (genera `electrode_.PA0` y `.PA1`–`.PA19`, no están versionados — pesan
   cientos de MB, ver `.gitignore`).
3. NO sincronizar `simulation/` con OneDrive/Google Drive mientras se
   corren evaluaciones — la interferencia de bloqueo de archivo sobre
   `electrode_.PA0` causó que ~330 de las 824 corridas originales fallaran
   (ver `report/informe.pdf` sección 2).

## Arquitectura

**Flujo de datos (`real_data.py`):** `beamline_results.csv` (494 corridas
reales válidas de SIMION) se parte con *farthest-point sampling
estratificado* — 45 puntos de señal + 55 puntos en cero elegidos por
dispersión, no al azar — en 100 puntos de entrenamiento (el presupuesto
declarado) y ~394 puntos de test libres (todo lo que no se eligió; datos
reales que ya existían no cuestan nada extra). El orden de columnas siempre
es `ELECTRODE_IDS = [3, 6, 9, 10, 11, 12, 15, 18]`.

**Modelo (`gp_model.py`): hurdle model, no un solo regresor.** Como la
mayor parte del espacio de voltajes da transmisión cero (una "zona muerta"),
un solo GP no puede representar bien tanto el precipicio como el pico. Dos
etapas:
- Etapa A `classifier` (GP classifier, RBF+ARD): P(señal > 0 | V), entrenado
  con todos los puntos, incluyendo los ceros.
- Etapa B `magnitude_model` (`BeamlineGP`, Matérn ν=1.5 ARD): magnitud
  esperada dado que hay señal, entrenado SOLO con puntos de señal (los ceros
  aplastarían el pico).
- `predict_combined()` los combina vía ley de varianza total:
  `E[T] = p·μ_B`, `Var[T] = p(1-p)·μ_B² + p·σ_B²`.
- Ambas etapas reciben features aumentadas con `V3²` y `V6²` (los electrodos
  3 y 6 son lentes Einzel; la potencia de enfoque escala con el cuadrado del
  voltaje según la electrostática de lente delgada) — ver
  `_augment_physics_features`.
- Los targets se transforman con `log1p` antes de escalarse para comprimir
  el rango 0–500; las predicciones invierten la transformación y se
  recortan a `[0, MAX_IONS=500]`.
- La elección de kernel (Matérn 1.5 sobre RBF/Matérn 2.5) y de las features
  log1p+físicas se hizo empíricamente contra el test real — ver
  `kernel_experiment.py` y `retrain_experiment.py` para las comparaciones
  (agregar más sondas de entrenamiento al modelo base en realidad *degrada*
  el R², por eso el modelo entregado sigue usando exactamente 116 puntos).

**Control (`control.py`): todo está restringido a una región de confianza.**
Optimizar `predict_combined` libremente sobre todo el espacio de ±1000V
encuentra voltajes donde el modelo está confiadamente (std bajo) equivocado
— verificado con SIMION real, que dio 0 iones donde el gemelo predecía ~35.
Por eso ambas tareas de control restringen
`scipy.optimize.minimize(method='L-BFGS-B')` a una caja de ±5V alrededor de
un *punto real observado* (el radio óptimo calibrado empíricamente corriendo
SIMION real a varios radios — ver la tabla en el docstring del módulo
`control.py`):
- `find_optimal_voltages`: caja centrada en `best_known_real_point()` (la
  fila con mayor transmisión observada en el CSV).
- `find_inverse_setpoint`: caja centrada en `nearest_real_point_to(target)`,
  NO en el pico — anclar en el pico y pedirle al modelo que interpole la
  ladera descendente (empinada) de la campana produjo un error real mucho
  mayor que anclar cerca del objetivo mismo.
- `verify_with_simion()` / `verify_gradient_with_simion()` llaman a SIMION
  real y nunca se invocan automáticamente desde `main.py` — se corren a
  mano cuando se quiera gastar evaluaciones de verificación (no cuentan
  contra el presupuesto de entrenamiento de 100 puntos).

**Manejo de ruido:** SIMION no expone una semilla de RNG para el muestreo
de la fuente de iones desde `--nogui fly`, así que los mismos voltajes dan
conteos distintos en corridas distintas (confirmado: 74/60/63 iones en el
mismo punto). Todo lo que necesita un número real de SIMION promedia varias
repeticiones (`evaluate_one_real_averaged`, `n_repeats=5` por defecto).

**Reproducibilidad / caché:** `data/gradient_probe_cache.npz` y
`data/gradient_eps_study_cache.npz` congelan resultados reales de SIMION de
corridas de sondeo hechas una sola vez (16 puntos del chequeo de gradiente,
y un estudio de barrido de eps) para que `main.py` reproduzca exactamente
los números del informe sin volver a consultar SIMION (que daría números
distintos cada vez por el ruido mencionado arriba). Borrar el caché o pasar
`--refresh-cache` lo regenera — pero eso requiere una instalación local de
SIMION funcionando y produce números distintos a los del informe.

**`prototype_synthetic/`** contiene simuladores sintéticos usados para
prototipar la arquitectura antes de tener datos reales — no forma parte del
gemelo entregado, se conserva solo como referencia. No construir sobre esto.

**`data_collection.expandir_activo()`** es un gancho opcional de active
learning, no usado por defecto: consultaría SIMION en los puntos de mayor
incertidumbre de `predict_combined` para ampliar el presupuesto de
entrenamiento más allá de 100. Solo llamarlo si se decide deliberadamente
gastar presupuesto real adicional.
