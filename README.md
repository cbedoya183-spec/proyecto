# Retrasos de vuelos en EE. UU. · Taller de Machine Learning

Universidad EIA · Machine Learning · Proyecto final (30 % de la nota) · Exposición presencial, sesión 9

**Equipo:** _TODO nombres (2 o 4 integrantes)_

## El problema

Dado un vuelo programado, ¿llegará con **15 minutos o más de retraso**? (clasificación binaria).
La predicción se hace **antes del despegue**: solo se usa lo que ya se sabe en ese momento (itinerario, ruta, fecha, congestión
del aeropuerto). Todo lo que se conoce después del despegue es una **fuga** y está excluido.

- **Datos:** [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights) (Kaggle). Ver `datos/LEEME.md`.
- **Ventana del taller:** mayo 2025 – mayo 2026 (13 meses). Los archivos se unen en una sola tabla y se recorta esa ventana.
- **Segunda fuente:** [OurAirports](https://ourairports.com/data/) (latitud, longitud y elevación de cada aeropuerto) y los festivos federales de EE. UU.

## Cómo correrlo

```bash
git clone <URL-del-repo> && cd taller-ml-vuelos
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python scripts/00_descargar.py        # Kaggle (kagglehub) + aeropuertos → datos/crudos y datos/externos
python scripts/01_construir_tabla.py  # une archivos, recorta fechas, arma la tabla → datos/procesados y datos/muestra
python scripts/02_entrenar_todo.py --rapido   # de principio a fin, con la muestra pequeña (el modo de la exposición)
python scripts/02_entrenar_todo.py            # comparación completa (N_MODELOS filas)
```

La muestra (`datos/muestra/vuelos_muestra.parquet`) la genera `01_construir_tabla.py` con los datos reales; **hay que hacer `git add` de ese archivo**.
Con ella, cualquiera que clone el repositorio puede correr los pasos 5 y 6 **sin descargar nada**: `python scripts/02_entrenar_todo.py --rapido`. Semilla fija (`SEMILLA = 42`): los números coinciden con las láminas.

## Estructura

```
├── README.md
├── requirements.txt · pyproject.toml · .gitignore
├── datos/
│   ├── LEEME.md            fuente, licencia y cómo descargar (lo que se muestra en la exposición)
│   ├── crudos/             archivos de Kaggle tal cual (NO se versiona: pesa demasiado)
│   ├── externos/           tabla de aeropuertos
│   ├── procesados/         tabla principal armada por nosotros (NO se versiona)
│   └── muestra/            muestra pequeña de la tabla principal (SÍ se versiona, sirve para correr en vivo)
├── notebooks/              un notebook por bloque de la ruta de los 6 pasos
│   ├── 01_problema_y_datos.ipynb       pasos 1-2  ¿ML? y datos
│   ├── 02_limpieza_y_columnas.ipynb    pasos 3-4  faltantes, fugas, columnas
│   ├── 03_modelos.ipynb                paso 5     varios modelos, misma validación
│   └── 04_evaluacion.ipynb             paso 6     métricas, elección, prueba final
├── src/vuelos/             el código reutilizable (los notebooks y los scripts lo importan)
│   ├── config.py           ventana de fechas, semilla, nombres de columnas: TODO se ajusta aquí
│   ├── descarga.py         paso 2
│   ├── datos.py            pasos 2-3: unir archivos, ventana, tabla principal, muestra
│   ├── limpieza.py         paso 3: imputación (dentro del Pipeline) y comparación de métodos
│   ├── columnas.py         paso 4: columnas nuevas y preprocesamiento (dentro del Pipeline)
│   ├── modelos.py          paso 5: los modelos y el Pipeline
│   └── evaluacion.py       paso 6: métricas, regla a mano, umbral, prueba final
├── scripts/                lo que se ejecuta desde la terminal
├── tests/                  prueba de humo con datos inventados (python -m pytest)
├── docs/
│   ├── decisiones.md       bitácora: cada decisión con su justificación
│   └── checklist_rubrica.md
├── modelos_guardados/      modelo_final.joblib (pequeño, SÍ se versiona)
└── resultados/             tabla_modelos.csv y figuras
```

## La ruta de los 6 pasos y dónde vive cada uno

| Paso | Qué se muestra | Código | Notebook |
|---|---|---|---|
| 1 · ¿ML? | Regla a mano como punto de comparación | `evaluacion.regla_a_mano` | 01 |
| 2 · Datos | Página de Kaggle, descarga, filas/columnas, etiqueta | `descarga.py`, `datos.cargar_ventana` | 01 |
| 3 · Limpieza | Faltantes, duplicados, fugas | `limpieza.py`, `datos.construir_tabla` | 02 |
| 4 · Columnas | Tabla antes/después, columnas creadas y descartadas | `columnas.py` | 02 |
| 5 · Modelos | Varios modelos, misma validación cruzada, dentro del Pipeline | `modelos.py` | 03 |
| 6 · Evaluación | Tabla con todos + línea base + regla a mano; elección argumentada | `evaluacion.py` | 04 |

## Decisiones de diseño que conviene saber

- **Fugas.** `retraso_salida`, `retraso_llegada`, `cancelado` y `desviado` se conocen después del evento. `preparar_xy` aborta si alguna
  entra como columna de entrada. El notebook 02 muestra el AUC con y sin fuga.
- **Etiqueta desbalanceada** (~20 % de retrasos). La línea base «siempre a tiempo» ya acierta ~80 %: la exactitud engaña. Se compara con **AUC**
  y el umbral de decisión se elige con la validación cruzada del entrenamiento (`evaluacion.umbral_optimo`), nunca con la prueba.
- **Congestión del aeropuerto** (`vuelos_origen_hora`, `vuelos_origen_dia`): se cuenta con todos los vuelos programados y **antes** de muestrear.
- **Muestreo.** La tabla completa tiene millones de filas: se explora completa, pero los modelos se comparan con `N_MODELOS` filas
  (estratificadas) y la exposición corre con `N_RAPIDO`. Ajustar en `config.py`.
- **Modelos.** Solo los vistos hasta la sesión 5 (línea base, logística, KNN, árbol, random forest, gradient boosting).
  XGBoost, SVM y red neuronal están listos en `modelos.modelos_extension()`: se activan con `INCLUIR_S7_S8 = True`.
- **Tamaño.** GitHub rechaza archivos de más de 100 MB: los datos crudos y procesados no se suben; se entregan con el enlace de descarga.
