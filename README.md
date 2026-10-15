# Retrasos de vuelos en EE. UU. · Proyecto final de Machine Learning

Universidad EIA · Machine Learning · Proyecto final (30 % de la nota) · Exposición presencial, sesión 9

**Equipo:** _completar nombres (2 o 4 integrantes)_

## El problema

Dado un vuelo programado, ¿llegará con **15 minutos o más de retraso**? Es una clasificación binaria y la
predicción se emite **a la hora de salida programada**: solo puede usar lo que ya se sabe en ese instante
(itinerario, ruta, calendario, congestión programada del aeropuerto y cómo venía la aeronave en su tramo
anterior). Todo lo que se mide durante o después del vuelo es **fuga** y está excluido.

- **Datos:** [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights)
  (Kaggle, a partir del reporte *On-Time Performance* del Bureau of Transportation Statistics, US DOT).
- **Periodo:** junio 2025 – mayo 2026 (12 meses calendario, un ciclo estacional completo).
- **Etiqueta:** `ARR_DEL15` (definición oficial del DOT: un vuelo está retrasado desde los 15 minutos).

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Descargar los CSV mensuales de Kaggle y dejarlos en datos/crudos/
#    (Jun-2025.csv … May-2026.csv). No se versionan: pesan varios GB.

# 2) Abrir el notebook y correrlo de principio a fin
jupyter lab notebooks/proyecto_final_vuelos.ipynb      # Kernel -> Restart & Run All
```

En `src/config.py`, `MODO_RAPIDO = True` es el modo de la exposición (unos minutos) y `MODO_RAPIDO = False`
reproduce los números del informe. La semilla está fija en 42, así que los números coinciden con los de las
láminas. La primera corrida deja una caché en `datos/procesados/*.parquet`; las siguientes arrancan de ahí.

## Estructura

```
├── README.md · requirements.txt · .gitignore
├── src/                   todo el código; el notebook solo cuenta la historia
│   ├── config.py          periodo, semilla, tamaños, listas de columnas: TODO se ajusta aquí
│   ├── datos.py           pasos 2-3: cargar los CSV, auditar y limpiar
│   ├── columnas.py        paso 4: columnas pre-vuelo, control de fugas, experimento de faltantes
│   ├── modelos.py         paso 5: regla a mano, catálogo de modelos, validación temporal
│   └── evaluacion.py      paso 6: métricas, umbral, prueba final, gráficas
├── notebooks/
│   └── proyecto_final_vuelos.ipynb     el único notebook: los 6 pasos, en orden
├── datos/crudos/          CSV de Kaggle tal cual (NO se versionan)
├── datos/procesados/      caché en parquet (NO se versiona)
├── modelos/               modelo_elegido.joblib
├── resultados/            tablas CSV y figuras PNG que produce el notebook
└── docs/                  ficha de datos y bitácora de decisiones
```

## La ruta de los 6 pasos y dónde vive cada uno

| Paso | Qué se muestra | Código |
|---|---|---|
| 1 · ¿ML? | Las cinco preguntas y la regla a mano como punto de comparación | `modelos.ReglaManual`, `modelos.calibrar_regla` |
| 2 · Datos | Página de Kaggle, licencia, cobertura mensual, qué es una fila | `datos.cargar`, `datos.cobertura_mensual` |
| 3 · Limpieza | Nulos, cancelados, duplicados, formatos, atípicos, tabla antes/después | `datos.limpiar`, `datos.foto` |
| 4 · Columnas | Columnas creadas, tabla de decisiones, faltantes, demostración de fuga | `columnas.py`, `evaluacion.demostrar_fuga` |
| 5 · Modelos | Ocho modelos, los mismos pliegues, preprocesamiento dentro del Pipeline | `modelos.comparar_modelos` |
| 6 · Evaluación | AUC, PR-AUC, F1, recall; elección argumentada; prueba final | `modelos.elegir_modelo`, `evaluacion.py` |

## Decisiones de diseño que conviene poder defender

- **Fugas.** `DEP_DELAY`, `ARR_DELAY`, `TAXI_*`, `WHEELS_*` y las cinco columnas de causas del retraso se
  conocen después del evento. A `X` solo pasa una lista explícita de columnas (`config.NUMERICAS` y
  `config.CATEGORICAS`), y un `assert` hace fallar el notebook si una prohibida se cuela. El notebook muestra
  el AUC **con y sin** fuga para que se vea la diferencia.
- **Cancelados y desviados.** No tienen etiqueta y **no se imputan**: taparlos con la mediana inventa una
  etiqueta falsa. Se marcan y se filtran al final del paso 4, porque sus horarios programados sí cuentan para
  la congestión y la rotación.
- **Etiqueta desbalanceada** (~20 % de retrasos). La línea base «siempre a tiempo» ya acierta ~80 %: la
  exactitud engaña. La métrica principal es el AUC, y el umbral de decisión se elige con la validación del
  entrenamiento, nunca con la prueba.
- **Validación temporal.** `TimeSeriesSplit` dentro del entrenamiento y el último 15 % del calendario como
  prueba final, tocada una sola vez. Un K-fold aleatorio entrenaría con el futuro.
- **Elección del modelo.** Entre los que quedan dentro de una desviación típica del mejor AUC, gana el más
  simple. Un empate estadístico se rompe por interpretabilidad y costo, no por el tercer decimal.
- **Muestreo.** La tabla completa tiene millones de filas: se explora entera, pero los modelos se comparan con
  `N_CV` filas y la exposición corre con el modo rápido. Se ajusta en `src/config.py`.
- **Tamaño.** GitHub rechaza archivos de más de 100 MB: los datos crudos y la caché no se suben, se entregan
  con el enlace de descarga.
