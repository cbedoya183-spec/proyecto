# ✈️ Predicción de retrasos en vuelos comerciales de EE.UU. (jun-2025 a may-2026)
### Proyecto final · Machine Learning · Especialización en Analítica y Ciencia de Datos · Universidad EIA

**Autores:** Sebastián Olaya Pérez · Carlos Daniel Bedoya Moreno · Santiago Felipe Espinel León · Julián Alzate Aristizábal

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)
![Fuente](https://img.shields.io/badge/fuente-BTS%20%2F%20DOT-blue.svg)
![Estado](https://img.shields.io/badge/reproducible-verificado-green.svg)

> **Pregunta:** con lo que se sabe **a la hora de salida programada** (itinerario, ruta, aerolínea, calendario, congestión y rotación de la aeronave), ¿llegará el vuelo **15 minutos o más tarde**?
> Clasificación binaria supervisada con la etiqueta oficial de la FAA/DOT (`ARR_DEL15`).

Esta entrega resuelve el proyecto completo siguiendo **los 6 pasos de la sesión 6** que pide la rúbrica, con comentarios línea a línea en el código y las decisiones justificadas en el notebook.

---

## 📋 Rúbrica y cómo la cubre esta entrega

| Criterio | Peso | Dónde se defiende |
|---|:---:|---|
| **Complejidad del problema** | 25 % | §1-2 · 12 CSV crudos (7,78 M filas, ~85 MB cada uno), lectura por bloques, tabla de rotación de aeronave armada por nosotros |
| **Datos y transformación** | 25 % | §3-4 · bitácora de limpieza, 4 imputadores comparados, tabla antes/después, 3 barreras anti-fuga + `assert` + demo empírica con `DEP_DELAY` |
| **Modelos** | 20 % | §5 · 8 modelos (línea base, regla a mano, logística, árbol, RF, HistGB, MLP, XGBoost), mismo Pipeline y mismos pliegues de `TimeSeriesSplit` |
| **Código en vivo** | 15 % | `Kernel → Restart & Run All` corre en ~3,5 min con `MODO_RAPIDO=True`, semilla 42, modelo guardado en `models/modelo_elegido.joblib` |
| **Métricas y elección** | 15 % | §6 · AUC + PR-AUC + F1 + Recall; umbral por F1 en validación (no en prueba); regla «dentro de 1 desviación típica, gana el más simple» |

---

## 🚀 Cómo correrlo

```bash
# 1) Dependencias
python -m venv .venv && .\.venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2) Descargar los 12 CSV mensuales de Kaggle a data/raw/
#    (Jun-2025.csv ... May-2026.csv). No se versionan: pesan ~1 GB en total.
#    Página: https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights

# 3) Abrir el notebook y correrlo de principio a fin
jupyter lab notebooks/proyecto_final_vuelos_us.ipynb     # Kernel -> Restart & Run All
```

En `src/config.py`:
- `MODO_RAPIDO = True`  → **modo exposición** (unos 3-4 minutos, 60k filas de validación y 200k de prueba).
- `MODO_RAPIDO = False` → **modo informe** (números del informe completo; tarda bastante más).

La semilla está fija en 42: los números del notebook coinciden con los de las láminas y con esta documentación.

La primera corrida deja una caché en `data/processed/*.parquet` (~90 s de lectura); las siguientes arrancan de ahí (<2 s).

---

## 📁 Estructura del repositorio

```text
├── README.md · requirements.txt · setup.py · .gitignore · LICENSE
├── src/                      todo el código; el notebook solo cuenta la historia
│   ├── config.py             constantes del proyecto en un solo lugar
│   ├── datos.py              pasos 2-3: cargar y limpiar los 12 CSV
│   ├── columnas.py           paso 4: columnas sin fuga + experimento de faltantes
│   ├── modelos.py            paso 5: catálogo, validación temporal y elección
│   └── evaluacion.py         paso 6: métricas, umbral, demo de fuga, gráficas
├── notebooks/
│   └── proyecto_final_vuelos_us.ipynb     el único notebook, con los 6 pasos
├── data/
│   ├── raw/                  CSV mensuales tal como salen de Kaggle (NO se versionan)
│   └── processed/            caché en parquet (NO se versiona)
├── models/                   modelo_elegido.joblib
├── reports/                  tablas CSV, resumen JSON y figuras PNG que produce el notebook
│   └── figuras/              gráficas (hipótesis, comparación, prueba, importancia)
├── docs/
│   ├── ficha_de_datos.md     paso 2 · fuente, licencia, qué es una fila, por qué complejo
│   ├── decisiones.md         pasos 3-4 · cada decisión con su conteo real
│   ├── diccionario_datos.md  significado de cada columna
│   └── guia_colaboracion.md  convenciones del equipo
└── Prediccion_retrasos_vuelos_EIA.pptx   presentación para la sesión 9
```

---

## 1 · ¿Hace falta Machine Learning?

El retraso de un vuelo depende de la interacción no lineal de:
- la **hora** (un avión hace varios tramos; el retraso de la mañana se propaga a la tarde),
- la **temporada** (tormentas en verano, nieve en invierno, Navidad, *Spring Break*),
- el **aeropuerto** (congestión programada de los *hubs*: ATL, ORD, DFW, DEN, CLT, LAX, JFK, MIA, MCO, EWR),
- la **aerolínea** y
- la **rotación de la aeronave** (el inbound que llega 40 minutos tarde difícilmente sale puntual).

Una regla a mano **no puede ponderar** todo eso. Nuestro punto de comparación:
> «Si el vuelo sale a las HH:00 o después, llegará retrasado». La hora `HH` se calibra con el entrenamiento (`modelos.calibrar_regla`).

**Línea base:** decir siempre «puntual» acierta ~78 % pero no detecta ni un retraso (recall = 0, F1 = 0). Por eso la **exactitud engaña** en este problema; la métrica principal es el **AUC**.

## 2 · Datos

| | |
|---|---|
| Publica | *Bureau of Transportation Statistics (BTS)*, U.S. DOT — *Reporting Carrier On-Time Performance* |
| Catálogo | Kaggle: [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights) |
| Licencia | *U.S. Government Work* (dominio público) |
| Periodo | 1-jun-2025 a 31-may-2026 (12 meses, un ciclo estacional completo) |
| Una fila | Un vuelo programado (origen → destino, aerolínea, fecha). Incluye cancelados y desviados. |
| Tamaño | 12 CSV crudos, **7.786.702 filas**, ~85 MB por mes |

**Complejidad del origen:** 12 archivos de varios GB en total → lectura por bloques (`chunksize=1_000_000`), reducción de tipos (`float64→float32`, `int→int32`, códigos a `category`) y caché en parquet para que todo quepa en una laptop común.

## 3 · Limpieza (bitácora completa en `docs/decisiones.md`)

Regla del paso: aquí solo se arregla lo que es un hecho objetivo de los datos. Nada que se *aprenda* (medias, medianas, codificaciones) se hace aquí: eso vive dentro del `Pipeline` del paso 5, que se reajusta en cada pliegue.

| Concepto | Filas |
|---|---:|
| Universo crudo | 7.786.702 |
| Cancelados | 145.007 |
| Desviados | 21.637 |
| Sin etiqueta `ARR_DEL15` | 166.647 |
| Operados con etiqueta | **7.620.055** |
| Duplicados exactos | 0 |
| Itinerarios imposibles (sin hora/duración/distancia) | 18 |

- Cancelados y desviados **no se imputan**: se marcan con `tiene_etiqueta = 0` y se filtran al final del paso 4 (sus horarios programados sí cuentan para la congestión y la rotación).
- `FL_DATE` mezcla dos formatos → la fecha se reconstruye desde `YEAR/MONTH/DAY_OF_MONTH`.
- Horas `hhmm` a minutos con `% 1440`: el `2400` del BTS cae en `0`.
- Atípicos de retraso se **conservan** (son reales: vuelos perdidos por tormentas).

## 4 · Columnas (tabla completa en `reports/tabla_columnas.csv`)

**Regla de oro:** una columna entra SOLO si su valor se conoce a la hora de salida programada.

Entran 11 numéricas + 4 categóricas:
- Del itinerario: `DISTANCE`, `CRS_ELAPSED_TIME`, `hora_sal_dec`, `hora_lleg`, `dia_semana`, `dias_a_festivo`.
- **Rotación de la aeronave (armada por nosotros):** `ret_prev_llegada`, `prev_disponible`, `holgura_prog`, `tramo_del_dia`.
- **Congestión programada:** `vuelos_origen_hora`, `vuelos_destino_hora`.
- Categóricas: `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `dia_semana`.

### Experimento de faltantes (`ret_prev_llegada`, 60k filas, ocultando 25 %)

| Método | MAE (min) |
|---|---:|
| **Mediana por hora de salida** ✅ | **18,44** |
| Mediana global | 18,57 |
| Regresión (HistGradientBoosting) | 19,94 |
| Media global | 20,25 |

Desviación típica de la variable: 71,76 min. La mediana por hora gana por poco; el Pipeline usa `SimpleImputer(strategy="median")` por simplicidad, reajustado por pliegue.

### Prevención de fuga: tres barreras + una demostración

1. Lista explícita de columnas (`cfg.NUMERICAS` + `cfg.CATEGORICAS`); el resto ni se lee del CSV.
2. Construcción con reglas temporales: `ret_prev_llegada` solo si el inbound aterrizó antes de ESTA salida programada; la congestión se cuenta sobre vuelos PROGRAMADOS (no sobre operados).
3. Preprocesamiento DENTRO del Pipeline, reajustado por pliegue de `TimeSeriesSplit`.

Más un `assert` (`columnas.verificar_sin_fuga()`) que aborta el notebook si una columna prohibida se cuela en `X`.

**Demostración empírica en vivo** (celda 4.4): el AUC con y sin `DEP_DELAY` enseña el salto que marca la fuga.

## 5 · Modelos y validación (resultados en `reports/tabla_modelos_cv.csv`)

- **Partición temporal:** entrenamiento = 85 % más antiguo (ordenado por fecha); **prueba = último 15 % del calendario** (corte en `2026-04-09`), que no interviene en la elección.
- **Validación:** `TimeSeriesSplit` de 3 pliegues dentro del entrenamiento, **la misma para todos**.
- **Métrica de elección:** ROC-AUC.
- **Umbral:** se elige en el último pliegue de validación maximizando el **F1 del retraso** (con 22 % de clase positiva, el 0,50 por defecto deja al modelo casi mudo).
- **Familias comparadas:** línea base, regla a mano, logística, árbol, Random Forest, HistGradientBoosting, red neuronal (MLP) y XGBoost. KNN y SVM-kernel se descartan por costo cuadrático con cientos de miles de filas; además no aportan una familia nueva.
- **Regla de elección:** mayor AUC de validación; si otro modelo queda dentro de **una desviación típica**, gana el más simple. Es la respuesta directa a *«no siempre gana el número más alto»* (lámina 9 de la rúbrica).

### Tabla de validación cruzada (3 pliegues de 60k, modo rápido)

| Modelo | AUC validación | ± desv. | PR-AUC | F1 | Recall | Precisión | Umbral |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,7989 | 0,0064 | 0,6578 | 0,599 | 0,563 | 0,640 | 0,32 |
| XGBoost | 0,7964 | 0,0076 | 0,6521 | 0,599 | 0,537 | 0,678 | 0,36 |
| **Random Forest** ✅ | **0,7943** | **0,0074** | **0,6447** | **0,593** | **0,539** | **0,659** | **0,33** |
| MLP (red neuronal) | 0,7823 | 0,0100 | 0,6225 | 0,580 | 0,535 | 0,632 | 0,32 |
| Árbol de decisión | 0,7813 | 0,0097 | 0,6325 | 0,593 | 0,586 | 0,599 | 0,27 |
| Regresión logística | 0,7367 | 0,0063 | 0,5523 | 0,511 | 0,442 | 0,608 | 0,29 |
| Regla a mano (hora ≥ 12) | 0,5794 | 0,0105 | 0,2424 | 0,381 | 0,701 | 0,261 | 0,50 |
| Línea base (siempre a tiempo) | 0,5000 | 0,0000 | 0,2105 | 0,000 | 0,000 | 0,000 | 0,50 |

**Elección argumentada:** el mejor AUC lo saca HistGradientBoosting (0,7989), pero XGBoost (0,7964) y Random Forest (0,7943) quedan dentro del margen de 1 desviación típica (0,0064 → umbral de empate en 0,7925). De los tres, **Random Forest es el más simple de interpretar** (importancias nativas, sin boosting iterativo) → **gana el empate**. El ajuste de hiperparámetros no se intenta porque los tres ya están empatados en validación.

## 6 · Evaluación (prueba final tocada UNA SOLA VEZ)

Partición temporal usada:

| Bloque | Fechas | Filas | Tasa de retraso |
|---|---|---:|---:|
| Entrenamiento + validación | 1-jun-2025 → 8-abr-2026 | 60.000 (muestra) | 23,07 % |
| **Prueba final** | 9-abr-2026 → 31-may-2026 | 200.000 (muestra) | 20,53 % |

### Resultados sobre la prueba (`reports/tabla_prueba_final.csv`)

| Modelo | AUC | PR-AUC | F1 | Recall | Precisión | Exactitud |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** ✅ | **0,806** | **0,659** | **0,586** | **0,524** | **0,666** | **0,848** |
| Regla a mano (hora ≥ 12) | 0,600 | 0,247 | 0,386 | 0,738 | 0,261 | 0,518 |
| Línea base (siempre a tiempo) | 0,500 | 0,205 | 0,000 | 0,000 | 0,000 | 0,795 |

- **F1 del retraso: 0,586** (vs 0,386 de la regla a mano y 0 de la línea base).
- **Recall 52 % · Precisión 67 %**: el umbral 0,333 captura la mitad de los retrasos reales con una precisión decente (dos de cada tres alertas son retrasos verdaderos).
- La **exactitud** del Random Forest (0,848) solo mejora a la línea base (0,795) en 5 puntos: con 20 % de retrasos esa métrica engaña, por eso vigilamos AUC, F1 y recall.

### Top variables (importancia por permutación, caída del AUC)

| Columna | Caída del AUC |
|---|---:|
| `ret_prev_llegada` (retraso del tramo anterior) | **+0,110** |
| `prev_disponible` (bandera: ¿ya aterrizó el inbound?) | **+0,052** |
| `holgura_prog` (minutos entre llegada previa y salida) | **+0,033** |
| `hora_sal_dec` | +0,013 |
| `DEST` (aeropuerto de destino) | +0,008 |

Las tres primeras son las **columnas que construimos**: la rotación de la aeronave es, por lejos, lo que más mueve el AUC. Confirma la hipótesis de la §1.

---

## 🧪 Reproducibilidad

- `src/config.py:SEMILLA = 42` fija toda la aleatoriedad del proyecto.
- `MODO_RAPIDO=True` corre toda la cadena en ~3,5 min (`Restart & Run All`) con 60k CV y 200k test.
- `MODO_RAPIDO=False` reproduce los números del informe completo.
- La caché parquet acelera las corridas sucesivas (<2 s para la carga).
- `models/modelo_elegido.joblib` guarda el Pipeline entrenado, el umbral, la hora de la regla y las columnas de `X`.
- Resumen reproducible en `reports/resumen_final.json`.

## 📚 Documentación adicional

- `docs/ficha_de_datos.md` — paso 2: fuente, licencia, qué es una fila, por qué el problema es complejo.
- `docs/decisiones.md` — pasos 3-4: cada decisión con su conteo real de la corrida.
- `docs/diccionario_datos.md` — significado y origen de cada columna.
- `docs/guia_colaboracion.md` — convenciones del equipo.

## 📄 Licencia

MIT (ver `LICENSE`). Los datos de BTS son dominio público (*U.S. Government Work*).
