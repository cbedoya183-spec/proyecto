# ✈️ Modelo Predictivo de Retrasos en Vuelos Comerciales de EE.UU. (2025–2026)
### Proyecto Final — Especialización en Analítica y Ciencia de Datos / Machine Learning
**Universidad EIA**  
**Autores:** Sebastián Olaya Pérez, Carlos Daniel Bedoya Moreno, Santiago Felipe Espinel León, Julián Alzate Aristizábal  

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![BTS DOT](https://img.shields.io/badge/source-BTS%20%2F%20DOT-blue.svg)
![Status](https://img.shields.io/badge/reproducibility-verified-green.svg)

---

## 📌 1. Descripción del Problema y Contexto de Negocio

El sistema de aviación comercial de los Estados Unidos representa la infraestructura de transporte más densa del mundo, operando más de 8 millones de vuelos comerciales al año. Los retrasos aéreos representan una pérdida económica anual superior a los **30.000 millones de dólares**, originando sobrecostos operativos para las aerolíneas (combustible adicional, horas extra de tripulación, reubicaciones y multas de la FAA), saturación de plataformas en los aeropuertos y pérdidas de conexiones para millones de pasajeros.

### ¿Por qué Machine Learning?
La puntualidad de un vuelo es el resultado de la interacción no lineal y dinámica de múltiples factores:
- **Estacionalidad climática:** Tormentas convectivas severas en verano y frentes árticos/nevadas en invierno.
- **Efecto cascada no estacionario:** Las aeronaves realizan múltiples trayectos diarios; una demora inicial de 15 minutos en la mañana se propaga y amplifica hacia la tarde y la noche.
- **Congestión topológica en red:** Retrasos concentrados en los principales *Hubs* del Sistema Nacional del Espacio Aéreo (NAS) de la FAA (Atlanta, O'Hare, Dallas-Fort Worth, Denver, JFK).

Una regla manual rígida no puede ponderar adecuadamente la combinación de estos factores. Machine Learning permite aprender fronteras de decisión y probabilidades calibradas a partir de registros históricos para anticipar el riesgo de demora con antelación.

---

## 🏛️ 2. Fuente Oficial de Datos y Periodo de Estudio

- **Entidad Oficial:** *Bureau of Transportation Statistics (BTS), U.S. Department of Transportation (DOT)*.
- **Sistema:** *Reporting Carrier On-Time Performance*.
- **Catálogo Kaggle:** [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights).
- **Licencia:** Obra del Gobierno Federal de los Estados Unidos (*U.S. Government Work* / Dominio Público), sin restricciones para investigación científica o uso comercial.
- **Periodo de Estudio Oficial:** **1 de junio de 2025 al 31 de mayo de 2026 (12 meses calendario completos)**.
  - Cubre un ciclo anual estacional completo: Verano (Jun–Ago 2025), Otoño (Sep–Nov 2025), Invierno (Dic 2025–Feb 2026) y Primavera (Mar–May 2026), incluyendo las temporadas de mayor congestión como Acción de Gracias, Navidad y *Spring Break*.
- **Granularidad:** Cada fila representa una operación aérea comercial regular doméstica en EE.UU.

---

## 🎯 3. Definición de la Variable Objetivo (Target)

Se formula un problema de **Clasificación Binaria Supervisada** basado en el estándar regulatorio oficial de la **Federal Aviation Administration (FAA)** y el **U.S. DOT**:

$$\text{Target } y = \begin{cases} 1 & \text{si } \text{ARR\_DELAY} \ge 15 \text{ minutos (Vuelo con retraso significativo)} \\ 0 & \text{si } \text{ARR\_DELAY} < 15 \text{ minutos (Vuelo puntual o adelantado)} \end{cases}$$

- **Justificación del umbral de 15 minutos:** Es el estándar internacional de la aviación comercial. Los vuelos que arriban con menos de 15 minutos de diferencia respecto al itinerario se consideran técnicamente a tiempo (*On-Time Arrival*), mientras que demoras $\ge 15$ minutos activan alertas de pérdida de conexiones, compensaciones a pasajeros y penalizaciones regulatorias.
- **Tratamiento de Cancelaciones y Desvíos:** Los vuelos con `CANCELLED == 1` o `DIVERTED == 1` no concluyen en su itinerario programado normal (`ARR_DELAY` es nulo). Se excluyen de la muestra de puntualidad regular para no introducir sesgo de proceso.

---

## 🛡️ 4. Prevención Absoluta de Data Leakage (Fuga de Información)

Para garantizar un modelo con utilidad real en producción, **toda la predicción se realiza exclusivamente con información disponible en la etapa de programación de itinerario (Pre-vuelo / Horarios CRS)**.

### Tabla de Decisiones y Exclusiones de Leakage:

| Categoría | Columnas | Decisión | Justificación Técnica |
|---|---|:---:|---|
| **Pre-Vuelo (Válidas)** | `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK`, `CRS_DEP_HOUR`, `CRS_ARR_HOUR`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DISTANCE`, `CRS_ELAPSED_TIME` | ✅ **CONSERVAR** | Disponibles con semanas o días de anticipación al vuelo. |
| **Operación en Tierra / Salida** | `DEP_TIME`, `DEP_DELAY`, `DEP_DEL15`, `TAXI_OUT` | ❌ **ELIMINAR** | Ocurren al salir de puerta; `DEP_DELAY` genera fuga trivial ($ArrDelay \approx DepDelay$, inflando artificialmente el $R^2$ a 0.97). |
| **En Vuelo** | `WHEELS_OFF`, `WHEELS_ON`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME` | ❌ **ELIMINAR** | Información generada en el aire o al aterrizar. |
| **Llegada y Causas** | `ARR_TIME`, `TAXI_IN`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | ❌ **ELIMINAR** | Consecuencias calculadas por la FAA con posterioridad al arribo. |

---

## 🏗️ 5. Estructura del Repositorio

El proyecto sigue una arquitectura modular para que la corrida se pueda repetir:

```text
Proyecto-final-Machine-Learning/
│
├── .gitignore                      # Reglas de exclusión para datos crudos y cachés
├── README.md                       # Documentación principal del proyecto
├── requirements.txt                # Dependencias fijadas del proyecto
├── setup.py                        # Instalación del paquete 'src' en modo editable
├── LICENSE                         # Licencia del proyecto
│
├── data/
│   ├── raw/                        # 12 CSVs mensuales de BTS (Jun-2025 a May-2026)
│   └── processed/                  # Dataset representativo (Parquet y CSV)
│
├── docs/
│   ├── diccionario_datos.md        # Catálogo de las 34 columnas y auditoría de leakage
│   └── guia_colaboracion.md        # Pautas de Git y ramas de trabajo
│
├── models/
│   ├── best_model.joblib           # Pipeline serializado del modelo campeón
│   ├── metrics_comparison.csv      # Tabla de la corrida del notebook
│   ├── hiperparametros.csv         # Búsqueda sobre entrenamiento
│   └── eleccion.txt                # Argumento escrito con esos números
│
├── notebooks/
│   └── proyecto_final_vuelos_us.ipynb # Notebook ejecutable con los 6 pasos oficiales
│
├── reports/
│   ├── auditoria_mensual_cruda.csv # Conteo de los CSV crudos, mes a mes
│   └── muestra_vs_mes.csv          # Vuelos operados vs filas tomadas de cada mes
│
└── src/
    ├── config.py                   # Rutas canónicas, periodo oficial y constantes
    ├── data/
    │   ├── download_data.py        # Descarga programática con Kaggle API y muestreo
    │   └── load_data.py            # Carga optimizada con fallback Parquet/CSV
    ├── features/
    │   ├── build_features.py       # Ingeniería de variables temporales y de red
    │   └── transformers.py         # Transformadores Scikit-Learn (RareCategoryGrouper, Cyclic)
    ├── models/
    │   ├── pipeline.py             # ColumnTransformer y mediana de duración por aerolínea
    │   ├── experiment.py           # Comparación, umbral y búsqueda de hiperparámetros
    │   ├── train.py                # Misma comparación desde la terminal
    │   └── evaluate.py             # F1 del retraso, ROC-AUC, PR-AUC y matrices
    └── utils/
        └── logger.py               # Configuración de registro profesional
```

---

## 6. Limpieza que sí se ve en los crudos

La tabla de modelado toma el 1 % de los vuelos operados de cada mes, al azar y con semilla 42: **76.201 filas**. Julio aporta 6.747 y febrero 5.543, porque julio operó más vuelos. El detalle está en `reports/muestra_vs_mes.csv`. Esa tabla ya llega sin cancelados. El efecto real de la limpieza está en `reports/auditoria_mensual_cruda.csv`, calculado sobre los 12 CSV:

| Concepto | Filas |
|---|---:|
| Universo crudo | 7.786.702 |
| Cancelados | 145.007 |
| Desviados | 21.637 |
| Sin `ARR_DEL15` | 166.645 |
| Operados, con target | 7.620.057 |
| Duplicados de la clave vuelo | 3 |
| Duración programada vacía | 4 |

Cancelados más desviados suman 166.644. Falta una fila, en 7,8 millones, que tiene el target vacío sin esas dos banderas. No se imputa: sin llegada al itinerario no hay etiqueta de puntualidad.

La duración programada casi no falta. Aun así, en el entrenamiento se escondió el 20 % de `CRS_ELAPSED_TIME`. La mediana por aerolínea bajó el error absoluto de 52,78 a 46,84 minutos (11,2 % menos que la mediana global; la media global se equivoca 54,37). Esa imputación vive dentro del Pipeline y se ajusta solo con cada pliegue.

---

## 7. Validación y modelos

Los vuelos se ordenan por `FL_DATE`. Un corte aleatorio entrenaría con 2026 para predecir 2025.

- **Entrenamiento:** 57.150 vuelos, 1 de junio de 2025 al 4 de marzo de 2026. Retraso: 22,70 %.
- **Prueba:** 19.051 vuelos, 4 de marzo al 31 de mayo de 2026. Retraso: 22,04 %. No elige el modelo ni el umbral.
- **Validación:** `TimeSeriesSplit` de 3 pliegues dentro del entrenamiento.

El corte es el último 25 % de las filas ya ordenadas por fecha. El 4 de marzo de 2026 queda partido: unos vuelos de ese día entrenan y otros se reservan para la prueba.

El umbral de cada modelo se elige en el último pliegue, maximizando el F1 del retraso. Con `class_weight='balanced'`, el 0,50 de scikit-learn no es el punto de operación. La exactitud a 0,50 no es la métrica de la decisión: la línea base ya acierta el 77,96 % de la prueba prediciendo siempre puntual, con recall 0.

Familias comparadas con los mismos datos y la misma validación cruzada:

1. Línea base: siempre la clase mayoritaria.
2. Regla manual: salida desde las 16:00 y origen o destino en un hub de la lista fija.
3. Regresión logística.
4. Árbol de decisión.
5. Random Forest.
6. Gradient boosting (`HistGradientBoosting`).
7. XGBoost.

KNN y SVM de kernel no entran: con decenas de miles de filas y aeropuertos en one-hot, cada predicción recorre el entrenamiento. La frontera lineal ya está en la logística. Una red neuronal no agrega una familia distinta del boosting en este problema.

---

## 8. Resultados de la corrida

El modelo se elige por ROC-AUC de la validación cruzada temporal. Si otro queda a menos de 0,01, se prefiere el más simple. Las columnas de precisión, recall y F1 son el reporte de la prueba, con el umbral ya fijado.

| Modelo | ROC-AUC validación | Umbral | Precisión | Recall | F1 retraso | ROC-AUC prueba |
|---|---:|---:|---:|---:|---:|---:|
| Regla manual | — | — | 0,3012 | 0,2608 | 0,2796 | — |
| Línea base | 0,5000 | 0,50 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| **Regresión logística** | **0,6193** | **0,30** | **0,2721** | **0,8073** | **0,4070** | **0,6524** |
| Árbol de decisión | 0,6107 | 0,40 | 0,2602 | 0,8244 | 0,3956 | 0,6207 |
| Random Forest | 0,6259 | 0,40 | 0,2713 | 0,7880 | 0,4037 | 0,6399 |
| Gradient boosting | 0,6273 | 0,40 | 0,2709 | 0,7923 | 0,4038 | 0,6374 |
| XGBoost | 0,6285 | 0,40 | 0,2661 | 0,8271 | 0,4026 | 0,6374 |

XGBoost tiene el mayor ROC-AUC de validación (0,6285). La logística queda en 0,6193. La diferencia es 0,0092, dentro de la banda de 0,01, y la logística es más simple. Esa diferencia también es menor que la desviación de los pliegues de la logística (0,0253). En la prueba, con el umbral ya fijado, el F1 de la logística es 0,4070. La prueba no se usó para elegir.

La logística, en su umbral 0,30, encuentra el 81 % de los retrasos de la prueba. La precisión es 0,27: de cada cuatro alertas, una es un retraso. La regla manual tiene F1 0,2796. El modelo la supera en el punto de operación.

### Hiperparámetros

La búsqueda usa otro `TimeSeriesSplit` de 2 pliegues, solo sobre el entrenamiento, 4 combinaciones por modelo. Se ajustan el campeón y el de mayor ROC-AUC de validación.

| Modelo | ROC-AUC base | ROC-AUC ajustado | Ganancia |
|---|---:|---:|---:|
| Regresión logística (`C` = 3,85) | 0,6030 | 0,6031 | 0,0001 |
| XGBoost | 0,6120 | 0,6103 | −0,0017 |

Hace falta una ganancia de al menos 0,005 para cambiar la configuración. No aparece. Se guarda la regresión logística base.

El 0,6030 de la logística en esta tabla no es el 0,6193 de la tabla anterior. El primero sale de los 2 pliegues usados para ajustar hiperparámetros. El segundo sale de los 3 pliegues usados para elegir la familia.

---

## 9. Instalación y ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/sebastia-olaya/Proyecto-final-Machine-Learning.git
cd Proyecto-final-Machine-Learning
```

### 2. Configurar entorno virtual e instalar dependencias
```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows:
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -e .
```

### 3. Descarga y preparación reproducible de datos
```bash
# Configura tu token de Kaggle si no lo has hecho:
# export KAGGLE_API_TOKEN="tu_token" o guárdalo en ~/.kaggle/access_token

# Descarga los 12 meses y genera el dataset reproducible:
python src/data/download_data.py --sample-fraction 0.01
```

### 4. La misma comparación desde la terminal
```bash
python src/models/train.py
```

`--quick` usa menos árboles. Los números del README salen de `python src/models/train.py`, con semilla 42. El notebook usa las mismas funciones.

### 5. Notebook
```bash
jupyter notebook notebooks/proyecto_final_vuelos_us.ipynb
```

La comparación de modelos tarda cerca de tres minutos. El notebook, con la auditoría y las gráficas, unos minutos más. La semilla es 42.
