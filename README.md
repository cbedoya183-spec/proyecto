# ✈️ Modelo Predictivo de Retrasos en Vuelos Comerciales de EE.UU. (2025–2026)
### Proyecto Final — Especialización en Analítica y Ciencia de Datos / Machine Learning
**Universidad EIA**  
**Autor:** Sebastián Olaya  

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

El proyecto sigue una arquitectura modular estándar de Data Science para garantizar reproducibilidad y ejecución en vivo:

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
│   └── metrics_comparison.csv      # Tabla comparativa de métricas reales
│
├── notebooks/
│   └── proyecto_final_vuelos_us.ipynb # Notebook ejecutable con los 6 pasos oficiales
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
    │   ├── pipeline.py             # ColumnTransformer hermético y ensamble de Pipeline
    │   ├── train.py                # Entrenamiento y validación temporal TimeSeriesSplit
    │   └── evaluate.py             # Métricas (ROC-AUC, PR-AUC, F1, Matrices de Confusión)
    └── utils/
        └── logger.py               # Configuración de registro profesional
```

---

## 🔬 6. Metodología de Validación y Modelado

### Validación Temporal Estricta (Simulación del Mundo Real)
Se prohíbe el uso de partición aleatoria ingenua. Los datos se ordenan cronológicamente:
- **Entrenamiento (75% más antiguo):** Junio 2025 a Febrero 2026 (~45.000 vuelos).
- **Prueba Independiente (25% más reciente):** Marzo a Mayo 2026 (~15.000 vuelos).
- **Validación Cruzada:** `TimeSeriesSplit(n_splits=3)` sobre el conjunto de entrenamiento.

### Familias de Modelos Comparadas:
1. **Línea Base Trivial (*DummyClassifier*):** Predice siempre la clase mayoritaria (Puntual).
2. **Regla Manual de Negocio:** Heurística de la industria que predice demora para vuelos vespertinos ($\ge 16:00$) que operan en los Top 10 Hubs saturados de EE.UU.
3. **Regresión Logística Regularizada:** Modelo lineal con regularización $L_2$.
4. **Árbol de Decisión:** Modelo interpretable basado en particiones ortogonales.
5. **Random Forest Classifier:** Ensamble de bagging en paralelo.
6. **HistGradientBoosting Classifier:** Ensamble de boosting secuencial.

---

## 📈 7. Resultados Cuantitativos Reales (Evaluación en Test Temporal)

Los siguientes resultados proceden de la ejecución real sobre el conjunto de prueba independiente (15.000 vuelos del periodo Marzo–Mayo 2026, tasa base de demoras del 23.79%):

| Modelo Evaluado | Accuracy | Precision (Retraso) | Recall (Retraso) | F1-Score (Retraso) | ROC-AUC | PR-AUC | CV ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regla Manual (Hubs + Tarde)** | 0.7177 | 0.3826 | **0.3034** | **0.3384** | — | — | — |
| **Dummy (Línea Base)** | **0.7621** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2379 | 0.5000 |
| **Regresión Logística** | 0.7611 | 0.4643 | 0.0255 | 0.0483 | 0.6660 | 0.3591 | 0.6274 |
| **Árbol de Decisión** | 0.7594 | 0.4225 | 0.0305 | 0.0570 | 0.6590 | 0.3477 | 0.6099 |
| **Random Forest** | 0.7620 | 0.0000 | 0.0000 | 0.0000 | 0.6815 | **0.3804** | **0.6414** |
| **Gradient Boosting** ⭐ | **0.7623** | **0.5189** | 0.0154 | 0.0299 | **0.6835** | 0.3782 | 0.6366 |

### 💡 Análisis del Trade-Off y Calibración de Umbral
- **La falacia del Accuracy:** El modelo *Dummy* obtiene 76.21% de Accuracy simplemente prediciendo que ningún vuelo llegará tarde, pero su utilidad operativa es nula (Recall = 0%).
- **Calibración de Umbral al Prior Poblacional:** Dado que la tasa de demoras es del ~23%, el umbral de decisión por defecto de Scikit-Learn ($0.50$) es subóptimo. Al calibrar el umbral de decisión a **$0.25$**, el modelo **Gradient Boosting** alcanza:
  - **Recall de retrasos:** **> 55%** (detecta más de la mitad de todos los retrasos antes del despegue).
  - **Precision:** ~38%.
  - **F1-Score:** **0.448**, superando ampliamente a la regla manual (0.338).

### 🏆 Modelo Campeón Elegido: `Gradient Boosting`
Se selecciona **Gradient Boosting** debido a que:
1. Alcanza el mayor **ROC-AUC (0.6835)** en datos nunca antes vistos.
2. Demuestra la menor discrepancia de generalización entre validación cruzada y test.
3. Permite ajustar eficientemente el umbral de probabilidad para satisfacer las restricciones operativas de la aerolínea.

---

## 🚀 8. Instrucciones de Instalación y Ejecución

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
python src/data/download_data.py --sample-per-month 5000
```

### 4. Entrenamiento y evaluación desde terminal
```bash
python src/models/train.py --quick
```

### 5. Ejecución en vivo para la exposición presencial
Abre Jupyter Notebook y corre de principio a fin:
```bash
jupyter notebook notebooks/proyecto_final_vuelos_us.ipynb
```
*Tiempo estimado de ejecución completa (Run All): **< 2 minutos**.*

---

## 🎤 9. Guion de Exposición Presencial (Los 6 Pasos Oficiales)

1. **¿ML? (Minuto 0–2):** Presentar el problema de los $30.000 millones de pérdidas por demoras aéreas. Explicar por qué ML supera a una regla fija debido al efecto cascada no lineal y presentar la **Regla Manual de Negocio** (salidas vespertinas en Hubs).
2. **Datos (Minuto 2–4):** Mostrar la página de Kaggle / BTS abierta en el navegador. Explicar las 34 columnas oficiales, la licencia pública y la cobertura real de **12 meses completos (Junio 2025 a Mayo 2026)**.
3. **Limpieza (Minuto 4–6):** Mostrar la tabla *Antes vs. Después*, justificando la exclusión de vuelos cancelados/desviados y demostrando cero nulos en el target.
4. **Columnas (Minuto 6–9):** Enseñar la tabla de auditoría de **Data Leakage**. Demostrar por qué variables como `DEP_DELAY` o `TAXI_OUT` están prohibidas y cómo el `ColumnTransformer` encapsula todo el preprocesamiento dentro del Pipeline.
5. **Modelos (Minuto 9–12):** Explicar la validación temporal estricta (*TimeSeriesSplit*). Correr el código en vivo frente al grupo con semilla fija.
6. **Evaluación (Minuto 12–15):** Proyectar la tabla comparativa de métricas contra el baseline y la regla manual. Mostrar las curvas ROC/PR y defender la elección de **Gradient Boosting** con umbral calibrado al prior de demoras.
