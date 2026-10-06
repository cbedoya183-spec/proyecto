# 🚦 Modelo Predictivo de Severidad de Siniestros Viales en Medellín
### Proyecto Final — Especialización en Analítica y Ciencia de Datos / Machine Learning

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0%2B-red.svg)
![License](https://img.shields.io/badge/license-GPLv3-green.svg)

---

## 📌 1. Descripción del Problema y Contexto de Negocio

En el Distrito de Medellín, la siniestralidad vial constituye una de las principales problemáticas de salud pública, congestión urbana y costo socioeconómico. Cientos de incidentes ocurren semanalmente, involucrando motociclistas, peatones, ciclistas y vehículos particulares en corredores neurálgicos como la Autopista Sur, la Avenida Regional y la Calle San Juan.

### El Reto de Analítica
Cuando un ciudadano llama a la **Línea Única de Seguridad y Emergencias (123)** o a la **Secretaría de Movilidad de Medellín** para reportar un accidente, los operadores cuentan únicamente con información preliminar: la ubicación (comuna, barrio, dirección), el tipo de incidente reportado (choque, atropello, caída) y la hora.

**Objetivo de Machine Learning:**  
Construir un pipeline de clasificación supervisada capaz de estimar en tiempo real la probabilidad de que un siniestro reporte **consecuencias críticas con víctimas humanas (`Con heridos` o `Con muertos`)** versus **únicamente daños materiales (`Solo daños`)**, permitiendo:
1. **Optimizar el despacho de ambulancias:** Priorizar el envío inmediato de unidades de Soporte Vital Avanzado (SVA) a los incidentes con mayor probabilidad de criticidad.
2. **Focalización preventiva:** Identificar puntos negros espaciotemporales (comunas y franjas horarias de alto riesgo) para desplegar agentes de tránsito e intervenciones de infraestructura.

---

## 🏛️ 2. Fuente de Datos Oficial

- **Entidad:** Secretaría de Movilidad de Medellín / MEData (Estrategia de Datos de la Alcaldía de Medellín).
- **Catálogo:** [Portal de Datos Abiertos de Medellín](http://medata.gov.co) y [Datos Abiertos Colombia (datos.gov.co)](https://www.datos.gov.co) (ID: `9wqu-juqb`).
- **Registros:** +350.000 incidentes georreferenciados históricos.

---

## 🏗️ 3. Estructura del Repositorio

El proyecto sigue una arquitectura modular estándar de Data Science para facilitar la colaboración en equipo y garantizar la reproducibilidad:

```text
Proyecto-final-Machine-Learning/
│
├── .gitignore                      # Reglas de exclusión para datos y modelos
├── README.md                       # Documentación principal del proyecto
├── requirements.txt                # Dependencias fijadas del proyecto
├── setup.py                        # Instalación del paquete 'src' en modo editable
├── LICENSE                         # Licencia del proyecto (GPL v3)
│
├── data/
│   ├── raw/                        # Datos crudos de MEData (ignorado por Git)
│   └── processed/                  # Datasets limpios y procesados (ignorado por Git)
│
├── docs/
│   ├── diccionario_datos.md        # Catálogo de variables y justificación de exclusiones
│   ├── guia_colaboracion.md        # Pautas de Git, ramas y onboarding del equipo
│   └── informe_academico.md        # Informe técnico final según la rúbrica docente
│
├── notebooks/
│   ├── 01_eda_comprension.ipynb    # Análisis exploratorio, geografía y temporalidad
│   ├── 02_pipeline_preprocesamiento.ipynb # Experimentación de transformadores
│   └── 03_modelado_evaluacion.ipynb       # Comparación de modelos, métricas y SHAP
│
└── src/
    ├── __init__.py
    ├── config.py                   # Rutas canónicas, semillas y esquemas
    ├── data/
    │   ├── download_data.py        # Descarga programática con soporte de muestra
    │   └── load_data.py            # Carga robusta con fallback de codificación
    ├── features/
    │   ├── build_features.py       # Extracción de variables temporales y espaciales
    │   └── transformers.py         # Transformadores compatibles con Scikit-Learn
    ├── models/
    │   ├── pipeline.py             # Ensamble de ColumnTransformer y Estimadores
    │   ├── train.py                # Entrenamiento y validación cruzada estratificada
    │   └── evaluate.py             # Curvas PR-AUC, ROC y matrices de confusión
    └── utils/
        ├── logger.py               # Configuración de registro profesional
        └── plots.py                # Visualizaciones estandarizadas
```

---

## 🚀 4. Instalación y Puesta en Marcha (Para el Equipo)

Sigue estos 4 pasos para configurar tu entorno local:

### 1. Clonar el repositorio
```bash
git clone https://github.com/sebastia-olaya/Proyecto-final-Machine-Learning.git
cd Proyecto-final-Machine-Learning
```

### 2. Crear y activar el entorno virtual
```bash
# En Windows:
python -m venv .venv
.\.venv\Scripts\activate

# En Linux / Mac:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias y paquete modular editable
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Descargar los datos oficiales de MEData
```bash
# Descarga el archivo oficial y crea una muestra rápida para desarrollo de 50.000 filas:
python src/data/download_data.py --sample-size 50000
```

---

## 🔬 5. Metodología de Machine Learning

1. **Aislamiento de Fuga de Datos (Data Leakage):**  
   Se excluyen identificadores posteriores al evento (`EXPEDIENTE`, `NRO_RADICADO`). Todo el preprocesamiento se realiza dentro de `Pipeline` ajustado únicamente en el conjunto de entrenamiento.
2. **Evaluación Rigurosa ante Desbalance:**  
   Dado que los siniestros graves son de alta criticidad, la optimización no se guía por Accuracy simple, sino por:
   - **PR-AUC (Área bajo la curva Precision-Recall)**
   - **F1-Score Ponderado y Macro**
   - **Recall de la clase crítica (`es_grave = 1`)**
3. **Modelos Comparados:**
   - Baseline: *DummyClassifier* & *Regresión Logística* regularizada.
   - Modelos Basados en Árboles: *Random Forest Classifier*.
   - Estado del Arte (Boosting): *XGBoost Classifier* / *LightGBM*.
4. **Interpretabilidad:**  
   Análisis de importancia de variables con *SHAP (SHapley Additive exPlanations)* para derivar conclusiones accionables para la Secretaría de Movilidad.

---

## 👥 6. Colaboración en Equipo

Por favor lee atentamente [docs/guia_colaboracion.md](docs/guia_colaboracion.md) antes de comenzar a trabajar.
- Crea ramas de trabajo independientes: `git checkout -b feature/nombre-tarea`.
- No hagas push de archivos en `data/` o `models/`.
- Usa commits semánticos (`feat:`, `fix:`, `docs:`, `refactor:`).
