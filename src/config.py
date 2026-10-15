"""
config.py — Configuración centralizada, rutas canónicas y constantes del proyecto.

Centraliza todas las rutas relativas al proyecto, fuente de datos en Kaggle,
semillas aleatorias y esquemas de columnas para garantizar reproducibilidad
entre todos los miembros del equipo y evaluadores.
"""

from pathlib import Path

# ==============================================================================
# RUTAS CANÓNICAS DEL PROYECTO
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "reports"

# ==============================================================================
# FUENTE OFICIAL DE DATOS: KAGGLE / BTS (Bureau of Transportation Statistics)
# ==============================================================================
KAGGLE_DATASET_SLUG = "umerhafeezsiddiqui/2021-2026-us-flights"

# Periodo disponible en los archivos locales: 1 de junio de 2025 al 31 de mayo de 2026.
# Mayo de 2025 no está descargado. Estos 12 meses cubren un ciclo estacional completo.
STUDY_PERIOD_FILES = [
    "Jun-2025.csv",
    "Jul-2025.csv",
    "Aug-2025.csv",
    "Sep-2025.csv",
    "Oct-2025.csv",
    "Nov-2025.csv",
    "Dec-2025.csv",
    "Jan-2026.csv",
    "Feb-2026.csv",
    "Mar-2026.csv",
    "Apr-2026.csv",
    "May-2026.csv",
]

# Rutas de archivos procesados y representativos para ejecución reproducible
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "us_flights_2025_2026_full.parquet"
SAMPLE_DATA_PATH = PROCESSED_DATA_DIR / "us_flights_sample_reproducible.csv"
SAMPLE_PARQUET_PATH = PROCESSED_DATA_DIR / "us_flights_sample_reproducible.parquet"

# ==============================================================================
# PARÁMETROS GLOBALES DE MACHINE LEARNING Y REPRODUCIBILIDAD
# ==============================================================================
RANDOM_STATE = 42
# Fracción aleatoria de los vuelos operados de cada mes. Conserva el peso del mes.
SAMPLE_FRACTION = 0.01
# Último tramo temporal, reservado para la prueba.
TEST_FRACTION = 0.25
# Pliegues de la validación cruzada temporal y de la búsqueda de hiperparámetros.
TIME_SPLITS = 3
TUNING_SPLITS = 2

# ==============================================================================
# DEFINICIÓN DEL TARGET Y ESQUEMA DE COLUMNAS
# ==============================================================================
# Variable objetivo principal (Clasificación binaria estándar FAA / DOT):
# 1 = Vuelo con retraso significativo en llegada (ArrDelay >= 15 min)
# 0 = Vuelo puntual o adelantado (ArrDelay < 15 min)
TARGET_COLUMN = "ARR_DEL15"
CONTINUOUS_TARGET_COLUMN = "ARR_DELAY"

# Columnas disponibles exclusivamente en etapa PRE-VUELO (Itinerario CRS):
PRE_FLIGHT_FEATURES = [
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_HOUR",
    "CRS_ARR_HOUR",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]

# Variables POST-DESPEGUE o CAUSALES que constituyen DATA LEAKAGE estricto
# Prohibidas para modelado predictivo antes del vuelo:
LEAKAGE_COLS = [
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DEL15",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",  # Fuga si se usa como feature; se conserva solo como derivador de target
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]

# Columnas administrativas / redundantes no generalizables:
ADMIN_COLS = [
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID",
]

