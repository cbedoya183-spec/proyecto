"""
config.py — Configuración centralizada, rutas canónicas y constantes del proyecto.

Centraliza todas las rutas relativas al proyecto, URLs de datos oficiales,
semillas aleatorias y esquemas de columnas para garantizar reproducibilidad
entre todos los miembros del equipo.
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

# Archivos específicos de datos
RAW_CSV_PATH = RAW_DATA_DIR / "incidentes_viales.csv"
SAMPLE_CSV_PATH = RAW_DATA_DIR / "incidentes_viales_sample.csv"
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "incidentes_viales_limpio.csv"

# ==============================================================================
# FUENTE OFICIAL DE DATOS ABIERTOS (MEData / Alcaldía de Medellín)
# ==============================================================================
MEDATA_CSV_URL = (
    "http://medata.gov.co/sites/default/files/distribution/1-023-25-000094/incidentes_viales.csv"
)

# ==============================================================================
# PARÁMETROS GLOBALES DE MACHINE LEARNING Y REPRODUCIBILIDAD
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5

# ==============================================================================
# ESQUEMA DE COLUMNAS ESTANDARIZADAS (ASCII-Friendly)
# ==============================================================================
STANDARD_COLUMNS = [
    "ANIO",
    "CBML",
    "CLASE_ACCIDENTE",
    "DIRECCION",
    "DIRECCION_ENCASILLADA",
    "DISENO",
    "EXPEDIENTE",
    "FECHA_ACCIDENTE",
    "FECHA_ACCIDENTES",
    "GRAVEDAD_ACCIDENTE",
    "MES",
    "NRO_RADICADO",
    "NUMCOMUNA",
    "BARRIO",
    "COMUNA",
    "LOCATION",
    "X",
    "Y",
]

# Variable objetivo original en MEData: 'Con heridos', 'Solo daños', 'Con muertos'
TARGET_COLUMN = "GRAVEDAD_ACCIDENTE"

# Variable objetivo binaria para clasificación supervisada:
# 1 = Severo / Crítico ('Con heridos' o 'Con muertos')
# 0 = No severo / Solo daños materiales ('Solo daños')
BINARY_TARGET_COLUMN = "es_grave"

# Identificadores administrativos que generan data leakage si se usan en el modelo
LEAKAGE_COLS = [
    "EXPEDIENTE",
    "NRO_RADICADO",
    "CBML",
    "FECHA_ACCIDENTES",  # Duplicado en formato ISO de FECHA_ACCIDENTE
]

# Columnas espaciales y geográficas
SPATIAL_COLS = ["COMUNA", "BARRIO", "NUMCOMUNA", "LOCATION", "X", "Y"]

# Columnas temporales
TEMPORAL_COLS = ["FECHA_ACCIDENTE", "MES", "ANIO"]

# Columnas del incidente y geometría de la vía
ROAD_COLS = ["CLASE_ACCIDENTE", "DISENO", "DIRECCION", "DIRECCION_ENCASILLADA"]
