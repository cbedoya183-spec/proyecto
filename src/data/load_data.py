"""
load_data.py — Funciones estandarizadas de carga y validación de datos.

Garantiza que cualquier módulo o notebook cargue los datos de manera consistente,
con validación de tipos, detección de codificación y estandarización de nombres
de columnas a formato canónico ASCII (ej. ANIO, DISENO).
"""

import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para ejecución directa
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    RAW_CSV_PATH,
    SAMPLE_CSV_PATH,
    TARGET_COLUMN,
    STANDARD_COLUMNS,
)
from src.utils.logger import get_logger

logger = get_logger("load_data")


def load_raw_data(use_sample: bool = True) -> pd.DataFrame:
    """
    Carga el dataset bruto desde data/raw/ con normalización de esquema.

    Args:
        use_sample: Si True, intenta cargar el dataset de muestra para desarrollo ágil.
                    Si False, carga el dataset completo de MEData.

    Returns:
        pd.DataFrame con los datos crudos estandarizados.
    """
    target_path = SAMPLE_CSV_PATH if use_sample and SAMPLE_CSV_PATH.exists() else RAW_CSV_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {target_path}. "
            "Ejecuta primero: python src/data/download_data.py"
        )

    logger.info(f"Cargando datos desde: {target_path.name}")

    # Cargar con codificación adecuada (latin-1 maneja acentos de datos colombianos)
    df = None
    for encoding in ["latin-1", "utf-8", "cp1252"]:
        try:
            df = pd.read_csv(target_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"No fue posible decodificar el archivo en {target_path}")

    # Estandarizar nombres de columnas a ASCII canónico si coincide el número de columnas
    if len(df.columns) == len(STANDARD_COLUMNS):
        df.columns = STANDARD_COLUMNS

    # Validación básica de estructura
    if TARGET_COLUMN not in df.columns:
        logger.warning(
            f"La columna objetivo '{TARGET_COLUMN}' no está explícita en las columnas. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    logger.info(f"Dataset cargado exitosamente: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
    return df
