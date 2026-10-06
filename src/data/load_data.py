"""
load_data.py — Funciones estandarizadas de carga y validación de datos.

Garantiza que cualquier módulo o notebook cargue los datos de manera consistente,
con validación de tipos, detección de codificación y control de versiones de muestra.
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
)
from src.utils.logger import get_logger

logger = get_logger("load_data")


def load_raw_data(use_sample: bool = True) -> pd.DataFrame:
    """
    Carga el dataset bruto desde data/raw/.

    Args:
        use_sample: Si True, intenta cargar el dataset de muestra para desarrollo ágil.
                    Si False, carga el dataset completo de MEData.

    Returns:
        pd.DataFrame con los datos crudos.
    """
    target_path = SAMPLE_CSV_PATH if use_sample and SAMPLE_CSV_PATH.exists() else RAW_CSV_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {target_path}. "
            "Ejecuta primero: python src/data/download_data.py"
        )

    logger.info(f"Cargando datos desde: {target_path.name}")

    # Intentar UTF-8 primero, luego Latin-1 (común en datos de entidades públicas en Colombia)
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(target_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"No fue posible decodificar el archivo en {target_path}")

    # Validación básica de estructura
    if TARGET_COLUMN not in df.columns:
        logger.warning(
            f"La columna objetivo '{TARGET_COLUMN}' no está explícita en las columnas. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    logger.info(f"Dataset cargado exitosamente: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
    return df
