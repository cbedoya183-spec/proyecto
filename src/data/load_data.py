"""
load_data.py — Funciones estandarizadas de carga y validación de datos para vuelos EE.UU.

Garantiza que cualquier módulo o notebook cargue los datos de manera consistente,
con soporte de Parquet ultrarrápido y CSV de respaldo para el periodo 2025–2026.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    SAMPLE_PARQUET_PATH,
    SAMPLE_DATA_PATH,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger("load_data")


def load_flight_data(use_parquet: bool = True) -> pd.DataFrame:
    """
    Carga el dataset de vuelos procesado y reproducible.

    Prioridad:
    1. Parquet en data/processed/ (carga en < 1 segundo).
    2. CSV en data/processed/.
    3. Si no existe, invoca la generación de muestra desde data/raw/.
    """
    if use_parquet and SAMPLE_PARQUET_PATH.exists():
        logger.info(f"Cargando dataset desde Parquet: {SAMPLE_PARQUET_PATH.name}")
        df = pd.read_parquet(SAMPLE_PARQUET_PATH)
    elif SAMPLE_DATA_PATH.exists():
        logger.info(f"Cargando dataset desde CSV: {SAMPLE_DATA_PATH.name}")
        df = pd.read_csv(SAMPLE_DATA_PATH, low_memory=False)
    else:
        logger.warning("No se encontró el dataset procesado. Generando automáticamente...")
        from src.data.download_data import build_stratified_sample_dataset
        build_stratified_sample_dataset()
        df = pd.read_parquet(SAMPLE_PARQUET_PATH)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        logger.info(f"Se descartaron columnas vacías: {empty_cols}")

    if TARGET_COLUMN not in df.columns:
        logger.warning(f"La columna objetivo '{TARGET_COLUMN}' no está presente en los datos.")

    logger.info(f"Dataset de vuelos cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas.")
    return df
