"""
download_data.py — Ingesta, descarga reproducible y muestreo de vuelos comerciales EE.UU. (2025–2026).

Descarga los 13 archivos mensuales oficiales (Mayo 2025 a Mayo 2026) desde Kaggle
(fuente primaria BTS / U.S. DOT) y genera un subconjunto representativo y reproducible
estratificado por mes y aerolínea para permitir ejecución ágil y evaluación rigurosa.

Uso:
    python src/data/download_data.py --sample-per-month 5000
    python src/data/download_data.py --download-all
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import (
    KAGGLE_DATASET_SLUG,
    STUDY_PERIOD_FILES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_PATH,
    SAMPLE_PARQUET_PATH,
    RANDOM_STATE,
)
from src.utils.logger import get_logger

logger = get_logger("download_data")


def ensure_kaggle_authenticated():
    """Verifica y asegura la autenticación con la API de Kaggle."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    token_file = Path.home() / ".kaggle" / "access_token"
    if not token and token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_monthly_file(filename: str, api=None, force: bool = False) -> Path:
    """Descarga un archivo mensual específico de Kaggle si no existe localmente."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DATA_DIR / filename

    if target_path.exists() and not force:
        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(f"Archivo {filename} ya existe ({size_mb:.2f} MB). Saltando descarga.")
        return target_path

    if api is None:
        api = ensure_kaggle_authenticated()

    logger.info(f"Descargando {filename} desde Kaggle ({KAGGLE_DATASET_SLUG})...")
    api.dataset_download_file(KAGGLE_DATASET_SLUG, filename, path=str(RAW_DATA_DIR))

    # Si se descarga como .zip, descomprimir
    zip_path = RAW_DATA_DIR / f"{filename}.zip"
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DATA_DIR)
        zip_path.unlink()

    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(f"Descarga exitosa de {filename} ({size_mb:.2f} MB).")
    else:
        logger.warning(f"No se encontró el archivo extraído {target_path}")

    return target_path


def download_study_period_files(force: bool = False):
    """Descarga todos los archivos correspondientes a los 13 meses del estudio."""
    api = ensure_kaggle_authenticated()
    logger.info(f"Descargando {len(STUDY_PERIOD_FILES)} meses del periodo (Mayo 2025 a Mayo 2026)...")
    for fname in STUDY_PERIOD_FILES:
        download_monthly_file(fname, api=api, force=force)


def build_stratified_sample_dataset(sample_per_month: int = 5000) -> Path:
    """
    Crea un dataset unificado representativo y estratificado por mes y aerolínea
    a partir de los archivos mensuales disponibles en data/raw.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    available_files = [f for f in STUDY_PERIOD_FILES if (RAW_DATA_DIR / f).exists()]
    if not available_files:
        logger.warning("No hay archivos mensuales en data/raw. Descargando primero...")
        download_study_period_files()
        available_files = [f for f in STUDY_PERIOD_FILES if (RAW_DATA_DIR / f).exists()]

    logger.info(
        f"Generando dataset reproducible con {sample_per_month} registros por mes "
        f"a través de {len(available_files)} meses disponibles..."
    )

    monthly_dfs = []
    rng = np.random.RandomState(RANDOM_STATE)

    for fname in available_files:
        fpath = RAW_DATA_DIR / fname
        logger.info(f"Procesando muestra de {fname}...")
        
        # Leer columnas relevantes directamente para ahorrar memoria
        df_month = pd.read_csv(fpath, low_memory=False)

        # Filtrar vuelos completados (excluir cancelados y desviados para puntualidad CRS)
        if "CANCELLED" in df_month.columns and "DIVERTED" in df_month.columns:
            completed = df_month[(df_month["CANCELLED"] == 0) & (df_month["DIVERTED"] == 0)].copy()
        else:
            completed = df_month.copy()

        # Filtrar registros con target válido
        completed = completed.dropna(subset=["ARR_DEL15"])

        # Muestreo aleatorio uniforme con semilla fija por mes
        n_take = min(len(completed), sample_per_month)
        sample_month = completed.sample(n=n_take, random_state=RANDOM_STATE) if n_take < len(completed) else completed

        monthly_dfs.append(sample_month)
        logger.info(f"  -> {len(sample_month)} registros seleccionados de {fname} (Aerolíneas: {sample_month['OP_UNIQUE_CARRIER'].nunique()})")

    unified_df = pd.concat(monthly_dfs, ignore_index=True)
    logger.info(f"Dataset total unificado: {len(unified_df):,} filas × {len(unified_df.columns)} columnas.")

    # Guardar en CSV y Parquet para máxima interoperabilidad y velocidad
    unified_df.to_csv(SAMPLE_DATA_PATH, index=False)
    unified_df.to_parquet(SAMPLE_PARQUET_PATH, index=False)
    logger.info(f"Dataset guardado en:")
    logger.info(f"  - CSV:     {SAMPLE_DATA_PATH}")
    logger.info(f"  - Parquet: {SAMPLE_PARQUET_PATH}")

    return SAMPLE_PARQUET_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestor de datos de vuelos US 2025-2026")
    parser.add_argument("--download-all", action="store_true", help="Descargar los 13 meses completos")
    parser.add_argument("--sample-per-month", type=int, default=5000, help="Registros por mes para la muestra")
    parser.add_argument("--force", action="store_true", help="Forzar re-descarga")
    args = parser.parse_args()

    if args.download_all:
        download_study_period_files(force=args.force)

    build_stratified_sample_dataset(sample_per_month=args.sample_per_month)
