"""
download_data.py — Ingesta, descarga reproducible y muestreo de vuelos comerciales EE.UU. (2025–2026).

Descarga los 12 archivos mensuales disponibles (junio 2025 a mayo 2026) desde Kaggle
(fuente primaria BTS / U.S. DOT) y toma, de cada mes, una fracción aleatoria de los
vuelos operados. El mes con más tráfico aporta más filas. Semilla fija.

Uso:
    python src/data/download_data.py --sample-fraction 0.01
    python src/data/download_data.py --download-all
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    ADMIN_COLS,
    KAGGLE_DATASET_SLUG,
    LEAKAGE_COLS,
    PRE_FLIGHT_FEATURES,
    STUDY_PERIOD_FILES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_DATA_PATH,
    SAMPLE_PARQUET_PATH,
    SAMPLE_FRACTION,
    RANDOM_STATE,
    TARGET_COLUMN,
)

# Columnas que el proyecto usa: itinerario, target, cancelación y las de fuga
# que el notebook muestra antes de descartarlas. El CSV crudo trae muchas más.
SAMPLE_COLUMNS = set(
    PRE_FLIGHT_FEATURES
    + LEAKAGE_COLS
    + ADMIN_COLS
    + [
        "FL_DATE",
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CANCELLED",
        "DIVERTED",
        TARGET_COLUMN,
        "ARR_DELAY",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
    ]
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
    """Descarga los archivos del periodo de estudio (junio 2025 a mayo 2026)."""
    api = ensure_kaggle_authenticated()
    logger.info(f"Descargando {len(STUDY_PERIOD_FILES)} meses del periodo (junio 2025 a mayo 2026)...")
    for fname in STUDY_PERIOD_FILES:
        download_monthly_file(fname, api=api, force=force)


def _completed_mask(chunk: pd.DataFrame) -> pd.Series:
    mask = chunk["ARR_DEL15"].notna()
    if "CANCELLED" in chunk.columns:
        mask &= chunk["CANCELLED"] == 0
    if "DIVERTED" in chunk.columns:
        mask &= chunk["DIVERTED"] == 0
    return mask


def _sample_month_file(fpath: Path, sample_fraction: float) -> tuple[pd.DataFrame, int, float]:
    """
    Dos lecturas por bloques. No carga el mes entero: primero cuenta los
    vuelos operados y luego se queda solo con las filas sorteadas.
    """
    reader_kwargs = dict(
        usecols=lambda col: col in SAMPLE_COLUMNS,
        chunksize=40_000,
        low_memory=True,
    )
    n_operados = 0
    delay_sum = 0.0
    for chunk in pd.read_csv(fpath, **reader_kwargs):
        done = chunk.loc[_completed_mask(chunk), "ARR_DEL15"]
        n_operados += int(len(done))
        delay_sum += float(done.sum())
        del chunk, done
    if n_operados == 0:
        raise ValueError(f"No hay vuelos operados en {fpath.name}")

    n_take = min(n_operados, max(1, int(round(n_operados * sample_fraction))))
    chosen = set(np.random.RandomState(RANDOM_STATE).choice(n_operados, size=n_take, replace=False).tolist())
    kept = []
    cursor = 0
    for chunk in pd.read_csv(fpath, **reader_kwargs):
        done = chunk.loc[_completed_mask(chunk)]
        if done.empty:
            continue
        positions = np.arange(cursor, cursor + len(done))
        pick = done.iloc[[i for i, pos in enumerate(positions) if pos in chosen]]
        if len(pick):
            kept.append(pick)
        cursor += len(done)
        del chunk, done, pick
    sample = pd.concat(kept, ignore_index=True)
    return sample, n_operados, delay_sum / n_operados


def build_stratified_sample_dataset(sample_fraction: float = SAMPLE_FRACTION) -> Path:
    """
    De cada mes toma una fracción aleatoria de los vuelos operados.
    Un mes con más vuelos aporta más filas. La semilla es fija.
    No estratifica por aerolínea: dentro del mes, cada vuelo operado
    tiene la misma probabilidad.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    available_files = [f for f in STUDY_PERIOD_FILES if (RAW_DATA_DIR / f).exists()]
    if not available_files:
        logger.warning("No hay archivos mensuales en data/raw. Descargando primero...")
        download_study_period_files()
        available_files = [f for f in STUDY_PERIOD_FILES if (RAW_DATA_DIR / f).exists()]

    logger.info(
        f"Muestra del {sample_fraction:.1%} de los vuelos operados de cada mes "
        f"({len(available_files)} archivos)..."
    )

    monthly_dfs = []
    coverage_rows = []

    for fname in available_files:
        fpath = RAW_DATA_DIR / fname
        logger.info(f"Procesando muestra de {fname}...")
        sample_month, n_operados, tasa_operados = _sample_month_file(fpath, sample_fraction)
        monthly_dfs.append(sample_month)
        coverage_rows.append(
            {
                "archivo": fname,
                "operados": n_operados,
                "muestra": len(sample_month),
                "fraccion": round(len(sample_month) / n_operados, 4) if n_operados else None,
                "tasa_retraso_operados": round(tasa_operados, 4),
                "tasa_retraso_muestra": round(float(sample_month["ARR_DEL15"].mean()), 4),
            }
        )
        logger.info(f"  -> {len(sample_month):,} de {n_operados:,} vuelos operados en {fname}")
        gc.collect()

    unified_df = pd.concat(monthly_dfs, ignore_index=True)
    logger.info(f"Dataset total unificado: {len(unified_df):,} filas × {len(unified_df.columns)} columnas.")

    unified_df.to_csv(SAMPLE_DATA_PATH, index=False)
    unified_df.to_parquet(SAMPLE_PARQUET_PATH, index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage_path = REPORTS_DIR / "muestra_vs_mes.csv"
    coverage.to_csv(coverage_path, index=False)
    logger.info(f"Comparación muestra vs mes guardada en {coverage_path}")
    logger.info(f"CSV: {SAMPLE_DATA_PATH}")
    logger.info(f"Parquet: {SAMPLE_PARQUET_PATH}")
    return SAMPLE_PARQUET_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestor de datos de vuelos US 2025-2026")
    parser.add_argument("--download-all", action="store_true", help="Descargar los 12 meses del periodo")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help="Fracción de vuelos operados que se toma de cada mes",
    )
    parser.add_argument("--force", action="store_true", help="Forzar re-descarga")
    args = parser.parse_args()

    if args.download_all:
        download_study_period_files(force=args.force)

    build_stratified_sample_dataset(sample_fraction=args.sample_fraction)
