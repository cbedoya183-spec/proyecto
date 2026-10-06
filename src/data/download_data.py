"""
download_data.py — Ingesta y descarga reproducible de datos abiertos de Medellín.

Descarga el conjunto oficial de Incidentes Viales de la Secretaría de Movilidad
de Medellín (MEData) y genera opcionalmente un subconjunto representativo
estratificado/aleatorio para agilizar el desarrollo y pruebas del equipo.

Uso:
    python src/data/download_data.py
    python src/data/download_data.py --sample-size 50000
    python src/data/download_data.py --force
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para ejecución directa
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    MEDATA_CSV_URL,
    RAW_CSV_PATH,
    SAMPLE_CSV_PATH,
    RAW_DATA_DIR,
    RANDOM_STATE,
)
from src.utils.logger import get_logger

logger = get_logger("download_data")


def download_medata_dataset(force: bool = False) -> Path:
    """
    Descarga el archivo CSV oficial de MEData si no existe localmente.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV_PATH.exists() and not force:
        file_size_mb = RAW_CSV_PATH.stat().st_size / (1024 * 1024)
        logger.info(
            f"El archivo ya existe localmente en {RAW_CSV_PATH} ({file_size_mb:.2f} MB). "
            "Usa --force para descargar nuevamente."
        )
        return RAW_CSV_PATH

    logger.info(f"Descargando dataset oficial desde: {MEDATA_CSV_URL}")
    logger.info(f"Destino: {RAW_CSV_PATH}")

    # Descarga con reporte de progreso
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            downloaded_mb = (block_num * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\rProgreso: {percent}% [{downloaded_mb:.1f} MB / {total_mb:.1f} MB]"
            )
            sys.stdout.flush()

    try:
        req = urllib.request.Request(
            MEDATA_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as response, open(RAW_CSV_PATH, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 64
            downloaded = 0
            block_num = 0
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                block_num += 1
                reporthook(block_num, block_size, total_size)
        print()  # Salto de línea
        file_size_mb = RAW_CSV_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Descarga completada con éxito ({file_size_mb:.2f} MB).")
        return RAW_CSV_PATH
    except Exception as e:
        logger.error(f"Error al descargar dataset: {e}")
        if RAW_CSV_PATH.exists():
            RAW_CSV_PATH.unlink()
        raise


def create_sample_dataset(sample_size: int = 50000) -> Path:
    """
    Crea una muestra representativa con semilla fija para acelerar
    el desarrollo y pruebas en el equipo de trabajo.
    """
    if not RAW_CSV_PATH.exists():
        download_medata_dataset()

    logger.info(f"Leyendo dataset completo para extraer muestra de {sample_size:,} registros...")
    
    # Detección de encoding compatible (latin-1 o utf-8)
    for encoding in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(RAW_CSV_PATH, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    total_rows = len(df)
    logger.info(f"Dataset total contiene {total_rows:,} filas x {df.shape[1]} columnas.")

    if total_rows > sample_size:
        df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
        logger.info(f"Muestra aleatoria generada con éxito ({sample_size:,} filas).")
    else:
        df_sample = df
        logger.warning("El dataset completo tiene menos registros que el tamaño de muestra solicitado.")

    df_sample.to_csv(SAMPLE_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"Muestra guardada en {SAMPLE_CSV_PATH}")
    return SAMPLE_CSV_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Descarga el dataset oficial de incidentes viales de Medellín."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar la descarga incluso si el archivo ya existe.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Número de filas para el dataset de muestra de desarrollo (default: 50000). Usa 0 para omitir.",
    )

    args = parser.parse_args()

    download_medata_dataset(force=args.force)

    if args.sample_size > 0:
        create_sample_dataset(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
