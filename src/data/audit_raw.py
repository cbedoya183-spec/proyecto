"""
audit_raw.py — Auditoría de cobertura y calidad sobre los CSV crudos de BTS.

Recorre los meses del periodo sin cargar el dataset completo en una sola tabla.
Guarda un resumen en reports/ con cancelados, desviados, nulos y duplicados.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import RAW_DATA_DIR, REPORTS_DIR, STUDY_PERIOD_FILES
from src.utils.logger import get_logger

logger = get_logger("audit_raw")

AUDIT_PATH = REPORTS_DIR / "auditoria_mensual_cruda.csv"

USECOLS = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CANCELLED",
    "DIVERTED",
    "ARR_DEL15",
    "ARR_DELAY",
    "CRS_DEP_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]


def _audit_file(path: Path) -> dict:
    rows = 0
    cancelled = 0
    diverted = 0
    null_target = 0
    operated = 0
    delayed = 0
    null_crs = 0
    null_elapsed = 0
    null_distance = 0
    dup_keys = 0
    date_min = None
    date_max = None
    seen_keys = set()

    reader = pd.read_csv(path, usecols=lambda c: c in USECOLS, chunksize=250_000, low_memory=False)
    for chunk in reader:
        rows += len(chunk)
        if "CANCELLED" in chunk.columns:
            cancelled += int((chunk["CANCELLED"] == 1).sum())
        if "DIVERTED" in chunk.columns:
            diverted += int((chunk["DIVERTED"] == 1).sum())
        if "ARR_DEL15" in chunk.columns:
            null_target += int(chunk["ARR_DEL15"].isna().sum())
            operated_mask = chunk["ARR_DEL15"].notna()
            operated += int(operated_mask.sum())
            delayed += int(chunk.loc[operated_mask, "ARR_DEL15"].sum())
        if "CRS_DEP_TIME" in chunk.columns:
            null_crs += int(chunk["CRS_DEP_TIME"].isna().sum())
        if "CRS_ELAPSED_TIME" in chunk.columns:
            null_elapsed += int(chunk["CRS_ELAPSED_TIME"].isna().sum())
        if "DISTANCE" in chunk.columns:
            null_distance += int(chunk["DISTANCE"].isna().sum())
        if "FL_DATE" in chunk.columns:
            parsed = pd.to_datetime(chunk["FL_DATE"], errors="coerce")
            if parsed.notna().any():
                local_min = parsed.min()
                local_max = parsed.max()
                date_min = local_min if date_min is None else min(date_min, local_min)
                date_max = local_max if date_max is None else max(date_max, local_max)
        if {"FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN"}.issubset(chunk.columns):
            keys = (
                chunk["FL_DATE"].astype(str)
                + "|"
                + chunk["OP_UNIQUE_CARRIER"].astype(str)
                + "|"
                + chunk["OP_CARRIER_FL_NUM"].astype(str)
                + "|"
                + chunk["ORIGIN"].astype(str)
            )
            before = len(seen_keys)
            seen_keys.update(keys.tolist())
            dup_keys += len(keys) - (len(seen_keys) - before)

    delay_rate = (delayed / operated) if operated else float("nan")
    return {
        "archivo": path.name,
        "filas": rows,
        "cancelados": cancelled,
        "desviados": diverted,
        "nulos_target": null_target,
        "operados_con_target": operated,
        "tasa_retraso_operados": round(float(delay_rate), 4) if pd.notna(delay_rate) else None,
        "nulos_crs_dep_time": null_crs,
        "nulos_crs_elapsed": null_elapsed,
        "nulos_distance": null_distance,
        "duplicados_clave_vuelo": dup_keys,
        "fecha_min": None if date_min is None else str(pd.Timestamp(date_min).date()),
        "fecha_max": None if date_max is None else str(pd.Timestamp(date_max).date()),
    }


def audit_study_period(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Audita cada CSV mensual del periodo y guarda el resumen."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in STUDY_PERIOD_FILES:
        path = raw_dir / fname
        if not path.exists():
            logger.warning(f"No está en disco: {fname}")
            rows.append({"archivo": fname, "filas": 0, "nota": "ausente"})
            continue
        logger.info(f"Auditando {fname}...")
        rows.append(_audit_file(path))
    report = pd.DataFrame(rows)
    report.to_csv(AUDIT_PATH, index=False)
    logger.info(f"Auditoría guardada en {AUDIT_PATH}")
    return report


if __name__ == "__main__":
    df = audit_study_period()
    print(df.to_string(index=False))
