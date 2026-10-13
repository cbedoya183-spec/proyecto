"""
build_features.py — Ingeniería de características para predicción de retrasos en vuelos comerciales (EE.UU.).

Contiene funciones modulares para:
1. Normalización del target de negocio ('ARR_DEL15': 0 vs 1).
2. Extracción de variables temporales y de calendario (día de semana, mes, estación, fin de semana).
3. Descomposición y fraccionamiento de horarios programados CRS (hora salida/llegada, bloque horario).
4. Categorización y variables de red de aviación (distancia, hubs congestionados, aerolíneas).
5. Aislamiento y exclusión estricta de variables de Data Leakage.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COLUMN,
    CONTINUOUS_TARGET_COLUMN,
    LEAKAGE_COLS,
    ADMIN_COLS,
)
from src.utils.logger import get_logger

logger = get_logger("build_features")

# Top 10 aeropuertos con mayor volumen de tráfico y congestión en EE.UU. (FAA/BTS)
TOP_HUBS = {"ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "JFK", "MIA", "MCO", "EWR"}


def clean_flight_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra vuelos completados (no cancelados ni desviados) y asegura la integridad
    de la variable objetivo binaria 'ARR_DEL15'.
    """
    df = df.copy()
    initial_rows = len(df)

    # Excluir vuelos cancelados o desviados
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]
    if "DIVERTED" in df.columns:
        df = df[df["DIVERTED"] == 0]

    # Eliminar nulos en el target si existieran
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    dropped = initial_rows - len(df)
    if dropped > 0:
        logger.info(f"Se descartaron {dropped} vuelos cancelados, desviados o con target ausente.")

    delayed_count = df[TARGET_COLUMN].sum()
    ontime_count = len(df) - delayed_count
    delay_rate = (delayed_count / len(df)) * 100
    logger.info(
        f"Target '{TARGET_COLUMN}' preparado: {delayed_count:,} retrasados ({delay_rate:.2f}%) vs "
        f"{ontime_count:,} puntuales ({100-delay_rate:.2f}%)."
    )
    return df


def extract_flight_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae componentes temporales pre-vuelo a partir de 'FL_DATE' y horarios programados CRS.
    """
    df = df.copy()

    # Conversión de FL_DATE
    if "FL_DATE" in df.columns:
        dt = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df["DAY_OF_WEEK"] = dt.dt.dayofweek  # 0: Lunes, 6: Domingo
        df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)
    else:
        if "DAY_OF_WEEK" not in df.columns and "DAY_OF_MONTH" in df.columns:
            df["DAY_OF_WEEK"] = df["DAY_OF_MONTH"] % 7
            df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)

    # Estación climática (afecta clima invernal y tormentas de verano)
    # 1: Invierno (Dic, Ene, Feb), 2: Primavera (Mar, Abr, May), 3: Verano (Jun, Jul, Ago), 4: Otoño (Sep, Oct, Nov)
    def asignar_estacion(mes):
        if mes in [12, 1, 2]:
            return "Invierno"
        elif mes in [3, 4, 5]:
            return "Primavera"
        elif mes in [6, 7, 8]:
            return "Verano"
        else:
            return "Otono"

    df["SEASON"] = df["MONTH"].apply(asignar_estacion)

    # Extraer horas enteras de los horarios programados en formato HHMM (CRS_DEP_TIME y CRS_ARR_TIME)
    def hhmm_to_hour(val):
        if pd.isna(val):
            return 12  # Mediodía como imputación razonable
        try:
            val_int = int(val)
            return min(23, max(0, val_int // 100))
        except (ValueError, TypeError):
            return 12

    df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"].apply(hhmm_to_hour)
    df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"].apply(hhmm_to_hour)

    # Franja horaria de salida (El efecto cascada de demoras crece en la tarde/noche)
    def categorizar_bloque_salida(hora):
        if 0 <= hora < 6:
            return "Madrugada_00_05"
        elif 6 <= hora < 12:
            return "Manana_06_11"
        elif 12 <= hora < 18:
            return "Tarde_12_17"
        else:
            return "Noche_18_23"

    df["DEP_TIME_BLOCK"] = df["CRS_DEP_HOUR"].apply(categorizar_bloque_salida)

    return df


def extract_route_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera características de red de transporte aéreo: distancia categorizada
    e indicadores de congestión en aeropuertos principales (Hubs).
    """
    df = df.copy()

    # Indicador de si el origen o destino es un Hub de alto tráfico
    df["ORIGIN_IS_HUB"] = df["ORIGIN"].isin(TOP_HUBS).astype(int)
    df["DEST_IS_HUB"] = df["DEST"].isin(TOP_HUBS).astype(int)
    df["IS_HUB_TO_HUB"] = ((df["ORIGIN_IS_HUB"] == 1) & (df["DEST_IS_HUB"] == 1)).astype(int)

    # Categorización de distancia de vuelo
    # Corta: < 500 millas, Media: 500-1500 millas, Larga: > 1500 millas
    def categorizar_distancia(dist):
        if pd.isna(dist):
            return "Media"
        if dist < 500:
            return "Corta"
        elif dist <= 1500:
            return "Media"
        else:
            return "Larga"

    df["DISTANCE_CAT"] = df["DISTANCE"].apply(categorizar_distancia)

    return df


def isolate_preflight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aísla las columnas predictoras disponibles en pre-vuelo y elimina
    de forma garantizada cualquier variable de fuga de información (data leakage).
    """
    df = df.copy()

    # Columnas prohibidas que deben removerse
    cols_to_drop = [c for c in (LEAKAGE_COLS + ADMIN_COLS) if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    logger.info(f"Columnas tras eliminación estricta de leakage: {len(df.columns)}")
    return df


def build_full_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ingeniería de características sobre el DataFrame.
    """
    logger.info(f"Iniciando ingeniería de características sobre {len(df):,} registros...")
    df_clean = clean_flight_target(df)
    df_feat = extract_flight_temporal_features(df_clean)
    df_feat = extract_route_network_features(df_feat)
    df_final = isolate_preflight_features(df_feat)
    logger.info("Ingeniería de características finalizada con éxito.")
    return df_final
