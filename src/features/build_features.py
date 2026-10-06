"""
build_features.py — Ingeniería de características para siniestros viales en Medellín.

Contiene funciones modulares para:
1. Normalización del target de negocio ('es_grave': 0 vs 1).
2. Extracción de características temporales (hora, día de la semana, fin de semana, franja horaria).
3. Transformaciones cíclicas de tiempo (seno/coseno de hora).
4. Normalización de características espaciales y de infraestructura vial (comuna, clase de accidente, diseño).
"""

import sys
from pathlib import Path

# Garantizar raíz en sys.path para importación
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import numpy as np
import pandas as pd

from src.config import (
    TARGET_COLUMN,
    BINARY_TARGET_COLUMN,
    LEAKAGE_COLS,
)
from src.utils.logger import get_logger

logger = get_logger("build_features")


def clean_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el target 'GRAVEDAD_ACCIDENTE' y crea la variable binaria 'es_grave'.

    1 = Severo / Crítico: 'Con heridos' o 'Con muertos'
    0 = No severo / Solo daños materiales: 'Solo daños'

    Elimina registros con target nulo si existieran.
    """
    df = df.copy()

    # Eliminar nulos en el target si los hay
    initial_rows = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    dropped = initial_rows - len(df)
    if dropped > 0:
        logger.warning(f"Se eliminaron {dropped} filas con target nulo.")

    target_str = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Detección robusta ante acentos o variaciones de codificación
    df[BINARY_TARGET_COLUMN] = np.where(
        target_str.str.contains("herid|muert", regex=True),
        1,
        0,
    )

    logger.info(
        f"Target '{BINARY_TARGET_COLUMN}' generado: "
        f"{df[BINARY_TARGET_COLUMN].sum():,} casos graves (1) vs "
        f"{(df[BINARY_TARGET_COLUMN] == 0).sum():,} casos solo daños (0)."
    )
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parsea la columna de fecha y extrae variables temporales de alto valor predictivo.
    """
    df = df.copy()

    # Conversión a datetime
    dt_series = pd.to_datetime(
        df["FECHA_ACCIDENTE"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df["hora"] = dt_series.dt.hour
    df["dia_semana"] = dt_series.dt.dayofweek  # 0: Lunes, 6: Domingo
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

    # Franja horaria para análisis de negocio y turnos de emergencia
    def categorizar_franja(hora):
        if pd.isna(hora):
            return "Desconocida"
        hora = int(hora)
        if 0 <= hora < 6:
            return "Madrugada (00-05)"
        elif 6 <= hora < 9:
            return "Pico Mañana (06-08)"
        elif 9 <= hora < 12:
            return "Mañana (09-11)"
        elif 12 <= hora < 15:
            return "Mediodía (12-14)"
        elif 15 <= hora < 18:
            return "Tarde (15-17)"
        elif 18 <= hora < 21:
            return "Pico Noche (18-20)"
        else:
            return "Noche (21-23)"

    df["franja_horaria"] = df["hora"].apply(categorizar_franja)

    # Componentes cíclicas de la hora (seno y coseno) para modelos lineales y distancias
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24.0)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24.0)

    # Componentes cíclicas del mes
    mes_val = df["MES"].fillna(dt_series.dt.month).fillna(1).astype(int)
    df["mes_sin"] = np.sin(2 * np.pi * mes_val / 12.0)
    df["mes_cos"] = np.cos(2 * np.pi * mes_val / 12.0)

    return df


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza categorías en COMUNA, CLASE_ACCIDENTE y DISENO,
    agrupando etiquetas de baja frecuencia y normalizando textos.
    """
    df = df.copy()

    # Normalización de COMUNA
    if "COMUNA" in df.columns:
        comunas_validas = [
            "La Candelaria", "Laureles Estadio", "Castilla", "El Poblado",
            "Guayabal", "Robledo", "Belén", "Aranjuez", "Buenos Aires",
            "San Javier", "Villa Hermosa", "Doce de Octubre", "Manrique",
            "Popular", "Santa Cruz", "San Antonio de Prado", "San Cristóbal",
            "Santa Elena", "San Sebastián de Palmitas", "Altavista",
        ]
        
        def normalizar_comuna(c):
            if pd.isna(c):
                return "DESCONOCIDA"
            c = str(c).strip()
            # Unificar Belén
            if "bel" in c.lower():
                return "Belén"
            for cv in comunas_validas:
                if cv.lower() in c.lower():
                    return cv
            return "OTRA_COMUNA"

        df["comuna_limpia"] = df["COMUNA"].apply(normalizar_comuna)

    # Normalización de CLASE_ACCIDENTE
    if "CLASE_ACCIDENTE" in df.columns:
        def normalizar_clase(cl):
            if pd.isna(cl):
                return "Otro"
            cl = str(cl).strip().lower()
            if "choque" in cl:
                return "Choque"
            elif "atropello" in cl:
                return "Atropello"
            elif "caida" in cl or "caída" in cl:
                return "Caída de Ocupante"
            elif "volcamiento" in cl:
                return "Volcamiento"
            elif "incendio" in cl:
                return "Incendio"
            else:
                return "Otro"

        df["clase_accidente_limpia"] = df["CLASE_ACCIDENTE"].apply(normalizar_clase)

    # Normalización de DISENO (geometría de vía)
    if "DISENO" in df.columns:
        def normalizar_diseno(d):
            if pd.isna(d):
                return "Tramo de via"
            d = str(d).strip().lower()
            if "tramo" in d:
                return "Tramo de via"
            elif "intersecci" in d:
                return "Intersección"
            elif "glorieta" in d:
                return "Glorieta"
            elif "puente" in d:
                return "Puente"
            elif "paso a nivel" in d:
                return "Paso a Nivel"
            elif "lote" in d:
                return "Lote o Predio"
            elif "tunel" in d or "túnel" in d:
                return "Túnel"
            else:
                return "Otro"

        df["diseno_limpio"] = df["DISENO"].apply(normalizar_diseno)

    # Extracción de tipo de vía a partir de DIRECCION (CR, CL, AV, TV, DG, etc.)
    if "DIRECCION" in df.columns:
        def extraer_tipo_via(dir_str):
            if pd.isna(dir_str):
                return "DESCONOCIDA"
            dir_str = str(dir_str).strip().upper()
            match = re.match(r"^(CR|CL|AV|TV|DG|CQ|AU)\b", dir_str)
            if match:
                return match.group(1)
            elif "AUTOPISTA" in dir_str:
                return "AU"
            elif "AVENIDA" in dir_str or "AV " in dir_str:
                return "AV"
            return "OTRO"

        df["tipo_via"] = df["DIRECCION"].apply(extraer_tipo_via)

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina identificadores administrativos posteriores al hecho (Data Leakage).
    """
    df = df.copy()
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        logger.info(f"Eliminando columnas con fuga de datos: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def prepare_full_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el flujo completo de ingeniería de características al dataframe crudo.
    """
    logger.info("Iniciando preparación e ingeniería de características...")
    df = clean_target_column(df_raw)
    df = extract_temporal_features(df)
    df = clean_categorical_features(df)
    df = drop_leakage_columns(df)
    logger.info(f"Dataset preparado: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
    return df
