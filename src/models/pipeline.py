"""
pipeline.py — Definición canónica del Pipeline de Preprocesamiento y Modelado.

Encapsula todo el preprocesamiento dentro de Scikit-Learn ColumnTransformer
para garantizar hermetismo absoluto y CERO Data Leakage entre entrenamiento y prueba.
"""

import sys
from pathlib import Path

# Garantizar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import RareCategoryGrouper
from src.utils.logger import get_logger

logger = get_logger("pipeline")

# ==============================================================================
# CARACTERÍSTICAS PREDICTORAS CANÓNICAS
# ==============================================================================
DEFAULT_NUMERICAL_FEATURES = [
    "hora_sin",
    "hora_cos",
    "dia_semana",
    "es_fin_de_semana",
    "mes_sin",
    "mes_cos",
    "X",
    "Y",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "clase_accidente_limpia",
    "diseno_limpio",
    "comuna_limpia",
    "tipo_via",
    "franja_horaria",
]


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    min_category_freq: float = 0.01,
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer hermético para preprocesamiento.

    Sub-pipelines:
    - Numérico: Imputación por mediana -> Estandarización (StandardScaler).
    - Categórico: Imputación por constante ('DESCONOCIDO') ->
                 Agrupación de categorías raras (< 1%) ->
                 OneHotEncoder (handle_unknown='ignore').
    """
    if numerical_cols is None:
        numerical_cols = DEFAULT_NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
            ("grouper", RareCategoryGrouper(min_freq=min_category_freq, fill_value="OTRO")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info(
        f"ColumnTransformer configurado: {len(numerical_cols)} variables numéricas "
        f"y {len(categorical_cols)} variables categóricas."
    )
    return preprocessor


def build_full_pipeline(
    classifier: BaseEstimator,
    preprocessor: Optional[ColumnTransformer] = None,
) -> Pipeline:
    """
    Ensambla el preprocesador hermético con cualquier clasificador de Scikit-Learn.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return full_pipeline


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Extrae los nombres finales de las columnas post-transformación
    para facilitar la interpretabilidad de coeficientes y feature importance.
    """
    if numerical_cols is None:
        numerical_cols = DEFAULT_NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_FEATURES

    feature_names = list(numerical_cols)

    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(encoded_cat_names)
    except Exception:
        # Fallback genérico si no ha sido ajustado
        pass

    return feature_names
