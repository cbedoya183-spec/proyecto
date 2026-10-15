"""
pipeline.py — Definición canónica del Pipeline de Preprocesamiento y Modelado para Vuelos EE.UU.

Encapsula todo el preprocesamiento dentro de Scikit-Learn ColumnTransformer
para garantizar hermetismo absoluto y CERO Data Leakage entre entrenamiento y prueba.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import GroupMedianImputer, RareCategoryGrouper
from src.utils.logger import get_logger

logger = get_logger("pipeline")

# ==============================================================================
# CARACTERÍSTICAS PREDICTORAS PRE-VUELO
# ==============================================================================
DEFAULT_NUMERICAL_FEATURES = [
    "CRS_DEP_HOUR",
    "CRS_ARR_HOUR",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "DAY_OF_WEEK",
    "MONTH",
    "IS_WEEKEND",
    "ORIGIN_IS_HUB",
    "DEST_IS_HUB",
    "IS_HUB_TO_HUB",
    "CRS_DEP_SIN",
    "CRS_DEP_COS",
    "MONTH_SIN",
    "MONTH_COS",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "DEP_TIME_BLOCK",
    "DISTANCE_CAT",
    "SEASON",
]


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    top_categories: int = 25,
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer hermético para preprocesamiento de vuelos.

    Sub-pipelines:
    - Numérico: Imputación por mediana -> Estandarización (StandardScaler).
    - Categórico: Imputación por constante ('DESCONOCIDO') ->
                 Agrupación de aeropuertos top (RareCategoryGrouper) ->
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
            ("grouper", RareCategoryGrouper(top_n=top_categories, fill_value="OTRO")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
    )

    logger.info(
        f"ColumnTransformer configurado: {len(numerical_cols)} variables numéricas "
        f"y {len(categorical_cols)} variables categóricas."
    )
    return preprocessor


class ToDense(BaseEstimator, TransformerMixin):
    """HistGradientBoosting no acepta la matriz dispersa del one-hot."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


def build_full_pipeline(
    model: BaseEstimator,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Ensambla el preprocesador hermético con un estimador Scikit-Learn en un solo Pipeline.
    """
    preprocessor = build_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )

    steps = [
        ("duracion_por_aerolinea", GroupMedianImputer()),
        ("preprocessor", preprocessor),
    ]
    if isinstance(model, HistGradientBoostingClassifier):
        steps.append(("denso", ToDense()))
    steps.append(("model", model))
    return Pipeline(steps=steps)
