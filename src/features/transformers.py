"""
transformers.py — Transformadores personalizados compatibles con Scikit-Learn.

Heredan de BaseEstimator y TransformerMixin para permitir su integración
directa dentro de Scikit-Learn Pipelines y ColumnTransformers, garantizando
que todo cálculo (frecuencias, categorías top) se aprenda estrictamente sobre X_train
sin generar Data Leakage.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger("transformers")


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías de baja frecuencia en una etiqueta 'OTRO' para controlar
    la alta cardinalidad en aeropuertos de origen y destino (IATA codes).

    Aprende el conjunto de categorías frecuentes exclusivamente en fit(X_train)
    y lo aplica en transform(), previniendo sobreajuste y desbordamiento dimensional.
    """

    def __init__(self, top_n: int = 25, fill_value: str = "OTRO"):
        self.top_n = top_n
        self.fill_value = fill_value
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.frequent_categories_ = {}
        for col in X_df.columns:
            top_cats = set(X_df[col].value_counts().nlargest(self.top_n).index)
            self.frequent_categories_[col] = top_cats
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            frequent = self.frequent_categories_.get(col, set())
            X_df[col] = X_df[col].apply(lambda val: val if val in frequent else self.fill_value)
        return X_df.values


class CyclicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma variables cíclicas periódicas (ej. hora del día [0-23] o mes [1-12])
    en sus componentes ortogonales seno y coseno para modelos lineales o de distancia.
    """

    def __init__(self, period: float = 24.0):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float)
        sin_comp = np.sin(2 * np.pi * X_arr / self.period)
        cos_comp = np.cos(2 * np.pi * X_arr / self.period)
        return np.hstack([sin_comp, cos_comp])
