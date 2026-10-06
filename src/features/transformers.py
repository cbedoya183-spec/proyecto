"""
transformers.py — Transformadores personalizados compatibles con Scikit-Learn.

Heredan de BaseEstimator y TransformerMixin para permitir su integración
directa dentro de Scikit-Learn Pipelines y ColumnTransformers, garantizando
que todo cálculo (frecuencias, medianas) se aprenda estrictamente sobre X_train
sin generar Data Leakage.
"""

import sys
from pathlib import Path

# Garantizar raíz en sys.path
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
    Agrupa categorías de baja frecuencia en una etiqueta 'OTRO' o valor especificado.
    
    Aprende el conjunto de categorías frecuentes exclusivamente en fit(X_train)
    y lo aplica en transform(), previniendo sobreajuste y problemas con categorías
    nuevas o poco representadas en el conjunto de prueba.
    """

    def __init__(self, min_freq: float = 0.01, fill_value: str = "OTRO"):
        self.min_freq = min_freq
        self.fill_value = fill_value
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.frequent_categories_ = {}
        for col in X_df.columns:
            freqs = X_df[col].value_counts(normalize=True)
            frequent = set(freqs[freqs >= self.min_freq].index)
            self.frequent_categories_[col] = frequent
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
    en sus componentes ortogonales seno y coseno.
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


class SpatialMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputa coordenadas geográficas (X, Y) utilizando la mediana aprendida
    por comuna sobre el conjunto de entrenamiento.
    """

    def __init__(self, comuna_col_idx: int = 0, coord_col_indices=(1, 2)):
        self.comuna_col_idx = comuna_col_idx
        self.coord_col_indices = coord_col_indices
        self.medians_per_comuna_ = {}
        self.global_medians_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        comuna_series = X_df.iloc[:, self.comuna_col_idx]

        self.medians_per_comuna_ = {}
        self.global_medians_ = {}

        for c_idx in self.coord_col_indices:
            coord_series = pd.to_numeric(X_df.iloc[:, c_idx], errors="coerce")
            self.global_medians_[c_idx] = coord_series.median()
            medians = coord_series.groupby(comuna_series).median()
            self.medians_per_comuna_[c_idx] = medians.to_dict()

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        comuna_series = X_df.iloc[:, self.comuna_col_idx]

        for c_idx in self.coord_col_indices:
            coord_series = pd.to_numeric(X_df.iloc[:, c_idx], errors="coerce")
            comuna_medians = self.medians_per_comuna_.get(c_idx, {})
            global_med = self.global_medians_.get(c_idx, 0.0)

            # Rellenar primero con la mediana de la comuna correspondiente
            filled = coord_series.fillna(comuna_series.map(comuna_medians))
            # Si aún hay nulos (ej. comuna desconocida), usar la mediana global de entrenamiento
            filled = filled.fillna(global_med)
            X_df.iloc[:, c_idx] = filled

        return X_df.values
