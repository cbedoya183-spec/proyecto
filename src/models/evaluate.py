"""
evaluate.py — Métricas de clasificación para retraso de llegada.

El positivo es el vuelo retrasado (ARR_DEL15 = 1). La exactitud se reporta,
pero no decide el modelo: con cerca del 77 % de vuelos puntuales, predecir
siempre «puntual» ya obtiene esa exactitud y no encuentra ningún retraso.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger("evaluate")

THRESHOLD_GRID = np.round(np.arange(0.15, 0.66, 0.05), 2)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
) -> Dict[str, float]:
    """Calcula exactitud, precisión, recall y F1 del retraso, más ROC-AUC y PR-AUC."""
    metrics = {
        "modelo": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_retraso": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_retraso": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_retraso": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    if y_proba is not None:
        try:
            proba_pos = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, proba_pos)), 4)
        except Exception as exc:
            logger.warning(f"No fue posible calcular ROC/PR AUC para {model_name}: {exc}")
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def best_f1_threshold(y_true, y_proba, grid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Elige el umbral que maximiza el F1 del retraso.
    Debe llamarse solo con predicciones de validación, nunca con la prueba.
    """
    if grid is None:
        grid = THRESHOLD_GRID
    proba = np.asarray(y_proba, dtype=float)
    y_arr = np.asarray(y_true)
    best = {"umbral": 0.5, "f1_validacion": 0.0}
    for threshold in grid:
        f1 = float(f1_score(y_arr, (proba >= threshold).astype(int), pos_label=1, zero_division=0))
        if f1 > best["f1_validacion"]:
            best = {"umbral": float(threshold), "f1_validacion": round(f1, 4)}
    return best


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Modelo",
    ax: Optional[plt.Axes] = None,
    figsize=(6, 5),
):
    """Matriz de confusión con conteos absolutos. Positivo = retraso de llegada."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Puntual", "Retrasado"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Condición real")
    ax.set_xlabel("Predicción")
    return fig


def threshold_table(y_true, y_proba, grid: Optional[List[float]] = None) -> pd.DataFrame:
    """Tabla de precisión, recall y F1 del retraso para varios umbrales."""
    if grid is None:
        grid = list(THRESHOLD_GRID)
    proba = np.asarray(y_proba, dtype=float)
    rows = []
    for threshold in grid:
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "umbral": float(threshold),
                "precision_retraso": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall_retraso": round(float(recall_score(y_true, pred, zero_division=0)), 4),
                "f1_retraso": round(float(f1_score(y_true, pred, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_true, pred)), 4),
            }
        )
    return pd.DataFrame(rows)
