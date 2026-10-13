"""
evaluate.py — Módulo integral de evaluación de métricas y visualizaciones diagnósticas.

Diseñado específicamente para el problema de clasificación de siniestros viales:
calcula métricas clave (PR-AUC, ROC-AUC, Recall de clase 1, F1-Score), genera
matrices de confusión, curvas Precision-Recall y simulaciones de umbral de decisión
para el caso de uso médico y de emergencias de la Línea 123 de Medellín.
"""

import sys
from pathlib import Path

# Garantizar raíz en sys.path
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger("evaluate")


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
) -> Dict[str, float]:
    """
    Calcula un diccionario exhaustivo de métricas de desempeño.
    """
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
            # Si y_proba tiene 2 columnas, tomar la columna 1 (probabilidad de clase retrasada)
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, proba_pos)), 4)
        except Exception as e:
            logger.warning(f"No fue posible calcular ROC/PR AUC para {model_name}: {e}")
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Modelo",
    ax: Optional[plt.Axes] = None,
    figsize=(6, 5),
):
    """
    Grafica la matriz de confusión normalizada con recuentos absolutos y porcentajes.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = ["0: Puntual", "1: Retrasado (>= 15 min)"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Condición Real", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicción del Modelo", fontsize=11, fontweight="bold")

    return fig


def plot_roc_and_pr_curves(
    models_dict: Dict[str, any],
    X_test,
    y_test,
    figsize=(14, 5),
):
    """
    Compara las curvas ROC y Precision-Recall de múltiples modelos en un solo gráfico.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = ["#7F8C8D", "#2980B9", "#27AE60", "#E74C3C", "#8E44AD"]

    for idx, (name, pipe) in enumerate(models_dict.items()):
        color = colors[idx % len(colors)]

        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_proba = pipe.decision_function(X_test)
        else:
            continue

        # 1. Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2)

        # 2. Curva Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2)

    # Configuración panel ROC
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio (AUC = 0.500)")
    axes[0].set_title("Curvas ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tasa Falsos Positivos (1 - Especificidad)")
    axes[0].set_ylabel("Tasa Verdaderos Positivos (Recall)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])

    # Configuración panel PR
    baseline_pr = float(y_test.mean())
    axes[1].axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.5, label=f"Línea Base ({baseline_pr:.3f})")
    axes[1].set_title("Curvas Precision-Recall (Foco en Clase Crítica)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Cobertura de Pacientes Graves)")
    axes[1].set_ylabel("Precision (Certeza Médica)")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])

    plt.tight_layout()
    return fig


def simulate_decision_thresholds(
    y_true,
    y_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Simula el impacto de ajustar el umbral de decisión para el despacho de emergencias.
    
    Analiza el costo de:
    - Falsos Negativos (FN): Accidentes graves reportados como 'leves' -> Peligro de muerte.
    - Falsos Positivos (FP): Accidentes leves clasificados como 'graves' -> Ambulancia despachada innecesariamente.
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []
    total_graves = int((y_true == 1).sum())
    total_leves = int((y_true == 0).sum())

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        rows.append({
            "Umbral": th,
            "Recall (Heridos Atendidos)": f"{rec*100:.1f}%",
            "Precision": f"{prec*100:.1f}%",
            "F1-Score": f"{f1:.3f}",
            "Falsos Negativos (Graves Sin Ambulancia)": fn,
            "Falsos Positivos (Despachos Innecesarios)": fp,
            "% Pacientes Graves Salva-vidas": f"{(tp/total_graves)*100:.1f}%",
        })

    return pd.DataFrame(rows)
