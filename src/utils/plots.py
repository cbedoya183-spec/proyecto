"""
plots.py — Gráficas del análisis exploratorio de retrasos de llegada.

Cada figura responde una pregunta concreta sobre los vuelos operados.
Describen el periodo completo. No se usan para elegir el modelo.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_delay_rate_by_month(df: pd.DataFrame, target: str = "ARR_DEL15"):
    """¿Cambia la tasa de retraso a lo largo del ciclo junio 2025–mayo 2026?"""
    work = df.copy()
    work["periodo"] = pd.to_datetime(
        {"year": work["YEAR"], "month": work["MONTH"], "day": 1}
    )
    rates = work.groupby("periodo")[target].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rates.index, rates.values, marker="o", color="#1f4e79")
    ax.set_title("Tasa de retraso de llegada por mes")
    ax.set_ylabel("% con llegada ≥ 15 min")
    ax.set_xlabel("Mes")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_delay_rate_by_hour(df: pd.DataFrame, target: str = "ARR_DEL15"):
    """¿Se acumula el retraso hacia la tarde, como supone la regla manual?"""
    rates = df.groupby("CRS_DEP_HOUR")[target].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rates.index.astype(int), rates.values, color="#4C78A8")
    ax.axvline(15.5, color="#E45756", linestyle="--", label="Corte de la regla: 16:00")
    ax.set_title("Tasa de retraso según la hora programada de salida")
    ax.set_xlabel("Hora programada (CRS)")
    ax.set_ylabel("% con llegada ≥ 15 min")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_delay_rate_by_carrier(df: pd.DataFrame, target: str = "ARR_DEL15"):
    """¿La aerolínea operadora cambia la tasa de retraso antes de despegar?"""
    rates = (
        df.groupby("OP_UNIQUE_CARRIER")[target]
        .agg(tasa="mean", vuelos="size")
        .sort_values("tasa", ascending=False)
    )
    rates["tasa_pct"] = rates["tasa"] * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=rates.index, y=rates["tasa_pct"], ax=ax, color="#F58518")
    ax.set_title("Tasa de retraso por aerolínea operadora")
    ax.set_xlabel("Código IATA")
    ax.set_ylabel("% con llegada ≥ 15 min")
    fig.tight_layout()
    return fig
