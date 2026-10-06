"""
plots.py — Funciones estandarizadas de visualización para EDA y Reportes.

Provee gráficos con estética moderna, paletas armoniosas y anotaciones claras
de porcentaje y volumen para su uso en notebooks y documentos ejecutivos.
"""

import sys
from pathlib import Path

# Garantizar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración estética global
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Paleta cromática corporativa / analítica
PALETTE_BINARY = ["#4A90E2", "#E74C3C"]  # Azul (Leve) / Rojo (Grave)
PALETTE_QUALITATIVE = sns.color_palette("deep")


def plot_target_distribution(df: pd.DataFrame, figsize=(12, 5)):
    """
    Grafica la distribución del target original (3 clases) y del target binario (es_grave).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Target original
    if "GRAVEDAD_ACCIDENTE" in df.columns:
        counts_orig = df["GRAVEDAD_ACCIDENTE"].value_counts()
        pcts_orig = (counts_orig / len(df) * 100).round(1)

        sns.barplot(
            x=counts_orig.index,
            y=counts_orig.values,
            ax=axes[0],
            palette="Blues_r",
            hue=counts_orig.index,
            legend=False,
        )
        axes[0].set_title("Distribución de Severidad Original (MEData)", fontsize=13, pad=12, fontweight="bold")
        axes[0].set_xlabel("Categoría Original")
        axes[0].set_ylabel("Cantidad de Incidentes")

        for i, (cnt, pct) in enumerate(zip(counts_orig.values, pcts_orig.values)):
            axes[0].text(
                i, cnt + (max(counts_orig.values) * 0.02),
                f"{cnt:,}\n({pct}%)",
                ha="center", fontsize=10, fontweight="bold",
            )
        axes[0].set_ylim(0, max(counts_orig.values) * 1.15)

    # 2. Target binario
    if "es_grave" in df.columns:
        counts_bin = df["es_grave"].value_counts().sort_index()
        pcts_bin = (counts_bin / len(df) * 100).round(1)
        labels_bin = ["0: Solo Daños (Leve)", "1: Con Heridos / Víctimas (Severo)"]

        sns.barplot(
            x=labels_bin,
            y=counts_bin.values,
            ax=axes[1],
            palette=PALETTE_BINARY,
            hue=labels_bin,
            legend=False,
        )
        axes[1].set_title("Target Binario de Priorización Vital ('es_grave')", fontsize=13, pad=12, fontweight="bold")
        axes[1].set_xlabel("Clase Modelada")
        axes[1].set_ylabel("Cantidad de Incidentes")

        for i, (cnt, pct) in enumerate(zip(counts_bin.values, pcts_bin.values)):
            axes[1].text(
                i, cnt + (max(counts_bin.values) * 0.02),
                f"{cnt:,}\n({pct}%)",
                ha="center", fontsize=10, fontweight="bold",
            )
        axes[1].set_ylim(0, max(counts_bin.values) * 1.15)

    plt.tight_layout()
    return fig


def plot_comuna_risk(df: pd.DataFrame, top_n: int = 12, figsize=(13, 6)):
    """
    Compara las comunas con mayor volumen de incidentes y su respectiva tasa de severidad.
    """
    col_comuna = "comuna_limpia" if "comuna_limpia" in df.columns else "COMUNA"
    if col_comuna not in df.columns or "es_grave" not in df.columns:
        return None

    # Filtrar desconocidas para foco territorial
    df_filtered = df[~df[col_comuna].isin(["DESCONOCIDA", "OTRA_COMUNA"])].copy()

    stats = (
        df_filtered.groupby(col_comuna)
        .agg(
            total=("es_grave", "count"),
            graves=("es_grave", "sum"),
            tasa_graves=("es_grave", "mean"),
        )
        .sort_values(by="total", ascending=False)
        .head(top_n)
    )
    stats["tasa_graves_pct"] = (stats["tasa_graves"] * 100).round(1)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    x = np.arange(len(stats))
    width = 0.55

    # Barras: Volumen total
    bars = ax1.bar(x, stats["total"], width=width, color="#4A90E2", alpha=0.75, label="Total Incidentes")
    # Línea: Tasa de severidad
    line = ax2.plot(x, stats["tasa_graves_pct"], color="#E74C3C", marker="o", linewidth=2.5, label="% Severos (Heridos/Muertos)")

    ax1.set_title(f"Top {top_n} Comunas de Medellín: Volumen de Siniestros vs % Severidad Vital", fontsize=14, pad=15, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats.index, rotation=35, ha="right", fontsize=10)
    ax1.set_ylabel("Volumen Total de Incidentes", color="#1f4e79", fontweight="bold")
    ax2.set_ylabel("Tasa de Severidad (% Con Heridos o Muertos)", color="#c0392b", fontweight="bold")
    ax2.grid(False)
    ax2.set_ylim(20, 85)

    for i, rate in enumerate(stats["tasa_graves_pct"]):
        ax2.annotate(f"{rate}%", (i, rate), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#900C3F", fontsize=9)

    plt.tight_layout()
    return fig


def plot_temporal_patterns(df: pd.DataFrame, figsize=(14, 5)):
    """
    Muestra la dinámica horaria y semanal de los accidentes y su tasa de criticidad.
    """
    if "hora" not in df.columns or "es_grave" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Patrón por hora del día
    df_temp = df.dropna(subset=["hora", "dia_semana"]).copy()
    hora_stats = df_temp.groupby("hora").agg(
        total=("es_grave", "count"),
        tasa_grave=("es_grave", "mean"),
    )
    hora_stats.index = [int(h) for h in hora_stats.index]
    hora_stats["tasa_pct"] = hora_stats["tasa_grave"] * 100

    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.bar(hora_stats.index, hora_stats["total"], color="#3498DB", alpha=0.6, label="Volumen")
    ax1_twin.plot(hora_stats.index, hora_stats["tasa_pct"], color="#E74C3C", marker="s", linewidth=2, label="% Severidad")

    ax1.set_title("Comportamiento por Hora del Día (0 a 23h)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hora del Día")
    ax1.set_ylabel("Cantidad de Incidentes")
    ax1_twin.set_ylabel("% Severos", color="#E74C3C")
    ax1_twin.grid(False)
    ax1.set_xticks(range(0, 24, 2))

    # 2. Patrón por día de la semana
    dias_nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_stats = df_temp.groupby("dia_semana").agg(
        total=("es_grave", "count"),
        tasa_grave=("es_grave", "mean"),
    )
    dia_stats["tasa_pct"] = dia_stats["tasa_grave"] * 100
    dia_stats.index = [dias_nombres[int(i)] for i in dia_stats.index]

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(dia_stats.index, dia_stats["total"], color="#2ECC71", alpha=0.6)
    ax2_twin.plot(dia_stats.index, dia_stats["tasa_pct"], color="#C0392B", marker="o", linewidth=2)

    ax2.set_title("Comportamiento por Día de la Semana", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Día de la Semana")
    ax2.set_ylabel("Cantidad de Incidentes")
    ax2_twin.set_ylabel("% Severos", color="#C0392B")
    ax2_twin.grid(False)
    ax2.set_xticks(range(len(dia_stats)))
    ax2.set_xticklabels(dia_stats.index, rotation=30, ha="right")

    plt.tight_layout()
    return fig


def plot_clase_accidente_risk(df: pd.DataFrame, figsize=(11, 5)):
    """
    Severidad según la dinámica del siniestro (Atropello, Caída, Volcamiento, Choque).
    """
    col = "clase_accidente_limpia" if "clase_accidente_limpia" in df.columns else "CLASE_ACCIDENTE"
    if col not in df.columns or "es_grave" not in df.columns:
        return None

    stats = (
        df.groupby(col)
        .agg(
            total=("es_grave", "count"),
            tasa_grave=("es_grave", "mean"),
        )
        .sort_values(by="tasa_grave", ascending=True)
    )
    stats["tasa_pct"] = (stats["tasa_grave"] * 100).round(1)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#E74C3C" if r >= 70 else "#F39C12" if r >= 50 else "#3498DB" for r in stats["tasa_pct"]]

    bars = ax.barh(stats.index, stats["tasa_pct"], color=colors, height=0.6)
    ax.set_title("Tasa de Severidad Vital por Dinámica del Siniestro", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("% Con Heridos o Fallecidos (Criticidad Vital)")
    ax.set_xlim(0, 105)

    for bar, pct, cnt in zip(bars, stats["tasa_pct"], stats["total"]):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct}% ({cnt:,} casos)",
            va="center", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    return fig
