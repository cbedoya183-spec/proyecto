"""Paso 6 · Evaluación: métricas adecuadas al problema y gráficas.

La clase «retrasado» es minoritaria (~20 %), así que la **exactitud engaña**: la
línea base «siempre a tiempo» acierta ~80 % sin predecir un solo retraso. La
métrica principal es el **AUC** (ordena el riesgo, no depende del umbral),
acompañada de **PR-AUC**, **recall** y **F1**.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config as cfg

AZUL, NARANJA = "#2b6cb0", "#c05621"


def auc_seguro(y, p) -> float:
    """`roc_auc_score` falla si un bloque tiene una sola clase. Devuelve NaN en ese caso."""
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def metricas(y, proba, pred) -> dict:
    """Diccionario de métricas estándar para clasificación binaria desbalanceada."""
    return {
        "AUC": auc_seguro(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "F1": f1_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precisión": precision_score(y, pred, zero_division=0),
        "Exactitud": float((pred == y).mean()),
    }


def umbral_optimo(y, proba) -> float:
    """Umbral que maximiza el F1 de la clase «retrasado», buscado en validación.

    Devuelve el corte asociado al mejor F1 de la curva precision-recall. Se usa
    como punto de operación: todo lo que esté por encima del umbral se predice
    como retraso.
    """
    precision, recall, cortes = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(cortes[int(np.argmax(f1[:-1]))]) if len(cortes) else 0.5


def demostrar_fuga(df: pd.DataFrame, columnas_legitimas, columna_fuga: str = "DEP_DELAY",
                   n: int = 80_000) -> pd.DataFrame:
    """Enseña, en vivo, qué pasa con el AUC si dejamos entrar una columna de fuga.

    Entrena un `HistGradientBoosting` con y sin `DEP_DELAY` y compara el AUC. El
    número CON fuga NO se puede obtener en la vida real (DEP_DELAY se mide al salir
    de la puerta, cuando ya se sabe el resultado). Es el ejemplo canónico de fuga.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import TargetEncoder

    demo = df.sample(min(n, len(df)), random_state=cfg.SEMILLA).sort_values("fecha")
    corte = int(len(demo) * 0.7)

    # Partimos las columnas en numéricas y categóricas para que el `HistGradientBoosting`
    # reciba todo como float. Las categóricas pasan por un `TargetEncoder` (igual que en
    # el pipeline oficial del paso 5); las numéricas, por una imputación por mediana.
    cat_set = set(cfg.CATEGORICAS)
    resultado = {}
    for nombre, cols in {"Sin fuga (solo columnas legítimas)": list(columnas_legitimas),
                         f"CON fuga (+ {columna_fuga})": list(columnas_legitimas) + [columna_fuga]}.items():
        num_cols = [c for c in cols if c not in cat_set]
        cat_cols = [c for c in cols if c in cat_set]
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", TargetEncoder(random_state=cfg.SEMILLA), cat_cols),
        ])
        m = make_pipeline(prep, HistGradientBoostingClassifier(max_iter=100, random_state=cfg.SEMILLA))
        # Las categóricas deben ser `str` para TargetEncoder.
        X_demo = demo[cols].copy()
        for c in cat_cols:
            X_demo[c] = X_demo[c].astype(str)
        m.fit(X_demo.iloc[:corte], demo[cfg.OBJETIVO].iloc[:corte])
        p = m.predict_proba(X_demo.iloc[corte:])[:, 1]
        resultado[nombre] = auc_seguro(demo[cfg.OBJETIVO].iloc[corte:], p)
    return pd.Series(resultado).to_frame("AUC").round(4)


def evaluar_prueba(y_test, p_modelo, umbral, nombre, p_regla, tasa_base) -> pd.DataFrame:
    """Tabla de la prueba final: el modelo elegido junto a la regla a mano y la línea base."""
    def fila(etiqueta, proba, pred):
        return {"modelo": etiqueta, **metricas(y_test, proba, pred)}

    return pd.DataFrame([
        fila(nombre, p_modelo, (p_modelo >= umbral).astype(int)),
        fila("Regla a mano", p_regla, (p_regla > 0.5).astype(int)),
        fila("Línea base (siempre a tiempo)",
             np.full(len(y_test), tasa_base), np.zeros(len(y_test), dtype=int)),
    ]).set_index("modelo").round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Gráficas
# ─────────────────────────────────────────────────────────────────────────────
def guardar(fig, nombre: str) -> None:
    fig.savefig(cfg.FIGURAS / f"{nombre}.png", bbox_inches="tight", dpi=110)


def grafica_hipotesis(df: pd.DataFrame):
    """¿Se ven en los datos los patrones que el modelo va a aprender? Mirar ANTES de modelar."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    por_hora = df.groupby("hora_sal")[cfg.OBJETIVO].mean()
    ax[0, 0].bar(por_hora.index, por_hora.values * 100, color=AZUL)
    ax[0, 0].set(title="Retraso según hora programada de salida", xlabel="hora", ylabel="% retrasados")

    por_dia = df.groupby("dia_semana")[cfg.OBJETIVO].mean()
    ax[0, 1].bar(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], por_dia.values * 100, color=AZUL)
    ax[0, 1].set(title="Retraso según día de la semana", ylabel="% retrasados")

    por_aer = df.groupby("OP_UNIQUE_CARRIER", observed=True)[cfg.OBJETIVO].agg(["mean", "size"])
    por_aer = por_aer[por_aer["size"] > por_aer["size"].median()].sort_values("mean")
    ax[1, 0].barh(por_aer.index.astype(str), por_aer["mean"] * 100, color=AZUL)
    ax[1, 0].set(title="Retraso por aerolínea", xlabel="% retrasados")

    con_prev = df[df["prev_disponible"] == 1]
    cortes = pd.cut(con_prev["ret_prev_llegada"], [-61, -5, 0, 15, 30, 60, 361])
    prop = con_prev.groupby(cortes, observed=True)[cfg.OBJETIVO].mean()
    ax[1, 1].bar(range(len(prop)), prop.values * 100, color=NARANJA)
    ax[1, 1].set_xticks(range(len(prop)))
    ax[1, 1].set_xticklabels([str(i) for i in prop.index], rotation=30, fontsize=8)
    ax[1, 1].set(title="Propagación: retraso del tramo anterior (min)", ylabel="% retrasados")

    plt.tight_layout()
    guardar(fig, "01_hipotesis")
    return fig


def grafica_comparacion(tabla: pd.DataFrame, tasa_retraso: float, referencias):
    """Las dos métricas que deciden (AUC y PR-AUC), con la variación entre pliegues."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))

    t = tabla.sort_values("AUC")
    colores = [NARANJA if i in referencias else AZUL for i in t.index]
    ax[0].barh(t.index, t["AUC"], xerr=t["AUC_std"], color=colores)
    ax[0].axvline(0.5, color="gray", ls="--")
    ax[0].set(title=f"ROC-AUC · validación temporal ({cfg.K_PLIEGUES} pliegues)",
              xlabel="AUC", xlim=(0.45, max(0.8, t["AUC"].max() + 0.04)))

    t2 = tabla.sort_values("PR-AUC")
    ax[1].barh(t2.index, t2["PR-AUC"],
               color=[NARANJA if i in referencias else AZUL for i in t2.index])
    ax[1].axvline(tasa_retraso, color="gray", ls="--")
    ax[1].set(title="PR-AUC (la línea gris es el azar)", xlabel="PR-AUC")
    ax[1].set_yticklabels([])

    plt.tight_layout()
    guardar(fig, "02_comparacion_modelos")
    return fig


def grafica_prueba(y_test, p_modelo, p_regla, umbral, nombre, tabla_test):
    """ROC, precisión-recall y matriz de confusión del modelo elegido en la prueba final."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    fpr, tpr, _ = roc_curve(y_test, p_modelo)
    fr, tr, _ = roc_curve(y_test, p_regla)
    ax[0].plot(fpr, tpr, color=AZUL, label=f"{nombre} (AUC {tabla_test.loc[nombre, 'AUC']:.3f})")
    ax[0].plot(fr, tr, color=NARANJA, label=f"Regla a mano (AUC {tabla_test.loc['Regla a mano', 'AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(title="Curva ROC (prueba temporal)", xlabel="falsos positivos", ylabel="verdaderos positivos")
    ax[0].legend(loc="lower right", fontsize=8)

    pr, rc, _ = precision_recall_curve(y_test, p_modelo)
    ax[1].plot(rc, pr, color=AZUL, label=f"PR-AUC {tabla_test.loc[nombre, 'PR-AUC']:.3f}")
    ax[1].axhline(y_test.mean(), color="gray", ls="--", label=f"azar = {y_test.mean():.2f}")
    ax[1].set(title="Precisión-Recall", xlabel="recall", ylabel="precisión")
    ax[1].legend(fontsize=8)

    cm = confusion_matrix(y_test, (p_modelo >= umbral).astype(int))
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[2].set(title=f"Matriz de confusión (umbral {umbral:.2f})", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["a tiempo", "retrasado"], yticklabels=["a tiempo", "retrasado"],
              xlabel="predicho", ylabel="real")
    ax[2].grid(False)

    plt.tight_layout()
    guardar(fig, "03_prueba_final")
    return fig, cm


def grafica_importancia(modelo, X_test, y_test, n: int = 20_000):
    """Importancia por permutación: baraja una columna a la vez y mide cuánto cae el AUC.

    Cuanto más cae el AUC al barajar, más depende el modelo de esa columna. Es
    agnóstica al tipo de modelo (funciona igual con lineales, árboles, boosting).
    """
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(cfg.SEMILLA)
    sub = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    imp = permutation_importance(modelo, X_test.iloc[sub], y_test[sub], scoring="roc_auc",
                                 n_repeats=3, random_state=cfg.SEMILLA, n_jobs=1)
    importancias = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importancias.index, importancias.values, color=AZUL)
    ax.set(title="Caída del AUC al barajar cada columna", xlabel="caída del AUC")
    plt.tight_layout()
    guardar(fig, "04_importancia")
    return fig, importancias.sort_values(ascending=False)
