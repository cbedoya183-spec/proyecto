"""Paso 5 · Modelos: varias familias, la misma vara para todos.

La misma validación, los mismos datos y todo el preprocesamiento DENTRO del Pipeline,
que se reajusta en cada pliegue y nunca ve las filas de validación.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

from src import config as cfg
from src.evaluacion import metricas, umbral_optimo

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    HAY_XGB = False

# Modelos que no dan una probabilidad útil: se reportan pero no se les busca umbral.
REFERENCIAS = ["Línea base (siempre a tiempo)", "Regla a mano"]


# ──────────────────────────────────────────────────────────────────────────────
# La regla a mano, como un modelo más
# ──────────────────────────────────────────────────────────────────────────────
class ReglaManual(BaseEstimator, ClassifierMixin):
    """«Si el vuelo sale a las HH:00 o después, llegará retrasado».

    Se escribe como estimador de sklearn para que pase por la MISMA validación cruzada
    y la MISMA tabla de métricas que los modelos de ML. Así la comparación es justa.
    """

    def __init__(self, hora=17):
        self.hora = hora

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (np.asarray(X["hora_sal_dec"]) >= self.hora).astype(float)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def calibrar_regla(horas_salida, y):
    """Busca la hora de corte que maximiza el F1. SOLO con datos de entrenamiento:
    elegirla mirando la prueba sería la misma trampa que entrenar con el futuro."""
    horas = np.arange(0, 24)
    f1s = [f1_score(y, (np.asarray(horas_salida) >= h).astype(int), zero_division=0) for h in horas]
    mejor = int(np.argmax(f1s))
    return int(horas[mejor]), float(f1s[mejor])


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline y catálogo
# ──────────────────────────────────────────────────────────────────────────────
def preprocesador(escalar):
    """Imputación, escalado y codificación por objetivo, todo DENTRO del Pipeline."""
    pasos = [SimpleImputer(strategy="median")] + ([StandardScaler()] if escalar else [])
    return ColumnTransformer([
        ("num", make_pipeline(*pasos), cfg.NUMERICAS),
        ("cat", TargetEncoder(random_state=cfg.SEMILLA), cfg.CATEGORICAS),
    ])


def catalogo(hora_regla=17):
    """nombre -> (estimador, ¿lleva preprocesamiento?, ¿escalar?, complejidad).

    Una familia distinta por entrada, no siete variantes del mismo árbol. KNN y SVM con
    kernel se descartan a propósito: con cientos de miles de filas su costo es cuadrático
    y no aportan una familia que no esté ya representada.
    """
    modelos = {
        "Línea base (siempre a tiempo)": (DummyClassifier(strategy="prior"), False, False, 0),
        "Regla a mano": (ReglaManual(hora=hora_regla), False, False, 1),
        "Regresión logística": (LogisticRegression(max_iter=500), True, True, 2),
        "Árbol de decisión": (DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=100, random_state=cfg.SEMILLA), True, False, 3),
        "Random forest": (RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=40,
            n_jobs=-1, random_state=cfg.SEMILLA), True, False, 5),
        "HistGradientBoosting": (HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=cfg.SEMILLA), True, False, 6),
        "Red neuronal (MLP)": (MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', early_stopping=True, max_iter=60,
            random_state=cfg.SEMILLA), True, True, 7),
    }
    if HAY_XGB:
        modelos["XGBoost"] = (XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
            random_state=cfg.SEMILLA), True, False, 6)
    return modelos


def construir(nombre, hora_regla=17):
    """Arma el Pipeline completo de un modelo del catálogo."""
    estimador, lleva_prep, escalar, _ = catalogo(hora_regla)[nombre]
    estimador = clone(estimador)
    return make_pipeline(preprocesador(escalar), estimador) if lleva_prep else estimador


# ──────────────────────────────────────────────────────────────────────────────
# Validación cruzada temporal
# ──────────────────────────────────────────────────────────────────────────────
def comparar_modelos(X, y, hora_regla=17, k=cfg.K_PLIEGUES, verbose=True):
    """Corre todos los modelos del catálogo con los MISMOS pliegues de TimeSeriesSplit.

    Barajar las filas al azar sería hacer trampa: un modelo entrenado con vuelos del 15 de
    marzo que predice vuelos del 14 estaría viendo el futuro. `X` debe venir ordenada por fecha.
    """
    tscv = TimeSeriesSplit(n_splits=k)
    pliegues = list(tscv.split(X))
    filas = []

    for nombre in catalogo(hora_regla):
        t0 = time.time()
        aucs, pr_aucs, probas_ult, y_ult = [], [], None, None

        for i, (tr, va) in enumerate(pliegues):
            modelo = construir(nombre, hora_regla).fit(X.iloc[tr], y[tr])
            p = modelo.predict_proba(X.iloc[va])[:, 1]
            m = metricas(y[va], p, (p >= 0.5).astype(int))
            aucs.append(m["AUC"])
            pr_aucs.append(m["PR-AUC"])
            if i == len(pliegues) - 1:
                probas_ult, y_ult = p, y[va]

        # El umbral se elige en el ÚLTIMO pliegue de validación, nunca en la prueba.
        # El 0,50 de sklearn no es una decisión de negocio: con 20 % de retrasos deja
        # al modelo casi mudo.
        if nombre in REFERENCIAS:
            umbral = 0.5
        else:
            umbral = umbral_optimo(y_ult, probas_ult)

        m = metricas(y_ult, probas_ult, (probas_ult >= umbral).astype(int))
        filas.append({
            "modelo": nombre,
            "AUC": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
            "PR-AUC": float(np.mean(pr_aucs)),
            "F1": m["F1"], "Recall": m["Recall"],
            "Precisión": m["Precisión"], "Exactitud": m["Exactitud"],
            "umbral": umbral, "seg": round(time.time() - t0, 1),
        })
        if verbose:
            print(f"  {nombre:<32} AUC {filas[-1]['AUC']:.4f}  F1 {m['F1']:.3f}  "
                  f"({filas[-1]['seg']:.0f} s)")

    return pd.DataFrame(filas).set_index("modelo").sort_values("AUC", ascending=False)


def elegir_modelo(tabla, hora_regla=17):
    """Elección argumentada, con la regla fijada ANTES de mirar la prueba:

    entre los modelos de ML cuyo AUC esté dentro de una desviación típica del mejor,
    gana EL MÁS SIMPLE. Un empate estadístico se rompe por interpretabilidad y costo,
    no por el tercer decimal.
    """
    comp = {n: v[3] for n, v in catalogo(hora_regla).items()}
    ml = tabla.drop(index=[r for r in REFERENCIAS if r in tabla.index])

    mejor = ml["AUC"].idxmax()
    margen = float(ml.loc[mejor, "AUC_std"])
    empatados = ml[ml["AUC"] >= ml.loc[mejor, "AUC"] - margen]
    elegido = min(empatados.index, key=lambda n: comp[n])

    texto = (
        f"Mejor AUC          : {mejor} ({ml.loc[mejor, 'AUC']:.4f})\n"
        f"Margen (1 desv.)   : {margen:.4f}\n"
        f"Empatados dentro del margen: {list(empatados.index)}\n"
        f"\n-> Modelo elegido  : {elegido}  (el más simple del empate)\n"
        f"Umbral de decisión : {tabla.loc[elegido, 'umbral']:.3f}  "
        f"(máximo F1 en validación, nunca en la prueba)"
    )
    if "Regla a mano" in tabla.index:
        texto += (f"\n\nVentaja sobre la regla a mano en validación: "
                  f"{tabla.loc[elegido, 'AUC'] - tabla.loc['Regla a mano', 'AUC']:+.3f} de AUC · "
                  f"{tabla.loc[elegido, 'F1'] - tabla.loc['Regla a mano', 'F1']:+.3f} de F1")
    return elegido, float(tabla.loc[elegido, "umbral"]), texto
