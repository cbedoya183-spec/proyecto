"""
train.py — Entrenamiento, validación temporal y selección del modelo campeón de puntualidad.

Compara de manera equitativa:
1. Dummy Classifier (Línea Base mayoritaria trivial)
2. Regla Manual de Negocio (Heurística: Salidas tarde/noche en hubs congestionados)
3. Regresión Logística regularizada
4. Árbol de Decisión (Interpretable)
5. Random Forest Classifier (Ensamble bagging)
6. HistGradientBoosting Classifier (Ensamble boosting de alta velocidad)

Aplica partición temporal estricta (Entrenamiento en 9 meses pasados, Prueba en 3 meses futuros)
y validación cruzada TimeSeriesSplit. Exporta métricas a models/metrics_comparison.csv
y el modelo campeón a models/best_model.joblib.

Uso:
    python src/models/train.py
    python src/models/train.py --quick
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data.load_data import load_flight_data
from src.features.build_features import build_full_feature_matrix
from src.models.evaluate import compute_classification_metrics
from src.models.pipeline import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERICAL_FEATURES,
    build_full_pipeline,
)
from src.utils.logger import get_logger

logger = get_logger("train")


def regla_manual_prediccion(df_subset: pd.DataFrame) -> np.ndarray:
    """
    Regla heurística manual de la industria aeronáutica:
    Predice 'Retrasado' si el vuelo está programado para salir a partir de las 16:00
    y opera en un hub de alta afluencia (Origen o Destino es Hub), debido al efecto
    cascada de demoras acumuladas durante el día.
    """
    cond_hora = df_subset["CRS_DEP_HOUR"] >= 16
    cond_hub = (df_subset["ORIGIN_IS_HUB"] == 1) | (df_subset["DEST_IS_HUB"] == 1)
    pred = (cond_hora & cond_hub).astype(int).values
    return pred


def get_candidate_models(quick: bool = False):
    """Retorna el catálogo de modelos candidatos de diversas familias algorítmicas."""
    n_est = 40 if quick else 100
    models = {
        "Dummy (Línea Base)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_est,
            max_depth=12,
            min_samples_leaf=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=n_est,
            max_depth=8,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def run_training_pipeline(quick: bool = False):
    """Ejecuta el flujo completo de modelado con validación temporal estricta."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando y preparando dataset de vuelos 2025–2026...")
    df_raw = load_flight_data(use_parquet=True)
    df_feat = build_full_feature_matrix(df_raw)

    # Ordenar estrictamente por fecha para validación temporal
    if "FL_DATE" in df_feat.columns:
        df_feat = df_feat.sort_values("FL_DATE").reset_index(drop=True)
    elif "YEAR" in df_feat.columns and "MONTH" in df_feat.columns:
        df_feat = df_feat.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)

    feature_cols = DEFAULT_NUMERICAL_FEATURES + DEFAULT_CATEGORICAL_FEATURES
    X = df_feat[feature_cols].copy()
    y = df_feat[TARGET_COLUMN].copy()

    # Partición Temporal: 75% más antiguo (Entrenamiento) vs 25% más reciente (Prueba)
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        f"Partición Temporal: Entrenamiento={len(X_train):,} vuelos (Periodo inicial), "
        f"Prueba={len(X_test):,} vuelos (Periodo reciente)."
    )
    logger.info(f"Tasa de retraso en Train: {y_train.mean()*100:.2f}% | en Test: {y_test.mean()*100:.2f}%")

    results = []
    trained_pipelines = {}

    # 1. Evaluar Regla Manual de Negocio en Test
    y_pred_regla = regla_manual_prediccion(X_test)
    metrics_regla = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred_regla,
        y_proba=None,
        model_name="Regla Manual (Hubs + Tarde)",
    )
    metrics_regla["cv_roc_auc"] = np.nan
    results.append(metrics_regla)
    logger.info(f"Regla Manual evaluada: Accuracy={metrics_regla['accuracy']}, F1={metrics_regla['f1_retraso']}")

    # 2. Evaluar Modelos dentro del Pipeline Hermético
    models = get_candidate_models(quick=quick)
    tscv = TimeSeriesSplit(n_splits=3)

    for name, model in models.items():
        logger.info(f"Entrenando {name} con Pipeline hermético...")
        pipe = build_full_pipeline(
            model=model,
            numerical_cols=DEFAULT_NUMERICAL_FEATURES,
            categorical_cols=DEFAULT_CATEGORICAL_FEATURES,
        )

        # Validación cruzada temporal sobre Train
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="roc_auc")
            cv_mean = round(float(cv_scores.mean()), 4)
        except Exception:
            cv_mean = np.nan

        # Ajuste final en Train y predicción en Test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                pass

        m = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba, model_name=name)
        m["cv_roc_auc"] = cv_mean
        results.append(m)
        trained_pipelines[name] = pipe

        logger.info(
            f"  {name} completado -> ROC-AUC Test: {m.get('roc_auc', 'N/A')}, "
            f"PR-AUC: {m.get('pr_auc', 'N/A')}, F1: {m.get('f1_retraso', 'N/A')}, Accuracy: {m.get('accuracy')}"
        )

    # 3. Consolidar y exportar tabla comparativa
    df_metrics = pd.DataFrame(results)
    metrics_path = MODELS_DIR / "metrics_comparison.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Tabla comparativa de métricas guardada en: {metrics_path}")

    # 4. Seleccionar modelo campeón (mayor ROC-AUC / PR-AUC equilibrado)
    ml_models = [m for m in results if m["modelo"] not in ["Dummy (Línea Base)", "Regla Manual (Hubs + Tarde)"]]
    best_entry = max(ml_models, key=lambda x: x["roc_auc"] if not np.isnan(x["roc_auc"]) else 0)
    best_name = best_entry["modelo"]
    best_pipe = trained_pipelines[best_name]

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipe, model_path)
    logger.info(f"Modelo campeón seleccionado: {best_name} (ROC-AUC={best_entry['roc_auc']}). Guardado en {model_path}.")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de puntualidad aérea")
    parser.add_argument("--quick", action="store_true", help="Modo rápido con menos estimadores")
    args = parser.parse_args()

    df_res = run_training_pipeline(quick=args.quick)
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA DE MODELOS — PUNTUALIDAD DE VUELOS EE.UU. (2025–2026)")
    print("=" * 80)
    print(df_res.to_string(index=False))
