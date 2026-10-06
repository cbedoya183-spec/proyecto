"""
train.py — Script de entrenamiento, validación cruzada y selección del modelo campeón.

Entrena y compara 4 modelos dentro del Pipeline hermético:
1. Dummy Classifier (Línea base trivial)
2. Regresión Logística (Línea base lineal regularizada)
3. Random Forest Classifier (Ensemble bagging de árboles)
4. XGBoost Classifier (Gradient Boosting de alta precisión)

Aplica Stratified K-Fold Cross Validation sobre X_train, evalúa en X_test independiente,
exporta la tabla comparativa de métricas a models/metrics_comparison.csv y serializa
el modelo ganador en models/best_model.joblib.

Uso:
    python src/models/train.py
    python src/models/train.py --quick
"""

import argparse
import sys
from pathlib import Path

# Garantizar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from src.config import (
    BINARY_TARGET_COLUMN,
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.load_data import load_raw_data
from src.features.build_features import prepare_full_dataset
from src.models.evaluate import compute_classification_metrics
from src.models.pipeline import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERICAL_FEATURES,
    build_full_pipeline,
)
from src.utils.logger import get_logger

logger = get_logger("train")


def get_candidate_models(quick: bool = False):
    """
    Retorna el diccionario de modelos candidatos a evaluar.
    """
    n_est = 50 if quick else 120
    models = {
        "Dummy (Línea Base)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_est,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_est,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def run_training_pipeline(quick: bool = False):
    """
    Ejecuta el ciclo completo de entrenamiento, evaluación y serialización.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando y preparando dataset...")
    df_raw = load_raw_data(use_sample=True)
    df_clean = prepare_full_dataset(df_raw)

    feature_cols = DEFAULT_NUMERICAL_FEATURES + DEFAULT_CATEGORICAL_FEATURES
    X = df_clean[feature_cols].copy()
    y = df_clean[BINARY_TARGET_COLUMN].copy()

    logger.info(
        f"Separando datos (Test={TEST_SIZE*100:.0f}%, Estratificado, Seed={RANDOM_STATE})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"X_train: {X_train.shape[0]:,} filas | X_test: {X_test.shape[0]:,} filas")

    candidates = get_candidate_models(quick=quick)
    trained_pipelines = {}
    metrics_list = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in candidates.items():
        logger.info(f"\n{'='*60}\nEntrenando: {name}...")
        pipeline = build_full_pipeline(classifier=clf)

        # 1. Cross Validation sobre X_train (Cero leakage)
        logger.info(f"Ejecutando {CV_FOLDS}-Fold Stratified Cross-Validation...")
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=["f1_macro", "average_precision"],
            n_jobs=-1,
        )
        cv_f1 = cv_results["test_f1_macro"].mean()
        cv_pr_auc = cv_results["test_average_precision"].mean()
        logger.info(f"CV F1-Macro: {cv_f1:.4f} | CV PR-AUC: {cv_pr_auc:.4f}")

        # 2. Ajuste final sobre todo X_train
        logger.info("Ajustando modelo sobre X_train completo...")
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # 3. Evaluación en Test Set independiente
        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )

        test_metrics = compute_classification_metrics(
            y_test, y_pred, y_proba=y_proba, model_name=name
        )
        test_metrics["cv_f1_macro"] = round(float(cv_f1), 4)
        test_metrics["cv_pr_auc"] = round(float(cv_pr_auc), 4)
        metrics_list.append(test_metrics)

        logger.info(
            f"Test Set -> Accuracy: {test_metrics['accuracy']:.4f} | "
            f"PR-AUC: {test_metrics['pr_auc']:.4f} | "
            f"Recall (Graves): {test_metrics['recall_grave']:.4f} | "
            f"F1 (Graves): {test_metrics['f1_grave']:.4f}"
        )

    # Consolidar tabla comparativa
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = MODELS_DIR / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nResumen comparativo guardado en {metrics_path}:\n{metrics_df.to_string(index=False)}")

    # Selección del modelo campeón (máximo PR-AUC en Test)
    valid_metrics = metrics_df.dropna(subset=["pr_auc"])
    champion_name = valid_metrics.sort_values(by="pr_auc", ascending=False).iloc[0]["modelo"]
    champion_pipeline = trained_pipelines[champion_name]

    champion_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(champion_pipeline, champion_path)
    logger.info(f"\n🏆 Modelo Campeón: '{champion_name}' guardado en {champion_path}")

    return trained_pipelines, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Entrena y compara modelos de ML.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecuta con menos estimadores para pruebas rápidas de desarrollo.",
    )
    args = parser.parse_args()
    run_training_pipeline(quick=args.quick)


if __name__ == "__main__":
    main()
