"""
train.py — Entrena, compara y guarda el modelo elegido.

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
import pandas as pd
from sklearn.base import clone

from src.config import MODELS_DIR, TARGET_COLUMN, TEST_FRACTION
from src.data.load_data import load_flight_data
from src.features.build_features import build_full_feature_matrix
from src.models.pipeline import build_full_pipeline
from src.models.experiment import (
    apply_tuning_decision,
    choose_champion,
    compare_models,
    explain_choice,
    get_candidate_models,
    regla_manual_prediccion,
    sort_chronologically,
    temporal_split,
    tune_top_models,
)
from src.models.pipeline import DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERICAL_FEATURES
from src.utils.logger import get_logger

logger = get_logger("train")


def run_training_pipeline(quick: bool = False, tune: bool = True):
    """Carga la muestra, compara modelos y guarda métricas y el campeón."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_flight_data(use_parquet=True)
    df_feat = sort_chronologically(build_full_feature_matrix(df_raw))
    feature_cols = DEFAULT_NUMERICAL_FEATURES + DEFAULT_CATEGORICAL_FEATURES
    X = df_feat[feature_cols].copy()
    y = df_feat[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test, split_idx = temporal_split(X, y, test_fraction=TEST_FRACTION)

    logger.info(
        "Entrenamiento %s vuelos | prueba %s vuelos | retraso train %.2f%% | test %.2f%%",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    if "FL_DATE" in df_feat.columns:
        dates = pd.to_datetime(df_feat["FL_DATE"], errors="coerce")
        logger.info(
            "Train %s a %s | Test %s a %s",
            dates.iloc[:split_idx].min().date(),
            dates.iloc[:split_idx].max().date(),
            dates.iloc[split_idx:].min().date(),
            dates.iloc[split_idx:].max().date(),
        )

    results, _pipes = compare_models(X_train, X_test, y_train, y_test, quick=quick)
    champion = choose_champion(results)
    tuning_note = "Búsqueda de hiperparámetros no solicitada."
    tuned_estimator = None

    if tune:
        tuning = tune_top_models(results, X_train, y_train, champion=champion, quick=quick)
        tuning.drop(columns=["estimador"], errors="ignore").to_csv(
            MODELS_DIR / "hiperparametros.csv", index=False
        )
        results, tuned_estimator, champion, tuning_note = apply_tuning_decision(
            results, tuning, champion, X_train, y_train, X_test, y_test
        )

    explanation = explain_choice(results, champion)
    logger.info(tuning_note)
    logger.info(explanation)

    export = results.drop(columns=[c for c in results.columns if c == "estimador"], errors="ignore")
    export.to_csv(MODELS_DIR / "metrics_comparison.csv", index=False)

    if tuned_estimator is not None:
        model_to_save = tuned_estimator
    else:
        model_to_save = build_full_pipeline(clone(get_candidate_models(y_train, quick=quick)[champion]))
        model_to_save.fit(X_train, y_train)
    joblib.dump(model_to_save, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "eleccion.txt").write_text(explanation + "\n\n" + tuning_note + "\n", encoding="utf-8")
    logger.info("Modelo guardado: %s", champion)
    return export, explanation, tuning_note


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de puntualidad de vuelos")
    parser.add_argument("--quick", action="store_true", help="Menos árboles, para un ensayo corto")
    parser.add_argument("--no-tune", action="store_true", help="Omite la búsqueda de hiperparámetros")
    args = parser.parse_args()
    table, explanation, note = run_training_pipeline(quick=args.quick, tune=not args.no_tune)
    print(table.to_string(index=False))
    print(explanation)
    print(note)
