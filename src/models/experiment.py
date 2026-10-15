"""
experiment.py — Comparación de modelos con validación temporal.

La decisión del modelo se toma con TimeSeriesSplit sobre el entrenamiento.
El bloque de prueba (los meses más recientes) se evalúa al reportar el resultado,
no para elegir hiperparámetros ni el ganador.

Si dos modelos quedan a menos de 0,01 de ROC-AUC en validación, se prefiere
el más simple. Un empate técnico no justifica el modelo más complejo.
"""

import gc
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, TEST_FRACTION, TIME_SPLITS, TUNING_SPLITS
from src.models.evaluate import best_f1_threshold, compute_classification_metrics
from src.models.pipeline import build_full_pipeline

AUC_TIE_BAND = 0.01
MIN_TUNING_GAIN = 0.005
SIMPLICITY = {
    "Regresión Logística": 0,
    "Árbol de Decisión": 1,
    "KNN": 2,
    "Random Forest": 3,
    "Gradient Boosting": 4,
    "XGBoost": 5,
}


def regla_manual_prediccion(df_subset: pd.DataFrame) -> np.ndarray:
    """
    Regla sin modelo: salida programada desde las 16:00 y origen o destino
    en un hub de la lista fija (ATL, ORD, DFW, DEN, CLT, LAX, JFK, MIA, MCO, EWR).
    La lista no se aprende del target. Es el número que el modelo tiene que vencer.
    """
    late = df_subset["CRS_DEP_HOUR"] >= 16
    hub = (df_subset["ORIGIN_IS_HUB"] == 1) | (df_subset["DEST_IS_HUB"] == 1)
    return (late & hub).fillna(False).astype(int).to_numpy()


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por fecha del vuelo. El corte temporal depende de este orden."""
    out = df.copy()
    if "FL_DATE" not in out.columns:
        return out.reset_index(drop=True)
    out["_fecha_orden"] = pd.to_datetime(out["FL_DATE"], errors="coerce")
    out = out.sort_values(["_fecha_orden"], kind="mergesort").drop(columns="_fecha_orden")
    return out.reset_index(drop=True)


def temporal_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = TEST_FRACTION):
    """Reserva el tramo final como prueba. No mezcla meses futuros en el entrenamiento."""
    split_idx = int(len(X) * (1.0 - test_fraction))
    return (
        X.iloc[:split_idx].copy(),
        X.iloc[split_idx:].copy(),
        y.iloc[:split_idx].copy(),
        y.iloc[split_idx:].copy(),
        split_idx,
    )


def get_candidate_models(y_train: pd.Series, quick: bool = False) -> Dict[str, object]:
    """
    Regresión logística, árbol, bosque, gradient boosting y XGBoost.

    KNN no entra: con decenas de miles de vuelos y aeropuertos en one-hot, cada predicción
    recorre el entrenamiento y no cabe en la misma validación temporal.
    SVM de kernel tampoco: la frontera lineal ya está en la regresión logística.
    Una red neuronal no agrega una familia distinta del boosting.
    """
    n_estimators = 40 if quick else 80
    positives = max(int((y_train == 1).sum()), 1)
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / positives

    return {
        "Dummy (Línea Base)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
            eval_metric="logloss",
        ),
    }


def _search_spaces(quick: bool = False) -> dict:
    n_hi = 80 if quick else 160
    return {
        "Regresión Logística": {
            "model__C": uniform(0.05, 4.0),
        },
        "Árbol de Decisión": {
            "model__max_depth": [4, 6, 8, 12],
            "model__min_samples_leaf": randint(10, 80),
        },
        "Random Forest": {
            "model__n_estimators": randint(40, n_hi),
            "model__max_depth": [8, 12, 16],
            "model__min_samples_leaf": randint(5, 40),
        },
        "Gradient Boosting": {
            "model__max_iter": randint(40, n_hi),
            "model__max_depth": [3, 6, 8],
            "model__learning_rate": uniform(0.03, 0.12),
        },
        "XGBoost": {
            "model__n_estimators": randint(40, n_hi),
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": uniform(0.03, 0.12),
        },
    }


def _threshold_from_validation(estimator, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Ajusta el umbral en el último pliegue temporal del entrenamiento."""
    splitter = TimeSeriesSplit(n_splits=TIME_SPLITS)
    train_idx, val_idx = list(splitter.split(X_train))[-1]
    pipe = build_full_pipeline(clone(estimator))
    _fit_release(pipe, X_train.iloc[train_idx], y_train.iloc[train_idx])
    if not hasattr(pipe, "predict_proba"):
        return 0.5
    proba = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
    return best_f1_threshold(y_train.iloc[val_idx], proba)["umbral"]


def _manual_rule_cv_f1(X_train: pd.DataFrame, y_train: pd.Series) -> float:
    splitter = TimeSeriesSplit(n_splits=TIME_SPLITS)
    scores = []
    for _, val_idx in splitter.split(X_train):
        pred = regla_manual_prediccion(X_train.iloc[val_idx])
        scores.append(
            compute_classification_metrics(y_train.iloc[val_idx], pred, model_name="regla")["f1_retraso"]
        )
    return round(float(np.mean(scores)), 4)


def _fit_release(estimator, X: pd.DataFrame, y: pd.Series):
    """Ajusta y, si el sistema no alcanza a reservar la matriz, reintenta una vez."""
    last_error = None
    for _ in range(2):
        try:
            estimator.fit(X, y)
            return estimator
        except MemoryError as exc:
            last_error = exc
            gc.collect()
    raise last_error


def _temporal_roc_auc(estimator, X: pd.DataFrame, y: pd.Series, n_splits: int) -> np.ndarray:
    """Un pliegue a la vez. Libera la matriz antes del siguiente."""
    scores = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(X):
        fold = build_full_pipeline(clone(estimator))
        _fit_release(fold, X.iloc[train_idx], y.iloc[train_idx])
        proba = fold.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(float(roc_auc_score(y.iloc[val_idx], proba)))
        del fold, proba
        gc.collect()
    return np.asarray(scores, dtype=float)


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    quick: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compara la regla manual y los modelos con el mismo corte temporal.
    El umbral de cada modelo se elige en validación. Las columnas de prueba
    quedan en la tabla para el reporte; la elección usa cv_roc_auc.
    """
    rows = []
    pipes: Dict[str, object] = {}
    splitter = TimeSeriesSplit(n_splits=TIME_SPLITS)

    started = time.perf_counter()
    pred_rule = regla_manual_prediccion(X_test)
    rule_metrics = compute_classification_metrics(
        y_test, pred_rule, model_name="Regla Manual (Hubs + Tarde)"
    )
    rule_metrics.update(
        {
            "cv_roc_auc": np.nan,
            "cv_roc_auc_std": np.nan,
            "cv_f1_umbral": _manual_rule_cv_f1(X_train, y_train),
            "umbral": np.nan,
            "f1_umbral_test": rule_metrics["f1_retraso"],
            "precision_umbral_test": rule_metrics["precision_retraso"],
            "recall_umbral_test": rule_metrics["recall_retraso"],
            "segundos": round(time.perf_counter() - started, 1),
        }
    )
    rows.append(rule_metrics)

    for name, estimator in get_candidate_models(y_train, quick=quick).items():
        print(f"Entrenando {name}...")
        started = time.perf_counter()
        pipe = build_full_pipeline(clone(estimator))
        try:
            cv_scores = _temporal_roc_auc(estimator, X_train, y_train, TIME_SPLITS)
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std = round(float(cv_scores.std()), 4)
        except Exception as exc:
            print(f"  Validación no disponible para {name}: {exc}")
            cv_mean, cv_std = np.nan, np.nan

        threshold = 0.5
        if name != "Dummy (Línea Base)":
            try:
                threshold = _threshold_from_validation(estimator, X_train, y_train)
            except Exception as exc:
                print(f"  Umbral por defecto para {name}: {exc}")

        gc.collect()
        _fit_release(pipe, X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        pred_default = pipe.predict(X_test)
        metrics = compute_classification_metrics(y_test, pred_default, proba, model_name=name)

        if proba is not None:
            pred_thr = (proba >= threshold).astype(int)
            tuned = compute_classification_metrics(y_test, pred_thr, proba, model_name=name)
        else:
            tuned = metrics

        metrics.update(
            {
                "cv_roc_auc": cv_mean,
                "cv_roc_auc_std": cv_std,
                "cv_f1_umbral": np.nan,
                "umbral": threshold,
                "f1_umbral_test": tuned["f1_retraso"],
                "precision_umbral_test": tuned["precision_retraso"],
                "recall_umbral_test": tuned["recall_retraso"],
                "segundos": round(time.perf_counter() - started, 1),
            }
        )
        rows.append(metrics)
        del pipe
        gc.collect()
        print(
            f"  CV ROC-AUC {cv_mean} | umbral {threshold} | "
            f"F1 test en ese umbral {tuned['f1_retraso']} | {metrics['segundos']} s"
        )

    return pd.DataFrame(rows), pipes


def choose_champion(results: pd.DataFrame) -> str:
    """
    Elige por ROC-AUC de validación temporal.
    Dentro de una banda de 0,01 se queda el modelo más simple.
    """
    ranked = results.dropna(subset=["cv_roc_auc"]).copy()
    ranked = ranked[~ranked["modelo"].isin(["Dummy (Línea Base)", "Regla Manual (Hubs + Tarde)"])]
    if ranked.empty:
        raise RuntimeError("Ningún modelo tiene ROC-AUC de validación.")
    best_auc = float(ranked["cv_roc_auc"].max())
    band = ranked[ranked["cv_roc_auc"] >= best_auc - AUC_TIE_BAND].copy()
    band["simplicidad"] = band["modelo"].map(lambda name: SIMPLICITY.get(name, 99))
    band = band.sort_values(["simplicidad", "cv_roc_auc"], ascending=[True, False])
    return str(band.iloc[0]["modelo"])


def tune_top_models(
    results: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    champion: str,
    quick: bool = False,
    n_iter: int = 4,
) -> pd.DataFrame:
    """
    Ajusta el campeón y, si es otro, el de mayor ROC-AUC.
    La base y el ajuste usan el mismo TimeSeriesSplit de 2 pliegues.
    """
    spaces = _search_spaces(quick=quick)
    ranked = results.dropna(subset=["cv_roc_auc"]).copy()
    ranked = ranked[ranked["modelo"].isin(spaces)].sort_values("cv_roc_auc", ascending=False)
    names = []
    if champion in spaces:
        names.append(champion)
    if not ranked.empty and ranked.iloc[0]["modelo"] not in names:
        names.append(str(ranked.iloc[0]["modelo"]))

    catalog = get_candidate_models(y_train, quick=quick)
    splitter = TimeSeriesSplit(n_splits=TUNING_SPLITS)
    rows = []

    for name in names:
        print(f"Buscando hiperparámetros de {name}...")
        started = time.perf_counter()
        base_scores = _temporal_roc_auc(catalog[name], X_train, y_train, TUNING_SPLITS)
        base_auc = float(np.nanmean(base_scores))
        search = RandomizedSearchCV(
            estimator=build_full_pipeline(clone(catalog[name])),
            param_distributions=spaces[name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=splitter,
            n_jobs=1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        rows.append(
            {
                "modelo": name,
                "cv_roc_auc_base": round(base_auc, 4),
                "cv_roc_auc_ajustado": round(float(search.best_score_), 4),
                "ganancia_cv": round(float(search.best_score_) - base_auc, 4),
                "mejores_parametros": str(
                    {k.replace("model__", ""): _jsonable(v) for k, v in search.best_params_.items()}
                ),
                "segundos": round(time.perf_counter() - started, 1),
                "estimador": search.best_estimator_,
            }
        )
        print(f"  Misma validación: base {base_auc:.4f} -> ajustado {search.best_score_:.4f}")
        del search
        gc.collect()

    return pd.DataFrame(rows)


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def apply_tuning_decision(
    results: pd.DataFrame,
    tuning: pd.DataFrame,
    champion: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Optional[object], str, str]:
    """
    Conserva el campeón si el ajuste no gana 0,005 de ROC-AUC en la misma validación.
    Cambia de familia solo si otro modelo ajustado supera al campeón por más de 0,01.
    """
    if tuning.empty:
        return results, None, champion, "No hubo modelos con búsqueda de hiperparámetros."

    champion_rows = tuning[tuning["modelo"] == champion]
    others = tuning[tuning["modelo"] != champion]
    champion_base = (
        float(champion_rows.iloc[0]["cv_roc_auc_base"]) if not champion_rows.empty else np.nan
    )
    best_other = others.sort_values("cv_roc_auc_ajustado", ascending=False).head(1)

    use_name = champion
    use_estimator = None
    if (
        not best_other.empty
        and pd.notna(champion_base)
        and float(best_other.iloc[0]["cv_roc_auc_ajustado"]) > champion_base + AUC_TIE_BAND
    ):
        use_name = str(best_other.iloc[0]["modelo"])
        use_estimator = best_other.iloc[0]["estimador"]
        gain = float(best_other.iloc[0]["ganancia_cv"])
        message = (
            f"{use_name} ajustado llega a ROC-AUC {float(best_other.iloc[0]['cv_roc_auc_ajustado']):.4f} "
            f"en la misma validación, más de {AUC_TIE_BAND:.2f} por encima de {champion} "
            f"({champion_base:.4f}). La banda de simplicidad ya no alcanza: se adopta {use_name} ajustado. "
            f"Ganancia interna del ajuste: {gain:.4f}."
        )
    elif not champion_rows.empty and float(champion_rows.iloc[0]["ganancia_cv"]) >= MIN_TUNING_GAIN:
        use_estimator = champion_rows.iloc[0]["estimador"]
        gain = float(champion_rows.iloc[0]["ganancia_cv"])
        message = (
            f"El ajuste de {champion} ganó {gain:.4f} de ROC-AUC en la misma validación temporal. "
            "Se guarda esa configuración."
        )
    else:
        parts = []
        for _, row in tuning.iterrows():
            gain = row["ganancia_cv"]
            base = row["cv_roc_auc_base"]
            adjusted = row["cv_roc_auc_ajustado"]
            if pd.notna(base) and pd.notna(adjusted) and pd.notna(gain):
                parts.append(
                    f"{row['modelo']} pasó de {float(base):.4f} a {float(adjusted):.4f} "
                    f"(ganancia {float(gain):.4f})"
                )
            else:
                parts.append(f"{row['modelo']} no tuvo una base comparable en ese pliegue")
        detail = "; ".join(parts) if parts else "no hubo filas de búsqueda"
        message = (
            "El ajuste no cambia la decisión. En la misma validación temporal, "
            f"{detail}. Hace falta al menos {MIN_TUNING_GAIN:.3f} para reemplazar la configuración base, "
            f"y ningún otro modelo superó al campeón por más de {AUC_TIE_BAND:.2f}. "
            "Se conserva la configuración base."
        )
        return results, None, champion, message

    raw_model = clone(use_estimator.named_steps["model"])
    threshold = _threshold_from_validation(raw_model, X_train, y_train)
    proba = use_estimator.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = compute_classification_metrics(y_test, pred, proba, model_name=f"{use_name} ajustado")
    tuned_cv = float(tuning.loc[tuning["modelo"] == use_name, "cv_roc_auc_ajustado"].iloc[0])
    metrics.update(
        {
            "cv_roc_auc": round(tuned_cv, 4),
            "cv_roc_auc_std": np.nan,
            "cv_f1_umbral": np.nan,
            "umbral": threshold,
            "f1_umbral_test": metrics["f1_retraso"],
            "precision_umbral_test": metrics["precision_retraso"],
            "recall_umbral_test": metrics["recall_retraso"],
            "segundos": float(tuning.loc[tuning["modelo"] == use_name, "segundos"].iloc[0]),
        }
    )
    updated = pd.concat([results, pd.DataFrame([metrics])], ignore_index=True)
    return updated, use_estimator, f"{use_name} ajustado", message


def explain_choice(results: pd.DataFrame, champion: str) -> str:
    """Arma el argumento con los números de la tabla, sin redondeos de memoria."""
    table = results.set_index("modelo")
    champ = table.loc[champion]
    rule = table.loc["Regla Manual (Hubs + Tarde)"]
    dummy = table.loc["Dummy (Línea Base)"]
    f1_model = float(champ["f1_umbral_test"])
    f1_rule = float(rule["f1_umbral_test"])
    comparison = (
        f"En la prueba, con el umbral elegido en validación ({champ['umbral']}), "
        f"el F1 del retraso es {f1_model:.4f} frente a {f1_rule:.4f} de la regla manual."
    )
    if f1_model > f1_rule:
        verdict = "Supera la regla en el punto de operación, que es el número que había que vencer."
    else:
        verdict = (
            "No supera el F1 de la regla en ese umbral. Aun así ordena mejor el riesgo: "
            f"su ROC-AUC de validación es {champ['cv_roc_auc']}, y la regla no produce un ranking. "
            "Sirve para priorizar vuelos, no para reemplazar la heurística como alerta dura."
        )
    std = champ["cv_roc_auc_std"]
    std_txt = "el ajuste no repite la desviación de los tres pliegues" if pd.isna(std) else f"desviación {std}"
    return (
        f"Campeón por validación temporal: {champion}. "
        f"ROC-AUC de validación {champ['cv_roc_auc']} ({std_txt}). "
        f"La línea base acierta el {float(dummy['accuracy']):.2%} de la prueba porque predice siempre puntual, "
        f"con recall de retraso {float(dummy['recall_retraso']):.2f}. "
        f"{comparison} {verdict}"
    )
