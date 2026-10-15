"""
imputation.py — Comparación de imputadores sobre el entrenamiento.

En los vuelos ya operados casi no hay vacíos en las columnas de itinerario.
Para decidir el imputador igual se esconde una parte de los valores conocidos
del entrenamiento y se mide el error absoluto. La prueba no participa.
"""

import numpy as np
import pandas as pd


def compare_numeric_imputers(
    values: pd.Series,
    groups: pd.Series,
    mask_rate: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Esconde una fracción de `values` y compara media global, mediana global
    y mediana por grupo. `values` y `groups` deben ser solo del entrenamiento.
    """
    rng = np.random.RandomState(random_state)
    observed = values.dropna()
    if observed.empty:
        raise ValueError("No hay valores observados para comparar imputadores.")

    hide_n = max(1, int(len(observed) * mask_rate))
    hide_idx = rng.choice(observed.index.to_numpy(), size=hide_n, replace=False)
    y_true = observed.loc[hide_idx].to_numpy(dtype=float)
    known = observed.drop(index=hide_idx)

    pred_mean = np.full(hide_n, float(known.mean()))
    pred_median = np.full(hide_n, float(known.median()))

    grouped = known.groupby(groups.loc[known.index]).median()
    pred_group = groups.loc[hide_idx].map(grouped).fillna(float(known.median())).to_numpy(dtype=float)

    rows = []
    for name, pred in (
        ("Media global", pred_mean),
        ("Mediana global", pred_median),
        ("Mediana por aerolínea", pred_group),
    ):
        mae = float(np.mean(np.abs(y_true - pred)))
        rows.append({"metodo": name, "mae": round(mae, 3), "valores_escondidos": hide_n})

    report = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_name = str(report.iloc[0]["metodo"])
    best = float(report.iloc[0]["mae"])
    baseline = float(report.loc[report["metodo"] == "Mediana global", "mae"].iloc[0])
    relative = (baseline - best) / baseline if baseline else 0.0
    if best_name == "Mediana por aerolínea" and relative >= 0.05:
        decision = (
            "La mediana por aerolínea baja el error "
            f"de {baseline:.3f} a {best:.3f} "
            f"({relative:.1%} menos) al esconder duración programada del entrenamiento. "
            "Esa imputación queda dentro del Pipeline, ajustada solo con cada pliegue. "
            "Una aerolínea que no estuvo en el entrenamiento usa la mediana global."
        )
    else:
        decision = (
            "Se mantiene la mediana global dentro del Pipeline. "
            f"El menor MAE fue {best_name} ({best:.3f}) y la mediana global quedó en {baseline:.3f}."
        )
    report.attrs["decision"] = decision
    return report
