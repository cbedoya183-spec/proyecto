"""Pasos 2 y 3 · Cargar los CSV crudos de Kaggle, auditarlos y limpiarlos.

Regla del paso 3: aquí solo se arregla lo que es un hecho objetivo de los datos
(formatos, duplicados, filas imposibles). Nada que se APRENDA de los datos
—medias, medianas, codificaciones— se hace aquí: eso vive dentro del Pipeline del
paso 5, que se reajusta en cada pliegue de la validación cruzada. Así evitamos la
fuga clásica de ajustar un `SimpleImputer` sobre el dataset entero antes de partir.
"""

import numpy as np
import pandas as pd

from src import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 · Carga
# ─────────────────────────────────────────────────────────────────────────────
def _armar_fecha(d: pd.DataFrame) -> pd.Series:
    """Reconstruye la fecha a partir de `YEAR/MONTH/DAY_OF_MONTH`.

    El campo `FL_DATE` que trae el CSV mezcla dos formatos (`06-01-25` y
    `6/1/2025 12:00:00 AM`). Para evitar ambigüedad día/mes, armamos la fecha
    desde las columnas numéricas, que son inequívocas.
    """
    partes = {
        nombre: pd.to_numeric(d[col], errors="coerce")
        for nombre, col in [("year", "YEAR"), ("month", "MONTH"), ("day", "DAY_OF_MONTH")]
    }
    completo = partes["year"].notna() & partes["month"].notna() & partes["day"].notna()
    fecha = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    fecha.loc[completo] = pd.to_datetime(
        {k: v[completo].astype("int32") for k, v in partes.items()}, errors="coerce")
    return fecha


def _reducir_tipos(d: pd.DataFrame) -> pd.DataFrame:
    """Mismos datos, menos memoria.

    Baja el ancho de los números (`float64` → `float32`, `int64` → `int32` o menor)
    y convierte los códigos de aerolínea/aeropuerto a `category`. Es equivalente en
    información pero reduce la huella de memoria para que los 12 CSV (~1 GB tras
    concatenar) quepan en una laptop común.
    """
    for c in d.columns:
        if c in cfg.CATEGORICAS_CRUDAS:
            d[c] = d[c].astype("category")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], downcast="float")
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], downcast="integer")
    return d


def cargar(usar_cache: bool = True) -> pd.DataFrame:
    """Lee todos los CSV de `data/raw/`, los concatena y recorta al periodo del estudio.

    Lee **por bloques** (`chunksize=1_000_000`) y reduce tipos sobre la marcha: así
    varios GB caben en memoria. El resultado se guarda en parquet para que las
    próximas corridas arranquen desde la caché.
    """
    cache = cfg.PROCESADOS / f"vuelos_{cfg.MES_INICIO}_{cfg.MES_FIN}.parquet"
    if usar_cache and cache.exists():
        print(f"Caché encontrada: {cache.name}")
        return pd.read_parquet(cache)

    archivos = sorted(cfg.CRUDOS.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay CSV en {cfg.CRUDOS}. Descárguelos de {cfg.URL_KAGGLE}")

    inicio, fin = pd.Period(cfg.MES_INICIO, "M"), pd.Period(cfg.MES_FIN, "M")
    bloques = []
    for archivo in archivos:
        # `usecols` filtra en el lector: no leemos las columnas de fuga que luego
        # tendríamos que descartar. Ahorra I/O y memoria.
        for trozo in pd.read_csv(archivo, usecols=lambda c: c in cfg.COLS_CRUDAS,
                                 chunksize=1_000_000, low_memory=False):
            trozo["fecha"] = _armar_fecha(trozo)
            periodo = trozo["fecha"].dt.to_period("M")
            trozo = trozo[(periodo >= inicio) & (periodo <= fin) & trozo["fecha"].notna()]
            if len(trozo):
                bloques.append(_reducir_tipos(trozo))
        print(f"  leído {archivo.name}")

    df = pd.concat(bloques, ignore_index=True)
    df.to_parquet(cache, index=False)
    print(f"Caché guardada en {cache.relative_to(cfg.RAIZ)}")
    return df


def cobertura_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba que los 12 meses del estudio están presentes y con un volumen comparable.

    Si falta un mes o llega a medias, cualquier comparación temporal queda sesgada
    (la validación temporal depende de que los meses pesen de forma similar).
    """
    tabla = (df.groupby(df["fecha"].dt.to_period("M").rename("periodo"))
               .agg(vuelos=("fecha", "size"), dias=("fecha", "nunique"),
                    aerolineas=("OP_UNIQUE_CARRIER", "nunique"),
                    aeropuertos=("ORIGIN", "nunique"))
               .reset_index())
    tabla["periodo"] = tabla["periodo"].astype(str)
    esperados = set(pd.period_range(cfg.MES_INICIO, cfg.MES_FIN, freq="M").astype(str))
    faltan = sorted(esperados - set(tabla["periodo"]))
    print(f"Meses esperados: {len(esperados)} · presentes: {len(tabla)} · "
          f"faltantes: {faltan if faltan else 'ninguno'}")
    return tabla


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 · Limpieza
# ─────────────────────────────────────────────────────────────────────────────
def foto(d: pd.DataFrame, etiqueta: str) -> pd.Series:
    """Resumen comparable de la tabla, para construir la tabla «antes y después»."""
    resumen = {
        "filas": len(d),
        "columnas": d.shape[1],
        "memoria (GB)": round(d.memory_usage(deep=True).sum() / 1e9, 3),
        # Duplicados usando la llave única del vuelo (ver cfg.LLAVE_VUELO).
        "duplicados": int(d.duplicated(subset=[c for c in cfg.LLAVE_VUELO if c in d.columns]).sum()),
        "cancelados": int((d["CANCELLED"] == 1).sum()) if "CANCELLED" in d else np.nan,
        "desviados": int((d["DIVERTED"] == 1).sum()) if "DIVERTED" in d else np.nan,
    }
    for c in ["CRS_ELAPSED_TIME", "ARR_DELAY", "ARR_DEL15", "TAIL_NUM"]:
        resumen[f"% nulos {c}"] = round(d[c].isna().mean() * 100, 2) if c in d else np.nan
    return pd.Series(resumen, name=etiqueta)


def nulos_por_columna(df: pd.DataFrame) -> pd.DataFrame:
    """Inventario de nulos por columna, útil para la lámina del paso 3.

    Las 5 columnas de causas (`CARRIER_DELAY`, etc.) superan el 50 % de nulos
    porque solo se llenan cuando YA hubo retraso: además de vacías son fuga.
    Por eso no se leen desde el CSV (ver `cfg.COLS_CRUDAS`).
    """
    nulos = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    return nulos[nulos > 0].to_frame("% nulos")


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpieza del paso 3. Devuelve `(df limpio, bitácora de conteos)`.

    Las decisiones y sus porqués están documentadas en `docs/decisiones.md`.
    """
    bitacora = {}
    antes = len(df)

    # 3.1 · Disponibilidad de la etiqueta.
    # Un vuelo cancelado o desviado nunca llegó: su ARR_DEL15 es NaN y no se puede
    # inventar con la mediana (sería inventarle una etiqueta). Pero tampoco se borra
    # todavía: su horario programado sigue contando para la congestión y la rotación
    # (el aeropuerto no sabía en el itinerario que el vuelo se iba a cancelar).
    df["tiene_etiqueta"] = (
        (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)
        & df["ARR_DEL15"].notna() & df["ARR_DELAY"].notna()
    )
    bitacora["cancelados"] = int((df["CANCELLED"] == 1).sum())
    bitacora["desviados"] = int((df["DIVERTED"] == 1).sum())
    bitacora["sin_etiqueta"] = int((~df["tiene_etiqueta"]).sum())

    # 3.1.b · Coherencia de la etiqueta: ¿ARR_DEL15 coincide con ARR_DELAY >= 15?
    ok = df["tiene_etiqueta"]
    bitacora["coherencia_etiqueta"] = float(
        (df.loc[ok, "ARR_DEL15"] == (df.loc[ok, "ARR_DELAY"] >= cfg.UMBRAL_RETRASO)).mean()
    )
    bitacora["tasa_retraso"] = float(df.loc[ok, "ARR_DEL15"].mean())

    # 3.2 · Itinerarios imposibles.
    # Sin hora programada, duración o distancia no hay vuelo que predecir. También
    # se eliminan valores no físicos (duración o distancia ≤ 0, hora > 2400).
    criticas = ["CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]
    invalidas = (
        df[criticas].isna().any(axis=1)
        | (df["CRS_ELAPSED_TIME"] <= 0) | (df["DISTANCE"] <= 0)
        | (df["CRS_DEP_TIME"] > 2400) | (df["CRS_ARR_TIME"] > 2400)
    )
    bitacora["itinerarios_invalidos"] = int(invalidas.sum())
    df = df.loc[~invalidas].reset_index(drop=True)
    for c in criticas:
        df[c] = df[c].astype("int32")

    # 3.3 · Duplicados exactos.
    # Clave: fecha + aerolínea + número de vuelo + origen + destino + hora programada.
    dup = df.duplicated(subset=cfg.LLAVE_VUELO, keep="first")
    bitacora["duplicados"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)

    bitacora["filas_antes"] = antes
    bitacora["filas_despues"] = len(df)
    return df, bitacora


def resumen_atipicos(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de los extremos de las variables numéricas.

    Los retrasos de más de un día (ARR_DELAY > 1440) son reales (vuelos
    perdidos por meteorología severa), no errores: no se borra ninguna fila.
    Como etiqueta no estorban porque solo importa el corte en 15 minutos.
    Como variable (en `ret_prev_llegada`) sí se recortan con **constantes
    fijas** [-60, 360] para que no distorsionen el ajuste; usar percentiles
    sería leer la distribución de los datos y meter fuga.
    """
    return (df.loc[df["tiene_etiqueta"], ["ARR_DELAY", "CRS_ELAPSED_TIME", "DISTANCE"]]
              .describe(percentiles=[0.01, 0.5, 0.99]).T)
