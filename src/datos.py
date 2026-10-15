"""Pasos 2 y 3 · Cargar los CSV crudos de Kaggle, auditarlos y limpiarlos.

Regla del paso 3: aquí solo se arregla lo que es un hecho objetivo de los datos
(formatos, duplicados, filas imposibles). Nada que se APRENDA de los datos
—medias, medianas, codificaciones— se hace aquí: eso vive dentro del Pipeline.
"""

import numpy as np
import pandas as pd

from src import config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 · Carga
# ──────────────────────────────────────────────────────────────────────────────
def _armar_fecha(d):
    """YEAR/MONTH/DAY_OF_MONTH es lo fiable; FL_DATE ('06-01-25') es ambiguo entre día y mes."""
    partes = {nombre: pd.to_numeric(d[col], errors="coerce")
              for nombre, col in [("year", "YEAR"), ("month", "MONTH"), ("day", "DAY_OF_MONTH")]}
    completo = partes["year"].notna() & partes["month"].notna() & partes["day"].notna()
    fecha = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    fecha.loc[completo] = pd.to_datetime(
        {k: v[completo].astype("int32") for k, v in partes.items()}, errors="coerce")
    return fecha


def _reducir_tipos(d):
    """Mismos datos, menos memoria: baja el ancho de los números y pasa los códigos a category."""
    for c in d.columns:
        if c in cfg.CATEGORICAS_CRUDAS:
            d[c] = d[c].astype("category")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], downcast="float")
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], downcast="integer")
    return d


def cargar(usar_cache=True):
    """Lee todos los CSV de datos/crudos/, los concatena y recorta al periodo del estudio.

    Lee por bloques y reduce tipos para que varios GB quepan en memoria. No se pierde
    ninguna celda: es la misma información con menos bytes.
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


def cobertura_mensual(df):
    """Comprueba que los 12 meses están y con un volumen parecido.

    Si falta un mes o llega a medias, cualquier comparación temporal queda sesgada.
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


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 · Limpieza
# ──────────────────────────────────────────────────────────────────────────────
def foto(d, etiqueta):
    """Resumen comparable de la tabla, para la tabla «antes y después»."""
    resumen = {
        "filas": len(d),
        "columnas": d.shape[1],
        "memoria (GB)": round(d.memory_usage(deep=True).sum() / 1e9, 3),
        "duplicados": int(d.duplicated(subset=[c for c in cfg.LLAVE_VUELO if c in d.columns]).sum()),
        "cancelados": int((d["CANCELLED"] == 1).sum()) if "CANCELLED" in d else np.nan,
        "desviados": int((d["DIVERTED"] == 1).sum()) if "DIVERTED" in d else np.nan,
    }
    for c in ["CRS_ELAPSED_TIME", "ARR_DELAY", "ARR_DEL15", "TAIL_NUM"]:
        resumen[f"% nulos {c}"] = round(d[c].isna().mean() * 100, 2) if c in d else np.nan
    return pd.Series(resumen, name=etiqueta)


def nulos_por_columna(df):
    """Inventario de nulos. Las causas del retraso superan el 50 % porque solo se llenan
    cuando YA hubo retraso: además de vacías, son fuga."""
    nulos = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    return nulos[nulos > 0].to_frame("% nulos")


def limpiar(df):
    """Limpieza del paso 3. Devuelve (df limpio, bitácora de conteos)."""
    bitacora = {}
    antes = len(df)

    # 3.1 · Etiqueta disponible. Un vuelo cancelado o desviado nunca llegó: no tiene
    # retraso de llegada y NO se le puede inventar uno con la mediana. Pero tampoco se
    # borra todavía: su horario programado cuenta para la congestión y la rotación (paso 4).
    df["tiene_etiqueta"] = ((df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)
                            & df["ARR_DEL15"].notna() & df["ARR_DELAY"].notna())
    bitacora["cancelados"] = int((df["CANCELLED"] == 1).sum())
    bitacora["desviados"] = int((df["DIVERTED"] == 1).sum())
    bitacora["sin_etiqueta"] = int((~df["tiene_etiqueta"]).sum())

    ok = df["tiene_etiqueta"]
    bitacora["coherencia_etiqueta"] = float(
        (df.loc[ok, "ARR_DEL15"] == (df.loc[ok, "ARR_DELAY"] >= cfg.UMBRAL_RETRASO)).mean())
    bitacora["tasa_retraso"] = float(df.loc[ok, "ARR_DEL15"].mean())

    # 3.2 · Itinerarios imposibles: sin hora, duración o distancia no hay vuelo que predecir.
    criticas = ["CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"]
    invalidas = (df[criticas].isna().any(axis=1)
                 | (df["CRS_ELAPSED_TIME"] <= 0) | (df["DISTANCE"] <= 0)
                 | (df["CRS_DEP_TIME"] > 2400) | (df["CRS_ARR_TIME"] > 2400))
    bitacora["itinerarios_invalidos"] = int(invalidas.sum())
    df = df.loc[~invalidas].reset_index(drop=True)
    for c in criticas:
        df[c] = df[c].astype("int32")

    # 3.3 · Duplicados del volcado (misma llave de vuelo dos veces).
    dup = df.duplicated(subset=cfg.LLAVE_VUELO, keep="first")
    bitacora["duplicados"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)

    bitacora["filas_antes"] = antes
    bitacora["filas_despues"] = len(df)
    return df, bitacora


def resumen_atipicos(df):
    """Los retrasos extremos (más de un día) son reales, no errores: no se borra ninguna fila.

    Como etiqueta no estorban (solo importa el corte en 15 minutos). Como variable
    —el retraso del tramo anterior— sí distorsionan, y por eso se recortan en el paso 4
    con constantes fijas, no con percentiles calculados sobre los datos.
    """
    return (df.loc[df["tiene_etiqueta"], ["ARR_DELAY", "CRS_ELAPSED_TIME", "DISTANCE"]]
              .describe(percentiles=[0.01, 0.5, 0.99]).T)
