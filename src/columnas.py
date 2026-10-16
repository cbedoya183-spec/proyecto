"""Paso 4 · Columnas: cuáles entran, cuáles se crean y cuáles se descartan.

Regla de oro anti-fuga: una columna entra SOLO si su valor se conoce a la hora de
salida programada. Entra todo lo que sale del itinerario publicado; no entra nada
que se mida durante o después del vuelo.

Las variables más informativas del proyecto (`ret_prev_llegada`, `holgura_prog`,
`vuelos_*_hora`, `tramo_del_dia`, `dias_a_festivo`) se construyen aquí. Ninguna
viene en los CSV: hay que deducirlas ordenando por aeronave y reconstruyendo los
husos horarios desde el propio itinerario, sin tabla externa.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src import config as cfg

# Recorte del retraso del tramo anterior. Son CONSTANTES FIJAS, no percentiles
# calculados sobre los datos. Si usáramos un percentil, el recorte dependería de
# la distribución de la muestra y filtraría información del conjunto completo.
RECORTE_PREV = (-60, 360)   # minutos: [-1 hora adelantado, +6 horas tarde]


def hhmm_a_min(serie):
    """Convierte horas en formato `hhmm` (entero) a minutos desde medianoche.

    El BTS usa `2400` para medianoche en vez de `0`. Con el módulo `% 1440` ese
    `2400` cae de nuevo en `0`, que es lo correcto para ordenar vuelos nocturnos.
    """
    s = np.asarray(serie, dtype="int64")
    return ((s // 100) * 60 + s % 100) % 1440


def crear_columnas(d: pd.DataFrame) -> pd.DataFrame:
    """Crea solo columnas conocidas a la hora de salida programada.

    Comentarios clave sobre fugas:
    - `ret_prev_llegada`: usa el retraso REAL del tramo anterior del mismo avión,
      pero **solo si ese tramo ya aterrizó antes de la salida programada de este
      vuelo** (`real_prev <= salida`). Si el tramo anterior aún está en el aire,
      la bandera `prev_disponible = 0` y la variable queda en NaN. Esto imita
      exactamente lo que sabría una aerolínea en la puerta: el inbound ya aterrizó
      o no, y si aterrizó, con cuánto retraso.
    - `vuelos_*_hora`: cuenta vuelos PROGRAMADOS (incluye los que luego se cancelan),
      porque las cancelaciones no se conocen de antemano. Es la congestión publicada.
    """
    d = d.copy()

    # --- Línea de tiempo en minutos ---
    # Convertimos cada vuelo a un entero: minutos desde el origen (la fecha mínima).
    # Esto nos permite ordenar, restar horas y calcular diferencias limpiamente.
    origen = pd.Timestamp(d["fecha"].min()).normalize()
    fecha_min = ((d["fecha"] - origen) // pd.Timedelta("1min")).to_numpy().astype("int64")

    dep_hm = hhmm_a_min(d["CRS_DEP_TIME"])
    arr_hm = hhmm_a_min(d["CRS_ARR_TIME"])
    duracion = d["CRS_ELAPSED_TIME"].to_numpy().astype("int64")

    dep_min = fecha_min + dep_hm                                   # salida programada, reloj del origen
    desfase = ((arr_hm - dep_hm - duracion + 720) % 1440) - 720    # huso destino - huso origen
    arr_prog_min = dep_min + duracion + desfase                    # llegada programada, reloj del destino
    arr_real_min = arr_prog_min + d["ARR_DELAY"].to_numpy()        # llegada real; NaN si no llegó

    # --- Calendario y hora del día ---
    d["hora_sal"] = (dep_hm // 60).astype("int8")
    d["hora_sal_dec"] = (dep_hm / 60).astype("float32")
    d["hora_lleg"] = (arr_hm // 60).astype("int8")
    d["dia_semana"] = d["fecha"].dt.dayofweek.astype("int8")

    # Días al festivo federal más cercano. Las vísperas y el día después saturan los
    # aeropuertos (Thanksgiving, Navidad, July 4th...).
    festivos = pd.DatetimeIndex(USFederalHolidayCalendar().holidays(
        start=d["fecha"].min() - pd.Timedelta(days=30),
        end=d["fecha"].max() + pd.Timedelta(days=30)))
    dias = d["fecha"].dt.normalize().to_numpy().astype("datetime64[D]").astype("int32")
    fest = np.sort(festivos.to_numpy().astype("datetime64[D]").astype("int32"))
    # Búsqueda binaria `searchsorted`: con 8 millones de filas una matriz de
    # distancias (`abs(dias[:, None] - fest[None, :])`) ocuparía cientos de MB
    # y se cae en memoria. Esto es O(N log F) en vez de O(N·F).
    pos = np.searchsorted(fest, dias)
    izq = np.abs(dias - fest[np.clip(pos - 1, 0, len(fest) - 1)])
    der = np.abs(dias - fest[np.clip(pos, 0, len(fest) - 1)])
    d["dias_a_festivo"] = np.clip(np.minimum(izq, der), 0, 15).astype("int8")

    # --- Congestión programada ---
    # Vuelos programados en el mismo aeropuerto y en la misma hora absoluta.
    # Se cuenta sobre la programación PUBLICADA (incluidos los que luego se cancelan),
    # porque el número de cancelados no se conoce de antemano.
    hora_abs = dep_min // 60
    d["vuelos_origen_hora"] = (d.groupby([d["ORIGIN"].astype(str), hora_abs], observed=True)
                                ["fecha"].transform("size").astype("int32"))
    hora_lleg_abs = arr_prog_min // 60
    d["vuelos_destino_hora"] = (d.groupby([d["DEST"].astype(str), hora_lleg_abs], observed=True)
                                 ["fecha"].transform("size").astype("int32"))

    # --- Rotación de la aeronave ---
    # Un avión que llega tarde en su tramo A→B sale tarde en el siguiente B→C.
    # Esta propagación es la variable más informativa del proyecto. No viene en los
    # datos: hay que ordenar por (TAIL_NUM, hora de salida) y usar las columnas
    # desplazadas (shift 1) para encontrar «el tramo inmediatamente anterior del
    # mismo avión».
    orden = np.lexsort((dep_min, pd.Categorical(d["TAIL_NUM"]).codes))
    tail = pd.Categorical(d["TAIL_NUM"]).codes[orden]
    mismo_avion = np.r_[False, tail[1:] == tail[:-1]]

    llegada_prev = np.r_[np.nan, arr_prog_min[orden][:-1]]
    real_prev = np.r_[np.nan, arr_real_min[orden][:-1]]
    salida = dep_min[orden].astype(float)

    # `holgura_prog`: minutos entre la llegada programada del tramo previo y la
    # salida programada de este tramo. Solo depende del itinerario publicado.
    holgura = np.where(mismo_avion, salida - llegada_prev, np.nan)

    # ANTIFUGA CRÍTICA: `ret_prev_llegada` solo se llena si el tramo anterior YA
    # había aterrizado antes de ESTA salida programada. Si el inbound sigue en el
    # aire, el dato no existe todavía y la bandera prev_disponible queda en 0.
    conocido = mismo_avion & (real_prev <= salida)
    retraso_prev = np.where(conocido, real_prev - llegada_prev, np.nan)

    # Número de tramo de esa aeronave ese día (1, 2, 3...).
    dia_orden = (dep_min[orden] // 1440)
    tramo = (pd.Series(1, index=np.arange(len(orden)))
               .groupby([tail, dia_orden]).cumcount().to_numpy() + 1)

    # Desordenar: volvemos al orden original del DataFrame.
    desordenar = np.empty(len(orden), dtype="int64")
    desordenar[orden] = np.arange(len(orden))
    d["holgura_prog"] = np.clip(holgura[desordenar], 0, 1440).astype("float32")
    # Recorte con constantes FIJAS (ver `RECORTE_PREV`).
    d["ret_prev_llegada"] = np.clip(retraso_prev[desordenar], *RECORTE_PREV).astype("float32")
    # El NaN *es* información: el inbound aún no aterrizó. La bandera lo captura.
    d["prev_disponible"] = (~np.isnan(retraso_prev[desordenar])).astype("int8")
    d["tramo_del_dia"] = np.clip(tramo[desordenar], 1, 12).astype("int8")

    d[cfg.OBJETIVO] = d["ARR_DEL15"]
    return d


def tabla_modelado(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra los vuelos sin etiqueta y deja solo las columnas que pasan al paso 5.

    Los cancelados y desviados ya cumplieron su función en la congestión y la
    rotación (sus horarios programados). A partir de aquí no tienen qué aportar
    porque no tienen etiqueta.
    """
    columnas = (["fecha", cfg.OBJETIVO, "DEP_DELAY", "hora_sal"]
                + cfg.NUMERICAS + cfg.CATEGORICAS)
    columnas = list(dict.fromkeys(columnas))  # sin repetidos, en orden
    tabla = df.loc[df["tiene_etiqueta"], columnas].reset_index(drop=True)
    tabla[cfg.OBJETIVO] = tabla[cfg.OBJETIVO].astype("int8")
    return tabla


def armar_X(d: pd.DataFrame) -> pd.DataFrame:
    """Matriz de entrada al Pipeline. Las categóricas van como `str`."""
    X = d[cfg.NUMERICAS + cfg.CATEGORICAS].copy()
    for c in cfg.CATEGORICAS:
        X[c] = X[c].astype(str)
    return X


def verificar_sin_fuga() -> None:
    """Hace fallar el notebook si alguna columna prohibida se coló entre las variables.

    Es la primera barrera anti-fuga: si en `cfg.NUMERICAS` o `cfg.CATEGORICAS` aparece
    cualquier nombre de `cfg.PROHIBIDAS` (por accidente al editar el archivo), el
    assert aborta todo. Más vale una excepción temprana que un AUC inflado.
    """
    fugadas = cfg.PROHIBIDAS & set(cfg.NUMERICAS + cfg.CATEGORICAS)
    assert not fugadas, f"¡Columna con fuga entre las variables!: {fugadas}"
    print(f"Verificación anti-fuga superada: {len(cfg.NUMERICAS)} numéricas + "
          f"{len(cfg.CATEGORICAS)} categóricas, ninguna prohibida.")


def tabla_decisiones() -> pd.DataFrame:
    """Tabla «antes y después»: qué entró, qué se creó y qué se descartó, con su motivo."""
    FUGA = "DESCARTADA · fuga (se mide durante o después del vuelo)"
    filas = [
        ("YEAR / MONTH / DAY_OF_MONTH / FL_DATE", "se transforma", "fecha · dia_semana · dias_a_festivo"),
        ("OP_UNIQUE_CARRIER", "ENTRA (categórica)", "codificación por objetivo dentro del Pipeline"),
        ("ORIGIN / DEST", "ENTRAN (categóricas)", "codificación por objetivo dentro del Pipeline"),
        ("CRS_DEP_TIME", "se transforma", "hora_sal · hora_sal_dec"),
        ("CRS_ARR_TIME", "se transforma", "hora_lleg · desfase horario"),
        ("CRS_ELAPSED_TIME / DISTANCE", "ENTRAN (numéricas)", "imputación y escalado dentro del Pipeline"),
        ("TAIL_NUM", "se usa, no entra", "ret_prev_llegada · holgura_prog · tramo_del_dia"),
        ("OP_CARRIER_FL_NUM", "DESCARTADA · identificador", "solo sirve para detectar duplicados"),
        ("ARR_DEL15", "ES LA ETIQUETA", cfg.OBJETIVO),
        ("ARR_DELAY", FUGA, "el del tramo ANTERIOR sí se usa: ret_prev_llegada"),
        ("DEP_DELAY / DEP_TIME / DEP_DEL15", FUGA, "DEP_DELAY se conserva solo para la demostración"),
        ("TAXI_* / WHEELS_* / ARR_TIME / AIR_TIME / ACTUAL_ELAPSED_TIME", FUGA, "no se leen de los CSV"),
        ("CARRIER_ / WEATHER_ / NAS_ / SECURITY_ / LATE_AIRCRAFT_DELAY", FUGA + " y > 50 % nulos",
         "son la CAUSA del retraso: solo existen si ya hubo retraso"),
        ("CANCELLED / DIVERTED", "filtro, no entran", "marcan los vuelos sin etiqueta"),
        ("(nueva) vuelos_origen_hora / vuelos_destino_hora", "ENTRAN", "congestión programada del aeropuerto"),
        ("(nueva) tramo_del_dia / holgura_prog", "ENTRAN", "rotación de la aeronave"),
        ("(nueva) ret_prev_llegada / prev_disponible", "ENTRAN", "propagación del retraso + bandera de faltante"),
        ("(nueva) dias_a_festivo", "ENTRA", "las vísperas de festivo saturan los aeropuertos"),
    ]
    return pd.DataFrame(filas, columns=["columna original", "decisión", "en qué se convierte"])


def comparar_imputacion(df: pd.DataFrame, semilla: int = cfg.SEMILLA,
                        n: int = 60_000, oculto: float = 0.25) -> pd.DataFrame:
    """Experimento controlado de imputación, estilo «edad del Titanic».

    `ret_prev_llegada` tiene NaN cuando no hay tramo anterior o éste aún no había
    aterrizado. Para medir qué método imputa mejor:
      1. Tomamos los vuelos donde la variable SÍ existe (prev_disponible = 1).
      2. Ocultamos a propósito el 25 % de esos valores.
      3. Estimamos con cuatro métodos y medimos el MAE en minutos.
      4. Gana el que menos se equivoca.
    """
    rng = np.random.default_rng(semilla)
    con_prev = df.loc[df["prev_disponible"] == 1]
    base = con_prev.sample(min(n, len(con_prev)), random_state=semilla).reset_index(drop=True)

    v = base["ret_prev_llegada"].to_numpy(dtype=float)
    tapado = rng.random(len(base)) < oculto
    visto = ~tapado

    mae = {
        "Media global": mean_absolute_error(v[tapado], np.full(tapado.sum(), v[visto].mean())),
        "Mediana global": mean_absolute_error(v[tapado], np.full(tapado.sum(), np.median(v[visto]))),
    }

    # Imputación por grupo (hora de salida): la distribución del retraso cambia con la hora.
    med_hora = base.loc[visto].groupby("hora_sal")["ret_prev_llegada"].median()
    pred = base.loc[tapado, "hora_sal"].map(med_hora).fillna(np.median(v[visto])).to_numpy()
    mae["Mediana por hora de salida"] = mean_absolute_error(v[tapado], pred)

    # Imputación por regresión (HistGradientBoosting con varias columnas auxiliares).
    auxiliares = ["hora_sal_dec", "DISTANCE", "holgura_prog", "vuelos_origen_hora",
                  "dias_a_festivo", "tramo_del_dia", "dia_semana"]
    Xr = base[auxiliares].copy()
    Xr["aerolinea"] = pd.Categorical(base["OP_UNIQUE_CARRIER"]).codes
    reg = HistGradientBoostingRegressor(max_iter=100, random_state=semilla)
    reg.fit(Xr[visto], v[visto])
    mae["Regresión (HistGB)"] = mean_absolute_error(v[tapado], reg.predict(Xr[tapado]))

    tabla = pd.Series(mae).sort_values().to_frame("MAE (min)").round(3)
    tabla.attrs["desviacion"] = float(v.std())
    return tabla
