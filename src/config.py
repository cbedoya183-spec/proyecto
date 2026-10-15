"""Todo lo ajustable del proyecto vive aquí. Si algo cambia, se cambia en un solo lugar."""

from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "datos" / "crudos"          # CSV tal como salen de Kaggle (no se versionan: pesan demasiado)
PROCESADOS = RAIZ / "datos" / "procesados"  # caché en parquet de la tabla ya limpia
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RESULTADOS / "figuras"

for _carpeta in (CRUDOS, PROCESADOS, MODELOS, RESULTADOS, FIGURAS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# ── Fuente de datos ──────────────────────────────────────────────────────────
KAGGLE_DATASET = "umerhafeezsiddiqui/2021-2026-us-flights"
URL_KAGGLE = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
SEMILLA = 42

# ── El problema ──────────────────────────────────────────────────────────────
# Etiqueta: ¿el vuelo llega con 15 minutos o más de retraso? (definición del US DOT)
# Momento de la predicción: a la hora de salida PROGRAMADA, antes de que el avión despegue.
UMBRAL_RETRASO = 15
OBJETIVO = "retraso15"

# ── Periodo del estudio (12 meses calendario completos) ──────────────────────
MES_INICIO, MES_FIN = "2025-06", "2026-05"

# ── Partición y tamaños ──────────────────────────────────────────────────────
FRACCION_PRUEBA = 0.15   # el último 15 % del calendario (~2 meses) es la prueba final

# MODO_RAPIDO = True es el modo de la exposición en vivo (unos minutos).
# MODO_RAPIDO = False reproduce los números del informe (tarda bastante más).
MODO_RAPIDO = True

if MODO_RAPIDO:
    N_CV, N_FINAL, N_TEST, K_PLIEGUES = 60_000, 200_000, 200_000, 3
else:
    N_CV, N_FINAL, N_TEST, K_PLIEGUES = 300_000, None, None, 5   # None = todas las filas

# ── Columnas ─────────────────────────────────────────────────────────────────
# Lo que el modelo puede ver: solo lo que se conoce a la hora de salida programada.
NUMERICAS = [
    "DISTANCE", "CRS_ELAPSED_TIME", "hora_sal_dec", "hora_lleg", "dias_a_festivo",
    "vuelos_origen_hora", "vuelos_destino_hora", "tramo_del_dia",
    "holgura_prog", "ret_prev_llegada", "prev_disponible",
]
CATEGORICAS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "dia_semana"]

# Lo que NUNCA puede entrar: se mide durante o después del vuelo.
PROHIBIDAS = {
    "DEP_DELAY", "DEP_TIME", "DEP_DEL15", "ARR_DELAY", "ARR_DEL15", "ARR_TIME",
    "TAXI_IN", "TAXI_OUT", "WHEELS_ON", "WHEELS_OFF", "AIR_TIME", "ACTUAL_ELAPSED_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "CANCELLED", "DIVERTED", OBJETIVO, "fecha",
}

# Columnas que se leen de los CSV crudos (el resto se descarta al leer).
COLS_CRUDAS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "FL_DATE",
    "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
    "CANCELLED", "DIVERTED", "ARR_DELAY", "ARR_DEL15", "DEP_DELAY",
]
CATEGORICAS_CRUDAS = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN", "DEST"]

# Identifica un vuelo de forma única (sirve para detectar duplicados).
LLAVE_VUELO = ["YEAR", "MONTH", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER",
               "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"]
