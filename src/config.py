"""Todo lo ajustable del proyecto vive aquí.

Si algo cambia (una ruta, la semilla, el periodo del estudio, la lista de columnas que
entra al modelo), se cambia en este archivo y el resto del código lo toma automáticamente.
No hay que tocar el notebook para cambiar un parámetro.
"""

from pathlib import Path

# ── Rutas del proyecto ──────────────────────────────────────────────────────
# `Path(__file__)` es la ruta de este archivo; `.parents[1]` sube a la raíz del repo.
RAIZ = Path(__file__).resolve().parents[1]

# CSV crudos descargados de Kaggle (varios GB; NO se versionan, están en .gitignore).
CRUDOS = RAIZ / "data" / "raw"

# Caché en parquet de la tabla ya limpia. Acelera recorridas sucesivas del notebook.
PROCESADOS = RAIZ / "data" / "processed"

# Modelo elegido guardado con joblib y tablas de resultados.
MODELOS = RAIZ / "models"

# Tablas CSV, JSON de resumen y figuras PNG que produce el notebook.
RESULTADOS = RAIZ / "reports"
FIGURAS = RESULTADOS / "figuras"

for _carpeta in (CRUDOS, PROCESADOS, MODELOS, RESULTADOS, FIGURAS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# ── Fuente de datos ─────────────────────────────────────────────────────────
# Catálogo público en Kaggle que reproduce el reporte oficial del Bureau of
# Transportation Statistics (US DOT) sobre puntualidad de vuelos.
KAGGLE_DATASET = "umerhafeezsiddiqui/2021-2026-us-flights"
URL_KAGGLE = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"

# ── Reproducibilidad ────────────────────────────────────────────────────────
# Semilla única para todo: muestreos, cortes temporales, inicialización de los modelos.
SEMILLA = 42

# ── El problema ─────────────────────────────────────────────────────────────
# Clasificación binaria. Un vuelo está retrasado si llega 15 o más minutos tarde que su
# itinerario (ARR_DEL15 = 1). Es la definición oficial del DOT. La predicción se emite
# a la hora de salida programada (CRS_DEP_TIME), antes de que el avión despegue.
UMBRAL_RETRASO = 15
OBJETIVO = "retraso15"

# ── Periodo del estudio ─────────────────────────────────────────────────────
# 12 meses calendario = un ciclo estacional completo: verano, otoño, invierno
# (incluye Acción de Gracias, Navidad) y primavera (incluye Spring Break).
MES_INICIO, MES_FIN = "2025-06", "2026-05"

# ── Partición y tamaños ─────────────────────────────────────────────────────
# El último 15 % del calendario (unos 2 meses) se guarda como prueba final.
# No se mira durante el entrenamiento; se usa una sola vez al final.
FRACCION_PRUEBA = 0.15

# MODO_RAPIDO = True   -> muestreo que permite correr toda la exposición en pocos minutos.
# MODO_RAPIDO = False  -> números del informe completo; tarda más.
MODO_RAPIDO = True

if MODO_RAPIDO:
    N_CV, N_FINAL, N_TEST, K_PLIEGUES = 60_000, 200_000, 200_000, 3
else:
    N_CV, N_FINAL, N_TEST, K_PLIEGUES = 300_000, None, None, 5  # None = todas las filas

# ── Columnas ────────────────────────────────────────────────────────────────
# Lo que el modelo puede ver: solo lo que se conoce a la hora de salida programada.
# Las columnas creadas por nosotros (ret_prev_llegada, holgura_prog, vuelos_*_hora,
# tramo_del_dia, dias_a_festivo) usan únicamente información del itinerario y, en el
# caso de la propagación de retraso del tramo anterior, exigen que el tramo previo
# ya haya aterrizado antes de la salida programada de este vuelo.
NUMERICAS = [
    "DISTANCE",            # millas del itinerario
    "CRS_ELAPSED_TIME",    # duración programada en minutos
    "hora_sal_dec",        # hora de salida programada, en horas decimales
    "hora_lleg",           # hora de llegada programada, entero (0..23)
    "dias_a_festivo",      # días al festivo federal más cercano (saturan los aeropuertos)
    "vuelos_origen_hora",  # congestión programada del aeropuerto de origen en esa hora
    "vuelos_destino_hora", # congestión programada del aeropuerto de destino en esa hora
    "tramo_del_dia",       # número de tramo de la aeronave ese día (1, 2, 3...)
    "holgura_prog",        # minutos entre la llegada programada previa y la salida
    "ret_prev_llegada",    # retraso real del tramo anterior (si ya aterrizó)
    "prev_disponible",     # bandera: ¿el tramo anterior ya había aterrizado?
]
CATEGORICAS = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "dia_semana"]

# Columnas prohibidas: se miden DURANTE o DESPUÉS del vuelo. Si alguna se cuela entre
# NUMERICAS/CATEGORICAS, la función `columnas.verificar_sin_fuga` hace fallar el notebook.
PROHIBIDAS = {
    "DEP_DELAY", "DEP_TIME", "DEP_DEL15",            # al salir de puerta
    "ARR_DELAY", "ARR_DEL15", "ARR_TIME",            # al llegar
    "TAXI_IN", "TAXI_OUT", "WHEELS_ON", "WHEELS_OFF",# tiempos en tierra y en el aire
    "AIR_TIME", "ACTUAL_ELAPSED_TIME",               # ya en vuelo o aterrizado
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",   # causas calculadas DESPUÉS del arribo
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "CANCELLED", "DIVERTED",                         # resultado del vuelo
    OBJETIVO, "fecha",                               # etiqueta y columna auxiliar
}

# Columnas que SÍ se leen de los CSV crudos. El resto se descarta al leer para no
# cargar en memoria datos que no vamos a usar.
# IMPORTANTE: DEP_DELAY y ARR_DELAY se leen SOLO para auditoría y para construir
# el retraso del tramo anterior (ret_prev_llegada). NO entran al modelo nunca.
COLS_CRUDAS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "FL_DATE",
    "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
    "CANCELLED", "DIVERTED",
    "ARR_DELAY", "ARR_DEL15", "DEP_DELAY",
]
CATEGORICAS_CRUDAS = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "ORIGIN", "DEST"]

# Identifica un vuelo de forma única (sirve para detectar duplicados exactos).
LLAVE_VUELO = ["YEAR", "MONTH", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER",
               "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"]

# ── Autoría ─────────────────────────────────────────────────────────────────
AUTORES = [
    "Sebastián Olaya Pérez",
    "Carlos Daniel Bedoya Moreno",
    "Santiago Felipe Espinel León",
    "Julián Alzate Aristizábal",
]
