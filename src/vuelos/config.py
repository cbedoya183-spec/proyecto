"""Todo lo que se puede ajustar del taller vive aquí. Si algo cambia, se cambia en un solo lugar."""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[2]
CRUDOS = RAIZ / "datos" / "crudos"          # archivos tal como salen de Kaggle (no se versionan)
PROCESADOS = RAIZ / "datos" / "procesados"  # tabla principal armada por nosotros (no se versiona)
EXTERNOS = RAIZ / "datos" / "externos"      # tablas auxiliares (aeropuertos)
MUESTRA = RAIZ / "datos" / "muestra"        # muestra pequeña, SÍ se versiona: sirve para correr en vivo
MODELOS = RAIZ / "modelos_guardados"
RESULTADOS = RAIZ / "resultados"

# ── Fuente de datos ──────────────────────────────────────────────────────────
KAGGLE_DATASET = "umerhafeezsiddiqui/2021-2026-us-flights"
URL_AEROPUERTOS = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# ── Ventana de tiempo del taller (incluye ambos extremos) ────────────────────
FECHA_INICIO = "2025-05-01"
FECHA_FIN = "2026-05-31"

# ── Reproducibilidad y tamaños ───────────────────────────────────────────────
SEMILLA = 42
N_MODELOS = 100_000   # filas que se usan para comparar modelos (la tabla completa se usa para explorar)
N_RAPIDO = 20_000     # filas del modo rápido (--rapido): lo que se corre en vivo en la exposición

# ── El problema ──────────────────────────────────────────────────────────────
# Etiqueta: ¿el vuelo llega con 15 minutos o más de retraso? (definición usual del DOT)
# Momento de la predicción: ANTES de que el vuelo despegue. Solo se puede usar lo que ya se sabe en ese momento.
UMBRAL_RETRASO = 15
OBJETIVO = "retraso15"

# ── Modelos ──────────────────────────────────────────────────────────────────
# XGBoost, SVM y red neuronal son de las sesiones 7 y 8. Se activan cuando el curso los haya visto.
INCLUIR_S7_S8 = False

# ── Columnas del archivo de Kaggle ───────────────────────────────────────────
# IMPORTANTE: los nombres de la IZQUIERDA son un punto de partida (convención del BTS de EE. UU.).
# Se confirman en notebooks/01_problema_y_datos.ipynb con df.columns y se corrigen aquí.
# El resto del código solo usa los nombres internos (derecha).
MAPA_COLUMNAS = {
    "FlightDate": "fecha",
    "Airline": "aerolinea",
    "Origin": "origen",
    "Dest": "destino",
    "CRSDepTime": "salida_prog",       # hora de salida PROGRAMADA, formato hhmm (1435 = 2:35 pm)
    "CRSArrTime": "llegada_prog",      # hora de llegada PROGRAMADA
    "CRSElapsedTime": "duracion_prog", # duración programada (min)
    "Distance": "distancia",           # millas
    "Cancelled": "cancelado",
    "Diverted": "desviado",
    "ArrDelay": "retraso_llegada",     # minutos: de aquí sale la etiqueta
    "DepDelay": "retraso_salida",      # minutos: FUGA (solo se conoce al despegar)
}

# Columnas que NUNCA entran como entrada del modelo (se conocen solo después del evento)
COLUMNAS_FUGA = ["retraso_llegada", "retraso_salida", "cancelado", "desviado"]

# ── Entradas del modelo ──────────────────────────────────────────────────────
# ENTRADAS: columnas de la tabla principal que recibe el Pipeline.
# NUM y CAT: columnas que el Pipeline le entrega al modelo, ya con las transformaciones de columnas.py.
ENTRADAS = [
    "fecha", "aerolinea", "origen", "destino", "salida_prog", "llegada_prog", "duracion_prog", "distancia",
    "vuelos_origen_hora", "vuelos_origen_dia",
    "lat_origen", "lon_origen", "elev_origen", "lat_destino", "lon_destino", "elev_destino",
]
NUM = [
    "log_distancia", "duracion_prog", "hora_salida", "hora_llegada", "fin_de_semana", "dias_a_festivo",
    "vuelos_origen_hora", "vuelos_origen_dia",
    "lat_origen", "lon_origen", "elev_origen", "lat_destino", "lon_destino", "elev_destino",
]
CAT = ["aerolinea", "origen", "destino", "franja_salida", "mes", "dia_semana"]
