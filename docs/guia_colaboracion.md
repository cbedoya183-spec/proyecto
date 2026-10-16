# Guía para clonar y correr el proyecto

## Entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Datos

Los 12 CSV crudos van en `data/raw/` y NO se versionan (pesan ~85 MB cada uno, ~1 GB en total).
Se descargan de Kaggle: [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights).

```
data/raw/
  Jun-2025.csv
  Jul-2025.csv
  Aug-2025.csv
  Sep-2025.csv
  Oct-2025.csv
  Nov-2025.csv
  Dec-2025.csv
  Jan-2026.csv
  Feb-2026.csv
  Mar-2026.csv
  Apr-2026.csv
  May-2026.csv
```

En la primera corrida, `src/datos.cargar()` lee los 12 archivos por bloques, los concatena y guarda la caché
en `data/processed/vuelos_2025-06_2026-05.parquet` (~100 s). Las corridas siguientes arrancan desde esa caché
en menos de 2 segundos.

## Correr la cadena completa

Toda la lógica vive en `src/`; el notebook solo cuenta la historia:

```powershell
jupyter notebook notebooks/proyecto_final_vuelos_us.ipynb
# Kernel → Restart & Run All
```

En `src/config.py`:
- `MODO_RAPIDO = True`  → modo exposición (3-4 min: 60.000 filas de CV, 200.000 de prueba).
- `MODO_RAPIDO = False` → modo informe completo (tarda bastante más).

Semilla fija en 42: todos los números coinciden con los del README y la presentación.

## Qué produce el notebook

- `data/processed/vuelos_2025-06_2026-05.parquet` (caché, ignorado por git).
- `models/modelo_elegido.joblib` (Pipeline + umbral + hora de la regla, ignorado por git).
- `reports/tabla_antes_despues.csv` · `tabla_columnas.csv` · `tabla_modelos_cv.csv` · `tabla_prueba_final.csv`
  · `comparacion_imputacion.csv` · `importancia_columnas.csv` · `resumen_final.json` (versionados).
- `reports/figuras/01_hipotesis.png` · `02_comparacion_modelos.png` · `03_prueba_final.png`
  · `04_importancia.png` (versionados).

## Qué no entra al repositorio

Ver `.gitignore`:
- `data/raw/` — CSV crudos (~1 GB).
- `data/processed/*.parquet` — caché.
- `models/*.joblib` — modelos entrenados.
- `.venv/`, `__pycache__/`, etc.

Las tablas de `reports/*.csv` y el JSON de resumen SÍ se versionan: son pequeñas y son la evidencia
reproducible de la corrida.
