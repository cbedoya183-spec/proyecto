# Guía para clonar y correr el proyecto

## Entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Datos

Los CSV crudos de `data/raw/` no van en Git: cada mes pesa cerca de 80 MB. La muestra reproducible sí puede regenerarse:

```powershell
python src/data/download_data.py --sample-fraction 0.01
python -m src.data.audit_raw
```

Hace falta un token de Kaggle solo si los CSV todavía no están en `data/raw/`.

## Entrenamiento

El notebook es la corrida oficial:

```powershell
jupyter notebook notebooks/proyecto_final_vuelos_us.ipynb
```

La misma comparación, desde la terminal:

```powershell
python src/models/train.py
```

`--quick` reduce árboles para un ensayo. Los números publicados salen de la corrida completa, con semilla 42.

## Qué no entra al repositorio

`data/raw/`, los parquet grandes y `models/*.joblib` están en `.gitignore`. `models/metrics_comparison.csv` y `reports/auditoria_mensual_cruda.csv` sí se versionan: son tablas pequeñas y son la evidencia de la evaluación.
