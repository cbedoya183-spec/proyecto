# Datos

## Fuente principal
- **Página:** https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights
- **Quién publica:** TODO (autor en Kaggle y organismo de origen; confirmarlo en la pestaña *Data* de la página)
- **Licencia:** TODO (campo *License* de la página)
- **Descarga:** `python scripts/00_descargar.py` (usa `kagglehub`; si pide credenciales, crear un token en Kaggle → *Settings* → *API* y guardar `kaggle.json` en `~/.kaggle/`; **nunca** subirlo al repositorio)
- **Uso en el taller:** solo mayo 2025 – mayo 2026. Los archivos comparten columnas y se unen en una sola tabla (`datos.cargar_ventana`).
- **Qué es una fila:** TODO · **Filas:** TODO · **Columnas:** TODO (se rellenan al correr el notebook 01)

## Fuente secundaria
- **OurAirports** (dominio público): https://davidmegginson.github.io/ourairports-data/airports.csv → `externos/airports.csv`

## Qué hay en cada carpeta
| Carpeta | Contenido | ¿Va a GitHub? |
|---|---|---|
| `crudos/` | Archivos de Kaggle sin tocar | No (pesan demasiado) |
| `externos/` | Tabla de aeropuertos | No (se descarga) |
| `procesados/` | `vuelos.parquet`, la tabla principal (1 fila por vuelo) | No |
| `muestra/` | `vuelos_muestra.parquet`, 20.000 filas de la tabla principal | **Sí** |
