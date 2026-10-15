"""Une todos los CSV mensuales de datos/crudos en un solo archivo: datos_anuales_unificados.csv

Se corre UNA vez, antes del notebook:   python scripts/combine_cvs.py
(Misma lógica de siempre; solo cambian la ruta, que ahora es relativa al repo, y que el archivo de
salida se excluye de la lectura para que volver a correr el script no duplique las filas.)
"""
from pathlib import Path

import pandas as pd

# 1. Carpeta donde están todos los archivos CSV (relativa al repositorio, funciona en cualquier PC)
RAIZ = Path(r"C:\Users\HP\Desktop\Proyecto-final-Machine-Learning\scripts\combine_cvs.py").resolve().parents[1]
carpeta = RAIZ / "datos" / "crudos"
NOMBRE_SALIDA = "datos_anuales_unificados.csv"

# 2. Buscar todos los CSV mensuales (se excluye el unificado: si ya existe de una corrida anterior,
#    leerlo junto con los mensuales duplicaría todas las filas)
archivos = sorted(p for p in carpeta.glob("*.csv") if p.name != NOMBRE_SALIDA)
if not archivos:
    raise SystemExit(f"No hay CSV mensuales en {carpeta}. Descárgalos de Kaggle y déjalos ahí.")

# 3. Leer y unir todos los archivos
lista_datos = []
for archivo in archivos:
    df = pd.read_csv(archivo, encoding="utf-8")
    lista_datos.append(df)
    print(f"  {archivo.name:<22} {len(df):>9,} filas")

# 4. Concatenar todos los DataFrames
df_unificado = pd.concat(lista_datos, ignore_index=True)

# 5. Guardar el resultado en un solo CSV
ruta_salida = carpeta / NOMBRE_SALIDA
df_unificado.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

# 6. Mostrar resultados
print(f"Archivos unidos: {len(archivos)}")
print(f"Total de registros: {len(df_unificado):,}")
print(f"Archivo guardado en: {ruta_salida}")