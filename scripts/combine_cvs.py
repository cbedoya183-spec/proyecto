
import pandas as pd
import glob
import os

# 1. Carpeta donde están todos los archivos CSV
carpeta = r"C:\Users\HP\Desktop\Proyecto-final-Machine-Learning\datos\crudos"

# 2. Buscar todos los archivos CSV de la carpeta
archivos = glob.glob(os.path.join(carpeta, "*.csv"))

# 3. Leer y unir todos los archivos
lista_datos = []

for archivo in archivos:
    df = pd.read_csv(archivo, encoding="utf-8")
    lista_datos.append(df)

# 4. Concatenar todos los DataFrames
df_unificado = pd.concat(lista_datos, ignore_index=True)

# 5. Guardar el resultado en un solo CSV
ruta_salida = os.path.join(carpeta, "datos_anuales_unificados.csv")
df_unificado.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

# 6. Mostrar resultados
print(f"Archivos unidos: {len(archivos)}")
print(f"Total de registros: {len(df_unificado)}")
print(f"Archivo guardado en: {ruta_salida}")