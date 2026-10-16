# Ficha de datos (pasos 1 y 2)

**Página:** https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights

**Publica:** usuario de Kaggle `umerhafeezsiddiqui`, a partir del reporte oficial *On-Time Performance* del
Bureau of Transportation Statistics (BTS), agencia del U.S. Department of Transportation. Los nombres de
columna (`FL_DATE`, `OP_UNIQUE_CARRIER`, `CRS_DEP_TIME`, `ARR_DEL15`...) son los del formulario oficial del BTS.

- **Licencia:** *U.S. Government Work*, dominio público (sin restricciones para investigación científica o
  uso comercial). Reproducido en Kaggle sin restricciones adicionales.
- **Cómo se descarga:** un CSV por mes (`Jun-2025.csv` ... `May-2026.csv`) desde la pestaña *Data* de Kaggle.
  Se dejan en `data/raw/`. `src/datos.py:cargar()` los lee por bloques y los concatena.
- **Qué es una fila:** un vuelo programado (un tramo origen → destino, de una aerolínea, en una fecha).
  Incluye los que terminaron cancelados o desviados.
- **Tamaño total:** 12 CSV crudos de ~85 MB cada uno. Tras unir y recortar al periodo del estudio:
  **7.786.702 filas** × 19 columnas (0,86 GB en memoria con tipos reducidos).
- **Periodo:** 1 de junio de 2025 a 31 de mayo de 2026 (12 meses, un ciclo estacional completo).
- **Fecha de descarga:** 11 de octubre de 2026.

## Etiqueta

`retraso15` = `ARR_DEL15` = 1 si el vuelo llegó con 15 minutos o más de retraso. Es la definición oficial
del U.S. DOT. Los cancelados y desviados no tienen llegada normal: quedan fuera de la etiqueta y se
filtran al final del paso 4 (sus horarios programados sí cuentan antes para la congestión y la rotación).

Tasa de retraso en la tabla de modelado (vuelos con etiqueta): **22,44 %** (`1.709.903` de `7.620.055`).

## ¿Por qué vale la pena predecirla?

- La **aerolínea** reasigna puertas y tripulación antes de que el retraso ocurra.
- El **pasajero** reorganiza una conexión con tiempo de reacción.
- Se predice **a la hora de salida programada**, así que hay tiempo real para actuar.

## Regla a mano (punto de comparación)

«Si el vuelo sale a las HH:00 o después, se retrasará». La hora de corte se busca con los datos de
entrenamiento (`modelos.calibrar_regla`) y la regla se mide con la misma validación cruzada que los
modelos. Esta corrida la calibró en **hora 12**, con F1 de 0,386 en la prueba.

## Por qué el problema es complejo (criterio «complejidad» de la rúbrica, 25 %)

- Datos crudos, no un archivo ya limpio: nulos, cancelados, desviados, horas en formato `hhmm` con el
  caso 2400, dos formatos de fecha mezclados.
- 12 archivos mensuales de ~85 MB: hay que leerlos por bloques (`chunksize=1_000_000`) y reducir tipos
  (`float64 → float32`, `int64 → int32`, códigos a `category`) para que quepan en memoria.
- **Tabla principal armada por nosotros:** la rotación de cada aeronave (el retraso de su tramo anterior,
  la holgura programada) no viene en los datos. Se deduce ordenando por `TAIL_NUM` y reconstruyendo los
  husos horarios desde el propio itinerario, sin tabla externa.
- Clases desbalanceadas (~22 % de retrasados) y **validación temporal** (entrenar con el pasado, probar
  con el futuro). Un K-fold aleatorio sería trampa.
- Riesgo de fuga muy alto: la mayoría de columnas del archivo se miden durante o después del vuelo.
  Hay 15 columnas prohibidas (`cfg.PROHIBIDAS`) con tres barreras de prevención y una demostración
  empírica en vivo que mide el salto de AUC si una de ellas se cuela.
