# Ficha de datos (pasos 1 y 2)

**Página:** https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights
**Publica:** usuario de Kaggle `umerhafeezsiddiqui`, a partir del reporte *On-Time Performance* del
Bureau of Transportation Statistics (BTS), agencia del U.S. Department of Transportation. Los nombres de
columna (`FL_DATE`, `OP_UNIQUE_CARRIER`, `CRS_DEP_TIME`, `ARR_DEL15`…) son los del formulario oficial del BTS.

- **Licencia:** ✍️ COMPLETAR con el campo «License» de la barra lateral de Kaggle. El reporte original del
  BTS es obra del gobierno federal de EE. UU.: dominio público.
- **Fecha de descarga:** ✍️ COMPLETAR.
- **Cómo se descarga:** un CSV por mes (`Jun-2025.csv` … `May-2026.csv`) desde la pestaña *Data*. Se dejan en
  `datos/crudos/`. `src/datos.py` los lee por bloques, los concatena y recorta el periodo.
- **Qué es una fila:** un vuelo programado (un tramo origen → destino, de una aerolínea, en una fecha).
  Incluye los que terminaron cancelados o desviados.
- **Tamaño:** ✍️ COMPLETAR con lo que imprime el notebook. Columnas originales: 34 (se leen 19).

## Etiqueta

`retraso15` = `ARR_DEL15` = 1 si el vuelo llegó con 15 minutos o más de retraso. Es la definición oficial del
DOT. Los cancelados y desviados no tienen llegada normal: quedan fuera de la etiqueta.

## ¿Por qué vale la pena predecirla?

- La aerolínea reasigna puertas y tripulación antes de que ocurra el problema.
- El pasajero reorganiza una conexión.
- Se predice a la hora de salida programada, así que hay tiempo de actuar.

## Regla a mano (punto de comparación)

«Si el vuelo sale a las HH:00 o después, se retrasará». La hora de corte se busca con los datos de
entrenamiento (`modelos.calibrar_regla`) y la regla se mide con la misma validación cruzada que los modelos.

## Por qué el problema es complejo (criterio «complejidad», 25 %)

- Datos crudos, no un archivo ya limpio: nulos, cancelados, desviados, horas en formato `hhmm` con el caso 2400.
- 12 archivos mensuales de varios GB: hay que leerlos por bloques y reducir tipos para que quepan en memoria.
- **Tabla principal armada por nosotros:** la rotación de cada aeronave (el retraso del tramo anterior) no
  viene en los datos; se deduce ordenando por `TAIL_NUM` y reconstruyendo los husos horarios desde el
  itinerario, sin tabla externa.
- Clases desbalanceadas (~20-25 % de retrasados) y validación temporal (entrenar con el pasado, probar con el futuro).
- Riesgo de fuga muy alto: la mayoría de columnas del archivo se miden durante el vuelo.
