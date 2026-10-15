# Qué se corrigió y qué se tomó de cada versión

## Errores corregidos de la versión original (`v1_Mio`)

| # | Error | Por qué importa | Cómo quedó |
|---|---|---|---|
| 1 | Se imputaban `ARR_DELAY`, `ARR_DEL15` y `DEP_DELAY` con la mediana | **El más grave.** `ARR_DEL15` es la etiqueta: imputarla le inventa un valor a cada vuelo cancelado y mete ruido falso en el objetivo. Además imputa `ARR_DELAY`, que es lo que se usa para reconstruir la llegada del tramo anterior: le inventa una hora de aterrizaje a un avión que nunca aterrizó. | No se imputa nada de eso. Los vuelos sin etiqueta se marcan con `tiene_etiqueta` y se filtran al final del paso 4 |
| 2 | Se imputaban `WHEELS_ON`, `WHEELS_OFF`, `TAXI_OUT`, `TAXI_IN`, `ARR_TIME`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME` | Son fuga: se miden durante el vuelo. Limpiarlas da a entender que podrían entrar al modelo | Ni se leen de los CSV. `config.PROHIBIDAS` + un `assert` que hace fallar el notebook si alguna se cuela |
| 3 | `SimpleImputer().fit_transform()` aplicado al `df` entero | Ajusta la mediana con todas las filas, incluidas las de prueba: fuga entre partición | Toda imputación, escalado y codificación va **dentro del `Pipeline`**, que se reajusta en cada pliegue |
| 4 | `hhmm_a_min` sin `% 1440` | El BTS usa `2400` para medianoche; sin el módulo devuelve 1440, una hora que no existe, y rompe el orden de la rotación | `((s // 100) * 60 + s % 100) % 1440` |
| 5 | Origen de tiempo fijo en `pd.Timestamp("2025-01-01")` | Si cambia el periodo del estudio, los minutos se vuelven negativos | El origen se toma de `d["fecha"].min()` |
| 6 | Umbral de decisión en 0,50 por defecto | Con 20-25 % de retrasos el modelo se queda casi mudo y el F1 se hunde | Umbral que maximiza el F1, buscado en el último pliegue de **validación**, nunca en la prueba |
| 7 | Filtros con constantes absolutas (`size > 20_000`) en las gráficas | Con muestra reducida la gráfica sale vacía y nadie se entera | Se usa la mediana del tamaño de grupo |
| 8 | Autoría equivocada en la cabecera de `v2` («Sebastián Olaya») | En un trabajo calificado es un problema real | Corregido |
| 9 | Cálculo de `dias_a_festivo` con una matriz `n × festivos` | Con 8 millones de filas son cientos de MB y se cae en memoria | Búsqueda binaria con `np.searchsorted` |
| 10 | `train_test_split` aleatorio (en la primera versión) | Entrenaría con marzo para predecir enero: ver el futuro | Corte temporal + `TimeSeriesSplit` |

## Qué se conservó de tu versión (lo bueno)

Todo esto es lo que hace que el proyecto puntúe alto en **complejidad** (25 %) y en **datos y
transformación** (25 %), y no estaba en el ejemplo de Sebastián:

- **La rotación de la aeronave.** `ret_prev_llegada`, `holgura_prog` y `tramo_del_dia`: no vienen en los datos,
  hay que ordenarlos por `TAIL_NUM` y deducir el desfase de husos horarios del propio itinerario. Es la tabla
  principal «armada por ustedes» que pide la lámina 5.
- **La congestión programada** del aeropuerto por hora, contada sobre vuelos programados —incluidos los que
  luego se cancelan— porque las cancelaciones no se conocen de antemano.
- **La demostración de fuga en vivo**: AUC con y sin `DEP_DELAY`. Es exactamente el ejemplo de los seguros de
  la lámina 6, pero con los datos propios.
- **El experimento de faltantes** con cuatro métodos y el 25 % de valores ocultos, igual que la edad en el
  Titanic. Mide el error en lugar de suponerlo.
- **La regla a mano escrita como estimador de sklearn**, que pasa por la misma validación cruzada que los
  modelos. Es más honesto que calcularla aparte.
- **La validación temporal estricta** y la prueba final tocada una sola vez.
- **La bandera `prev_disponible`**: el faltante no es aleatorio y eso informa.
- **El resumen reproducible** en JSON al final.

## Qué se tomó del ejemplo de Sebastián

- **La estructura.** El código vive en `src/` repartido en cinco módulos que siguen los seis pasos; el
  notebook solo cuenta la historia y muestra resultados. Pasó de 60 celdas con 59 000 caracteres a **34 celdas
  con unas 20 líneas de código cada una**. En la exposición se lee lo que hay en pantalla, no se bucea en una
  celda de 80 líneas.
- **`config.py` como único lugar de ajuste**: periodo, semilla, tamaños, listas de columnas.
- **La narrativa por paso**: cada bloque arranca con un markdown que explica la decisión *antes* de la celda
  que la ejecuta, no después.
- **Una sola tabla de métricas** donde conviven los modelos, la línea base y la regla a mano, con la columna
  de validación como la que decide y la de prueba como reporte.
- **La regla de desempate explícita** («dentro de una desviación típica, gana el más simple»), fijada antes de
  mirar la prueba. Es la respuesta directa a la lámina 9: *no siempre gana el número más alto*.
- **La auditoría de la tabla cruda** y la tabla «antes y después» como un objeto comparable (`datos.foto`).

## Lo que falta hacer antes de la exposición

1. Descargar los CSV de Kaggle a `datos/crudos/` y correr el notebook completo en modo rápido.
2. Rellenar los `✍️ COMPLETAR` de `docs/ficha_de_datos.md` (licencia y fecha de descarga) y los `____` de
   `docs/decisiones.md` con los números de la corrida.
3. Escribir las cinco conclusiones de la última celda con los números reales.
4. Poner los nombres del equipo en el `README.md` y en la cabecera del notebook.
5. Ensayar `Kernel → Restart & Run All` cronometrado. Si tarda demasiado, bajar `N_CV` en `src/config.py`.
