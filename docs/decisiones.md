# Bitácora de decisiones (pasos 3 y 4)

Cada decisión se imprime con su conteo en el notebook. Copiar aquí las cifras de **la corrida propia**.

## Limpieza

1. **Nulos por columna.** Las cinco columnas de causas del retraso (`CARRIER_DELAY`, `WEATHER_DELAY`,
   `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`) superan el 50 % de nulos porque solo se llenan cuando
   ya hubo retraso. Además de vacías son fuga: no se leen del CSV.
2. **Cancelados y desviados.** No tienen etiqueta. Se marcan con `tiene_etiqueta` y se filtran AL FINAL del
   paso 4, porque sus horarios programados cuentan para la congestión y la rotación.
   Cancelados: ____ % · Desviados: ____ %
   ⚠️ **El error que hay que evitar:** imputarles `ARR_DELAY`/`ARR_DEL15` con la mediana. Eso inventa una
   etiqueta falsa y, peor, inventa una hora de llegada al tramo anterior de la aeronave, que es la variable
   más predictiva del proyecto.
3. **Itinerarios imposibles** (sin hora, duración o distancia; valores ≤ 0; horas > 2400): se eliminan.
   Filas eliminadas: ______
4. **Coherencia de la etiqueta.** `ARR_DEL15` coincide con `(ARR_DELAY >= 15)` en ____ % de las filas.
5. **Duplicados** (fecha + aerolínea + número de vuelo + origen + destino + hora): ______ filas.
6. **Formatos.** Fecha armada con `YEAR`/`MONTH`/`DAY_OF_MONTH` (`FL_DATE` es ambiguo). Horas `hhmm` a
   minutos con `% 1440`, de modo que el 2400 del BTS cae en 0.
7. **Atípicos.** Los retrasos de más de un día son reales y no se borran. Como variable
   (`ret_prev_llegada`) se recortan a [-60, 360] minutos con **constantes fijas**, no con percentiles
   calculados sobre los datos: por eso no hay fuga.

## Faltantes (experimento del paso 4.2)

- Variable con faltantes: `ret_prev_llegada` (no hay tramo anterior, o aún no había aterrizado).
- El faltante **no es aleatorio** → se conserva la bandera `prev_disponible`.
- Se oculta el 25 % de los valores conocidos y se mide el MAE de cuatro métodos:
  Media global: ____ · Mediana global: ____ · Mediana por hora: ____ · Regresión (HistGB): ____
  Desviación típica de la variable: ____ min
- **Decisión en el Pipeline:** mediana + bandera. Justificación: ______________

## Transformaciones dentro del Pipeline (sin fuga entre pliegues)

- Numéricas: `SimpleImputer(mediana)` y, para la logística y la red neuronal, `StandardScaler`.
- Categóricas (aerolínea, origen, destino, día de la semana): `TargetEncoder`, ajustado solo con el
  entrenamiento de cada pliegue.
- Se descarta el mes como variable: con partición temporal, los meses de la prueba no existen en el entrenamiento.

## Prevención de fugas: tres barreras

1. **Al armar la tabla** — solo pasa la lista explícita de `config.NUMERICAS` y `config.CATEGORICAS`.
2. **Al construir** — `ret_prev_llegada` solo se llena si el tramo anterior ya había aterrizado antes de esta
   salida programada; la congestión se cuenta sobre vuelos PROGRAMADOS (incluidos los que luego se cancelan),
   porque el número de cancelados no se conoce de antemano.
3. **En el Pipeline** — imputación, escalado y codificación se reajustan en cada pliegue.

Más un `assert` que hace fallar el notebook si una columna prohibida se cuela en `X`.

**Demostración de fuga:** AUC sin `DEP_DELAY` = ____ · con `DEP_DELAY` = ____

## Tabla antes / después

La genera el notebook en `resultados/tabla_antes_despues.csv` y `resultados/tabla_columnas.csv`.
