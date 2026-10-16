# Bitácora de decisiones (pasos 3 y 4)

Cifras de la corrida real en `MODO_RAPIDO = True`, semilla 42, 12 meses de BTS (2025-06 a 2026-05).

## Limpieza (paso 3)

1. **Nulos por columna.** Las cinco columnas de causas del retraso (`CARRIER_DELAY`, `WEATHER_DELAY`,
   `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`) superan el 50 % de nulos porque solo se llenan
   cuando ya hubo retraso. Además de vacías son fuga: **ni se leen** del CSV (`cfg.COLS_CRUDAS`).
2. **Cancelados y desviados.** No tienen etiqueta de llegada. Se marcan con `tiene_etiqueta = 0` y se
   filtran AL FINAL del paso 4, porque sus horarios programados cuentan para la congestión y la rotación.
   - Cancelados: **145.007** (1,86 %)
   - Desviados: **21.637** (0,28 %)
   - ⚠️ *El error que hay que evitar:* imputarles `ARR_DELAY` o `ARR_DEL15` con la mediana. Eso inventa
     una etiqueta falsa y, peor, una hora de llegada al tramo anterior de la aeronave, que es justamente
     la variable más predictiva del paso 4 (`ret_prev_llegada`).
3. **Itinerarios imposibles** (sin hora programada, sin duración, sin distancia; valores ≤ 0; horas > 2400):
   se eliminan. Filas eliminadas: **18**.
4. **Coherencia de la etiqueta.** `ARR_DEL15` coincide con `(ARR_DELAY >= 15)` en el **100,00 %** de las
   filas con etiqueta (verificación cruzada entre las dos columnas del BTS).
5. **Duplicados exactos** (clave: fecha + aerolínea + número de vuelo + origen + destino + hora programada):
   **0** filas.
6. **Formatos.** Fecha reconstruida con `YEAR/MONTH/DAY_OF_MONTH` (el `FL_DATE` del CSV mezcla
   `06-01-25` con `6/1/2025 12:00:00 AM`). Horas `hhmm` a minutos con `% 1440`, de modo que el `2400`
   del BTS cae en `0`.
7. **Atípicos.** Los retrasos de más de un día son reales (vuelos perdidos por tormentas severas) y no
   se borran. Como variable (`ret_prev_llegada`) se recortan a `[-60, 360]` minutos con **constantes
   fijas**, no con percentiles calculados sobre los datos: por eso el recorte no introduce fuga.

Tabla «antes y después» (ver `reports/tabla_antes_despues.csv`):

| | ANTES | DESPUÉS | cambio |
|---|---:|---:|---:|
| filas | 7.786.702 | 7.786.684 | −18 |
| columnas | 19 | 20 | +1 (bandera `tiene_etiqueta`) |
| memoria (GB) | 0,86 | 0,92 | +0,06 |
| duplicados | 0 | 0 | 0 |
| cancelados | 145.007 | 145.002 | −5 |
| desviados | 21.637 | 21.626 | −11 |

## Faltantes (experimento del paso 4.3)

- **Variable con faltantes:** `ret_prev_llegada` (no hay tramo anterior, o aún no había aterrizado).
- El faltante **no es aleatorio** (depende de la hora, del aeropuerto, del día): por eso se conserva la
  bandera `prev_disponible`, que llega a ser la **segunda variable más importante** del modelo.
- Se oculta el 25 % de los valores conocidos y se mide el MAE de cuatro métodos (ver
  `reports/comparacion_imputacion.csv`). Desviación típica de la variable: **71,76 min**.

| Método | MAE (min) |
|---|---:|
| **Mediana por hora de salida** ✅ | **18,44** |
| Mediana global | 18,57 |
| Regresión (HistGradientBoosting) | 19,94 |
| Media global | 20,25 |

- **Decisión en el Pipeline:** `SimpleImputer(strategy="median")` dentro del `ColumnTransformer`
  (se reajusta en cada pliegue). El experimento aquí sirve para **argumentar la decisión**, no para
  reemplazar la imputación por pliegue. La mediana por hora es la mejor en el experimento pero exige
  más código; la mediana global pierde apenas 0,13 min y es más simple.

## Transformaciones dentro del Pipeline (sin fuga entre pliegues)

- **Numéricas**: `SimpleImputer(strategy="median")` → `StandardScaler` (solo para la logística y el MLP).
- **Categóricas** (aerolínea, origen, destino, día de la semana): `TargetEncoder` con cross-fitting
  interno (sklearn ≥ 1.3), ajustado solo con el entrenamiento de cada pliegue.
- Se descarta el **mes** como variable: con partición temporal, los meses de la prueba no existen en el
  entrenamiento y el modelo recibiría una categoría desconocida.

## Prevención de fugas: tres barreras + una demostración

1. **Al armar la tabla** — solo pasa la lista explícita de `cfg.NUMERICAS` y `cfg.CATEGORICAS`. El resto
   ni se lee del CSV (`cfg.COLS_CRUDAS` filtra en el lector).
2. **Al construir** — `ret_prev_llegada` solo se llena si el tramo anterior del mismo avión ya había
   aterrizado antes de ESTA salida programada (`real_prev <= dep_min`). La congestión (`vuelos_*_hora`)
   se cuenta sobre vuelos PROGRAMADOS (incluidos los que luego se cancelan), porque el número de
   cancelados no se conoce de antemano.
3. **En el Pipeline** — imputación, escalado y codificación se reajustan en cada pliegue de
   `TimeSeriesSplit`.

Además, un **`assert`** en `columnas.verificar_sin_fuga()` hace fallar el notebook si una columna
prohibida se cuela en `X` (barrera defensiva).

**Demostración empírica de fuga** (celda 4.4 del notebook, ver `reports/demo_fuga` o la ejecución):
al dejar entrar `DEP_DELAY` el AUC de un `HistGradientBoosting` da el salto característico de la fuga.
Es exactamente el ejemplo canónico de la rúbrica (sesión 6, seguros: `decision_renovacion` → AUC 1,000).

## Tabla por columna

Ver `reports/tabla_columnas.csv`. Resumen por tipo:

- **Transformadas** del itinerario crudo: `fecha` (de `YEAR/MONTH/DAY_OF_MONTH`), `hora_sal` y `hora_sal_dec`
  (de `CRS_DEP_TIME`), `hora_lleg` (de `CRS_ARR_TIME`).
- **Entran al modelo**: `DISTANCE`, `CRS_ELAPSED_TIME`, `hora_sal_dec`, `hora_lleg`, `dias_a_festivo`,
  `vuelos_origen_hora`, `vuelos_destino_hora`, `tramo_del_dia`, `holgura_prog`, `ret_prev_llegada`,
  `prev_disponible`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `dia_semana` (11 numéricas + 4 categóricas).
- **Usadas pero no entran**: `TAIL_NUM` (para `ret_prev_llegada` y `tramo_del_dia`), `FL_DATE`.
- **Descartadas por fuga** (15 columnas): `DEP_DELAY`, `DEP_TIME`, `DEP_DEL15`, `ARR_DELAY`, `ARR_TIME`,
  `TAXI_IN`, `TAXI_OUT`, `WHEELS_ON`, `WHEELS_OFF`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME`, `CARRIER_DELAY`,
  `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`.
- **Descartadas por identificador**: `OP_CARRIER_FL_NUM` (solo sirve para detectar duplicados).
