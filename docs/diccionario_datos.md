# 📚 Diccionario de datos — Retrasos de vuelos comerciales de EE. UU. (2025–2026)

**Fuente primaria:** *Bureau of Transportation Statistics (BTS), U.S. Department of Transportation (DOT)*
**Sistema:** *Reporting Carrier On-Time Performance*
**Catálogo Kaggle:** [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights)
**Licencia:** *U.S. Government Work* (dominio público)

---

## 1 · Variable objetivo

| Variable | Tipo | Descripción | Valores | Rol |
|---|---|---|---|---|
| `ARR_DEL15` | Binaria (int) | Retraso en llegada ≥ 15 minutos (definición oficial FAA/DOT). | `0` = puntual · `1` = retrasado | **Target de la clasificación** |
| `retraso15` | Binaria (int) | Alias interno de `ARR_DEL15` que vive en `cfg.OBJETIVO`. | Mismo que arriba | Nombre único que recibe el Pipeline |
| `ARR_DELAY` | Flotante | Diferencia en minutos entre la hora real de llegada y la programada. | continuo | **No se usa como X**: solo como referencia para `ret_prev_llegada` del siguiente tramo |

---

## 2 · Columnas que SÍ entran al modelo

11 numéricas + 4 categóricas. Todas se conocen a la hora de salida programada.

### Numéricas (`cfg.NUMERICAS`)

| Columna | Fuente | Descripción |
|---|---|---|
| `DISTANCE` | CSV directo | Distancia programada del tramo (millas). |
| `CRS_ELAPSED_TIME` | CSV directo | Duración programada del vuelo (minutos). |
| `hora_sal_dec` | derivada | Hora de salida programada en horas decimales (0,0 – 23,99). |
| `hora_lleg` | derivada | Hora de llegada programada (entero 0–23). |
| `dias_a_festivo` | derivada | Distancia mínima en días al festivo federal más cercano (0–15). |
| `vuelos_origen_hora` | **construida** | Número de vuelos programados en el aeropuerto de origen en la misma hora absoluta. |
| `vuelos_destino_hora` | **construida** | Número de vuelos programados en el aeropuerto de destino en la misma hora absoluta. |
| `tramo_del_dia` | **construida** | Número de tramo que lleva la aeronave ese día (1 = primer vuelo, 2, 3…). |
| `holgura_prog` | **construida** | Minutos entre la llegada programada del tramo previo y la salida programada actual (0–1440). |
| `ret_prev_llegada` | **construida** | Retraso REAL del tramo anterior del mismo avión, en minutos, solo si ya aterrizó antes de ESTA salida programada. Recortado a `[-60, 360]` con constantes fijas. |
| `prev_disponible` | **construida** | Bandera (0/1): ¿el tramo anterior ya había aterrizado antes de esta salida programada? El faltante NO es aleatorio. |

### Categóricas (`cfg.CATEGORICAS`)

| Columna | Fuente | Descripción |
|---|---|---|
| `OP_UNIQUE_CARRIER` | CSV directo | Código IATA de la aerolínea (WN, AA, DL, UA, …). |
| `ORIGIN` | CSV directo | Código IATA del aeropuerto de origen. |
| `DEST` | CSV directo | Código IATA del aeropuerto de destino. |
| `dia_semana` | derivada | Día de la semana (0 = lunes, 6 = domingo). |

**Transformación dentro del Pipeline (reajustada por pliegue):**

- Numéricas: `SimpleImputer(strategy="median")` + `StandardScaler` (solo para los modelos lineales y la red neuronal).
- Categóricas: `TargetEncoder(random_state=42)` con cross-fitting interno (sklearn ≥ 1.3) que evita leakage del target.

---

## 3 · Columnas prohibidas por fuga (`cfg.PROHIBIDAS`)

NO se leen del CSV o se descartan explícitamente. Si alguna se cuela en `X`, el `assert` de
`columnas.verificar_sin_fuga()` aborta el notebook.

| Columna | Momento en que se conoce | Por qué es fuga |
|---|---|---|
| `DEP_TIME` | al salir de puerta | Posterior a la salida programada. |
| `DEP_DELAY` | al salir de puerta | $ArrDelay \approx DepDelay$: AUC se dispara ($R^2 \approx 0{,}97$). |
| `DEP_DEL15` | al salir de puerta | Derivada de `DEP_DELAY`. |
| `TAXI_OUT` | en pista | Posterior a la salida de puerta. |
| `WHEELS_OFF` | en el aire | Posterior al despegue. |
| `WHEELS_ON` | al tocar pista | Final del vuelo. |
| `TAXI_IN` | en pista de destino | Posterior al aterrizaje. |
| `ARR_TIME` | en puerta de destino | Determina directamente el target. |
| `ACTUAL_ELAPSED_TIME` | fin del vuelo | Se conoce al concluir. |
| `AIR_TIME` | fin del vuelo | Conocido tras el vuelo. |
| `CARRIER_DELAY` | post-vuelo | Causa calculada por la FAA a posteriori (>50 % de nulos). |
| `WEATHER_DELAY` | post-vuelo | Causa calculada por la FAA a posteriori (>50 % de nulos). |
| `NAS_DELAY` | post-vuelo | Causa calculada por la FAA a posteriori (>50 % de nulos). |
| `SECURITY_DELAY` | post-vuelo | Causa calculada por la FAA a posteriori (>50 % de nulos). |
| `LATE_AIRCRAFT_DELAY` | post-vuelo | Causa calculada por la FAA a posteriori (es exactamente la señal que capturamos de forma anti-fuga con `ret_prev_llegada`). |

Las cuatro últimas tampoco se leen del CSV (ver `cfg.COLS_CRUDAS`): esa es la barrera 1 de prevención.

---

## 4 · Columnas que SE USAN pero NO entran como `X`

| Columna | Para qué sirve |
|---|---|
| `TAIL_NUM` | Identificador de la aeronave; se usa para ordenar por avión y construir `ret_prev_llegada`, `holgura_prog`, `tramo_del_dia`. **No entra al modelo** (cardinalidad extrema sin generalización). |
| `FL_DATE` | Fecha del vuelo; se reconstruye desde `YEAR/MONTH/DAY_OF_MONTH` y se usa para la validación temporal. **No entra al modelo**. |
| `CANCELLED` | Marca el filtro `tiene_etiqueta = 0`. |
| `DIVERTED` | Marca el filtro `tiene_etiqueta = 0`. |
| `OP_CARRIER_FL_NUM` | Número de vuelo; solo para la llave única de duplicados (`cfg.LLAVE_VUELO`). |

---

## 5 · Imputación

- Los vuelos operados tienen `CRS_DEP_TIME`, `CRS_ARR_TIME`, `DISTANCE` y `CRS_ELAPSED_TIME` casi siempre completos (`ARR_DELAY` y `ARR_DEL15` tienen 2,14 % de nulos exclusivamente por cancelaciones/desvíos, que se filtran).
- `ret_prev_llegada` tiene nulos cuando no hay tramo anterior o el inbound aún no aterrizó. El faltante **NO es aleatorio** → se conserva la bandera `prev_disponible`.
- El experimento del paso 4 compara **cuatro métodos** sobre 60.000 filas con el 25 % de los valores ocultos:
  - Mediana por hora de salida: 18,44 min MAE (gana).
  - Mediana global: 18,57 min MAE.
  - Regresión (HistGradientBoosting): 19,94 min MAE.
  - Media global: 20,25 min MAE.
- El Pipeline usa `SimpleImputer(strategy="median")` reajustado por pliegue: la mediana global pierde solo 0,13 min frente a la mediana por hora y es más simple.
