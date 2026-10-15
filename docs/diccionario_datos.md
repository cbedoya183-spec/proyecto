# 📚 Diccionario de Datos — Retrasos de Vuelos Comerciales en EE.UU. (2025–2026)

**Fuente Primaria:** *Bureau of Transportation Statistics (BTS), U.S. Department of Transportation (DOT)*  
**Sistema:** *Reporting Carrier On-Time Performance*  
**Catálogo Kaggle:** [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights)  
**Licencia:** Obra del Gobierno Federal de EE.UU. (*U.S. Government Work*) / Dominio Público

---

## 1. Variable Objetivo (Target)

| Variable | Tipo | Descripción | Valores posibles | Rol en el Proyecto |
|---|---|---|---|---|
| `ARR_DEL15` | Binaria (Entero) | Indicador oficial regulatorio de la FAA y el DOT de retraso en llegada $\ge 15$ minutos. | `0` = Puntual o adelantado<br>`1` = Retraso significativo ($\ge 15$ min) | **Target Principal de Clasificación** |
| `ARR_DELAY` | Flotante (Continuo)| Diferencia en minutos entre la hora de llegada real y la programada CRS. | $(-\infty, +\infty)$ minutos | **Target Continuo de Referencia** (usado para validar umbral) |

---

## 2. Variables Predictoras Válidas en Pre-Vuelo (Itinerario CRS)

Estas columnas están disponibles **antes del vuelo** (fase de programación/itinerario) y constituyen las características legítimas para el modelado:

| Columna | Tipo | Descripción | Tratamiento en Pipeline |
|---|---|---|---|
| `MONTH` | Entero (1–12) | Mes del año programado del vuelo. | Codificación de estacionalidad (invierno, verano, festividades). |
| `DAY_OF_MONTH` | Entero (1–31) | Día calendario del mes. | Patrones de inicio y fin de mes. |
| `DAY_OF_WEEK` | Entero (0–6) | Día de la semana (0: Lunes, 6: Domingo). | Captura diferencias entre días laborales y fin de semana. |
| `IS_WEEKEND` | Binario (0/1) | Indicador derivado de sábado o domingo. | Variable dicotómica de demanda de ocio vs. negocios. |
| `SEASON` | Categórica | Estación climática (Invierno, Primavera, Verano, Otoño). | One-Hot Encoding en ColumnTransformer. |
| `OP_UNIQUE_CARRIER` | Categórica | Código IATA de la aerolínea operadora (ej. WN, AA, DL, UA). | One-Hot Encoding (factor de flota y modelo operativo). |
| `ORIGIN` | Categórica (IATA) | Código de 3 letras del aeropuerto de origen. | Agrupación de aeropuertos top + One-Hot Encoding. |
| `DEST` | Categórica (IATA) | Código de 3 letras del aeropuerto de destino. | Agrupación de aeropuertos top + One-Hot Encoding. |
| `CRS_DEP_HOUR` | Entero (0–23) | Hora programada de salida (extraída de `CRS_DEP_TIME`). 2400 se lee como 00. Los vacíos no se rellenan aquí. | Imputación por mediana dentro del Pipeline, ajustada solo con entrenamiento. |
| `CRS_DEP_SIN`, `CRS_DEP_COS` | Flotante | Seno y coseno de la hora programada (periodo 24). | La hora 23 queda junto a la 0 para los modelos lineales. |
| `MONTH_SIN`, `MONTH_COS` | Flotante | Seno y coseno del mes (periodo 12). | Diciembre queda junto a enero. |
| `CRS_ARR_HOUR` | Entero (0–23) | Hora programada de llegada (extraída de `CRS_ARR_TIME`). | Escalado estándar (congestión de arribo). |
| `DEP_TIME_BLOCK` | Categórica | Franja horaria de salida (Madrugada, Mañana, Tarde, Noche). | One-Hot Encoding. |
| `CRS_ELAPSED_TIME` | Flotante | Duración programada total del vuelo en minutos. | Imputación por mediana + StandardScaler. |
| `DISTANCE` | Flotante | Distancia geográfica entre origen y destino (millas). | Escalado estándar. |
| `DISTANCE_CAT` | Categórica | Categoría de distancia (Corta $<500$ mi, Media, Larga $>1500$ mi). | One-Hot Encoding. |
| `ORIGIN_IS_HUB` | Binario (0/1) | Indicador de si el origen es uno de los 10 hubs más congestionados. | Predictor de congestión de pista y espacio aéreo. |
| `DEST_IS_HUB` | Binario (0/1) | Indicador de si el destino es uno de los 10 hubs más congestionados. | Predictor de congestión de rampa y puertas. |
| `IS_HUB_TO_HUB` | Binario (0/1) | Vuelo que conecta dos hubs principales simultáneamente. | Interacción de alta saturación operativa. |

---

## 3. Variables de Fuga de Datos (Data Leakage) — Excluidas Estrictamente del Modelo

Las siguientes columnas **NO DEBEN SER USADAS** como predictoras porque se generan durante o después de la operación del vuelo, revelando el resultado futuro:

| Columna | Significado | Momento en que se Conoce | Motivo de Exclusión Técnica |
|---|---|---|---|
| `DEP_TIME` | Hora real de salida de puerta | Tras el abordaje | Revela si el vuelo ya salió tarde de puerta. |
| `DEP_DELAY` | Retraso real en salida (minutos) | Al salir de puerta | Fuga severa: $ArrDelay \approx DepDelay$ (da $R^2 \approx 0.97$). |
| `DEP_DEL15` | Retraso salida $\ge 15$ min | Al salir de puerta | Variable derivada del retraso de salida. |
| `TAXI_OUT` | Tiempo de carreteo de salida | Tras dejar la puerta | Información post-despacho en pista. |
| `WHEELS_OFF` | Hora real de despegue | En el aire | Ocurre después de la salida programada. |
| `WHEELS_ON` | Hora real de aterrizaje | Al tocar pista | Final del vuelo. |
| `TAXI_IN` | Tiempo de carreteo de llegada | En pista de destino | Información posterior al aterrizaje. |
| `ARR_TIME` | Hora real de llegada | En puerta de destino | Determina directamente la variable objetivo. |
| `ACTUAL_ELAPSED_TIME` | Duración real del vuelo | Fin del vuelo | Se conoce solo al concluir la operación. |
| `AIR_TIME` | Tiempo neto en el aire | Fin del vuelo | Conocido tras el vuelo. |
| `CARRIER_DELAY` | Retraso imputado a aerolínea | Post-vuelo | Causa calculada por la FAA a posteriori. |
| `WEATHER_DELAY` | Retraso meteorológico | Post-vuelo | Causa calculada por la FAA a posteriori. |
| `NAS_DELAY` | Retraso por espacio aéreo | Post-vuelo | Causa calculada por la FAA a posteriori. |
| `SECURITY_DELAY` | Retraso por seguridad | Post-vuelo | Causa calculada por la FAA a posteriori. |
| `LATE_AIRCRAFT_DELAY` | Retraso por avión entrante | Post-vuelo | Causa calculada por la FAA a posteriori. |

---

## 4. Columnas Administrativas / Filtros

| Columna | Significado | Tratamiento |
|---|---|---|
| `CANCELLED` | Vuelo cancelado (1/0) | Utilizado para filtrar vuelos no operados. |
| `DIVERTED` | Vuelo desviado a aeropuerto alterno (1/0) | Utilizado para filtrar operaciones incompletas. |
| `OP_CARRIER_FL_NUM` | Número de vuelo comercial | Descartado (identificador administrativo no generalizable). |
| `TAIL_NUM` | Matrícula de la aeronave | Descartado (cardinalidad extrema sin generalización en prueba). |
| `ORIGIN_AIRPORT_ID` | Identificador numérico de origen | Descartado (redundante con `ORIGIN`). |
| `DEST_AIRPORT_ID` | Identificador numérico de destino | Descartado (redundante con `DEST`). |
| `S` | Columna vacía heredada del CSV | Descartada: todos sus valores son nulos. |

---

## 5. Imputación

En los vuelos operados, `CRS_DEP_TIME` y `DISTANCE` vienen completos. `CRS_ELAPSED_TIME` falta en cuatro filas de todo el año. Aun así se esconde el 20 % de la duración programada del entrenamiento y se compara la media, la mediana global y la mediana por aerolínea. La mediana por aerolínea baja el error y es la que usa el Pipeline (`GroupMedianImputer`), ajustada solo con cada pliegue. El resto de las variables numéricas usa la mediana global dentro del `ColumnTransformer`.
