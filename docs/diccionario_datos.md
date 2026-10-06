# Diccionario de Datos — Siniestros Viales en Medellín

**Fuente:** Secretaría de Movilidad de Medellín / MEData (Alcaldía de Medellín)  
**Catálogo oficial:** [Portal de Datos Abiertos MEData](http://medata.gov.co) | Dataset ID: `9wqu-juqb`

---

## 1. Variable Objetivo (Target)

| Variable | Tipo | Descripción | Valores posibles | Rol en el Proyecto |
|---|---|---|---|---|
| `GRAVEDAD_ACCIDENTE` | Categórica (String) | Nivel máximo de severidad resultante del siniestro vial. | `Solo daños`, `Con heridos`, `Con muertos` | **Target Original** de MEData |
| `es_grave` | Binaria (Entero) | Indicador derivado de criticidad vital para clasificación supervisada. | `1` = Severo (`Con heridos` o `Con muertos`)<br>`0` = Leve / No severo (`Solo daños`) | **Target Primario de ML** para priorización de emergencias |

---

## 2. Variables Predictoras (Features Iniciales)

### 2.1 Variables Espaciales y Geográficas
| Columna | Tipo | Descripción | Tratamiento en Pipeline |
|---|---|---|---|
| `COMUNA` | Categórica | Nombre de la comuna o corregimiento de Medellín (16 comunas urbanas + 5 corregimientos). | Imputación categórica + Target / One-Hot Encoding |
| `NUMCOMUNA` | Categórica / Entero | Código numérico oficial de la comuna (ej. `10` = La Candelaria, `14` = El Poblado). | Consistencia con `COMUNA` |
| `BARRIO` | Categórica | Nombre del barrio (alta cardinalidad: ~249 barrios). | Frecuencia de accidentalidad o agrupación de cola larga |
| `LOCATION` | String | Coordenadas en formato WGS84 `[Longitud, Latitud]`. | Extracción de `lat` y `lon` numéricas |
| `X` | Float | Coordenada plana Magna-Sirgas Este. | Normalización / Imputación por mediana de comuna |
| `Y` | Float | Coordenada plana Magna-Sirgas Norte. | Normalización / Imputación por mediana de comuna |

### 2.2 Variables de Infraestructura y Vía
| Columna | Tipo | Descripción | Tratamiento en Pipeline |
|---|---|---|---|
| `CLASE_ACCIDENTE` | Categórica | Dinámica del siniestro (`Choque`, `Atropello`, `Volcamiento`, `Caída Ocupante`, `Incendio`, `Otro`). | One-Hot Encoding |
| `DISEÑO` | Categórica | Tipo de infraestructura geométrica (`Tramo de via`, `Interseccion`, `Glorieta`, `Puente`, `Paso a nivel`, etc.). | One-Hot Encoding / Agrupación de infrecuentes |
| `DIRECCION` | Texto | Nomenclatura del punto del siniestro vial. | Extracción de tipo de vía (`CR`, `CL`, `TV`, `DG`, `AV`) |

### 2.3 Variables Temporales
| Columna | Tipo | Descripción | Feature Engineering requerido |
|---|---|---|---|
| `FECHA_ACCIDENTE` | String (Fecha/Hora) | Fecha y hora registrada en formato `DD/MM/YYYY HH:MM:SS`. | Conversión a `datetime64`, extracción de: `hora`, `dia_semana`, `es_fin_de_semana`, `franja_horaria`, componentes cíclicas seno/coseno |
| `MES` | Entero | Mes del siniestro (1-12). | Estacionalidad |
| `AÑO` | Entero | Año del reporte. | Control de tendencias temporales |

---

## 3. Variables con Fuga de Datos (Data Leakage) — Excluidas del Modelo

Las siguientes columnas **NO DEBEN SER USADAS** como variables predictoras porque representan información administrativa generada **después** de que el siniestro fue atendido e investigado:

| Columna | Justificación de Exclusión |
|---|---|
| `EXPEDIENTE` | Número de expediente judicial/administrativo asignado tras la investigación pericial. |
| `NRO_RADICADO` | Identificador de radicación en el sistema contravencional de tránsito. |
| `CBML` | Código catastral de manzana que presenta alta redundancia e identificadores espurios. |
| `FECHA_ACCIDENTES` | Campo redundante que duplica `FECHA_ACCIDENTE` en formato ISO. |
