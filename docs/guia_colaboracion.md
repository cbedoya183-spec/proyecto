# Guía de Colaboración y Buenas Prácticas del Equipo

Este documento establece las pautas y estándares técnicos para que todos los miembros del equipo trabajen de manera armónica, ordenada y sin conflictos en el repositorio.

---

## 1. Configuración del Entorno de Desarrollo (Onboarding)

Cada integrante del equipo debe ejecutar los siguientes pasos una sola vez al clonar el repositorio:

### Paso 1: Clonar y entrar al repositorio
```bash
git clone https://github.com/sebastia-olaya/Proyecto-final-Machine-Learning.git
cd Proyecto-final-Machine-Learning
```

### Paso 2: Crear y activar entorno virtual de Python (>= 3.10)
**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar dependencias y paquete modular en modo editable
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
> **¿Por qué `pip install -e .`?**  
> Permite importar cualquier función o clase desde `src` (por ejemplo: `from src.config import RANDOM_STATE`) desde cualquier notebook o script sin trucos de `sys.path.append()`.

### Paso 4: Descargar los datos oficiales
```bash
python src/data/download_data.py --sample-size 50000
```
> Esto descarga el CSV completo oficial de MEData y genera una muestra de 50.000 registros para que el equipo pueda iterar de forma fluida sin ralentizar sus computadores.

---

## 2. Estrategia de Ramas en Git (Git Flow Ligero)

1. **Rama `main` (Producción / Entrega Académica):**
   - Siempre debe contener código probado, que ejecute de principio a fin sin errores.
   - **Nadie hace commits directos a `main`** para cambios grandes.

2. **Ramas de funcionalidad (`feature/...`):**
   - Cada tarea o módulo se desarrolla en su propia rama:
     - `feature/eda-visual`
     - `feature/preprocesamiento-pipeline`
     - `feature/modelo-baseline`
     - `feature/xgboost-tuning`
     - `feature/documentacion-informe`

3. **Flujo de trabajo para un cambio:**
   ```bash
   git checkout -b feature/nombre-de-tu-modulo
   # Trabajas, pruebas...
   git add .
   git commit -m "feat(modulo): descripción clara del cambio"
   git push origin feature/nombre-de-tu-modulo
   # Abres un Pull Request (PR) en GitHub para revisión del equipo
   ```

---

## 3. Convención de Mensajes de Commit (Conventional Commits)

Usa prefijos semánticos para que el historial del proyecto sea claro y profesional:
- `feat:` Nueva funcionalidad (ej. `feat(data): agregar script de descarga de MEData`).
- `fix:` Corrección de un error (ej. `fix(features): corregir timezone en timestamp`).
- `docs:` Cambios en documentación o README (ej. `docs: actualizar diccionario de datos`).
- `refactor:` Mejora de código sin cambiar funcionalidad (ej. `refactor(pipeline): encapsular ColumnTransformer`).
- `test:` Adición o mejora de pruebas.

---

## 4. Reglas de Oro Técnicas del Proyecto

1. 🚫 **NUNCA subir datos crudos o modelos pesados a Git:**
   - La carpeta `data/raw/`, `data/processed/` y `models/*.joblib` están en `.gitignore`.
   - Si creas un archivo de datos intermedio, asegúrate de que no se versione en Git.
2. 🚫 **NUNCA hacer transformaciones fuera del Pipeline de Scikit-Learn:**
   - Cualquier cálculo de media, mediana, codificación One-Hot o Target Encoding **debe** ir dentro de un `Pipeline` o `ColumnTransformer` para evitar **Data Leakage**.
3. 🧹 **Limpieza en Notebooks:**
   - Antes de commitear un notebook a Git, haz: **Kernel -> Restart & Run All** para verificar que corra limpio de arriba a abajo, o limpia las salidas si son demasiado pesadas.
4. 🔒 **Semilla Global:**
   - Usa siempre `from src.config import RANDOM_STATE` para cualquier modelo, split o submuestreo (`random_state=RANDOM_STATE`).
