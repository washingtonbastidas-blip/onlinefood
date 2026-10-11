# Online Food · Explorador de pedidos con agente de IA local

Proyecto Final Integrador del módulo **Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos** (Maestría en Inteligencia Artificial y Ciencia de Datos, Universidad Casa Grande · C2026 P1).

Aplicación modular en **Streamlit** que carga, limpia, explora, segmenta y modela el *Online Food Dataset* de Kaggle, e integra un **agente de IA local** que usa **Qwen3 8B** a través de **Ollama** para interpretar los resultados calculados por Python.

> El proyecto toma como base el demo del curso (Streamlit + Ollama + Scikit-learn) y lo adapta al nuevo dataset, a un problema de clasificación y al modelo `qwen3:8b`.

---

## 1. Problema analítico

¿Qué características demográficas, socioeconómicas y de experiencia del cliente se asocian con que un pedido de comida a domicilio se concrete (`Output = Yes`)? ¿Es posible anticiparlo con un modelo de clasificación interpretable?

## 2. Dataset

| Campo | Detalle |
|---|---|
| Nombre | Online Food Dataset |
| Fuente | https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset |
| Archivo | `data/onlinefoods.csv` (incluido, 36 KB) |
| Registros | 388 originales → 285 tras eliminar 103 duplicados exactos |
| Variables | Edad, género, estado civil, ocupación, ingreso mensual, nivel educativo, tamaño familiar, latitud, longitud, código postal, `Output` (Yes/No) y `Feedback` (Positive/Negative) |

Si el CSV no estuviera en el repositorio, descárgalo desde el enlace anterior y colócalo en `data/onlinefoods.csv`.

## 3. Funcionalidades

- **Datos:** vista previa, registro de limpieza (espacios, columna duplicada `Unnamed: 12`, error tipográfico `Self Employeed`, duplicados), calidad de datos y resúmenes numéricos y categóricos.
- **Exploración:** distribución del objetivo, histogramas, boxplots, frecuencias, correlaciones de Spearman y ubicación geográfica.
- **Segmentación:** ranking de asociación con chi-cuadrado y V de Cramér, tasa de pedidos concretados por segmento y mapa de calor (Seaborn) cruzando dos variables.
- **Modelo:** Regresión Logística y Random Forest con *pipeline* de Scikit-learn, métricas en prueba, validación cruzada de 5 particiones, matriz de confusión, curva ROC e importancia por permutación.
- **Agente:** `LocalDataAgent` planifica, ejecuta herramientas autorizadas, consulta a Qwen3 y verifica las cifras de la respuesta. Muestra el contexto enviado y, opcionalmente, el razonamiento del modelo.
- **Respaldo:** si Ollama no está disponible, la app sigue funcionando y el agente devuelve un resumen calculado sin LLM.

## 4. Estructura del proyecto

```text
proyecto_final_onlinefoods/
├── app.py                  # Interfaz Streamlit (solo orquesta la UI)
├── README.md
├── ARCHITECTURE.md
├── START_HERE.txt
├── requirements.txt
├── .gitignore
├── run_app.bat             # Arranque rápido en Windows
├── run_app.command         # Arranque rápido en macOS / Linux
├── .streamlit/config.toml  # Tema visual
├── data/
│   └── onlinefoods.csv
├── assets/
│   └── arquitectura.png
└── src/
    ├── __init__.py
    ├── config.py           # Rutas, columnas, modelo por defecto, parámetros de Ollama
    ├── data_loader.py      # Carga, validación de esquema, limpieza (CleaningLog)
    ├── eda.py              # Perfilado, segmentación, chi², correlaciones (DatasetProfiler)
    ├── models.py           # Clasificadores de Scikit-learn y comparación
    ├── visualizations.py   # Gráficos Plotly y Seaborn
    ├── llm_client.py       # OllamaClient: HTTP, modo thinking de Qwen3, errores
    └── agent.py            # LocalDataAgent: herramientas, prompt, verificación, respaldo
```

## 5. Requisitos previos

- Python **3.10 a 3.13**
- [Ollama](https://ollama.com/download) **0.9 o superior** (recomendado, para controlar el modo *thinking* de Qwen3; con versiones anteriores la app usa el interruptor `/no_think`)
- Unos 6 GB libres en disco y 8 GB de RAM o más para `qwen3:8b`

## 6. Instalación

### 6.1 Crear y activar el entorno virtual

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.2 Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6.3 Instalar Ollama y descargar el modelo

1. Descarga e instala Ollama desde https://ollama.com/download y ábrelo (o ejecuta `ollama serve`).
2. Descarga el modelo usado en el proyecto:

```bash
ollama pull qwen3:8b
```

3. Comprueba que esté disponible:

```bash
ollama list
```

En equipos con menos recursos puede usarse `qwen3:4b` (`ollama pull qwen3:4b`) y seleccionarlo en la barra lateral. También puede fijarse otro modelo con la variable de entorno `OLLAMA_MODEL`.

## 7. Ejecución

```bash
python -m streamlit run app.py
```

La aplicación se abre en http://localhost:8501. También puedes usar `run_app.bat` (Windows) o `run_app.command` (macOS).

## 8. Uso del agente

1. Verifica en la barra lateral el mensaje **"Ollama activo · qwen3:8b listo"**.
2. Abre la pestaña **🤖 Agente** y elige una pregunta sugerida o escribe la tuya, por ejemplo:
   - *¿Qué perfiles de clientes concretan más pedidos y cuáles menos?*
   - *¿Influye la retroalimentación del cliente en el pedido?*
   - *Interpreta las métricas del modelo de clasificación.*
3. Revisa la respuesta, la verificación de cifras y el expansor **Contexto enviado al modelo**.
4. Activa **Modo razonamiento (thinking)** si quieres respuestas más elaboradas; tardan más.

La primera consulta puede tardar más porque Ollama carga el modelo en memoria.

## 9. Configuración opcional

| Variable de entorno | Valor por defecto | Uso |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | Modelo inicial de la barra lateral |
| `OLLAMA_HOST` | `http://localhost:11434` | URL del servicio de Ollama |
| `OLLAMA_TIMEOUT` | `300` | Segundos máximos de espera por respuesta |

## 10. Problemas conocidos y limitaciones

- **"Ollama no detectado":** abre la aplicación de Ollama o ejecuta `ollama serve`.
- **"Falta el modelo":** ejecuta `ollama pull qwen3:8b`.
- **Respuestas lentas:** desactiva el modo *thinking* o usa `qwen3:4b`.
- **Mapa base en blanco:** la opción "Mostrar mapa base" necesita Internet para descargar las teselas.
- La muestra es pequeña (285 registros tras limpiar) y proviene de una sola ciudad (Bangalore); los resultados describen asociaciones, no causalidad.
- La documentación de Kaggle describe `Output` de forma ambigua; en este proyecto se interpreta como "pedido concretado (Yes/No)".
- El despliegue obligatorio es local: Streamlit Community Cloud no puede acceder a un Ollama que corre en `localhost`.

## 11. Créditos

- Dataset: sudarshan24byte (2024). *Online Food Dataset*. Kaggle.
- Base del proyecto: demo del curso "Streamlit + Ollama + Scikit-learn".
- Modelo: Qwen3 (Alibaba Cloud), distribuido mediante la biblioteca de Ollama.
