# Arquitectura

![Arquitectura](assets/arquitectura.png)

```text
data/onlinefoods.csv / CSV cargado
              ↓
     src/data_loader.py   (carga · validación · limpieza · CleaningLog)
              ↓
        src/eda.py        (perfil · segmentos · chi² · DatasetProfiler)
       ↙       ↓       ↘
visualizations  models.py   agent.py  (LocalDataAgent + herramientas autorizadas)
   (Plotly/        (LogReg /     ↓
    Seaborn)        RF)     llm_client.py (OllamaClient · think on/off)
       ↘            ↓       ↙        ↕ HTTP /api/chat
              app.py (Streamlit)   Ollama local · qwen3:8b
```

## Ciclo del agente

1. **Planificar:** `LocalDataAgent.plan()` detecta palabras clave y variables mencionadas
   en la pregunta y elige herramientas del registro (`AgentTool`).
2. **Ejecutar:** cada herramienta es una función de `eda.py` o `models.py`; solo se
   ejecutan herramientas registradas (autorizadas).
3. **Redactar:** se envía a Qwen3 un contexto JSON compacto (nunca el dataset completo)
   con un *system prompt* que exige usar solo esas cifras.
4. **Verificar:** `verify_numbers()` comprueba que las cifras de la respuesta existan en
   el contexto y marca las que no.

## Paradigmas

- **Imperativo:** validaciones, limpieza y planificación con `if`/`for`.
- **Funcional:** funciones puras de análisis y visualización que reciben datos y devuelven tablas o figuras.
- **Orientado a objetos:** `OllamaClient`, `LocalDataAgent`, `DatasetProfiler`, `CleaningLog`, `AgentTool`, `AgentAnswer`.
