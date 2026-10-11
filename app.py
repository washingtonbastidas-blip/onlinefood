"""Interfaz Streamlit del Proyecto Final Integrador.

Este archivo solo orquesta la interfaz: toda la lógica de datos, análisis,
modelos, gráficos y agente vive en los módulos de `src/`.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agent import LocalDataAgent
from src.config import (
    ASSETS_DIR, DATASET_NAME, DATASET_URL, DEFAULT_MODEL, TARGET, label,
)
from src.data_loader import (
    add_target_flag, clean_data, dataset_profile, load_default_data,
    load_uploaded_data, validate_schema,
)
from src.eda import (
    DatasetProfiler, add_age_group, association_ranking, categorical_summary,
    missing_report, numeric_by_target, numeric_summary, target_correlations,
    target_distribution, target_rate_by, top_locations,
)
from src.llm_client import OllamaClient
from src.models import MODEL_NAMES, compare_models, default_features, train_classifier
from src import visualizations as viz

st.set_page_config(page_title="Online Food · IA + Datos", page_icon="🍔", layout="wide")


# ---------------------------------------------------------------------------
# Caché: evita recalcular en cada interacción (envuelve funciones de src/)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_clean(raw: pd.DataFrame, drop_duplicates: bool):
    return clean_data(raw, drop_duplicates)


@st.cache_data(show_spinner="Entrenando modelo...")
def cached_model(df: pd.DataFrame, features: tuple[str, ...], model_type: str):
    return train_classifier(df, TARGET, list(features), model_type)


@st.cache_data(show_spinner="Comparando modelos...")
def cached_comparison(df: pd.DataFrame, features: tuple[str, ...]):
    return compare_models(df, TARGET, list(features))


def show_df(data: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(data, width="stretch", hide_index=True, **kwargs)


# ---------------------------------------------------------------------------
# Encabezado y barra lateral
# ---------------------------------------------------------------------------
st.title("🍔 Online Food · Explorador de pedidos con agente local")
st.caption("Streamlit + Scikit-learn + Ollama (Qwen3) · Proyecto Final Integrador")

with st.sidebar:
    st.header("1. Datos")
    uploaded = st.file_uploader("CSV opcional (mismo esquema)", type=["csv"])
    drop_dups = st.checkbox("Eliminar registros duplicados", value=True,
                            help="El CSV original contiene 103 filas idénticas.")

try:
    raw_df = load_uploaded_data(uploaded) if uploaded else load_default_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()

missing_cols = validate_schema(raw_df)
if missing_cols:
    st.error("El CSV no tiene el esquema del Online Food Dataset. Faltan: " + ", ".join(missing_cols))
    st.dataframe(raw_df.head(20), width="stretch")
    st.stop()

df, cleaning_log = cached_clean(raw_df, drop_dups)
df = add_target_flag(df)
analysis_df = df.drop(columns=["target_flag"])
profiler = DatasetProfiler(analysis_df)
profile = dataset_profile(analysis_df)

with st.sidebar:
    st.header("2. Modelo predictivo")
    all_features = default_features(analysis_df)
    model_type = st.selectbox("Algoritmo", list(MODEL_NAMES), index=1, format_func=MODEL_NAMES.get)
    features = st.multiselect("Variables del modelo", all_features, default=all_features,
                              format_func=label)

    st.header("3. Agente (Ollama)")
    probe = OllamaClient(DEFAULT_MODEL)
    ollama_ok, installed_models, ollama_detail = probe.status()
    options = list(dict.fromkeys([DEFAULT_MODEL] + installed_models))
    model_name = st.selectbox("Modelo local", options, index=0,
                              help="Por defecto qwen3:8b. Se listan los modelos instalados en Ollama.")
    think_mode = st.toggle("Modo razonamiento (thinking)", value=False,
                           help="Qwen3 razona antes de responder: más profundo pero más lento.")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.1)

    client = OllamaClient(model_name)
    if ollama_ok and client.has_model(installed_models):
        st.success(f"Ollama activo · {model_name} listo")
        st.caption(f"Versión de Ollama: {client.version()}")
    elif ollama_ok:
        st.warning(f"Ollama activo, pero falta el modelo. Ejecuta:\n\n`ollama pull {model_name}`")
    else:
        st.warning("Ollama no detectado")
        st.caption("La aplicación sigue funcionando; el agente usará el resumen de respaldo.")

model_ready = ollama_ok and client.has_model(installed_models)

model_result, comparison, model_error = None, None, ""
if features:
    try:
        model_result = cached_model(analysis_df, tuple(features), model_type)
        comparison = cached_comparison(analysis_df, tuple(features))
    except Exception as exc:  # noqa: BLE001
        model_error = str(exc)
else:
    model_error = "Selecciona al menos una variable para entrenar el modelo."

# ---------------------------------------------------------------------------
# Indicadores principales
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Registros", profile["rows"], delta=f"-{cleaning_log.duplicates_removed} duplicados"
          if cleaning_log.duplicates_removed else None, delta_color="off")
c2.metric("Variables", profile["columns"])
c3.metric("Valores nulos", profile["missing"])
c4.metric("Pedidos concretados", f"{profiler.positive_rate:.1f}%")
c5.metric("F1 del modelo", f"{model_result['metrics']['f1']:.3f}" if model_result else "—")

tabs = st.tabs(["📄 Datos", "📊 Exploración", "🎯 Segmentación", "📈 Modelo", "🤖 Agente", "🧩 Arquitectura"])

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset de trabajo")
    if uploaded is None:
        st.info(f"Dataset incluido: **{DATASET_NAME}** (Kaggle). Fuente: {DATASET_URL}")
    else:
        st.info("Se está utilizando el CSV cargado desde la interfaz.")
    show_df(analysis_df.head(25))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Limpieza aplicada")
        st.caption(f"{cleaning_log.rows_before} registros originales → {cleaning_log.rows_after} analizados")
        for step in cleaning_log.steps:
            st.markdown(f"- {step}")
    with col_b:
        st.markdown("#### Calidad de datos")
        show_df(missing_report(analysis_df), height=300)

    st.markdown("#### Resumen numérico")
    show_df(numeric_summary(analysis_df))
    st.markdown("#### Resumen categórico")
    show_df(categorical_summary(analysis_df))

# ---------------------------------------------------------------------------
# 2. Exploración
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Exploración visual")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.plotly_chart(viz.target_donut(target_distribution(analysis_df)), width="stretch")
    with col_b:
        num_cols = ["Age", "Family size"]
        xcol = st.selectbox("Variable numérica", num_cols, format_func=label)
        st.plotly_chart(viz.histogram(analysis_df, xcol), width="stretch")

    cat_cols = [c for c in analysis_df.select_dtypes(exclude="number").columns if c != TARGET]
    col_a, col_b = st.columns(2)
    with col_a:
        ccol = st.selectbox("Variable categórica", cat_cols, format_func=label)
        st.plotly_chart(viz.category_counts(analysis_df, ccol), width="stretch")
    with col_b:
        bcol = st.selectbox("Numérica según pedido concretado", num_cols, format_func=label, key="box")
        st.plotly_chart(viz.box_by_target(analysis_df, bcol), width="stretch")
        show_df(numeric_by_target(analysis_df))

    st.markdown("#### Correlaciones con el objetivo (Spearman)")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        corr = target_correlations(analysis_df)
        corr["variable"] = corr["variable"].map(label)
        show_df(corr)
    with col_b:
        st.plotly_chart(viz.correlation_heatmap(analysis_df), width="stretch")

    st.markdown("#### Distribución geográfica")
    basemap = st.checkbox("Mostrar mapa base (requiere Internet)", value=False)
    st.plotly_chart(viz.location_scatter(analysis_df, use_basemap=basemap), width="stretch")
    st.caption("Códigos postales con al menos 5 registros:")
    show_df(top_locations(analysis_df))

# ---------------------------------------------------------------------------
# 3. Segmentación
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("¿Qué perfiles concretan más pedidos?")
    ranking = association_ranking(analysis_df)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(viz.association_bar(ranking), width="stretch")
    with col_b:
        st.markdown("**Hallazgos calculados (sin IA)**")
        for finding in profiler.key_findings(top=3):
            st.markdown(f"- {finding}")

    with st.expander("Tabla completa: chi-cuadrado y V de Cramér"):
        show_df(ranking)
        st.caption("V de Cramér: 0.1 débil · 0.3 moderada · ≥ 0.5 fuerte. "
                   "Celdas esperadas < 5 indican que el p-valor debe leerse con cautela.")

    seg_options = ["Age group"] + cat_cols + ["Family size"]
    seg = st.selectbox("Variable de segmentación", seg_options, format_func=label,
                       index=seg_options.index("Occupation"))
    seg_df = add_age_group(analysis_df)
    rate_table = target_rate_by(seg_df, seg)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.target_rate_bar(rate_table, seg, profiler.positive_rate), width="stretch")
    with col_b:
        st.plotly_chart(viz.stacked_share(seg_df, seg), width="stretch")
    show_df(rate_table)

    st.markdown("#### Cruce de dos variables (Seaborn)")
    pair_opts = [c for c in cat_cols if c != "Feedback"] + ["Feedback"]
    col_a, col_b = st.columns(2)
    row_var = col_a.selectbox("Filas", pair_opts, index=pair_opts.index("Occupation"), format_func=label)
    col_var = col_b.selectbox("Columnas", pair_opts, index=pair_opts.index("Monthly Income"), format_func=label)
    if row_var == col_var:
        st.info("Elige dos variables distintas.")
    else:
        st.pyplot(viz.seaborn_rate_heatmap(analysis_df, row_var, col_var), width="stretch")

# ---------------------------------------------------------------------------
# 4. Modelo
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader(f"Modelo de clasificación · {MODEL_NAMES[model_type]}")
    if model_result is None:
        st.warning(f"No fue posible entrenar el modelo: {model_error}")
    else:
        m = model_result["metrics"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{m['accuracy']:.3f}", delta=f"{m['accuracy'] - m['baseline_accuracy']:+.3f} vs. base")
        k2.metric("Precisión", f"{m['precision']:.3f}")
        k3.metric("Recall", f"{m['recall']:.3f}")
        k4.metric("F1", f"{m['f1']:.3f}")
        k5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        st.caption(
            f"Entrenamiento: {model_result['n_train']} · Prueba: {model_result['n_test']} (estratificado, "
            f"semilla 42). Línea base (siempre 'Yes'): {m['baseline_accuracy']:.3f}. "
            f"F1 en validación cruzada 5-fold: {model_result['cv']['f1']['media']:.3f} ± "
            f"{model_result['cv']['f1']['desv']:.3f}"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(viz.confusion_matrix_fig(model_result["confusion_matrix"]), width="stretch")
        with col_b:
            st.plotly_chart(viz.roc_fig(model_result["roc"], m["roc_auc"]), width="stretch")
        st.plotly_chart(viz.importance_bar(model_result["importance"]), width="stretch")
        st.markdown("#### Comparación de modelos (mismas variables)")
        show_df(comparison)

# ---------------------------------------------------------------------------
# 5. Agente
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Agente local para interpretación")
    st.caption(
        "El agente elige herramientas autorizadas, ejecuta los cálculos en Python y envía a "
        f"**{model_name}** solo el contexto resumido (nunca el dataset completo)."
    )
    agent = LocalDataAgent(client, analysis_df, cleaning_log, model_result, comparison)
    history = st.session_state.setdefault("agent_history", [])

    presets = {
        "📌 Hallazgos principales": "Resume los principales hallazgos del dataset.",
        "👥 Perfiles de clientes": "¿Qué perfiles de clientes concretan más pedidos y cuáles menos?",
        "💼 Ocupación e ingreso": "¿Cómo influyen la ocupación y el ingreso mensual en los pedidos?",
        "📈 Métricas del modelo": "Interpreta las métricas del modelo de clasificación.",
        "💡 Recomendaciones": "¿Qué recomendaciones de negocio se pueden plantear con estos resultados?",
    }
    st.markdown("**Preguntas sugeridas**")
    preset_cols = st.columns(len(presets))
    clicked = None
    for col, (button_label, text) in zip(preset_cols, presets.items()):
        if col.button(button_label, help=text, width="stretch"):
            clicked = text

    typed = st.chat_input("Escribe tu pregunta sobre el dataset...")
    question = typed or clicked

    if question:
        spinner = (f"Consultando {model_name} (la primera respuesta puede tardar mientras se carga el modelo)..."
                   if model_ready else "Calculando resumen de respaldo...")
        with st.spinner(spinner):
            answer = agent.ask(question, think=think_mode, temperature=temperature,
                               history=history, use_llm=model_ready)
        history.append(answer)

    if not history:
        st.info("Elige una pregunta sugerida o escribe la tuya.")

    for item in reversed(history):
        with st.chat_message("user"):
            st.markdown(item.question)
        with st.chat_message("assistant"):
            if item.source == "ollama":
                st.caption(f"🟢 {item.model} · {item.seconds} s · herramientas: {', '.join(item.tools_used)}")
            else:
                st.caption("🟠 Respaldo sin LLM · herramientas: " + ", ".join(item.tools_used))
                if item.error:
                    st.error(item.error)
            st.markdown(item.answer)
            v = item.verification
            if v.get("cifras_revisadas"):
                msg = f"✔ Verificación: {v['verificadas']}/{v['cifras_revisadas']} cifras citadas existen en el contexto."
                if v["no_verificadas"]:
                    msg += f" Revisar: {', '.join(v['no_verificadas'])}"
                st.caption(msg)
            if item.thinking:
                with st.expander("🧠 Razonamiento de Qwen3"):
                    st.markdown(item.thinking)
            with st.expander("🔧 Contexto enviado al modelo"):
                st.code(json.dumps(item.context, ensure_ascii=False, indent=2, default=str), language="json")

    if history and st.button("Limpiar conversación"):
        st.session_state["agent_history"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# 6. Arquitectura
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Arquitectura modular")
    diagram = ASSETS_DIR / "arquitectura.png"
    if diagram.exists():
        st.image(str(diagram), width="stretch")
    st.markdown(
        """
- **Imperativo:** flujo de carga, validaciones, limpieza y planificación del agente (`if/for`).
- **Funcional:** funciones puras de perfilado, segmentación, métricas y gráficos en `eda.py` y `visualizations.py`.
- **Orientado a objetos:** `OllamaClient`, `LocalDataAgent`, `DatasetProfiler`, `CleaningLog`, `AgentTool`.
- **Modularidad:** cada responsabilidad vive en un archivo de `src/`; `app.py` solo arma la interfaz.
        """
    )
    st.markdown("#### Herramientas autorizadas del agente")
    show_df(pd.DataFrame([{"herramienta": t.name, "descripción": t.description}
                          for t in agent.tools.values()]))

st.markdown("---")
st.caption("Proyecto académico · Los resultados describen asociaciones en una muestra de 285 clientes; "
           "no implican causalidad.")
