import altair as alt
import pandas as pd
import streamlit as st

try:
    from data import DATASET_SOURCE, dataset_profile, dataset_sample, dataset_summary, evaluation_sample
    from model import LABELS, MODEL_NAME, classify_text, evaluate_messages
except ImportError:
    from app.data import DATASET_SOURCE, dataset_profile, dataset_sample, dataset_summary, evaluation_sample
    from app.model import LABELS, MODEL_NAME, classify_text, evaluate_messages


st.set_page_config(page_title="Detector de Spam SMS", page_icon=":email:", layout="wide")

st.title("Detector de Spam SMS")
st.caption(f"Modelo Hugging Face: {MODEL_NAME}")

tab_eda, tab_evaluation, tab_chat = st.tabs(
    [
        "Analisis exploratorio de datos",
        "Resultados de evaluacion del modelo",
        "Clasificacion de mensajes",
    ]
)


@st.cache_data(show_spinner=False)
def load_dataset_assets():
    return dataset_summary(), dataset_profile(), pd.DataFrame(dataset_sample(12))


@st.cache_data(show_spinner=False)
def run_model_evaluation(limit: int):
    sample = evaluation_sample(limit)
    evaluation = evaluate_messages(sample)
    prediction_rows = []

    for row, prediction in zip(sample.to_dict(orient="records"), evaluation["predictions"]):
        scores = prediction["scores"]
        prediction_rows.append(
            {
                "label_real": row["label"],
                "label_predicha": prediction["label"],
                "correcta": row["label"] == prediction["label"],
                "confianza": prediction["confidence"],
                "score_nospam": scores.get("NOSPAM", 0),
                "score_spam": scores.get("SPAM", 0),
                "message": row["message"],
            }
        )

    return evaluation, pd.DataFrame(prediction_rows)


def classification_report_dataframe(report: dict) -> pd.DataFrame:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"clase": label, **values})
        else:
            rows.append({"clase": label, "precision": values, "recall": values, "f1-score": values})
    return pd.DataFrame(rows)


def prediction_badge(label: str) -> str:
    return "SPAM" if label == "SPAM" else "NOSPAM"


def default_chat_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "Listo para clasificar mensajes como SPAM o NOSPAM.",
        }
    ]


def clear_chat_history() -> None:
    st.session_state.messages = default_chat_messages()


def render_chat_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.write(message["content"])
        if "confidence" in message:
            st.progress(float(message["confidence"]))


with tab_eda:
    st.subheader("SMS Spam Collection de UCI")
    st.caption(DATASET_SOURCE)

    with st.spinner("Cargando dataset..."):
        summary, profile, sample_rows = load_dataset_assets()

    distribution = profile["label_distribution"]
    length_by_label = profile["length_by_label"]
    common_terms = profile["common_terms"]
    dataframe = profile["dataframe"]

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric("Mensajes", f"{summary['rows']:,}")
    metric_b.metric("SPAM", f"{summary['labels'].get('SPAM', 0):,}", f"{summary['spam_rate']:.1%}")
    metric_c.metric("NOSPAM", f"{summary['labels'].get('NOSPAM', 0):,}", f"{summary['nospam_rate']:.1%}")
    metric_d.metric("Promedio palabras", summary["avg_words"])
    metric_e.metric("Promedio caracteres", summary["avg_characters"])

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Distribucion de clases**")
        label_chart = (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("label:N", title="Etiqueta"),
                y=alt.Y("count:Q", title="Mensajes"),
                color=alt.Color(
                    "label:N",
                    title="Etiqueta",
                    scale=alt.Scale(domain=LABELS, range=["#2563eb", "#dc2626"]),
                ),
                tooltip=["label", "count"],
            )
            .properties(height=280)
        )
        st.altair_chart(label_chart, use_container_width=True)

    with right:
        st.markdown("**Longitud de mensajes por clase**")
        length_chart = (
            alt.Chart(dataframe)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("label:N", title="Etiqueta"),
                y=alt.Y("characters:Q", title="Caracteres"),
                color=alt.Color(
                    "label:N",
                    legend=None,
                    scale=alt.Scale(domain=LABELS, range=["#2563eb", "#dc2626"]),
                ),
                tooltip=["label", "characters", "words"],
            )
            .properties(height=280)
        )
        st.altair_chart(length_chart, use_container_width=True)

    st.markdown("**Metricas descriptivas por clase**")
    st.dataframe(length_by_label, use_container_width=True, hide_index=True)

    terms_left, sample_right = st.columns([1, 1])
    with terms_left:
        st.markdown("**Terminos frecuentes del corpus**")
        terms_chart = (
            alt.Chart(common_terms)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("term:N", title="Termino", sort="-x"),
                x=alt.X("count:Q", title="Frecuencia"),
                tooltip=["term", "count"],
            )
            .properties(height=360)
        )
        st.altair_chart(terms_chart, use_container_width=True)

    with sample_right:
        st.markdown("**Muestra del dataset**")
        st.dataframe(sample_rows, use_container_width=True, hide_index=True)


with tab_evaluation:
    st.subheader("Evaluacion del modelo sobre SMS Spam Collection")
    st.caption("La muestra se selecciona de forma estratificada para incluir mensajes SPAM y NOSPAM.")

    summary, _, _ = load_dataset_assets()
    controls_left, controls_right = st.columns([2, 1])
    with controls_left:
        evaluation_limit = st.slider(
            "Cantidad de mensajes para evaluar",
            min_value=15,
            max_value=min(1000, summary["rows"]),
            value=300,
            step=5,
        )
    with controls_right:
        run_evaluation = st.button("Evaluar modelo", type="primary", use_container_width=True)

    if run_evaluation:
        with st.spinner("Ejecutando inferencia sobre la muestra del dataset..."):
            evaluation, predictions = run_model_evaluation(evaluation_limit)

        report = classification_report_dataframe(evaluation["classification_report"])
        spam_report = report[report["clase"].eq("SPAM")]
        nospam_report = report[report["clase"].eq("NOSPAM")]

        metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
        metric_a.metric("Mensajes evaluados", evaluation["count"])
        metric_b.metric("Exactitud", f"{evaluation['accuracy']:.2%}")
        metric_c.metric(
            "Precision SPAM",
            f"{float(spam_report['precision'].iloc[0]):.2%}" if not spam_report.empty else "0.00%",
        )
        metric_d.metric(
            "Recall SPAM",
            f"{float(spam_report['recall'].iloc[0]):.2%}" if not spam_report.empty else "0.00%",
        )
        metric_e.metric(
            "F1 NOSPAM",
            f"{float(nospam_report['f1-score'].iloc[0]):.2%}" if not nospam_report.empty else "0.00%",
        )

        matrix = pd.DataFrame(
            evaluation["confusion_matrix"],
            index=["Real NOSPAM", "Real SPAM"],
            columns=["Predicho NOSPAM", "Predicho SPAM"],
        )
        matrix_rows = matrix.reset_index().melt(id_vars="index", var_name="predicho", value_name="mensajes")
        matrix_chart = (
            alt.Chart(matrix_rows)
            .mark_rect(cornerRadius=3)
            .encode(
                x=alt.X("predicho:N", title="Prediccion"),
                y=alt.Y("index:N", title="Etiqueta real"),
                color=alt.Color("mensajes:Q", title="Mensajes", scale=alt.Scale(scheme="blues")),
                tooltip=["index", "predicho", "mensajes"],
            )
            .properties(height=260)
        )
        text_chart = (
            alt.Chart(matrix_rows)
            .mark_text(fontSize=18, fontWeight="bold")
            .encode(x="predicho:N", y="index:N", text="mensajes:Q", color=alt.value("#111827"))
        )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Matriz de confusion**")
            st.altair_chart(matrix_chart + text_chart, use_container_width=True)
        with right:
            st.markdown("**Reporte de clasificacion**")
            st.dataframe(report.round(4), use_container_width=True, hide_index=True)

        st.markdown("**Predicciones evaluadas**")
        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "correcta": st.column_config.CheckboxColumn("correcta"),
                "confianza": st.column_config.ProgressColumn(
                    "confianza",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "score_nospam": st.column_config.NumberColumn("score_nospam", format="%.4f"),
                "score_spam": st.column_config.NumberColumn("score_spam", format="%.4f"),
            },
        )

        correct_examples = predictions[predictions["correcta"]].head(5)
        incorrect_examples = predictions[~predictions["correcta"]].head(5)
        example_left, example_right = st.columns([1, 1])
        with example_left:
            st.markdown("**Ejemplos correctos**")
            st.dataframe(correct_examples, use_container_width=True, hide_index=True)
        with example_right:
            st.markdown("**Ejemplos incorrectos**")
            if incorrect_examples.empty:
                st.success("No se encontraron errores en esta muestra.")
            else:
                st.dataframe(incorrect_examples, use_container_width=True, hide_index=True)
    else:
        st.info("Selecciona el tamano de muestra y ejecuta la evaluacion para cargar el modelo.")


with tab_chat:
    st.subheader("Clasificador interactivo")
    st.caption("Escribe un mensaje y el sistema respondera con la prediccion generada por el modelo.")

    if "messages" not in st.session_state:
        st.session_state.messages = default_chat_messages()

    render_chat_message(st.session_state.messages[0])

    has_chat_history = len(st.session_state.messages) > 1
    if has_chat_history:
        st.button(
            "Nueva consulta y limpiar historial",
            help="Reinicia el chat para clasificar otro mensaje desde cero.",
            on_click=clear_chat_history,
        )
    else:
        st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)

    prompt = st.chat_input("Escribe aqui el mensaje a clasificar")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Clasificando..."):
            result = classify_text(prompt)

        if result is None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "No pude clasificar un mensaje vacio. Escribe un texto para analizar.",
                }
            )
        else:
            scores = result["scores"]
            response = (
                f"Prediccion: **{prediction_badge(result['label'])}**\n\n"
                f"Confianza: **{result['confidence']:.2%}**\n\n"
                f"Scores: NOSPAM {scores.get('NOSPAM', 0):.2%} | "
                f"SPAM {scores.get('SPAM', 0):.2%}"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "confidence": result["confidence"],
                }
            )

        st.rerun()

    for message in st.session_state.messages[1:]:
        render_chat_message(message)
