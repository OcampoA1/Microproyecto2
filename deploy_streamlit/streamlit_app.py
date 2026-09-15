from pathlib import Path

import pandas as pd
import streamlit as st

from src.model_controller import ModelController, PredictionInputError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resources" / "models" / "modelo_ods.joblib"

st.set_page_config(
    page_title="Clasificador de textos por ODS",
    page_icon="🌎",
    layout="centered",
)


@st.cache_resource
def load_controller() -> ModelController:
    return ModelController.from_file(MODEL_PATH)


st.title("🌎 Clasificador de textos por ODS")
st.caption("Procesamiento de lenguaje natural · TF-IDF · LSA/SVD · LinearSVC calibrado")

try:
    controller = load_controller()
except Exception as exc:
    st.error(f"No fue posible cargar el modelo: {exc}")
    st.stop()

predict_tab, about_tab = st.tabs(["Clasificar texto", "Acerca del modelo"])

with predict_tab:
    st.write(
        "Ingrese un texto en español para estimar el Objetivo de Desarrollo "
        "Sostenible con el que guarda mayor relación semántica."
    )
    st.info(
        "El conjunto de entrenamiento contiene los ODS 1 a 16. "
        "El ODS 17 no es una salida posible del modelo."
    )

    with st.form("prediction_form"):
        text = st.text_area(
            "Texto para clasificar",
            height=190,
            placeholder=(
                "Ejemplo: La expansión de sistemas de energía solar puede mejorar "
                "el acceso a electricidad limpia en comunidades rurales."
            ),
        )
        submitted = st.form_submit_button("Clasificar texto", type="primary", width="stretch")

    if submitted:
        try:
            prediction = controller.predict(text, top_k=3)
        except PredictionInputError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"No fue posible generar la predicción: {exc}")
        else:
            left, right = st.columns([1, 2])
            left.metric("ODS predicho", f"ODS {prediction.ods}")
            right.markdown("**Objetivo**")
            right.markdown(f"### {prediction.name}")
            st.progress(
                prediction.probability,
                text=f"Probabilidad calibrada: {prediction.probability:.1%}",
            )

            if prediction.low_confidence:
                st.warning(
                    "La probabilidad es baja. Amplíe el texto y solicite revisión humana."
                )

            alternatives = pd.DataFrame([
                {
                    "ODS": f"ODS {item.ods}",
                    "Objetivo": item.name,
                    "Probabilidad": item.probability,
                }
                for item in prediction.alternatives
            ])
            st.subheader("Alternativas principales")
            st.dataframe(
                alternatives.style.format({"Probabilidad": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )

with about_tab:
    metadata = controller.metadata
    metrics = metadata.get("test_metrics", {})
    st.subheader("Metodología")
    st.markdown(
        """
        El modelo utiliza una bolsa de palabras ponderada con TF-IDF, reducción de
        dimensionalidad mediante SVD truncado, estandarización y una máquina de
        soporte vectorial lineal. Sus márgenes fueron calibrados con el método
        sigmoidal para producir probabilidades estimadas.
        """
    )

    if metrics:
        columns = st.columns(3)
        columns[0].metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
        columns[1].metric("F1 macro", f"{metrics.get('F1 macro', 0):.3f}")
        columns[2].metric("Balanced accuracy", f"{metrics.get('Balanced accuracy', 0):.3f}")

    st.subheader("Uso responsable")
    st.markdown(
        """
        - La predicción es una ayuda analítica y no sustituye la valoración de especialistas.
        - Un texto real puede relacionarse con varios ODS, aunque el modelo entregue uno.
        - La probabilidad calibrada cuantifica incertidumbre; no garantiza que la clase sea correcta.
        - Textos muy diferentes del corpus de entrenamiento pueden ser menos fiables.
        """
    )

    st.subheader("Información académica")
    st.markdown(
        """
        - **Autores:** Alejandro Ocampo Rojas y Mateo Alvarez Lopera
        - **Institución:** Universidad de los Andes
        - **Asignatura:** Machine Learning No Supervisado
        """
    )

st.divider()
st.caption(
    "Microproyecto 2 · Universidad de los Andes · "
    "Alejandro Ocampo Rojas y Mateo Alvarez Lopera"
)
