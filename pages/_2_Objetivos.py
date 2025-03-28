from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

st.title("Capacitación para Analista de I+D")
st.divider()

st.header("Objetivos")
st.subheader("Objetivos generales")
st.write(
    """
    1. Fortalecer las bases de Python enfocadas en la manipulación de datos y el desarrollo de modelos de Machine Learning.
    2. Introducir fundamentos de estadística y matemáticas aplicados al Machine Learning.
    3. Demostrar el flujo de trabajo de un proyecto de ML/IA: desde la obtención y transformación de datos hasta la interpretación y despliegue de resultados.
    4. Sentar las bases teóricas para abordar técnicas de ML tradicionales y algunas más avanzadas como son: redes neuronales, series de tiempo, modelado de tópicos, entre otras, que se verán en cursos posteriores.
    """
)
st.subheader("Objetivos específicos")
st.write(
    """
    - Ofrecer un propedéutico para perfiles menos familiarizados con Python, que incluya los siguientes temas:
        - configuración de un ambiente de trabajo y repaso de python3,
        - fundamentos de Análisis Exploratorio de Datos (EDA), y,
        - estadística descriptiva e inferencial aplicada.
    - Enseñar los fundamentos de Machine Learning clásico, que incluya los siguientes temas:
        - aprendizaje supervisado y no supervisado,
        - problemas de regresión y clasificación,
        - métricas de rendimiento de los modelos,
        - algoritmo de árboles de decisión, concepto de _ensemble_ y Random Forest,
        - introducción a los conceptos de regularización y sobreajuste,
        - penalizaciones _Lasso_ y _Ridge_,
        - ejemplo básico de NLP (Natural Language Processing). 
    """
)