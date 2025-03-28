from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

st.title("Capacitación para Analista de I+D")
st.divider()

st.header("Metodología")
st.subheader("Teoría y práctica")
st.write(
    """
    Cada módulo combinará una porción teórica y práctica orientada a:
    - entender los conceptos del tema, la motivación detrás de su uso y el fundamento matemático básico (cuando aplique),
    - cuadernos de práctica (Jupyter) para aprender a programar la teoría.
    """
)
st.subheader("Casos de uso")
st.write(
    """
    Se usarán conjuntos de datos públicos para aplicar la teoría a casos de uso comunmente aceptados."
    """
)
st.subheader("Proyecto integrador")
st.write(
    """
    Cada módulo incluirá un proyecto integrador, que incluya:
    - extracción y limpieza de datos desde SQL,
    - preparación y entrenamiento de modelos en Python,
    - visualización y entendimiento de los resultados.
    """
)