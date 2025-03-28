from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

st.title("Capacitación para Analista de I+D")
st.divider()

st.header("Introducción")
st.write(
    """
    Este curso de capacitación está diseñado para facilitar la trancisión de un perfil técnico-analista hacia un Analista de I+D.

    Pretendemos, en términos generales:
    - Enseñar un manejo de Python que permita resolver problemas utilizando imágenes, texto, bases estructuradas, series de tiempo, incluso cuando no existen suficientes datos.
    - Algoritmos útiles para resolver la inmensa mayoría de las posibles tareas a las que se puede enfrentar un analista.
    - Una explicación matemática sencila de cómo funcionan los algoritmos que le permitan al analista interpretar sus resultados, así como distinguir las ventajas y retos en cada caso.
    - Proveer una mezcla de teoría simple de casos de uso, teoría matemática y práctica, para acelerar el aprendizaje.
    """
)