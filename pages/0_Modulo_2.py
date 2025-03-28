from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

st.title("Módulo 2")
st.write("Contenido próximamente...")
