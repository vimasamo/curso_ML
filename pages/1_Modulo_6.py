from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

st.subheader("Introducción a la regularización y NLP básico")
st.write("Contenido próximamente...")
