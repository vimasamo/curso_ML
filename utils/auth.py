from utils.defs import *

if "usuario" not in st.session_state:
    st.session_state.usuario = ""

st.title("Capacitación para Analista de I+D")
st.divider()
st.header("Inicio de Sesión")
with st.form("login"):
    email = st.text_input("Ingresa tu correo electrónico", value="", placeholder="correo@ejemplo.com")

    if st.form_submit_button("Ingresa"):

        if email:
            authorized = st.secrets["auth"]["authorized_emails"]
            if email in authorized:
                st.session_state.access_granted = True
                st.session_state.usuario = email
                st.rerun()  # Forzar recarga para ocultar pantalla de login
            else:
                msg(2)
                st.stop()
        elif not email:
            msg(1)
        else:
            st.stop()
