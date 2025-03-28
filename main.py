from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

lin = st.Page("utils/auth.py", title="Login")

bienvenida = st.Page("pages/_1_Bienvenida.py", title="Bienvenida", url_path="bienvenida")
objetivos = st.Page("pages/_2_Objetivos.py", title="Objetivos", url_path="objetivos")
metodologia = st.Page("pages/_3_Metodologia.py", title="Metodología", url_path="metodologia-del-curso")

mod1 = st.Page("pages/0_Modulo_1.py", title="Repaso de python", url_path="repaso-de-python")
mod2 = st.Page("pages/0_Modulo_2.py", title="EDA", url_path="eda")
mod3 = st.Page("pages/0_Modulo_3.py", title="Estadística descriptiva / inferencial", url_path="estadistica-descriptiva-inferencial")

mod4 = st.Page("pages/1_Modulo_4.py", title="Fundamentos de Machine Learning", url_path="fundamentos-de-machine-learning")
mod5 = st.Page("pages/1_Modulo_5.py", title="Árboles de decisión y Random Forest", url_path="arboles-de-decision-y-bosques-aleatorios")
mod6 = st.Page("pages/1_Modulo_6.py", title="Regularización y NLP básico", url_path="introduccion-a-la-regularizacion-y-nlp-basico")

mod7 = st.Page("pages/2_Modulo_7.py", title="Topic modeling y LDA", url_path="topic-modeling-lda")
mod8 = st.Page("pages/2_Modulo_8.py", title="Monte Carlo y Cadenas de Markov", url_path="monte-carlo-y-cadenas-de-markov")
mod9 = st.Page("pages/2_Modulo_9.py", title="Micro-redes y Q-Learning", url_path="micro-redes-y-qlearning")
mod10 = st.Page("pages/2_Modulo_10.py", title="Deep Learning y NLP", url_path="deep-learning-y-nlp")

lout = st.Page(logout, title="Salir")

if st.session_state.access_granted:
    pg = st.navigation(
        pages={
            "Inicio": [bienvenida, objetivos,metodologia],
            # "Propedéutico": [mod1, mod2, mod3],
            "Machine Learning Clásico": [mod4, mod5, mod6],
            # "Modelos Avanzados": [mod7, mod8, mod9, mod10],
            "Cuenta": [lout]
        },
        expanded=True
    )

else:
    pg = st.navigation([lin])

pg.run()
