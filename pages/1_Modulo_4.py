from utils.defs import *

if "access_granted" not in st.session_state or not st.session_state["access_granted"]:
    st.session_state.access_granted = False

@st.cache_data
def load_df():
    df_iris = load_iris(as_frame=True)
    df_housing = fetch_california_housing(as_frame=True)
    return df_iris, df_housing

iris, housing = load_df()

st.subheader("Fundamentos de Machine Learning")
st.divider()

with st.expander("**¿Qué es el Machine Learning?**"):
    st.write(
        """
        El Machine Learning (ML), o Aprendizaje Automático, es una rama de la Inteligencia artificial (IA) que estudia como dotar a las máquinas de capacidad de aprendizaje, basándose en algoritmos capaces de identificar patrones en grandes bases de datos y aprender de ellos.
        """
    )
    st.image(
        image='https://www.masterdatascienceucm.com/wp-content/uploads/2020/12/Conceptos-IA-Machine-Learning-y-Deep-Learning.png',
        caption="Conceptos de IA, Machine Learning y Deep Learning. Imagen de Máster Ciencia de Datos UCM (2020)."
    )

with st.expander("**Tipos de aprendizaje automático**"):
    st.write(
        """
        Un sistema de ML se nutre de experiencias y evidencias en forma de datos, con los cuales aprenderá a interpretar por si mismo patrones y/o comportamientos.
        
        Gracias a este _aprendizaje_, las máquinas son capaces de crear predicciones y aportar soluciones en un campo concreto.

        A partir de un gran número de ensayos y errores, se puede elaborar un modelo de predicción lo más ajustado posible al mínimo error y generalizar un comportamiento ya observado, por ejemplo, se puede predecir el valor un acción del mercado de valores en el futuro si se analiza y entrena un modelo con una cantidad de datos suficientes del pasado.

        En general, podemos decir que exiten tres tipos de aprendizaje automático:
        """
    )
    st.image(
        image='https://www.masterdatascienceucm.com/wp-content/uploads/2020/12/tipos-de-machine-learning.png.webp',
        caption="Tipos de Machine Learning. Imagen de Máster Ciencia de Datos UCM (2020)."
    )
    st.write(
        """
        Durante la primera parte del curso abordaremos los primeros dos: **supervisado** y **no supervisado**.

        En los modelos avanzados daremos un vistazo al **aprendizaje por refuerzo**.
        """
    )

with st.expander("**Aprendizaje supervisado**"):
    
    tab1, tab2, tab3 = st.tabs(["Teoría básica","Práctica clasificación","Práctica regresión"])
    
    with tab1:
        st.write(
            """
            Anteriormente mencionamos que el sistema de ML necesita datos de los cuales aprenderá. El aprendizaje supervisado consiste en aprender de datos de los cuáles ya se saben los resultados, es decir, _datos etiquetados_.

            Dichos datos pueden ser:
            - formato tabular (tablas): donde algunas columnas serán las características y una columna será la etiqueta o valor a predecir,
            - imágenes: que representen un concepto o clase, y cada imagen sepamos qué representa,
            - objetos más complejos: correos electrónicos, grafos, etc.



            Durante el curso, diremos que el apredizaje supervisado servirá para resolver dos distintos tipos de problemas:  
            """
        )
        st.image(
            image='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*1pz7fPDG8KRE9VenFXFdhw.png',
            caption="Diferencias entre IA, Machine Learning y Deep Learning. Imagen de Medium (s.f.)."
        )
        st.write(
            """
            **Clasificación**
            
            Los problemas de clasificación tratan sobre asignar una etiqueta a un ejemplo no etiquetado. La detección de correo electrónico no deseado es un ejemplo famoso de clasificación.

            Diríamos que una etiqueta pertenece un conjunto finito de clases.
            Si el conjunto tiene dos clases (spam / no spam, enfermo / sano), hablaríamos de un problema de clasificación binaria (también llamado **binomial**).
            La _clasificación multiclase_ (también llamada **multinomial**) obecede a problemas de clasificación con tres o más clases.

            Dicho de una forma más técnica, el modelo de ML aprenderá a predecir la clase (respuesta) a la cuál pertenecen los datos basándose únicamente en sus características (predictores).

            **Regresión**

            Los problemas de regresión se usan para predecir un número real dado un ejemplo con ciertas características, por ejemplo, predecir el precio de una casa basándose en datos como: ubicación, tamaño en m2, no. de habitaciones, etc.

            En términos muy simplificados, _podríamos_ decir que mientras la regresión predice números, la clasificación predice etiquetas.
            
            **_Pregunta de exámen..._**
            
            _Si tengo un modelo de ML que tiene como posibles respuestas únicamente 0 y 1, ¿es un modelo de clasificación o de regresión?_
            """
        )
    
    with tab2:

        st.markdown(
"""
##### =====================================
##### EJERCICIOS DE MACHINE LEARNING BÁSICO
##### =====================================
"""
        )
        st.divider()
        
        st.write(
            """
           **ESTRUCTURA DEL EJERICIO**

            1. Preparar los datos
                - Obtener un conjunto de datos etiquetado
                - Determinar cuáles son las características _predictoras_
                - Determinar cuál es la catacterística a predecir
            2. Elegir un algoritmo de clasificación y entrenarlo
                - kNN (k-Nearest Neighbours | k-vecinos más cercanos) para clasificación
            3. Evaluar el modelo
                - Accuracy, Precision, Recall, F1-Score
            """
        )

        st.divider()

        st.write(
            """
            **DEFINICIÓN DEL PROBLEMA**
            
            Basándonos en el dataset **Iris** (https://www.kaggle.com/datasets/uciml/iris), intentaremos clasificar las flores basádas en sus características.

            Exploremos el dataset (solo mostraremos las 30 filas al azar):
            """
        )

        st.dataframe(iris.frame.sample(30), hide_index=True)
        targets = {k:v for k, v in enumerate(iris.target_names)}
        col1, col2, col3 = st.columns([1,2,1])
        col2.caption(
            f"""
            - Las columnas **sepal lenght**, **sepal width**, **petal length** y **petal width** serán los predictores.
                - sepal lenght = largo del sépalo de la flor
                - sepal width = ancho del sépalo de la flor
                - petal lenght = largo del pétalo de la flor
                - petal width = ancho del pétalo de la flor
            - La columna **target** será la clase a encontrar (nombre del tipo de flor).
                - 0 = {targets[0]}
                - 1 = {targets[1]}
                - 2 = {targets[2]}
            - Aunque en el dataset el _target_ es un número, internamente el algoritmo hace uso de un diccionario para tratarlo como una etiqueta
            """
        )
        col1.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Petal-sepal.jpg")
        col3.image("https://www.math.umd.edu/~petersd/666/html/iris_with_labels.jpg")

        st.divider()
        
        with st.container():
            st.caption("usa este control para manipular el *vector_caracteristicas* del código")
            col1, col2, col3, col4 = st.columns(4)
            sepal_lenght = col1.number_input("sepal_lenght", value=7.7)
            sepal_width = col2.number_input("sepal_width", value=3.8)
            petal_lenght = col3.number_input("petal_lenght", value=6.7)
            petal_width = col4.number_input("petal_width", value=2.2)
        
        st.divider()
        
        st.write(
            """
            Código para entrenar este modelo en python:
            """
        )
        class_train_code = f"""
# importar las librerías
import numpy as np  # para manipulación de datos
from sklearn.datasets import load_iris  # importar el dataset desde internet
from sklearn.neighbors import KNeighborsClassifier  # importar la clase para declarar el modelo
from sklearn.model_selection import train_test_split  # para separar el dataset en entrenamiento y validación

# cómo importar los datos
iris = load_iris()  # load_iris() devuelve un diccionario...
X_predictores = iris.data  # el elemento ["data"] contiene los predictores...
y_target = iris.target  # el elemento ["target"] contiene únicamente la columna target...

# dividir en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(
    X_predictores, y_target, test_size=0.25  # se usará 25% para validación
)

# definir el modelo a utilizar (kNN)
modelo = KNeighborsClassifier()

# entrenar el modelo
modelo.fit(X = X_train, y = y_train)

# hacer predicciones con el modelo entrenado
vector_caracteristicas = np.array([{sepal_lenght}, {sepal_width}, {petal_lenght}, {petal_width}]).reshape(1, -1)
prediccion = modelo.predict(vector_caracteristicas)
print("La clase predicha por el modelo es:", prediccion)
            """
        st.code(class_train_code)
        btn_class_train_code = st.button(f"▶️ Ejecutar código", key="btn_class_train_code")
        if btn_class_train_code:
            modelo = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=0.25
            )
            modelo.fit(X = X_train, y = y_train)
            vector_caracteristicas = np.array([sepal_lenght, sepal_width, petal_lenght, petal_width]).reshape(1, -1)
            prediccion = modelo.predict(vector_caracteristicas)
            st.code(f"La clase predicha por el modelo es: {prediccion[0]}")
        
        st.divider()

        st.write("Código adicional para evaluar el modelo en python")

        class_eval_code = """
# importar funciones para evaluar el modelo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predecir sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))  # Qué tan frecuentemente acierta el modelo en general.
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))  # Qué porcentaje de las predicciones positivas fueron correctas.
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))  # Qué porcentaje de las verdaderas clases positivas fueron encontradas.
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))  # Promedio armónico entre Precision y Recall. Útil cuando hay clases desbalanceadas.
# average='macro': Calcula la métrica de cada clase por separado y luego saca el promedio.

# Reporte más completo
print('Reporte completo:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
"""
        st.code(class_eval_code)
        btn_class_eval_code = st.button(f"▶️ Ejecutar código", key="btn_class_eval_code")
        # st.caption("cada vez que ejecutas esta parte del código, se re-entrena el modelo y puedes obtener resultados distintos")
        if btn_class_eval_code:
            modelo = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=0.25
            )
            modelo.fit(X = X_train, y = y_train)
            # Evaluar el modelo
            y_pred = modelo.predict(X_test)

            acc_res =  accuracy_score(y_test, y_pred)
            pres_res = precision_score(y_test, y_pred, average='macro')
            rec_res = recall_score(y_test, y_pred, average='macro')
            f1_res = f1_score(y_test, y_pred, average='macro')
            class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

            resultados = f"""
Accuracy: {acc_res:.4f}  # Qué tan frecuentemente acierta el modelo en general.
Precision (macro): {pres_res:.4f}  # Qué porcentaje de las predicciones positivas fueron correctas.
Recall (macro): {rec_res:.4f}  # Qué porcentaje de las verdaderas clases positivas fueron encontradas.
F1 Score (macro): {f1_res:.4f}  # Promedio armónico entre Precision y Recall. Útil cuando hay clases desbalanceadas.

Reporte completo:
{class_report}
"""
            st.code(resultados)

        st.divider()

        st.write(
            """
            **RECURSOS ADICIONALES**

            - https://youtu.be/0p0o5cmgLdE?si=G_369t76jahqTTby
            - https://youtu.be/FHHuo7xEeo4?si=nOzjBsgL4ZGcHhbQ

            """
        )


    with tab3:

        st.markdown(
"""
##### =====================================
##### EJERCICIOS DE MACHINE LEARNING BÁSICO
##### =====================================
"""
        )
        st.divider()
        
        st.write(
            """
            **ESTRUCTURA DEL EJERCICIO**

            1. Preparar los datos
                - Obtener un conjunto de datos etiquetado
                - Determinar cuáles son las características _predictoras_
                - Determinar cuál es la catacterística a predecir
            2. Elegir un algoritmo de clasificación y entrenarlo
                - Linear regression (regresión lineal) para regresión.
            3. Evaluar el modelo
                - Accuracy, Precision, Recall, F1-Score
            """
        )
        st.divider()

        st.write(
            """
            **DEFINICIÓN DEL PROBLEMA**

            Basándonos en el dataset **California Housing Prices** (https://www.kaggle.com/datasets/camnugent/california-housing-prices), intentaremos predecir el precio de la casa basándonos en sus características.

            Exploremos el dataset (solo mostraremos las 30 filas al azar):
            """
        )

        st.dataframe(housing.frame.sample(30), hide_index=True)
        col1, col2 = st.columns([3,1])
        col1.caption(
            f"""
            - Cada fila del dataset representa una "cuadra" de un vecindario, motivo por el cual muchas columnas son un promedio.
            - Las columnas **MedInc**, **HouseAge**, **AveRooms**, **AveBedrms**, **Population**, **AveOccup**, **Latitude** y **Longitude** serán los predictores.
                - MedInc = ingreso medio anual de los habitantes de las casas (medido en decenas de miles de dólares)
                - HouseAge = edad media de las casas (entre más bajo más nueva)
                - AveRooms = número total de espacios promedio de las casas (recámaras, cocina, estancia, etc.)
                - AveBedrms = número total de habitaciones promedio de las casas
                - Population = total de personas viviendo en la cuadra
                - AveOccup = personas promedio viviendo en cada casa de la cuadra
                - Latitude = valor de que tan al norte está la casa (entre más alto más al norte)
                - Longitude = valor de que tan al oeste está la casa (entre más alto más al oeste)
            - La columna **MedHouseVal** representa el valor numérico a predecir: Valor promedio de las casas en la cuadra (medido en decenas de miles de dólares)
            """
        )
        col2.image("https://storage.googleapis.com/kaggle-datasets-images/1446091/2391843/2b2935fb8f7e9cf17349d6dd4ec36570/dataset-card.jpg?t=2021-07-03-15-46-44")

        st.divider()

        with st.container():
            st.caption("usa este control para manipular el *vector_caracteristicas* del código")
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            MedInc = col1.number_input("MedInc", value=3.6)
            HouseAge = col2.number_input("HouseAge", value=28)
            AveRooms = col3.number_input("AveRooms", value=5.6)
            AveBedrms = col4.number_input("AveBedrms", value=1)
            Population = col5.number_input("Population", value=1613)
            AveOccup = col6.number_input("AveOccup", value=3.75)
            Latitude = col7.number_input("Latitude", value=33.87)
            Longitude = col8.number_input("Longitude", value=-118.07)
        st.divider()
        
        st.write(
            """
            Código para entrenar este modelo en python:
            """
        )
        reg_train_code = f"""
# importar las librerías
import numpy as np  # para manipulación de datos
from sklearn.datasets import fetch_california_housing  # importar el dataset desde internet
from sklearn.linear_model import LinearRegression  # importar la clase para declarar el modelo
from sklearn.model_selection import train_test_split  # para separar el dataset en entrenamiento y validación

# cómo importar los datos
housing = fetch_california_housing()  # fetch_california_housing() devuelve un diccionario...
X_predictores = housing.data  # el elemento ["data"] contiene los predictores...
y_target = housing.target  # el elemento ["target"] contiene únicamente la columna target...

# dividir en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(
    X_predictores, y_target, test_size=0.25  # se usará 25% para validación
)

# definir el modelo a utilizar (kNN)
modelo = LinearRegression()

# entrenar el modelo
modelo.fit(X = X_train, y = y_train)

# hacer predicciones con el modelo entrenado
vector_caracteristicas = np.array([{MedInc}, {HouseAge}, {AveRooms}, {AveBedrms}, {Population}, {AveOccup}, {Latitude}, {Longitude}]).reshape(1, -1)
prediccion = modelo.predict(vector_caracteristicas)
print("El precio predicho por el modelo es:", prediccion[0])
            """
        st.code(reg_train_code)
        btn_reg_train_code = st.button(f"▶️ Ejecutar código", key="btn_reg_train_code")
        if btn_reg_train_code:
            modelo = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(
                housing.data, housing.target, test_size=0.25
            )
            modelo.fit(X = X_train, y = y_train)
            vector_caracteristicas = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]).reshape(1, -1)
            prediccion = modelo.predict(vector_caracteristicas)
            st.code(f"El precio predicho por el modelo es: {prediccion[0]}")
        
        st.divider()

        st.write("Código adicional para evaluar el modelo en python")

        class_eval_code = """
# importar funciones para evaluar el modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predecir sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")  # En promedio, cuánto se equivoca el modelo. Cuanto más cercano a cero mejor.
print(f"MSE: {mse:.4f}")  # Igual que MAE, pero penaliza más los errores grandes.
print(f"RMSE: {rmse:.4f}")  # Como MAE, pero tiene en cuenta la escala de los datos.
print(f"R²: {r2:.4f}")  # Cuánto del comportamiento del precio se explica con las variables. Cuanto más cerca de 1, mejor.
"""
        st.code(class_eval_code)
        btn_reg_eval_code = st.button(f"▶️ Ejecutar código", key="btn_reg_eval_code")
        # st.caption("cada vez que ejecutas esta parte del código, se re-entrena el modelo y puedes obtener resultados distintos")
        if btn_reg_eval_code:
            modelo = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(
                housing.data, housing.target, test_size=0.25
            )
            modelo.fit(X = X_train, y = y_train)
            y_pred = modelo.predict(X_test)
            
            # Evaluar el modelo
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            resultados = f"""
MAE: {mae:.4f}  # En promedio, cuánto se equivoca el modelo. Cuanto más cercano a cero mejor.
MSE: {mse:.4f}  # Igual que MAE, pero penaliza más los errores grandes.
RMSE: {rmse:.4f}  # Como MAE, pero tiene en cuenta la escala de los datos.
R²: {r2:.4f}  # Cuánto del comportamiento del precio se explica con las variables. Cuanto más cerca de 1, mejor.
"""
            st.code(resultados)

        st.divider()

        st.write(
            """
            **RECURSOS ADICIONALES**

            - https://youtu.be/0p0o5cmgLdE?si=G_369t76jahqTTby
            - https://youtu.be/FHHuo7xEeo4?si=nOzjBsgL4ZGcHhbQ

            """
        )