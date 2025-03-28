import streamlit as st
from code_editor import code_editor

import io, contextlib, traceback, nbformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    layout="wide"
)

@st.dialog("Error")
def msg(tipo):
    if tipo==1:
        st.warning("Ingresa un usuario")
    if tipo==2:
        st.error("El usuario no existe")

@st.dialog("Logout")
def logout():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()