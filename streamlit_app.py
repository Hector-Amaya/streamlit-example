import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background: #212529;
        color: #ffffff;
    }
    .Widget>label {
        color: #ffffff;
    }
    .st-ck {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Esto crea una barra lateral en la aplicación
st.sidebar.title('Opciones')

# Agrega elementos a la barra lateral
option = st.sidebar.radio(
    'Selecciona una opción',
    ('Inicio', 'Estadíticas descriptivas', 'Patrones', 'Anomalías', 'Análisis de Estacionariedad')
)


# Mostrar el contenido principal en función de la opción seleccionada
if option == 'Inicio':
    st.markdown("<center><h1>Universidad de Colima</h1>", unsafe_allow_html=True)
    st.markdown("<center><h3>Campus Coquimatlán</h3>", unsafe_allow_html=True)
    st.markdown("<center><h3>Facultad de Ingeniería Mecánica y Eléctrica</h3>", unsafe_allow_html=True)
    st.markdown("<center><h2><b>Proyecto de Ejercicios de la 2da parcial en Streamlit</b></h2>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4><b>Materia:</b> Análisis de Series Temporales</h4>", unsafe_allow_html=True)
    st.markdown("<h4><b>Maestro:</b> Mata López Walter Alexander</h4>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4><b>Alumno:</b> Amaya González Héctor Eduardo</h4>", unsafe_allow_html=True)
    st.markdown("<h4><b>No. de Cuenta:</b> 20186366</h4>", unsafe_allow_html=True)
    st.markdown("<h4><b>Carrera:</b> Ingeniería en Computación Inteligente</h4>", unsafe_allow_html=True)
    st.markdown("<h4><b>Semestre y Grupo:</b> 6°B</h4>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4><b>Fecha de Entrega:</b> 28/04/2024</h4>", unsafe_allow_html=True)


elif option == 'Estadíticas descriptivas':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')




elif option == 'Patrones':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')    



elif option == 'Anomalías':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')





elif option == 'Análisis de Estacionariedad':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')
