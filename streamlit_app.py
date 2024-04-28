import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
# import statistics
# import seaborn as sns
# from statsmodels.tsa.seasonal import seasonal_decompose


# Esto crea una barra lateral en la aplicación
st.sidebar.title('Opciones')

# Agrega elementos a la barra lateral
option = st.sidebar.radio(
    'Selecciona una opción',
    ('Inicio', 'Estadíticas descriptivas', 'Patrones', 'Anomalías', 'Análisis de Estacionariedad')
)


# Mostrar el contenido principal en función de la opción seleccionada
if option == 'Inicio':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')

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
