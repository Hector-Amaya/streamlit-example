import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import statistics
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

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
    st.markdown("""
    #### **Problema**

    **Contexto:** Una empresa desea realizar un análisis estadístico de los salarios anuales de sus empleados. 
    El propósito de este análisis es obtener una mejor comprensión de la distribución de los ingresos entre los empleados, 
    lo que permitirá a la empresa tomar decisiones informadas respecto a la equidad salarial y la estructura de compensaciones.

    **Objetivo:** Como parte de un proyecto de análisis de datos, se te ha asignado la tarea de calcular las estadísticas 
    descriptivas básicas de los salarios anuales en la empresa. Específicamente, deberás calcular la media, mediana, moda, 
    varianza y desviación estándar de los salarios. Además, deberás interpretar estas estadísticas para discutir la equidad 
    salarial y la dispersión de los salarios.
    """)

    st.markdown("""
    #### **Instrucciones**

    ### 1. **Generar Datos:** Utiliza el siguiente código en Python para generar una muestra aleatoria de salarios anuales. Esta muestra simulará los salarios anuales de los empleados de la empresa.
    """)

    # Configurar la semilla del generador aleatorio para reproducibilidad
    np.random.seed(29)

    # Generar datos de salarios anuales (simulados)
    salarios = np.random.normal(loc=50000, scale=15000, size=250) 
    # media=50000, desv_std=15000, n=250

    # Asegurarse de que todos los salarios sean positivos
    salarios = np.abs(salarios)
    st.write("Los salarios son:")
    st.write(salarios)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### 2. **Calcular Estadísticas Descriptivas:** 

    - Calcula la media, mediana, moda, varianza y desviación estándar de los salarios generados.

    - Puedes usar las librerías numpy para media, mediana, varianza y desviación estándar, y scipy.stats o statistics para la moda.
    """)

    #Media
    media = np.mean(salarios)
    st.write("La media de los salarios es: {:.2f}".format(media))

    #Mediana
    mediana = np.median(salarios)
    st.write("La mediana de los salarios es: {:.2f}".format(mediana))

    #Moda
    moda = statistics.mode(salarios)
    st.write("La moda de los salarios es: {:.2f}".format(moda))

    #Varianza
    varianza = np.var(salarios)
    st.write("La varianza de los salarios es: {:.2f}".format(varianza))

    #Desvición Estándar
    desv_est = np.std(salarios)
    st.write("La desviación estándar de los salarios es: {:.2f}".format(desv_est))

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### 3. **Interpretar los Resultados:**

    - Discute qué te indican la media y la mediana sobre la tendencia central de los salarios.

    - Analiza la moda y qué dice sobre los salarios más comunes dentro de la empresa.

    - Interpreta la varianza y la desviación estándar en términos de dispersión de los salarios. 
    ¿Los salarios están agrupados cerca de la media o dispersos?
    """)

    #Media, Moda y Mediana gráficadas 
    indices = np.arange(len(salarios))
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, salarios, edgecolors='black')
    plt.axhline(media, color='r', linestyle='-', label=f'Media: {media:.2f}')
    plt.axhline(mediana, color='b', linestyle='--', label=f'Mediana: {mediana:.2f}')
    plt.axhline(moda, color='g', linestyle='--', label=f'moda: {moda:.2f}')
    plt.title('Salarios de 250 empleados')
    plt.xlabel('Cantidad de empleados')
    plt.ylabel('Salarios')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Mostrar el gráfico
    plt.legend()
    st.pyplot(plt)

    st.markdown("""
    ##### La media indica que la mayoria de los trabajadores tiene un salario similar haciendo que el salario promedio de los empleados sea de 49,622.77.

    ##### La mediana indica que si acomodamos todos los salarios en orden, el salario que queda justo enmedio es de 50,066.49, el hecho de que sea un valor cercano a la media indica que los salarios están más agurpados cerca de la media.
    
    ##### La moda representa el salario que más se repite entre los trabajadores, por lo que el salario más común es 43,737.77
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    plt.figure(figsize=(8, 6))
    plt.boxplot(salarios, vert=False)
    plt.xlabel('Salarios')
    plt.title('Salarios de 250 empleados')
    plt.grid()
    st.pyplot(plt)

    indices = np.arange(len(salarios))
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, salarios, edgecolors='black')
    plt.axhline(media, color='r', linestyle='-', label=f'Media: {media:.2f}')
    plt.axhline(media + desv_est, color='g', linestyle='--', label=f'+1 Desv. Est.: {media + desv_est:.2f}')
    plt.axhline(media - desv_est, color='g', linestyle='--', label=f'-1 Desv. Est.: {media - desv_est:.2f}')
    plt.title('Salarios de 250 empleados')
    plt.xlabel('Cantidad de empleados')
    plt.ylabel('Salarios')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Mostrar el gráfico
    plt.legend()
    st.pyplot(plt)

    st.markdown("""
    ##### En estas dos gráficas de arriba se puede apreciar que los datos si están agrupados cerca de la media, y esto lo podemos hacer gracias a la desviación estandar.

    ##### En la primera gráfica se puede apreciar gracias a la caja, ya que eso representa donde están agrupados los datos, los cuales están donde está la media, las lineas representan que ahí hay algunos datos más dispersos y los círculos de los lados representan que hay unos valores atípicos, lo cuales están muy alejados de la media, por eso se consideran valores atípicos.

    ##### En la segunda gráfica al marcar con lineas las desviaciones podemos apreciar que los datos si están más agrupados cerca de la media, y los que se salen de ese rango de las desviaciones, son valores que se encuentran más dispersos y lejanos de la media, aunque la mayoría está más cerca de la media.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### 4. **Informe:**

    - Escribe un informe breve que incluya los cálculos realizados, las gráficas
    pertinentes (como histogramas o gráficos de cajas), y una discusión sobre las
    implicaciones de estas estadísticas en términos de equidad salarial y política
    de remuneraciones de la empresa.
    """)

    st.markdown("""
    ### **Informe Anual de los salarios de los empleados**

    En este trabajo calculamos la media, mediana, moda, varianza y desviación estándar de los salarios de 
    250 empleados para discutir sobre las implicaciones de están estadísticas en términos de equidad salarial y 
    política de remuneraciones de la empresa.

    Media: 49622.77

    Mediana: 50066.49

    Moda: 43737.76810543171

    Varianza: 227508380.75

    Desviación Estándar: 15083.38
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    En la siguiente gráfica se puede apreciar la distribución de los datos, 
    como los valores de la media, mediana y moda concuerdan con las freciencias de los datos, 
    y como a partir de la media la dispersión de los datos hacia la derecha y la izquierda es muy equitativa.
    """)

    # Crear el histograma
    fig, ax = plt.subplots()
    ax.hist(salarios, bins=10, alpha=0.5, color='blue', edgecolor='black')
    ax.axvline(media, color='r', linestyle='-', label=f'Media: {media:.2f}')
    ax.axvline(mediana, color='b', linestyle='--', label=f'Mediana: {mediana:.2f}')
    ax.axvline(moda, color='g', linestyle='--', label=f'Moda: {moda:.2f}')
    ax.set_title('Salarios de 250 empleados')
    ax.set_xlabel('Sueldos')
    ax.set_ylabel('Frecuencia')
    ax.grid()
    ax.legend()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    En la siguiente gráfica podemos apreciar la densidad de los datos, esto representa que son pocos de 
    datos se encuentran de 0 a 2,000 y de 8,000 a 100,000; y a partir de ahí los datos van compactandose hasta quedar de 
    3,800 a 5,500; pero esa densidad lo que indicae es que la mayoría de datos se encuentra por la media.
    """)

    fig, ax = plt.subplots()
    sns.kdeplot(salarios, fill=True, color="r", alpha=0.5)
    plt.title('Densidad del salario de 250 empleados')
    plt.xlabel('Sueldos')
    plt.ylabel('Densidad')
    plt.grid(fig)

    # Mostrar el gráfico en Streamlit
    st.pyplot()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    **Discusión**

    En cuanto a equidad salarial estas estadísticas índican que la mayoría de empleados tienen un salario similar, 
    y eso lo demuestra la media y las desviaciones ya que hay una mayor agrupación de datos ahí, hay algunas personas con un 
    salario más bajo, y otras con un salario más alto, pero eso suele ser normal en los trabajos, ya que demuestran o que 
    tienen un puesto con pocas obligaciones o un puesto con muchas; pero si hablamos de equidad salarial, 
    habría que ver si son injusto esos salarios de acuerdo a la carga de trabajo de los empleados.

    En cuanto a la política de remuneraciones de la empresa, si consideramos que si hay una equidad salarial, 
    la política de remuneraciones de la empresa está bien redactada, y no se está incumpliendo.
    """)




elif option == 'Patrones':
    st.markdown("""
    ## Problema de identificación de patrones en datos de ventas mensuales

    En una empresa de venta al por menor, se ha recopilado un conjunto de datos que registra las ventas mensuales de 
    varios productos durante un periodo de varios años. Tu tarea es analizar estos datos para identificar 
    posibles patrones o tendencias en las ventas mensuales.
    """)

    st.markdown("""
    ### **Parte 1: Generación de Datos**

    Utiliza un programa en Python para generar datos simulados que representen las ventas mensuales de 
    varios productos a lo largo de un período de tiempo. Los datos deben incluir al menos 3 productos 
    diferentes y abarcar un periodo de al menos 3 años.
    """)

    # Definir los productos y el rango de fechas
    productos = ['Xbox Series X', 'PlayStation 5', 'Nintendo Switch']
    fechas = pd.date_range(start='1/1/2019', end='12/31/2021', freq='ME')

    # Generar datos de ventas aleatorios para cada producto en cada mes
    ventas = pd.DataFrame(index=fechas)
    for producto in productos:
        ventas[producto] = np.random.randint(50, 500, size=len(fechas))

    st.write("Los datos de ventas mensuales son: ")
    st.write(ventas)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### **Parte 2: Análisis de Datos**

    Una vez que hayas generado los datos, realiza un análisis para identificar posibles 
    patrones en las ventas mensuales. Algunas preguntas que podrías explorar incluyen:
    - ¿Hay algún patrón estacional en las ventas de ciertos productos?
    - ¿Se observa alguna tendencia de crecimiento o decrecimiento en las ventas a lo largo del tiempo?
    - ¿Existen meses específicos en los que las ventas tienden a ser más altas o más bajas?
    """)

    ## Descomposición de la serie temporal
    result = seasonal_decompose(ventas['Xbox Series X'], model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 8) # Ajustar tamaño de la figura para mejor visualización

    # Mostrar la figura en Streamlit
    st.pyplot(fig)






elif option == 'Anomalías':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')





elif option == 'Análisis de Estacionariedad':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')
