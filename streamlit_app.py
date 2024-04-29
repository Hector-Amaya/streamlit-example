import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import statistics
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import random

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

    st.title("Análisis de Serie Temporal de Ventas del Xbox Series X")
    st.write("A continuación, se muestra el descomposición estacional de las ventas del Xbox Series X.")

    ## Descomposición de la serie temporal
    result = seasonal_decompose(ventas['Xbox Series X'], model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 8) # Ajustar tamaño de la figura para mejor visualización

    # Mostrar la figura en Streamlit
    st.pyplot(fig)

    st.markdown("""
    #### Análisis de las ventas del producto Xbox Series X

    La primera gráfica muestra fluctuaciones a lo largo del tiempo con picos y valles. 
    En la segunda, parece haber una tendencia general al alza hasta principios de 2021 
    seguida por una ligera disminución. Esto podría indicar un crecimiento gradual en los 
    datos relacionados con la Xbox Series X durante ese período.

    La tercera gráfica parece indicar patrones estacionales con picos y caídas 
    regulares que se repiten a lo largo del tiempo. Esto podría estar relacionado con 
    eventos específicos, como lanzamientos de juegos o temporadas de vacaciones.

    La cuarta gráfica muestra puntos dispersos que representan el residuo después de 
    eliminar la tendencia y la estacionalidad. Estos puntos no muestran un patrón claro, 
    lo cual es esperado en un componente residual. Puede haber factores no modelados o 
    aleatorios que contribuyan a esta variabilidad.
    """)

    # Interfaz de usuario
    st.title("Análisis de Serie Temporal de Ventas de PlayStation 5")
    st.write("A continuación, se muestra el descomposición estacional de las ventas de PlayStation 5.")

    # Descomposición estacional
    result = seasonal_decompose(ventas['PlayStation 5'], model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 8)
    
    st.pyplot(fig)

    st.markdown("""
    #### Análisis de las ventas del producto PlayStation 5

    La primera gráfica muestra una serie temporal con fluctuaciones a lo largo del tiempo. 
    En la segunda se observa una tendencia clara hacia abajo de mediados de 2020 en adelante. 
    A mediados de 2021 se puede apreciar una alza en la tendencia.
    
    La tercera gráfica presenta un patrón repetitivo con picos y valles. 
    Esto sugiere que hay una estacionalidad en los datos, es decir, ciertos patrones que 
    se repiten en intervalos regulares.
    
    La cuarta gráfica muestra puntos dispersos alrededor del eje cero. 
    Estos puntos representan las fluctuaciones aleatorias o el ruido en los datos después 
    de haber eliminado la tendencia y la estacionalidad. No se observa un patrón claro en 
    estos residuos.
    """)

    st.title("Análisis de Serie Temporal de Ventas de Nintendo Switch")
    st.write("A continuación, se muestra el descomposición estacional de las ventas de Nintendo Switch.")
    
    # Descomposición estacional
    result = seasonal_decompose(ventas['Nintendo Switch'], model='additive')
    fig = result.plot()
    fig.set_size_inches(10, 8)
    
    st.pyplot(fig)

    st.markdown("""
    #### Análisis de las ventas del producto Nintendo Switch

    La primera gráfica muestra fluctuaciones en las ventas a lo largo del tiempo. 
    En la segunda gráfica no se observa una tendencia clara ya que a mediados del 2019 a 
    mediados de 2020 la tendencia iba a la alta, pero decreció por 4 meses, volvió a 
    despuntar hacia arriba hasta que al final de 2020 fue decreciendo; a mediados de 2021 
    empezó a aumentar.
    
    La tercera gráfica presenta un patrón repetitivo con picos y valles. 
    Hay una estacionalidad en los datos de las ventas.
    
    La cuarta gráfica muestra puntos dispersos alrededor del eje cero. 
    Estos puntos representan las fluctuaciones aleatorias o el ruido en los datos. 
    No se observa un patrón claro en estos residuos.
    """)


    
    st.markdown("""
    ### **Parte 3: Informe de Resultados**

    Escribe un informe que resuma tus hallazgos. 
    Incluye gráficos o visualizaciones que ayuden a ilustrar los patrones identificados 
    en los datos. Además, discute cualquier insight o conclusión que hayas obtenido del 
    análisis de los datos.
    """)

    st.markdown("""
    #### **Informe**

    Como se aprecia en las gráficas las ventas son normales, 
    manteniendo periodos altos y bajos a lo largo de los 3 años, 
    si acaso al PlayStation le fue mal los 5 primeros meses del 2021, 
    la tendencia en los 3 productos es similar en cuanto a comportamiento, 
    ya que empieza siendo alta, decrece bastante y luego vuelve a levantarse, 
    eso demuestra que en los periodos de 2019 y 2021 las ventas fueron bastante bien, 
    el problema surge en 2020 que es el tiempo donde la tendencia es muy baja, 
    y aunque hay momentos donde intenta elevarse, no lo logra del todo hasta 2021.
    
    En cuanto a la estacionalidad, a pesar de que 2020 fue el año más duro según la tendencia, 
    la estacionalidad en los tres productos se mantuvo uniforme y repetitiva. 
    El residou en los tres producos se mantuvo pequeño, lo cual indica que la tendencia y 
    estacionalidad fueron capturados correctamente.
    """)

    for producto in productos:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(list(ventas.index), list(ventas[producto]), label=producto)  # Convertir ventas[producto] a lista
        ax.set_title(f'Ventas mensuales de {producto}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Ventas')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for producto in productos:
        ax.plot(list(ventas.index), list(ventas[producto]), label=producto)
    ax.set_title('Ventas mensuales')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.legend()
    ax.grid(True)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)




elif option == 'Anomalías':
    st.markdown("""
    ## Problema para la detección de anomalías en sensores de temperatura en una planta de manufactura

    En una planta de manufactura, se utilizan sensores para monitorear la temperatura 
    en diferentes partes de la línea de producción. Estos sensores registran lecturas de 
    temperatura cada hora durante un periodo de varios meses. Tu tarea es analizar estos 
    datos para identificar posibles anomalías en las lecturas de temperatura que puedan 
    indicar problemas en el proceso de fabricación o fallos en los equipos.
    """)

    st.markdown("""
    ### **Parte 1: Generación de Datos**

    Utiliza un programa en Python para generar datos simulados que representen las lecturas 
    de temperatura de los sensores durante varios meses. Los datos deben incluir una variedad 
    de lecturas normales, así como algunas lecturas anómalas que simulen problemas en el 
    proceso de fabricación o fallos en los equipos.
    """)

    # Definir el rango de fechas
    fechas = pd.date_range(start='1/1/2021', end='05/01/2021', freq='H')
    
    # Generar lecturas de temperatura normales con una media de 20 y una desviación estándar de 2
    temperaturas = np.random.normal(20, 2, size=len(fechas))
    
    # Añadir algunas lecturas anómalas
    for i in range(len(temperaturas)):
        if random.random() < 0.01:  # 1% de probabilidad de anomalía
            temperaturas[i] += np.random.normal(0, 10)  # Añadir ruido con media 0 y desviación estándar 10
    
    # Crear un DataFrame de pandas con los datos
    df = pd.DataFrame({'Fecha': fechas, 'Temperatura': temperaturas})
    
    # Mostrar el DataFrame
    st.write(df)

    st.markdown("""
    ### **Parte 2: Análisis de Datos**

    
    Una vez que hayas generado los datos, realiza un análisis para identificar posibles 
    anomalías en las lecturas de temperatura. Algunas preguntas que podrías explorar incluyen:
    
    - ¿Existen lecturas de temperatura que se desvíen significativamente del rango 
    esperado para esa área de la planta?
    
    - ¿Hay algún patrón o tendencia en las lecturas anómalas?
    
    - ¿Qué características tienen las lecturas anómalas en comparación 
    con las lecturas normales?
    """)

    # Utilizar Isolation Forest para detectar anomalías
    iso_forest = IsolationForest(contamination=0.02) # Suponemos que aproximadamente el 2% de los datos son anomalías
    anomalies = iso_forest.fit_predict(df[['Temperatura']])
    df['Anomaly'] = anomalies == -1
    
    # Interfaz de usuario
    st.title("Detección de Anomalías en Temperaturas Diarias")
    st.write("A continuación se muestra un gráfico de las temperaturas diarias con anomalías detectadas:")
    
    # Gráfico de transacciones y anomalías
    plt.figure(figsize=(15, 6))
    plt.plot(df['Fecha'].tolist(), df['Temperatura'].tolist(), label='Temperatura')
    plt.scatter(df.loc[df['Anomaly'], 'Fecha'].tolist(), df.loc[df['Anomaly'], 'Temperatura'].tolist(),
                color='red', label='Anomalía', marker='x', s=100)  # Marcar anomalías con una X roja
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de Temperatura')
    plt.title('Temperaturas Diarias con Anomalía Detectada')
    plt.legend()
    plt.grid(True)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)    

    st.write("Las Temperaturas anómalas son:")
    df_anomalias = pd.DataFrame(df.loc[df['Anomaly']])
    st.write(df_anomalias)

    st.write("Información de las Temperaturas:")
    st.write(df_anomalias['Temperatura'].describe())


    st.markdown("""
    #### **Análisis**

    Podemos ver en la gráfica que a lo largo de 4 meses las temperaturas más normales 
    rondan entre 16 a 23 grados, por ende, los valores más alejados de ese rango pueden 
    considerarse como anomalías. Se puede apreciar que si hay más valores que se alejan 
    del rango, pero en la gráfica solo se contemplan como anomalías las que tienen una 'X'.
    
    Podemos apreciar que si hay lecturas que se desvian significativamente del rango esperado, 
    en este caso son 58 las anomalías presentes en las gráficas, curiosamente a pesar de ser 
    valores anómalos, como hay valores de temperaturas muy bajos y muy altos, 
    lograron hacer una media de 19.2753 para los valores anómalos. 
    El valor anómalo más bajo que se registró fue de 2.5466 grados y el valor más alto anómalo 
    que se registró es 49.0574 grados.
    
    En cuanto a si hay un patrón o tendencia en los valores anómalos, 
    por encima de la media de todos los datos la mayoría de valores anómalos se muestran 
    entre 24 a 26, de ahí hay valores más altos que sobrepasan los 40 grados. 
    Por debajo de la media la mayoría de valores ronda desde los 14 a los 10 grados, 
    habiendo anomalías de hasta 2 grados.
    """)

    st.markdown("""
    ### **Parte 3: Informe de Resultados**

    Escribe un informe que resuma tus hallazgos. 
    Incluye gráficos o visualizaciones que ayuden a identificar las anomalías en los datos 
    de temperatura. Además, discute cualquier insight o conclusión que hayas obtenido del 
    análisis de los datos y cómo podrían utilizarse para mejorar el mantenimiento preventivo o 
    la eficiencia en la planta de manufactura.
    """)

    st.markdown("""
    #### **Informe de Resultados**

    Uno pensaría que todo el tiempo la temperatura en la planta está en orden, 
    que sería una temperatura aproximada de 20 grados, pero gracias a este análisis y a está 
    gráfica generada con el algoritmo Isolation Forest podemos apreciar que a lo largo del 
    tiempo, en este caso 4 meses, hay anomalías en la temperatura, datos que están por 
    debajo y por encima del rango de temperatura normal en la planta.
    
    Los datos anómalos registrados por debajo del rango normal, 
    ronda de los 14 grados a los 2 grados; y los datos anómalos por encima del rango normal, 
    ronda de los 24 a los 49 grados, estás anomalías son peligrosas ya que si la planta se 
    debe de mantener en una cierta temperatura, el que existan temperaturas muy por debajo y 
    muy por arriba puede provocar un problema en la línea de producción.
    
    #### Conclusión
    
    Estas anomlías en las temperaturas puede ser un gran riesgo para la línea de producción, 
    por lo que al saber que existe este problema debemos de solucionarlo, 
    y para eso, en vez de solo tener un aire acondicionado que lo pongas en 20 grados, 
    podríamos hacer un Agente Inteligente que se encargue de recibir la temperatura del 
    ambiente y si la temperatura empieza a salir del rango normal, que empiece a sacar más 
    aire más frío para que se regule en la temperatura deseada.
    """)

    #Media
    media = np.mean(df['Temperatura'])
    st.write("La media de la temperatura es: {:.2f}".format(media))

    # Calcular la media móvil de un lado
    media_movil_un_lado = df['Temperatura'].rolling(window=3, min_periods=1).mean()
    
    # st.write("Media Móvil de Un Lado:", media_movil_un_lado.tolist())


    # Utilizar Isolation Forest para detectar anomalías
    iso_forest = IsolationForest(contamination=0.02) # Suponemos que aproximadamente el 2% de los datos son anomalías
    anomalies = iso_forest.fit_predict(df[['Temperatura']])
    df['Anomaly'] = anomalies == -1
    
    # Interfaz de usuario
    st.title("Detección de Anomalías en Temperaturas Diarias")
    st.write("A continuación se muestra un gráfico de las temperaturas diarias con anomalías detectadas:")
    
    # Gráfico de transacciones y anomalías
    plt.figure(figsize=(15, 6))
    plt.plot(df['Fecha'].tolist(), df['Temperatura'].tolist(), label='Temperatura')
    plt.scatter(df.loc[df['Anomaly'], 'Fecha'].tolist(), df.loc[df['Anomaly'], 'Temperatura'].tolist(),
                color='red', label='Anomalía', marker='x', s=100)  # Marcar anomalías con una X roja
    plt.plot(df['Fecha'].tolist(), media_movil_un_lado, label='Media Móvil de un lado')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de Temperatura')
    plt.title('Temperaturas Diarias con Anomalía Detectada')
    plt.legend()
    plt.grid(True)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)  

    

elif option == 'Análisis de Estacionariedad':
    st.title('Ayuda')
    st.write('¡Aquí puedes encontrar ayuda y soporte!')
